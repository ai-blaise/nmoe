"""Persistent Dispatch Mode for Decode.

Task 5.2.8: Implement persistent kernel-like behavior for decode phase
to minimize kernel launch overhead.

This implementation uses a different approach than true persistent kernels:
- Pre-allocated work queues
- Double-buffered inputs/outputs
- Overlapped compute and data transfer
- Event-based synchronization
"""

import torch
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass


def _canonical_device(device: Union[str, torch.device]) -> torch.device:
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return d


@dataclass
class PersistentWorkItem:
    """Work item for persistent dispatch."""
    x: torch.Tensor          # Input hidden states
    eid: torch.Tensor        # Expert IDs
    gates: torch.Tensor      # Gating weights
    W1: torch.Tensor         # Gate projection weights [E, H, Dff]
    W3: torch.Tensor         # Up projection weights [E, H, Dff]
    W2: torch.Tensor         # Down projection weights [E, Dff, H]
    output: torch.Tensor     # Pre-allocated output buffer
    done_event: torch.cuda.Event  # Signaled when work is complete
    W_cache: Optional[torch.Tensor] = None  # Weight cache for blockscaled


class PersistentDispatchQueue:
    """Queue-based persistent dispatch for decode.

    Instead of launching a new kernel for each decode step, this class
    manages a queue of work items with pre-allocated buffers and overlapped
    execution.

    Benefits:
    - Reduces kernel launch overhead
    - Enables pipelining of compute and memory operations
    - Better GPU utilization for small batches
    """

    def __init__(
        self,
        rdep,  # nmoe.rdep.Rdep instance
        max_batch_size: int = 64,
        dim: int = 4096,
        n_buffers: int = 2,  # Double buffering
    ):
        self.rdep = rdep
        self.max_batch_size = max_batch_size
        self.dim = dim
        self.n_buffers = n_buffers

        # Pre-allocate buffer pools
        self._input_buffers: List[torch.Tensor] = []
        self._eid_buffers: List[torch.Tensor] = []
        self._gate_buffers: List[torch.Tensor] = []
        self._events: List[torch.cuda.Event] = []
        self._topk: Optional[int] = None
        self._gates_dtype: Optional[torch.dtype] = None

        # Separate stream for async execution (initialized per device).
        self._compute_stream: Optional[torch.cuda.Stream] = None
        self._device: Optional[torch.device] = None

        # Buffer management
        self._current_buffer = 0
        self._initialized = False

    def initialize(self, device: Union[str, torch.device] = "cuda") -> None:
        """Initialize buffer pools.

        Args:
            device: Target device for buffers. Can be string ("cuda", "cuda:0")
                   or torch.device instance.
        """
        target_device = _canonical_device(device)
        rdep_device = getattr(self.rdep, "_device", None)
        if rdep_device is not None and target_device != rdep_device:
            raise RuntimeError(
                f"PersistentDispatchQueue device {target_device} must match RDEP device {rdep_device}."
            )
        if self._initialized:
            if self._device is not None and target_device != self._device:
                raise RuntimeError(
                    f"PersistentDispatchQueue already initialized on {self._device}, "
                    f"cannot reinitialize on {target_device}."
                )
            return

        self._device = target_device
        self._compute_stream = torch.cuda.Stream(device=target_device)

        for _ in range(self.n_buffers):
            self._input_buffers.append(
                torch.empty(self.max_batch_size, self.dim,
                           dtype=torch.bfloat16, device=target_device)
            )
            evt = torch.cuda.Event()
            # Mark buffer as initially available.
            evt.record(torch.cuda.current_stream(device=target_device))
            self._events.append(evt)

        self._initialized = True

    def _ensure_routing_buffers(self, topk: int, gates_dtype: torch.dtype) -> None:
        if self._device is None:
            raise RuntimeError("PersistentDispatchQueue must be initialized before routing buffer setup.")
        if self._topk is None:
            self._topk = int(topk)
            self._gates_dtype = gates_dtype
            for _ in range(self.n_buffers):
                self._eid_buffers.append(
                    torch.empty(
                        self.max_batch_size,
                        self._topk,
                        dtype=torch.int32,
                        device=self._device,
                    )
                )
                self._gate_buffers.append(
                    torch.empty(
                        self.max_batch_size,
                        self._topk,
                        dtype=self._gates_dtype,
                        device=self._device,
                    )
                )
            return
        if int(topk) != self._topk:
            raise RuntimeError(
                f"PersistentDispatchQueue topk mismatch: expected {self._topk}, got {int(topk)}."
            )
        if gates_dtype != self._gates_dtype:
            raise RuntimeError(
                f"PersistentDispatchQueue gates dtype mismatch: expected {self._gates_dtype}, got {gates_dtype}."
            )

    def dispatch_async(
        self,
        x: torch.Tensor,
        eid: torch.Tensor,
        gates: torch.Tensor,
        W1: torch.Tensor,
        W3: torch.Tensor,
        W2: torch.Tensor,
        W_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.cuda.Event]:
        """Submit work and return immediately with output buffer and event.

        Args:
            x: [T, D] input hidden states
            eid: [T, K] expert IDs
            gates: [T, K] gating weights
            W1: [E, H, Dff] gate projection weights
            W3: [E, H, Dff] up projection weights
            W2: [E, Dff, H] down projection weights
            W_cache: Pre-computed weight cache for blockscaled mode (optional)

        Returns:
            output: Output tensor from rdep dispatch (ready when event fires)
            event: CUDA event that will be recorded when dispatch completes
        """
        if not self._initialized:
            self.initialize(x.device)
        assert self._device is not None
        if x.device != self._device:
            raise RuntimeError(
                f"PersistentDispatchQueue is bound to {self._device}, got input on {x.device}."
            )
        for name, tensor in (
            ("eid", eid),
            ("gates", gates),
            ("W1", W1),
            ("W3", W3),
            ("W2", W2),
        ):
            if tensor.device != self._device:
                raise RuntimeError(
                    f"PersistentDispatchQueue expects {name} on {self._device}, got {tensor.device}."
                )
        if W_cache is not None and W_cache.device != self._device:
            raise RuntimeError(
                f"PersistentDispatchQueue expects W_cache on {self._device}, got {W_cache.device}."
            )

        T = x.shape[0]
        if T > self.max_batch_size:
            raise ValueError(f"Batch size {T} exceeds max {self.max_batch_size}")
        if eid.dim() != 2 or gates.dim() != 2:
            raise ValueError("eid and gates must be rank-2 tensors with shape [T, K].")
        if eid.shape[0] != T or gates.shape[0] != T:
            raise ValueError(
                f"eid/gates token dimension mismatch: x has T={T}, eid has {eid.shape[0]}, gates has {gates.shape[0]}."
            )
        K = int(eid.shape[1])
        if gates.shape[1] != K:
            raise ValueError(
                f"eid/gates top-k mismatch: eid has K={K}, gates has K={gates.shape[1]}."
            )
        self._ensure_routing_buffers(K, gates.dtype)

        # Get current buffer
        buf_idx = self._current_buffer
        self._current_buffer = (self._current_buffer + 1) % self.n_buffers

        # Keep dependency on device stream; avoid host blocking sync.
        cur_stream = torch.cuda.current_stream(device=x.device)
        cur_stream.wait_event(self._events[buf_idx])

        # Copy input to buffer (on current/default stream)
        self._input_buffers[buf_idx][:T].copy_(x)
        self._eid_buffers[buf_idx][:T, :K].copy_(eid)
        self._gate_buffers[buf_idx][:T, :K].copy_(gates)

        # Execute dispatch on separate stream
        # CRITICAL: Synchronize compute stream with current stream to ensure
        # the copy above completes before dispatch reads from the buffer
        self._compute_stream.wait_stream(cur_stream)

        assert self._compute_stream is not None
        with torch.cuda.stream(self._compute_stream):
            # Use the buffered input with all required weight tensors
            output = self.rdep.dispatch(
                self._input_buffers[buf_idx][:T],
                self._eid_buffers[buf_idx][:T, :K],
                self._gate_buffers[buf_idx][:T, :K],
                W1,
                W3,
                W2,
                W_cache,
            )
            # Record completion event
            self._events[buf_idx].record()

        return output, self._events[buf_idx]

    def dispatch_sync(
        self,
        x: torch.Tensor,
        eid: torch.Tensor,
        gates: torch.Tensor,
        W1: torch.Tensor,
        W3: torch.Tensor,
        W2: torch.Tensor,
        W_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Submit work and wait for result (convenience method)."""
        output, event = self.dispatch_async(x, eid, gates, W1, W3, W2, W_cache)
        torch.cuda.current_stream(device=output.device).wait_event(event)
        return output

    def flush(self) -> None:
        """Wait for all pending work to complete."""
        if self._compute_stream is None:
            return
        self._compute_stream.synchronize()


class PersistentDecodeRunner:
    """High-level interface for persistent decode mode.

    Wraps the dispatch queue with a simple interface for decode iterations.
    """

    def __init__(
        self,
        rdep,
        max_batch_size: int = 64,
        dim: int = 4096,
    ):
        self.queue = PersistentDispatchQueue(rdep, max_batch_size, dim)
        self._in_decode_mode = False

    def enter_decode_mode(self, device: Optional[Union[str, torch.device]] = None) -> None:
        """Enter persistent decode mode."""
        if device is None:
            device = "cuda"
        self.queue.initialize(device=device)
        self._in_decode_mode = True

    def exit_decode_mode(self) -> None:
        """Exit persistent decode mode."""
        self.queue.flush()
        self._in_decode_mode = False

    def decode_step(
        self,
        x: torch.Tensor,
        eid: torch.Tensor,
        gates: torch.Tensor,
        W1: torch.Tensor,
        W3: torch.Tensor,
        W2: torch.Tensor,
        W_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Execute one decode step.

        Args:
            x: [T, D] input hidden states
            eid: [T, K] expert IDs
            gates: [T, K] gating weights
            W1: [E, H, Dff] gate projection weights
            W3: [E, H, Dff] up projection weights
            W2: [E, Dff, H] down projection weights
            W_cache: Pre-computed weight cache for blockscaled mode (optional)

        Returns:
            [T, D] output tensor
        """
        if not self._in_decode_mode:
            # Auto-enter decode mode to avoid silently bypassing persistent path.
            self.enter_decode_mode(device=x.device)

        return self.queue.dispatch_sync(x, eid, gates, W1, W3, W2, W_cache)

    def __enter__(self):
        self.enter_decode_mode()
        return self

    def __exit__(self, *args):
        self.exit_decode_mode()
