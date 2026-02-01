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
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass
import threading
import queue


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
        self._output_buffers: List[torch.Tensor] = []
        self._events: List[torch.cuda.Event] = []

        # Separate stream for async execution
        self._compute_stream = torch.cuda.Stream()

        # Buffer management
        self._current_buffer = 0
        self._initialized = False

    def initialize(self, device: Union[str, torch.device] = "cuda") -> None:
        """Initialize buffer pools.

        Args:
            device: Target device for buffers. Can be string ("cuda", "cuda:0")
                   or torch.device instance.
        """
        if self._initialized:
            return

        # P3.2: Normalize device to torch.device for consistent handling
        if isinstance(device, str):
            device = torch.device(device)

        for _ in range(self.n_buffers):
            self._input_buffers.append(
                torch.empty(self.max_batch_size, self.dim,
                           dtype=torch.bfloat16, device=device)
            )
            self._output_buffers.append(
                torch.empty(self.max_batch_size, self.dim,
                           dtype=torch.bfloat16, device=device)
            )
            self._events.append(torch.cuda.Event())

        self._initialized = True

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
            output: Pre-allocated output buffer (will be filled when event fires)
            event: CUDA event that will be recorded when dispatch completes
        """
        if not self._initialized:
            self.initialize(x.device)

        T = x.shape[0]
        if T > self.max_batch_size:
            raise ValueError(f"Batch size {T} exceeds max {self.max_batch_size}")

        # Get current buffer
        buf_idx = self._current_buffer
        self._current_buffer = (self._current_buffer + 1) % self.n_buffers

        # Wait for previous use of this buffer to complete
        self._events[buf_idx].synchronize()

        # Copy input to buffer (on current/default stream)
        self._input_buffers[buf_idx][:T].copy_(x)

        # Execute dispatch on separate stream
        # CRITICAL: Synchronize compute stream with current stream to ensure
        # the copy above completes before dispatch reads from the buffer
        self._compute_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self._compute_stream):
            # Use the buffered input with all required weight tensors
            output = self.rdep.dispatch(
                self._input_buffers[buf_idx][:T],
                eid,
                gates,
                W1,
                W3,
                W2,
                W_cache,
            )
            # Copy to output buffer
            self._output_buffers[buf_idx][:T].copy_(output)
            # Record completion event
            self._events[buf_idx].record()

        return self._output_buffers[buf_idx][:T], self._events[buf_idx]

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
        event.synchronize()
        return output.clone()

    def flush(self) -> None:
        """Wait for all pending work to complete."""
        torch.cuda.current_stream().wait_stream(self._compute_stream)
        for event in self._events:
            event.synchronize()


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

    def enter_decode_mode(self) -> None:
        """Enter persistent decode mode."""
        self.queue.initialize()
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
            # Fallback to regular dispatch
            return self.queue.rdep.dispatch(x, eid, gates, W1, W3, W2, W_cache)

        return self.queue.dispatch_sync(x, eid, gates, W1, W3, W2, W_cache)

    def __enter__(self):
        self.enter_decode_mode()
        return self

    def __exit__(self, *args):
        self.exit_decode_mode()


def benchmark_persistent_vs_regular(
    rdep,
    batch_size: int = 32,
    dim: int = 256,
    n_experts: int = 8,
    topk: int = 2,
    intermediate_dim: int = 1024,
    n_iters: int = 100,
) -> dict:
    """Benchmark persistent mode vs regular dispatch.

    Args:
        rdep: Rdep instance
        batch_size: Number of tokens per batch
        dim: Hidden dimension
        n_experts: Number of experts
        topk: Number of experts per token
        intermediate_dim: MLP intermediate dimension
        n_iters: Number of benchmark iterations

    Returns:
        dict with regular_ms, persistent_ms, speedup
    """
    import time

    # Create test inputs
    x = torch.randn(batch_size, dim, dtype=torch.bfloat16, device='cuda')
    eid = torch.randint(0, n_experts, (batch_size, topk), device='cuda')
    gates = torch.softmax(torch.randn(batch_size, topk, device='cuda'), dim=-1).bfloat16()

    # Create expert weights
    W1 = torch.randn(n_experts, dim, intermediate_dim, dtype=torch.bfloat16, device='cuda')
    W3 = torch.randn(n_experts, dim, intermediate_dim, dtype=torch.bfloat16, device='cuda')
    W2 = torch.randn(n_experts, intermediate_dim, dim, dtype=torch.bfloat16, device='cuda')

    # Warmup regular
    for _ in range(10):
        rdep.dispatch(x, eid, gates, W1, W3, W2)
    torch.cuda.synchronize()

    # Benchmark regular
    start = time.perf_counter()
    for _ in range(n_iters):
        rdep.dispatch(x, eid, gates, W1, W3, W2)
    torch.cuda.synchronize()
    regular_time = (time.perf_counter() - start) / n_iters * 1000  # ms

    # Warmup persistent
    runner = PersistentDecodeRunner(rdep, max_batch_size=batch_size, dim=dim)
    with runner:
        for _ in range(10):
            runner.decode_step(x, eid, gates, W1, W3, W2)
    torch.cuda.synchronize()

    # Benchmark persistent
    with runner:
        start = time.perf_counter()
        for _ in range(n_iters):
            runner.decode_step(x, eid, gates, W1, W3, W2)
    torch.cuda.synchronize()
    persistent_time = (time.perf_counter() - start) / n_iters * 1000  # ms

    return {
        'regular_ms': regular_time,
        'persistent_ms': persistent_time,
        'speedup': regular_time / persistent_time if persistent_time > 0 else 0,
    }


if __name__ == "__main__":
    from nmoe.rdep import Rdep

    dim = 256
    n_experts = 8
    intermediate_dim = 1024
    topk = 2

    rdep = Rdep(dim=dim, n_local=n_experts, topk=topk, profile="bf16")
    results = benchmark_persistent_vs_regular(
        rdep,
        dim=dim,
        n_experts=n_experts,
        intermediate_dim=intermediate_dim,
        topk=topk,
    )

    print(f"Regular dispatch: {results['regular_ms']:.3f} ms")
    print(f"Persistent dispatch: {results['persistent_ms']:.3f} ms")
    print(f"Speedup: {results['speedup']:.2f}x")
