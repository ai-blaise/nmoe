# 16xB200 Speed Task Board (NVFP4 MoE)

Purpose:
- Track all high-priority speed work for 16-node x 8-GPU B200 training.
- Keep one explicit plan with measurable acceptance criteria.
- Prevent silent fallback paths and regression drift.
- Include rollout, logging, and infrastructure tracks so performance work is
  operable and debuggable in production.

Status legend:
- `[ ]` Todo
- `[~]` In progress
- `[x]` Done
- `[!]` Blocked

Scope:
- Kernels, launch/runtime, build flags, benchmark gating, rollout, and forensics.
- Primary path assumes no admin-side infrastructure changes.
- Separate admin-change track is included for TCPXO/MTU migration.

## A) Guardrails and No-Fallback Baseline

- [x] A-001 Enforce fused router in training entrypoint.
  - Files: `nmoe/train.py`, `nmoe/config.py`
  - Acceptance: launch fails if `use_fused_router=false`; model is created with `use_fused_router=true`.

- [x] A-002 Enforce launch guardrails for ECO CUDA path.
  - Files: `../nmoe-multinode/orchestrate.py`
  - Acceptance: launch fails unless `eco_enabled=true`, `eco_fused_backward=true`, `eco_require_cuda=true`, `use_fused_router=true`.

- [x] A-003 Pin production training config to no-fallback values.
  - Files: `configs/dsv3_reap_sft_16node.toml`
  - Acceptance: config includes `use_fused_router=true`, `eco_enabled=true`, `eco_fused_backward=true`, `eco_require_cuda=true`.

- [x] A-004 Propagate the new `nmoe` commit to all nodes and verify file-level parity.
  - Files: `../nmoe-multinode/orchestrate.py` flow (`provision`)
  - Acceptance: all nodes show same `git rev-parse HEAD` and expected guardrail code strings.
  - Status note: `uv run ./orchestrate.py provision --clone-only --force` completed 16/16 and strict parity verified all nodes at `662b162332bdf9117ee08507e41b11a7156daa07`.

- [x] A-005 Enforce NVFP4 no-fallback guard inside fused MoE autograd entrypoints.
  - Files: `nmoe/moe.py`
  - Acceptance: `_MoEBlockscaledFused.forward/backward` fail fast when `_nvfp4_primary=true` and fused ECO is not attached.
  - Status note: added explicit runtime guards in both forward and backward to forbid non-ECO NVFP4 primary execution outside higher-level model/train checks.

- [x] A-006 Enforce BF16 fused-router backward symbol at launch.
  - Files: `nmoe/train.py`
  - Acceptance: launch validation fails if `fused_router_backward_bf16` is missing from `rdep` shared object.
  - Status note: required router symbol list now includes `fused_router_backward_bf16`, preventing silent BF16->FP32 fallback drift from stale binaries.

- [x] A-007 Fail fast on distributed communicator warmup failures.
  - Files: `nmoe/train.py`
  - Acceptance: DP/EP warmup errors abort launch immediately (no continue-on-failure behavior).
  - Status note: DP and EP warmup `except` handlers now raise runtime errors under no-fallback policy.

- [x] A-008 Enforce TOML-only production training config input.
  - Files: `nmoe/train.py`
  - Acceptance: `nmoe.train` exits when config path is not `.toml`.
  - Status note: main entrypoint now rejects non-TOML paths before loading config.

- [x] A-009 Fail fast on negative RDEP dispatch return codes in MoE autograd paths.
  - Files: `nmoe/moe.py`
  - Acceptance: negative `dispatch_meta_*` return values raise runtime errors immediately instead of entering the `M_recv==0` collectiveness path.
  - Status note: both blockscaled forward and BF16 backward dispatch callsites now reject `<0` codes with detailed mode/world context; distributed `M_recv==0` backward path now preserves collective `dGate` contributions instead of silently returning zeros.

- [x] A-010 Enforce primary extension path loading for fused router/aux-loss kernels.
  - Files: `nmoe/fused_router.py`, `nmoe/fused_aux_loss.py`
  - Acceptance: loaders prefer the active `nmoe.csrc.rdep` extension path and only allow standalone legacy `.so` candidates via explicit opt-in env flags.
  - Status note: loaders are now strict-primary only with ABI/version checks (`rdep.abi_version()==1`) and writable-path rejection, and standalone env toggles hard-error in production (`NMOE_ROUTER_BWD_ALLOW_STANDALONE`, `NMOE_AUX_LOSS_ALLOW_STANDALONE`). Router backward still enforces BF16 symbol path (`fused_router_backward_bf16`) with FP32 fallback disabled.

- [x] A-011 Enable hybrid split-dGate backward path (remove unsupported-mode trap).
  - Files: `nmoe/moe.py`, `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`, `nmoe/csrc/rdep_nvshmem.cuh`
  - Acceptance: distributed backward uses split dGate path in both IPC and hybrid modes without `Ye_pad` dependency; no hybrid-mode aborts in split wrappers.
  - Status note: hybrid now supports `gather_dy_nogate_dist_bf16` + `send_dgate_dist_bf16_out_bf16`; MoE backward unified on split dGate flow so the undefined `Ye_pad` hybrid crash path is removed.

- [x] A-012 Require fused ECO in blockscaled MoE backward.
  - Files: `nmoe/moe.py`
  - Acceptance: blockscaled MoE backward cannot downshift into non-ECO weight-gradient path.
  - Status note: backward now hard-fails when `fused_eco` is missing.

- [x] A-013 Disable NVFP4 BF16 re-quantization fallback in cache refresh.
  - Files: `nmoe/model.py`
  - Acceptance: NVFP4-primary cache refresh fails fast when direct NVFP4 buffers are missing.
  - Status note: `refresh_weight_cache()` now hard-errors only for invalid `_nvfp4_primary` states without complete W1/W3/W2 NVFP4 buffer triplets; startup/scratch BF16 quantization path remains allowed.

- [x] A-014 Avoid invalid EP-group hard-fail in distributed load aggregation fallback.
  - Files: `nmoe/model.py`
  - Acceptance: distributed forward does not raise when EP group is absent (EP=1-style topology); EP all-reduce is only run when EP group exists.
  - Status note: router-load aggregation now conditionally all-reduces on EP group presence.

- [x] A-015 Canonicalize packed `cu_seqlens` for downstream MLA call path.
  - Files: `nmoe/model.py`
  - Acceptance: when packed mode is enabled, attention layers receive CUDA int32 contiguous `cu_seqlens` tensors (matching MLA varlen contract).
  - Status note: model forward now propagates canonicalized per-batch `cu_seqlens` tensors to block attention calls.

## B) Highest ROI Hot-Path Sync Removal

- [x] B-001 Remove per-layer GPU->CPU sync from MoE load CV metric.
  - Files: `nmoe/model.py` around load CV calculation.
  - Change: remove `.item()` in hot path; keep tensor on device and convert only in low-frequency logging.
  - Acceptance: no `.item()` in MoE forward hot path; p50 step-time improves; no metric correctness loss.
  - Status note: `MoE.forward()` no longer computes CV reductions at all; CV is derived from cached `last_loads` only when `mean_expert_load_cv` is queried (logging path), and converted to Python once there.

- [x] B-002 Remove Python spin wait in MoE forward for dispatch event.
  - Files: `nmoe/moe.py` around `while not evt.query()`.
  - Change: avoid host polling if C path already synchronizes/guarantees readiness.
  - Acceptance: no busy-wait loop in forward; same outputs and no race failures.
  - Status note: busy-spin query loops removed; forward now reads pinned `M_pad` directly without extra Python stream synchronization, and `offs_pad` device buffers are reused instead of per-call allocation.

- [~] B-003 Remove host sync and D2H metadata reads in RDEP dispatch.
  - Files: `nmoe/csrc/rdep.cu` dispatch metadata path.
  - Change: keep `M_recv/M_pad` on device; remove forced stream sync + host reads from hot path.
  - Acceptance: no D2H metadata transfer/sync in trace during steady-state; p50 and p95 step-time improve.
  - Status note: BF16 2-phase now computes `M_recv` on GPU and reads back one int instead of copying/summing `recv_counts[world]`; blockscaled IPC path is now also 2-phase by default; `dispatch_meta_bf16` now uses 2-phase in multi-rank IPC instead of legacy remote-atomic dispatch. 2-phase `M_recv` handoff is now pinned-host-only with hard validation (`cudaHostGetFlags`) and pageable fallback stack copies removed in IPC/NVSHMEM paths, with validation performed at the D2H handoff point (after required collectives) to avoid rank-asymmetric barrier hangs. Negative dispatch return codes are now propagated before zero-token fast paths in BF16/blockscaled dispatch APIs. Hybrid path removed one host D2H counter read by using device-side dynamic forwarding/merge counters in NVSHMEM dispatch. Additional hot-path cleanup applied (`eids/gates` direct flat indexing and blockscaled dispatch write-order fence), and blockscaled path no longer does host-scalar H2D patching for `offs_pad[last]` (device-side tail write kernel). Runtime toggle drift removed by locking 2-phase dispatch to a compile-time constant in `rdep.cu`. Hybrid BF16/blockscaled paths now use deterministic aligned `M_pad` bounds (device tail patch) to avoid stale async `M_pad` handoff and remove the second host sync. BF16 IPC path now also removes exact-`M_pad` D2H+sync by using deterministic aligned `M_pad` bound + device tail patch. Added explicit `eid/dest/local_eid` bounds guards in 2-phase dispatch kernels to prevent OOB writes from invalid routing IDs. Also removed unnecessary full-buffer memset/prefill in blockscaled gather/materialization when `M_pad == M_recv` (padding-free case). Full host-sync elimination remains (`M_recv` handoff still host-visible).

- [~] B-004 Remove host wait gate in MoE backward offset handoff.
  - Files: `nmoe/moe.py`, `nmoe/csrc/gemm.cu`, `nmoe/csrc/bindings.cpp`
  - Change: eliminate `copy_event.synchronize()` by switching BF16 wgrad entrypoints to fully device-resident offset metadata.
  - Acceptance: no host event wait in backward hot path; stable numerics and lower CPU idle in traces.
  - Status note: Python-side `copy_event` wait/copy-stream path is removed; backward now stages `offs_pad` through `bf16_prepare_offs_pad_host(...)` and hot grouped-wgrad calls use host-offs entrypoints directly (`bf16_wgrad_*_host_offs`). Full device-resident grouped metadata build is still pending.

- [x] B-012 Guard router-load post-processing when load tracking is disabled.
  - Files: `nmoe/model.py`, `nmoe/opt.py`, `nmoe/metrics.py`
  - Acceptance: no `last_loads=None` exception/log churn in post-step bias update or router metrics collection.
  - Status note: MoE now only populates `last_loads` when router-bias updates are enabled; post-step `router.update_bias` skips when disabled/no-loads; router metrics collector now guards `last_loads is not None`.

- [x] B-013 Reduce ECO async queue polling overhead with tail-work completion sentinel.
  - Files: `nmoe/eco.py`
  - Acceptance: no per-entry `all(work.is_completed() ...)` scans in steady-state queue checks.
  - Status note: pending all-reduce entries now track `tail_work`; `_drain_completed` checks one completion handle and `_drain_one` waits on the sentinel only.

- [x] B-014 Cache ZeRO-2 timer context and dtype buckets; remove repeated hot-path lookups.
  - Files: `nmoe/zero2.py`
  - Acceptance: no per-step timer import path and reduced repeated dtype bucketing/call overhead.
  - Status note: `cuda_time` resolver is now cached, dtype grouping is cached per param-group, AdamW chunk path hoists stream/pointer stride values, and chunk iteration avoids temporary list allocation.

- [x] B-015 Optimize expert DP gradient reduction and fail fast on stale post-hook states.
  - Files: `nmoe/opt.py`
  - Acceptance: expert gradient all-reduce uses coalesced collective when available; post-step quant/bias hook failures are hard errors.
  - Status note: added `all_reduce_coalesced` fast path for expert grads and converted post-hook warning-and-continue behavior into fail-fast runtime errors.

- [x] B-016 Normalize legacy BF16 dispatch counter readback to stream-ordered D2H.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: no blocking default-stream `cudaMemcpy` in legacy dispatch branch.
  - Status note: legacy path now uses caller-provided host scratch + `cudaMemcpyAsync(..., stream)` followed by one explicit stream sync.

- [x] B-006 Cache MoE distributed-path capability checks in module globals.
  - Files: `nmoe/moe.py`
  - Acceptance: no repeated `dist.get_world_size()` / `hasattr(_C, ...)` checks in forward/backward steady-state path.
  - Status note: `is_dist` now uses `rdep.world > 1`, NVTX gating is cached once, and required BF16 dGate bindings are validated via cached module booleans.

- [x] B-007 Cache NVTX enable checks once per module (remove per-range env/capability parsing).
  - Files: `nmoe/model.py`, `nmoe/moe.py`, `nmoe/eco.py`, `nmoe/attention/mla.py`, `nmoe/rdep.py`, `nmoe/fused_router.py`
  - Acceptance: no hot-path `os.getenv('NMOE_NVTX', ...)` / `hasattr(torch.cuda.nvtx, ...)` checks inside each `_nvtx(...)` invocation.
  - Status note: all key training hot modules now compute `_NVTX_ENABLED` at import and use fast branch-only `_nvtx` helpers.

- [x] B-008 Remove duplicated per-step model parameter scans in optimizer section.
  - Files: `nmoe/train.py`
  - Acceptance: single `params_with_grad` list is reused for grad norm + grad clip path.
  - Status note: train step now computes `_grad_params` once, uses `torch.isfinite(grad_norm).item()` for robust NaN/Inf guard, and avoids a second full `model.parameters()` walk in fused clip flow.

- [x] B-009 Hoist blockscaled expert compute import out of MoE forward hot path.
  - Files: `nmoe/moe.py`
  - Acceptance: no import statement executed inside `MoE` forward pass.
  - Status note: `expert_blockscaled` now imports at module load instead of inside `_MoEBlockscaledFused.forward`.

- [x] B-010 Prebind RDEP dispatch implementation and remove per-call profile branch.
  - Files: `nmoe/rdep.py`
  - Acceptance: `Rdep.dispatch()` calls preselected dispatch impl (`bf16` vs blockscaled) instead of branching on profile each invocation.
  - Status note: `_dispatch_impl` is now selected in `Rdep.__init__`, and `dispatch()` calls it directly.

- [x] B-011 Cache MoE layer list and reuse it across model hot/near-hot paths.
  - Files: `nmoe/model.py`
  - Acceptance: no repeated per-call MoE discovery list comprehensions in `forward`, stats, and runtime counter aggregation.
  - Status note: `Transformer` now stores `self._moe_layers` once at init and reuses it in load aggregation, dropped-token stats, aux-loss collection, and runtime counters.

- [~] B-005 Remove remaining host stream synchronizations from steady-state RDEP paths.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Change: replace host `cudaStreamSynchronize` waits with stream-ordered device-side handoff/barrier/counter logic where correctness allows.
  - Acceptance: no avoidable host stream synchronization in dispatch/return/backward hot path traces.
  - Status note: removed explicit host stream waits from hybrid backward hot path (`gather_dy_hybrid_bf16` and `scatter_dx_hybrid_bf16`) and kept execution fully stream-ordered via `quiet + hybrid_barrier_on_stream`. Dispatch now also avoids forced stream synchronize in meta-only hybrid mode (sync retained only when `Xe` materialization is requested). Hybrid return-scatter paths now use device-counter dynamic forwarding/scatter kernels (removed host D2H `nvshmem_ret/ipc_ret` reads), and hybrid barrier helper now skips no-op IPC/NVSHMEM barriers when single local rank or single node. Dispatch-side `M_pad` host-sync handoff was removed via deterministic bound; remaining hot host sync point is `M_recv` handoff in dispatch. Added pinned-host hard validation for hybrid `M_pad_out` handoff to prevent pageable async-copy stalls. Tightened backward pre-memset synchronization scope in hybrid path by downgrading local IPC tok-gate clear barriers from hybrid-wide to local IPC-only barriers.

- [x] B-017 Remove redundant gate cast round-trip and reduce packed cu_seqlens host handoff sync points.
  - Files: `nmoe/moe.py`, `nmoe/attention/mla.py`
  - Acceptance: no `gates BF16 -> FP32` conversion chain in MoE forward dispatch metadata path; packed MLA does a single batched starts/totals host transfer and a single optional monotonic validation sync.
  - Status note: blockscaled MoE forward now uses `gates.detach().contiguous().float()` directly; packed FlashMLA path combines starts/totals D2H into one transfer and consolidates monotonic checks into one terminal validation sync.

- [x] B-018 Trim IPC/NVSHMEM barrier and memcpy overhead in steady-state dispatch paths.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: IPC barrier launch uses local-world-sized CTA (not fixed 256), direct IPC write no longer falls back to blocking `cudaMemcpy`, and no full Xe zero-fill when `M_pad == M_recv`.
  - Status note: IPC barriers now launch with dynamic thread count (`32..256`) from peer-count; `rdep_direct_ipc_write` now always uses `cudaMemcpyAsync(..., stream ? stream : 0)`; BF16 gather/materialization zero-fill is now conditional on real padding, and BF16/blockscaled gather paths now clear/fill only alignment gap rows (derived from `offsets` + `offs_pad`) instead of full `[M_pad,*]` memsets for multi-expert layouts.

- [x] B-019 Restore strict dropped-token accounting and routing bounds safety in hybrid/2-phase dispatch.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: BF16 2-phase dispatch increments dropped counters on overflow; hybrid dispatch rejects invalid `eid/dest/local_eid` before pointer/PE access.
  - Status note: BF16 2-phase now threads `dropped_off` into kernel path and increments dropped on capacity overflow; hybrid BF16/blockscaled kernels now guard negative/out-of-range expert IDs and invalid local expert IDs before IPC/NVSHMEM writes.

- [x] B-020 Restore expert-load CV metric scale for normalized load tensors.
  - Files: `nmoe/model.py`
  - Acceptance: `mean_expert_load_cv` reflects real coefficient of variation when loads are normalized probabilities.
  - Status note: CV denominator now clamps with epsilon (`1e-12`) instead of `1.0`, avoiding order-of-magnitude under-reporting.

- [x] B-021 Remove packed RoPE position-id GPU scalar syncs from transformer forward.
  - Files: `nmoe/model.py`
  - Acceptance: packed `cu_seqlens` validation avoids `cuda` scalar `.item()` in model hot path.
  - Status note: per-sample `cu_seqlens` checks now run on CPU mirrors, with one non-blocking H2D transfer for `searchsorted` only.

- [x] B-022 Set RDEP default profile to production NVFP4.
  - Files: `nmoe/rdep.py`
  - Acceptance: new `Rdep(...)` instances default to `profile='nvfp4'` (no stale BF16 default).
  - Status note: constructor default and in-file usage examples now target NVFP4.

- [x] B-023 Collapse packed FlashMLA `cu_seqlens` micro-copies into a single merged transfer.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: no per-sample tiny `cu_seqlens -> cuda` transfers in packed FlashMLA path.
  - Status note: merged `cu_seqlens` now builds on CPU and performs one non-blocking H2D copy for the varlen launch.

- [x] B-024 Remove host-blocking event synchronizations from persistent decode dispatch path.
  - Files: `nmoe/persistent_dispatch.py`
  - Acceptance: persistent `dispatch_async/dispatch_sync` do not call host `Event.synchronize()` in steady-state decode path.
  - Status note: switched to stream `wait_event` ordering, removed redundant output-buffer copy in async path, and auto-enters decode mode on first `decode_step()` to avoid silent fallback bypass.

- [x] B-025 Add single-rank fast-exit in legacy BF16 return-scatter path.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: legacy return-scatter path skips mailbox drain/D2H counter handoff when `world==1`.
  - Status note: added early return after primary scatter kernel in single-rank mode.

- [x] B-026 Harden 2-phase count/write handoff and trim zero-value global atomics.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: count/write phase has an explicit cross-rank barrier when `recv_counts` is reused for send/recv staging; shared->global count reduction avoids `atomicAdd(..., 0)`.
  - Status note: removed send/recv mailbox aliasing by staging send counts in `local_counters` (BF16 + blockscaled), then publishing into `recv_counts`; also guarded shared->global count atomics with non-zero checks.

- [x] B-027 Enforce persistent decode single-device stream correctness.
  - Files: `nmoe/persistent_dispatch.py`
  - Acceptance: queue rejects mixed-device tensor inputs; compute stream waits on device-correct current stream (no implicit default-device stream usage).
  - Status note: added strict per-tensor device invariants and switched to `torch.cuda.current_stream(device=x.device)` for copy->compute dependencies.

- [x] B-028 Make CUDA graph capture/replay stream/device contracts explicit.
  - Files: `nmoe/rdep.py`
  - Acceptance: graph warmup/capture uses device-scoped streams, avoids extra host synchronize, and replay rejects mismatched device/shape/dtype inputs.
  - Status note: warmup stream now binds to `x.device`, capture uses stream dependency instead of host synchronize, and replay enforces strict input contracts to prevent hidden copy/downshift paths.

- [x] B-031 Reset IPC barrier signal slots during alloc/rebind.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: reusing an existing shared IPC buffer cannot leave stale barrier signal values that bypass low-phase waits after phase counter reset.
  - Status note: both BF16 and blockscaled alloc paths now clear local `barrier_off` signal slots (`MAX_RANKS` ints).

- [x] B-032 Validate blockscaled pack stride divisibility at alloc time.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: blockscaled alloc fails fast when `H` is incompatible with active pack factor (`fp8:2`, `nvfp4:4`) instead of allowing implicit floor division behavior.
  - Status note: added hard alloc-time guard on `H % pack_factor == 0`.

- [x] B-033 Enforce hybrid bootstrap group topology contract.
  - Files: `nmoe/rdep.py`
  - Acceptance: hybrid init fails fast unless EP group rank/world match global rank/world (no subgroup hybrid bootstrap ambiguity).
  - Status note: `_setup_hybrid()` now hard-rejects non-global EP groups for hybrid mode.

- [x] B-034 Pin IPC-handle exchange tensors to explicit CUDA device.
  - Files: `nmoe/rdep.py`
  - Acceptance: IPC handle int32 tensors are created on explicit current CUDA device (no implicit `.cuda()` default-device placement drift).
  - Status note: `_ipc_handle_to_int32` now accepts explicit `device` and `_setup_ipc` passes one canonical CUDA device.

- [x] B-035 Enforce global-world EP topology for multi-rank IPC/hybrid mode.
  - Files: `nmoe/rdep.py`
  - Acceptance: multi-rank (`ipc|hybrid`) RDEP init fails fast unless EP group rank/world match global rank/world mapping.
  - Status note: added init-time contract check and mirrored hybrid bootstrap guard.

- [x] B-036 Scope CUDA-graph capture/replay to captured device context.
  - Files: `nmoe/rdep.py`
  - Acceptance: CUDA graph warmup/capture/replay execute under explicit `torch.cuda.device(x.device)` context.
  - Status note: capture warmup/graph sections and replay now explicitly enter the captured device context.

- [x] B-037 Validate local-rank/device binding at RDEP init.
  - Files: `nmoe/rdep.py`
  - Acceptance: when `LOCAL_RANK` is set, RDEP init fails fast if current CUDA device index does not match.
  - Status note: constructor now checks `LOCAL_RANK` against `torch.cuda.current_device()` and aborts on mismatch.

- [x] B-038 Remove hybrid local-handle rank-contiguity assumption.
  - Files: `nmoe/rdep.py`
  - Acceptance: hybrid local IPC handle selection no longer depends on `rank // local_world`; it derives node-local peers from gathered hostnames and validates local handle count.
  - Status note: `_setup_hybrid()` now all-gathers hostnames, filters handles by hostname equality, and validates expected `local_world` count.

- [x] B-039 Enforce RDEP dispatch tensor-device invariants.
  - Files: `nmoe/rdep.py`
  - Acceptance: `Rdep.dispatch()` fails fast when `x/eid/gates/W*/W_cache` device differs from the initialized RDEP device.
  - Status note: added explicit per-input device guards at dispatch entry.

- [x] B-040 Order hybrid local IPC handles by gathered `LOCAL_RANK`.
  - Files: `nmoe/rdep.py`
  - Acceptance: hybrid local handle list passed to NVSHMEM open is ordered by unique local rank indices and validated for completeness.
  - Status note: hybrid bootstrap now all-gathers `LOCAL_RANK`, validates uniqueness/range, and sorts local handles by local rank before open.

- [x] B-041 Bind persistent decode queue device to RDEP device.
  - Files: `nmoe/persistent_dispatch.py`
  - Acceptance: persistent queue init rejects device values that do not match `rdep._device`.
  - Status note: added init-time queue/RDEP device consistency guard.

- [x] B-042 Stage decode routing tensors (`eid/gates`) in persistent queue-owned buffers.
  - Files: `nmoe/persistent_dispatch.py`
  - Acceptance: async decode dispatch does not consume caller-owned `eid/gates` directly on compute stream; queue stages them in ring buffers with fixed `topk`/dtype contract.
  - Status note: added per-buffer `eid/gates` staging tensors and strict `topk`/dtype invariants.

- [x] B-043 Initialize/open blockscaled NVSHMEM hybrid path explicitly.
  - Files: `nmoe/rdep.py`
  - Acceptance: hybrid mode with `profile in {fp8,nvfp4}` allocates blockscaled NVSHMEM state and opens/syncs blockscaled IPC handles explicitly.
  - Status note: hybrid bootstrap now calls `nvshmem_alloc_blockscaled` and `nvshmem_open_ipc_handles_blockscaled`/sync alongside BF16 path.

- [x] B-044 Reject unsupported BF16 RDEP profile at construction.
  - Files: `nmoe/rdep.py`
  - Acceptance: `Rdep(profile='bf16')` fails fast at init (instead of deferring to runtime dispatch failure).
  - Status note: constructor now raises for BF16 profile and enforces `fp8|nvfp4` production contract.

- [x] B-045 Enforce persistent decode routing-shape consistency.
  - Files: `nmoe/persistent_dispatch.py`
  - Acceptance: `dispatch_async` fails fast when `eid/gates` shapes are not `[T,K]`-consistent (no silent gate slicing/truncation).
  - Status note: added rank/shape/token/top-k checks before staging `eid/gates`.

- [x] B-046 Make MoE offs-copy cache keys device-aware.
  - Files: `nmoe/moe.py`
  - Acceptance: cached offs-copy events and pinned offs buffers are keyed by `(device, stream)` (no cross-device stream-id aliasing).
  - Status note: `_get_cached_offs_copy_event` and `_get_cached_pinned_offs` now include CUDA device id in cache key.

- [x] B-047 Fail fast on unsupported hybrid `n_local` at Python init.
  - Files: `nmoe/rdep.py`
  - Acceptance: hybrid mode initialization aborts before CUDA launch when `n_local > 256`.
  - Status note: constructor now raises with explicit guidance for hybrid mode if local expert count exceeds current padded-mapping kernel limit.

- [x] B-048 Require coalesced expert DP all-reduce API in production path.
  - Files: `nmoe/opt.py`
  - Acceptance: expert DP sync fails fast when `torch.distributed.all_reduce_coalesced` is unavailable (no per-gradient fallback loop).
  - Status note: removed per-gradient all-reduce fallback branch in expert gradient reduction.

- [x] B-049 Move MoE offs D2H host wait to wgrad point-of-use.
  - Files: `nmoe/moe.py`
  - Acceptance: backward does not block on offs-copy readiness before dGate/recompute work; host wait occurs immediately before first wgrad call.
  - Status note: `copy_event.synchronize()` moved from pre-wgrad region to W2 wgrad launch boundary.

- [x] B-050 Clamp MoE backward `max_pad` to dispatcher-safe bound.
  - Files: `nmoe/moe.py`
  - Acceptance: backward gather path does not request `M_pad` beyond dispatcher max bound (`capacity + E*(align-1)`).
  - Status note: added max-bound clamp and sanity guard (`max_pad >= M_recv`) before gather/meta usage.

- [x] B-051 Use per-stream pinned host scratch for dispatch `M_pad/M_recv` handoff.
  - Files: `nmoe/moe.py`
  - Acceptance: MoE forward/backward dispatch metadata no longer shares one global pinned scalar across concurrent CUDA streams.
  - Status note: added `_get_cached_pinned_m_host(...)` keyed by `(device, stream)` and switched blockscaled forward/BF16 backward dispatch metadata calls to use stream-scoped pinned host scratch.

- [x] B-052 Unblock hybrid blockscaled backward from BF16-profile metadata assumptions.
  - Files: `nmoe/moe.py`, `nmoe/csrc/rdep.cu`, `nmoe/csrc/bindings.cpp`, `nmoe/csrc/quant.cu`
  - Acceptance: hybrid blockscaled backward does not call BF16-only hybrid metadata/gather entrypoints; it gathers blockscaled activations and dequantizes to BF16 on-GPU before grouped BF16 backward math.
  - Status note: hybrid backward now uses `dispatch_meta_blockscaled` + `gather_xe_blockscaled` + new CUDA bindings (`dequant_fp8_to_bf16`, `dequant_nvfp4_to_bf16`); hybrid meta/pad helper wrappers (`gather_meta_sorted_bf16`, `gather_from_pad_bf16`, `scatter_sorted_to_pad_bf16`) are profile-agnostic.

- [!] B-053 Enable distributed IPC blockscaled metadata path in backward.
  - Files: `nmoe/moe.py`
  - Acceptance: distributed backward in IPC mode with blockscaled profiles (`fp8`/`nvfp4`) does not downshift to `dispatch_meta_bf16` + `gather_xe_bf16`.
  - Status note: blocked for correctness now: IPC backward metadata consumers are BF16-state keyed (`g_bf16.dest/order`); enabling blockscaled dispatch metadata in IPC without matching blockscaled gather-meta/gather-dy/scatter-dx consumers can misroute gradients.

- [x] B-054 Remove IPC `dX_sorted` staging copy in distributed backward.
  - Files: `nmoe/moe.py`, `nmoe/csrc/rdep.cu`, `nmoe/csrc/bindings.cpp`
  - Acceptance: IPC distributed backward can scatter input gradients directly from `dX_pad` without `gather_from_pad_bf16` temporary.
  - Status note: added `scatter_dx_dist_from_pad_bf16` (IPC fast path), with Python call-site switching to direct padded scatter when binding is available.

- [x] B-055 Gate EP load all-reduce out of the forward hot path by default.
  - Files: `nmoe/model.py`
  - Acceptance: router-load EP all-reduce is no longer unconditional in every training forward pass.
  - Status note: EP load aggregation is now opt-in via `NMOE_ROUTER_LOADS_EP_ALLREDUCE=1`; default path keeps local counters and avoids per-step sync/cast overhead.

- [x] B-056 Enforce IPC distributed `dX` no-fallback fast path.
  - Files: `nmoe/moe.py`
  - Acceptance: IPC distributed backward cannot silently downshift into `gather_from_pad_bf16 + scatter_dx_dist_bf16` when direct padded scatter binding is missing.
  - Status note: mode `ipc` now hard-requires `scatter_dx_dist_from_pad_bf16` and raises on missing binding; sorted-path scatter remains hybrid-only.

- [x] B-057 Guard hybrid blockscaled dispatch on NVSHMEM blockscaled profile state.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: hybrid blockscaled dispatch returns explicit error when NVSHMEM is initialized in BF16 profile state.
  - Status note: added profile guard (`profile < 0 -> return -2`) in both `rdep_dispatch_meta_blockscaled` and `rdep_dispatch_blockscaled` hybrid entrypoints.

- [x] B-058 Remove redundant BF16 hybrid bootstrap allocation/IPC-handle exchange for blockscaled runs.
  - Files: `nmoe/rdep.py`
  - Acceptance: hybrid setup in `fp8`/`nvfp4` profile does not allocate BF16 NVSHMEM buffers or perform BF16 IPC-handle all-gather/open/sync.
  - Status note: `_setup_hybrid` now allocates/exchanges only the active profile path; blockscaled profile performs blockscaled-only IPC handle open/sync.

- [x] B-059 Keep dropped-token safety checks available with low-overhead sampling (opt-in).
  - Files: `nmoe/moe.py`, `nmoe/csrc/rdep.cu`
  - Acceptance: dropped-token guard is opt-in (`NMOE_CHECK_DROPPED_TOKENS=1`) and sampled to avoid per-step host-sync overhead.
  - Status note: default remains disabled on production hot path; when enabled, sampling uses configurable interval (`NMOE_CHECK_DROPPED_TOKENS_INTERVAL`, default `1000`) and dropped-counter reads use async event-polled D2H telemetry in `rdep_get_dropped_*` (no `cudaStreamSynchronize` on caller stream).

- [x] B-060 Remove host-synchronized `M_ret` readback in legacy BF16 return path.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: return scatter in distributed legacy BF16 path does not perform D2H + `cudaStreamSynchronize` to obtain receive count.
  - Status note: added `k_scatter_received_bf16_dynamic` that reads `counter_off` on-device and clamps to capacity, eliminating the host sync in `rdep_return_scatter`.

- [x] B-061 Cache pinned-host scratch validation in dispatch hot path.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: steady-state dispatch does not call `cudaHostGetFlags` repeatedly for the same pinned host scalar pointers.
  - Status note: `validate_pinned_host_int` now keeps a thread-local small pointer cache (IPC + NVSHMEM paths) and validates each stable host scratch pointer once, preserving fail-fast behavior on unknown/unpinned pointers.

- [x] B-062 Cap CTA launch counts across warp-stride gather/return kernels.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: hot warp-stride gather/return kernels avoid pathological over-launch at large `M_recv` while preserving full coverage via kernel-internal warp-stride loops.
  - Status note: switched multiple BF16/blockscaled gather and return-scatter callsites from raw `std::max(1, work/threads)` launch sizing to `cap_warp_stride_blocks(...)`, including `k_gather_bf16`, `k_gather_blockscaled`, `k_gather_from_pad_bf16`, `k_scatter_sorted_to_pad_bf16`, `k_return_write_tokslot_*`, `k_return_scatter_bf16`, and `k_return_scatter_from_pad_atomic`, plus backward warp-stride wrappers (`gather_dy*`, `dgate_from_adA`, `scatter_dx*`, IPC `send_dx*`/`gather_dy*` distributed paths), matching hybrid NVSHMEM warp-stride wrapper callsites, and dGate send/collect wrappers.

- [x] B-063 Cap CTA launch counts in 2-phase dispatch count/write kernels.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: 2-phase dispatch count and write phases do not over-launch CTAs on large token batches.
  - Status note: `dispatch_2phase_bf16` and `dispatch_2phase_blockscaled` now cap `k_count_dispatch_*` and `k_dispatch_2phase_*` launch grids via `cap_warp_stride_blocks(...)`.

- [x] B-064 Convert metadata extract/gather helpers to grid-stride + capped CTAs.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: metadata helper kernels (`extract_local_eid*`, `gather_meta_sorted*`) do not require full uncapped grids and preserve full coverage under CTA caps.
  - Status note: kernels now use grid-stride loops and all hot callsites compute launch size via `cap_warp_stride_blocks(...)`.

- [ ] B-029 Make 2-phase dispatch staging safe for concurrent stream usage.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: shared phase counters and recv scratch used by 2-phase dispatch cannot be clobbered by overlapping multi-stream dispatch calls.

- [~] B-065 Remove dispatch `M_recv` host-stream sync handoff in IPC/hybrid paths.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`, `nmoe/moe.py`
  - Acceptance: no `cudaStreamSynchronize(stream)` in dispatch hot path to read counters; host handoff is overlapped or deferred without correctness drift.
  - Status note: replaced dispatch counter readbacks with event-scoped blocking reads (`cudaEventRecord + cudaEventSynchronize`) in IPC + hybrid dispatch paths, removing direct `cudaStreamSynchronize(stream)` from hot dispatch entrypoints. Full overlap/deferred host handoff remains pending.

- [x] B-066 Overlap grouped-wgrad offset staging via async prepare path.
  - Files: `nmoe/moe.py`, `nmoe/csrc/gemm.cu`, `nmoe/csrc/bindings.cpp`
  - Acceptance: offset staging is queued once before grouped-wgrad calls and hot wgrad entrypoints consume staged host offsets without Python copy-stream/event synchronization.
  - Status note: `bf16_prepare_offs_pad_host(...)` now performs async staged D2H with event readiness; MoE backward reuses prepared host offsets across W2/W13 grouped-wgrad calls via host-offs entrypoints.

- [x] B-067 Enforce NVFP4-primary backward metadata path (no BF16 dispatch/gather fallback).
  - Files: `nmoe/moe.py`, `nmoe/model.py`
  - Acceptance: NVFP4-primary backward fails fast unless `mode='hybrid'`, `world>1`, and `rdep.profile='nvfp4'`; BF16 metadata/gather path is unreachable in production NVFP4-primary mode.
  - Status note: added hard guards before backward dispatch to forbid BF16 metadata fallback for NVFP4-primary, and added forward-time precondition checks on RDEP mode/profile/world for NVFP4-primary runs.

- [x] B-068 Restrict fused ECO path to NVFP4-primary only and hard-fail invalid mixed mode.
  - Files: `nmoe/moe.py`
  - Acceptance: `fused_eco` cannot execute when `_nvfp4_primary` is false; backward cannot consume empty placeholder weights in non-primary mode.
  - Status note: added explicit forward/backward guards requiring `fused_eco` + `moe_ref._nvfp4_primary`, eliminating invalid placeholder-weight path.

- [x] B-069 Make NVFP4 init/load cache lifecycle strict and checkpoint-safe.
  - Files: `nmoe/model.py`
  - Acceptance: `init_weights()` does not hard-fail before NVFP4 checkpoint load; cache is invalidated when NVFP4 buffers are replaced.
  - Status note: NVFP4 non-primary `init_weights()` now defers cache build (`_W_cache=None`) until buffers are loaded; `set_nvfp4_buffers(...)` now deletes stale cache to force rebuild from new packed buffers.

- [x] B-070 Remove dead NVSHMEM host-barrier wrapper surface.
  - Files: `nmoe/csrc/rdep_nvshmem.cu`, `nmoe/csrc/bindings.cpp`
  - Acceptance: unused `rdep_nvshmem_barrier` C-wrapper and declaration are removed; no runtime callsites remain.
  - Status note: removed dead `barrier()` host wrapper and `rdep_nvshmem_barrier` export/declaration; steady-state synchronization remains only on active IPC/NVSHMEM barrier paths.

- [x] B-071 Enforce fail-fast behavior for dropped-token overflow in production dispatch.
  - Files: `nmoe/moe.py`
  - Acceptance: dropped-token detection cannot warn-and-continue; overflow raises runtime error on strict path.
  - Status note: periodic dropped-token sampling now raises a hard error when `dropped > 0`, removing silent degraded-throughput/correctness continuation.

- [x] B-072 Disable NVFP4 backward BF16 metadata/gather fallback unconditionally.
  - Files: `nmoe/moe.py`
  - Acceptance: any `rdep.profile='nvfp4'` backward call that is not on hybrid distributed blockscaled metadata path fails fast.
  - Status note: added explicit guard requiring `use_dist_blockscaled_meta` whenever profile is NVFP4; BF16 backward metadata/gather fallback is now unreachable.

- [x] B-099 Label distributed backward IPC barrier sites for deterministic timeout forensics.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: hot distributed backward BF16/blockscaled barriers emit non-null site labels in RDEP barrier trace/watchdog logs.
  - Status note: replaced unlabeled `ipc_barrier_*` calls in gather-dY/dGate/scatter-dX distributed paths with `ipc_barrier_*_site(...)` labels (pre/post-zero, post-send, post-gather) so phase timeouts map to exact stages.

- [x] B-100 Keep router aux-loss differentiable unless MoE FFN checkpointing is active.
  - Files: `nmoe/model.py`
  - Acceptance: `gradient_checkpointing=true` without `checkpoint_moe_ffn=true` no longer detaches aux-loss path; router aux receives gradients.
  - Status note: `_aux_loss_detached_for_checkpoint` now keys off `checkpoint_moe_ffn` only, fixing unintended aux-loss detach in common NVFP4 runs.

- [x] B-101 Enforce venv-only torchrun and runtime parity checks before launch/provision success.
  - Files: `../nmoe-multinode/agent.py`, `../nmoe-multinode/orchestrate.py`
  - Acceptance: agent launch defaults to `<nmoe_dir>/.venv/bin/torchrun`; provision/launch parity checks fail on repo/env/extension drift across nodes.
  - Status note: added strict venv torchrun selection (system fallback opt-in only), expanded per-node probe to include flash-attn SHA + Python/PyTorch/marker/torchrun source, and wired parity validation into `repo-parity --strict` and post-provision checks.

- [x] B-102 Add step-0 NaN localization diagnostics for dense-grad failures.
  - Files: `nmoe/train.py`
  - Acceptance: when enabled, backward emits anomaly trace for one micro-step and dense-grad NaN failures include bad-rank localization/details without adding steady-state overhead.
  - Status note: added `NMOE_NAN_TRACE_STEP0` / `NMOE_NAN_TRACE_MICRO` scoped anomaly tracing, bad-rank all-gather context on grad-norm failure, and FP64 reference norm fallback detail when fused norm is non-finite but elementwise grad scans are inconclusive.

- [x] B-073 Replace full `dYe_pad` zero-fill with selective padding-row zero kernel.
  - Files: `nmoe/moe.py`, `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`, `nmoe/csrc/rdep_nvshmem.cuh`, `nmoe/csrc/bindings.cpp`
  - Acceptance: backward padded buffer path no longer uses `torch.zeros(max_pad, H)`; only expert-alignment gap rows are zeroed via CUDA kernel.
  - Status note: added `zero_padding_rows_bf16` binding and hybrid wrapper; backward now allocates `dYe_pad` with `torch.empty`, scatters sorted rows, and zeroes only padding gaps based on dispatch offsets.

- [x] B-074 Fix hybrid dGate routing source in SonicMoE backward wrapper.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: `dgate_from_adA_bf16` cannot silently early-return in hybrid mode and always uses the active mode's dispatch `dest` mapping.
  - Status note: `rdep_dgate_from_adA_bf16` now has explicit hybrid branch using `nvshmem::g_nvshmem.dest` and fail-fast checks instead of BF16-state early return.

- [x] B-075 Harden IPC padded dX scatter fast path shape contracts.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: `scatter_dx_dist_from_pad_bf16` rejects invalid `H` shape/state before vectorized `int4` row accesses.
  - Status note: added fail-fast checks for `H % 8 == 0` and per-mode `H` state match (`g_bf16.H` / `g_block.H`) in IPC padded dX scatter wrapper.

- [x] B-076 Remove hybrid backward `dX` pad->sorted gather copy.
  - Files: `nmoe/moe.py`, `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`, `nmoe/csrc/rdep_nvshmem.cuh`
  - Acceptance: hybrid distributed `dX` path consumes padded `dX_pad` directly (no intermediate `dX_sorted` allocation + gather kernel).
  - Status note: added `scatter_dx_hybrid_bf16_from_pad` NVSHMEM path and wired `rdep_scatter_dx_dist_from_pad_bf16` to use it in hybrid mode; `moe.py` now uses from-pad scatter for both IPC and hybrid distributed modes.

- [x] B-030 Propagate hybrid backward precondition failures as hard runtime errors.
  - Files: `nmoe/csrc/rdep_nvshmem.cu`, `nmoe/csrc/rdep.cu`
  - Acceptance: `gather_dy_hybrid_bf16` / `scatter_dx_hybrid_bf16` precondition failures cannot early-return silently; call path fails fast and surfaces rank-local error context.
  - Status note: hybrid gather/scatter precondition checks now `abort()` on violation (`init`, IPC-sync, `K/H`, tok-slot bounds) instead of silent returns, and distributed BF16 wrappers now hard-fail on invalid mode/state/topology instead of returning no-op.

- [x] B-077 Remove full blockscaled SF memsets in gather paths (IPC + hybrid).
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: `Xe_sf_out` initialization no longer performs full `cudaMemsetAsync(...,127, M_pad*Hsf)` in blockscaled gather paths; only per-expert padding gap rows are initialized to 127 directly in CUTLASS MMA-swizzled layout.
  - Status note: added `k_fill_blockscaled_padding_sf_swizzled` (IPC) and `k_fill_blockscaled_padding_sf_swizzled_hybrid` (NVSHMEM) with async wrappers; both dispatch/gather entrypoints now use padding-only swizzled SF fill before gather.

- [x] B-078 Convert token-slot reduce/scatter kernels to grid-stride + capped launches.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: token-slot reduction/scatter kernels are no longer forced to `<<<T,256>>>`/uncapped launch shape; kernels preserve full coverage via grid-stride token loops and launch grids use `cap_warp_stride_blocks(...)`.
  - Status note: converted `k_reduce_tokslot_gate_bf16`, `k_reduce_tokslot_sum_bf16`, `k_reduce_dx_tokslot_hybrid`, `k_scatter_gate_bf16`, `k_scatter_gate_bf16_out_bf16`, and IPC fallback `k_send_dgate_ipc_bf16` to grid-stride forms and capped all wrapper callsites.

- [x] B-079 Remove dispatch overflow clamp fail-open behavior.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: dispatch paths no longer silently truncate `M_recv` via `std::min(M_recv, capacity)`; overflow is surfaced as explicit runtime error.
  - Status note: replaced all hot dispatch clamp sites (BF16/blockscaled IPC + hybrid NVSHMEM) with hard overflow checks that return error and include `M_recv/capacity` context. Hybrid blockscaled merge now uses exact counter sum (`k_add_counter_sum`) instead of pre-check clamp so overflow cannot be masked before host-side strict check. Remaining dynamic NVSHMEM `nv_count > capacity` clamps now atomically account overflow into `dropped` before clamp to avoid silent truncation.

- [x] B-080 Fix hybrid `dX` from-pad sender guard to avoid valid-row drop.
  - Files: `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: `scatter_dx_hybrid_bf16_from_pad` sender kernel does not gate padded row indices by dispatch capacity (`capacity`) and cannot silently drop valid `pad_i` rows when `M_pad > capacity`.
  - Status note: removed `pad_i >= capacity` reject in `k_send_dx_tokslot_hybrid_from_pad`; kernel now validates negative indices only and trusts padded-index mapping contract.

- [x] B-081 Narrow dispatch radix-sort key range for local expert IDs.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: dispatch sort passes use `end_bit=ceil(log2(n_local))` for `local_eid` keys instead of fixed 32-bit range.
  - Status note: added `radix_sort_end_bit_for_range(...)` helper and switched BF16/blockscaled IPC and hybrid local-eid sort callsites to bounded key bits.

- [x] B-082 Remove search-heavy padded mapping fill in IPC/hybrid dispatch.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: padded `dest` mapping no longer performs per-row expert-search loops (`binary search` or `O(M*n_local)` scan); mapping uses sorted expert-id keys directly.
  - Status note: mapping now uses split prefix+fill kernels (`k_compute_padded_prefix*` + `k_fill_dest_from_sorted_eid*`) and computes `dest[i]` in O(1) from sorted expert IDs and offset starts.

- [x] B-083 Skip empty-slot payload loads in tok-slot gate reduction.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: `k_reduce_tokslot_gate_bf16` does not load/decode `tok_y` when slot gate is exactly zero.
  - Status note: added `if (g == 0.0f) continue;` fast skip in slot accumulation loop.

- [x] B-084 Remove redundant pre-count IPC barrier in 2-phase dispatch.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: 2-phase BF16/blockscaled dispatch does not issue an extra global barrier between local counter reset and local count kernel launch.
  - Status note: removed the initial `ipc_barrier_*` before phase-1 count; barriers after count/offset/data exchange remain unchanged.

- [x] B-085 Complete split padded-mapping rollout in IPC dispatch callsites.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: IPC dispatch/meta paths no longer reference removed `k_compute_padded_mapping`; all callsites use `k_compute_padded_prefix + k_fill_dest_from_sorted_eid`.
  - Status note: replaced remaining stale launches in BF16/blockscaled dispatch and meta entrypoints; removed build-break symbol drift from partial rollout.

- [x] B-086 Fold `offs_pad` tail override into padded-prefix kernel.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: dispatch paths that over-approximate `M_pad` do not launch a separate `<<<1,1>>>` tail-fix kernel.
  - Status note: `k_compute_padded_prefix` now accepts optional `override_total` and writes final `offs_pad[n_local-1]` inline; removed `k_set_last_offs_pad` launch points and deleted the now-dead kernel.

- [x] B-087 Expand async device-int poll slot table to reduce forced sync eviction.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: async counter read helpers have enough slot headroom to avoid frequent all-slots-pending eviction sync in deep pipelines.
  - Status note: increased thread-local slot table capacity from 16 to 64 for both IPC and hybrid counter-read paths.

- [x] B-088 Replace token-count histogram offset kernel with sorted-key lower-bound offsets.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: per-dispatch offset generation no longer scans all `M_recv` tokens with shared-memory atomics; offsets are generated directly from sorted keys in parallel over experts.
  - Status note: `k_compute_offsets*` now computes offsets via lower-bound binary search on `sorted_eid` for `e in [0..n_local]` (`O(n_local log M)`), and all callsites dropped dynamic shared-memory histogram launches.

- [x] B-089 Harden local-expert configuration contracts and remove stale hybrid cap.
  - Files: `nmoe/rdep.py`, `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: invalid `n_local <= 0` cannot reach dispatch kernels; hybrid mode no longer rejects `n_local > 256` due removed legacy mapping constraint.
  - Status note: added fail-fast `n_local > 0` validation in Python constructor and all IPC/NVSHMEM alloc entrypoints; removed outdated hybrid dispatch `n_local > 256` runtime checks.

- [x] B-090 Warp-aggregate 2-phase count kernel atomics into shared destination counters.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: phase-1 count kernels avoid one shared-memory atomic per active thread and aggregate by destination per warp.
  - Status note: `k_count_dispatch_bf16` and `k_count_dispatch_block` now use `__ballot_sync + __match_any_sync` leader aggregation (one atomic per unique destination per warp iteration).

- [x] B-091 Broadcast per-destination recv offset once per warp in 2-phase dispatch write kernels.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: phase-3 dispatch write kernels do not issue redundant per-lane global loads of `my_recv_offsets[dest]`.
  - Status note: `k_dispatch_2phase_bf16` and `k_dispatch_2phase_blockscaled` now load `base_offset` in lane 0 and `__shfl_sync` broadcast to the warp.

- [x] B-092 Tighten hybrid return-scatter launch bounds to token upper bound.
  - Files: `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: hybrid IPC scatter-after-forward kernels are not launched at raw `capacity` when `T*K` is smaller.
  - Status note: `k_scatter_received_hybrid_bf16_dynamic` launch sizing in both return paths now uses `min(capacity, T*K)` upper bound.

- [x] B-093 Disable distributed layout-reuse restore path by default in MoE backward.
  - Files: `nmoe/moe.py`
  - Acceptance: distributed backward does not enter saved-layout restore/consensus path unless explicitly enabled; collective scope remains EP-consistent when enabled.
  - Status note: added `NMOE_ENABLE_LAYOUT_REUSE_DIST` (default `0`), gated distributed layout-reuse on this flag, and scoped consensus all-reduces to `rdep.ep_group`.

- [x] B-094 Add explicit ECO async all-reduce wait diagnostics.
  - Files: `nmoe/eco.py`
  - Acceptance: when enabled, stalls in pending DP all-reduce drain path emit actionable step/sequence/param queue context before blocking wait.
  - Status note: added `NMOE_ECO_WAIT_DEBUG` and a pre-wait diagnostic log in `_drain_one()` before `tail.wait()`.

- [x] B-093 Optimize quantized CT->MMA scale handling and expert lookup hint path.
  - Files: `nmoe/csrc/quant.cu`
  - Acceptance: strided quant expert lookup uses fast stride hint with safe fallback; CT->MMA kernels remove redundant per-lane global-scale decode/divide and use fast group-scale leader path for pow2 groups.
  - Status note: added `find_expert_for_row_hint(...)` across quant/SwiGLU strided kernels; CT->MMA kernels now broadcast `inv_gs` from lane 0 and use pow2-group leader broadcast fast path (fallback remains `__match_any_sync`); CT->BF16 decode path now skips second group-scale decode for even `group_size`.

- [x] B-094 Harden quant runtime architecture guard against silent misbuild targets.
  - Files: `nmoe/csrc/quant.cu`
  - Acceptance: quant entrypoints fail fast when loaded kernel binaries are below `sm_100`; zero-size `ct_nvfp4_to_bf16` requests preserve no-op success behavior.
  - Status note: `require_sm100_or_newer` now validates kernel `binaryVersion >= 100` via `cudaFuncGetAttributes(...)`, and `ct_nvfp4_to_bf16` keeps `M<=0 || K<=0` early-return before architecture guard.

- [x] B-095 Parallelize 2-phase recv-offset exchange prefix in IPC dispatch.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: phase-2 offset exchange no longer runs serial prefix/write loop on thread 0.
  - Status note: `k_compute_and_write_offsets_bf16` and `k_compute_and_write_offsets_block` now run one-warp prefix scans and per-lane source writes, preserving existing write contract (`src_recv_offsets[my_rank]`) and total-recv counter semantics.

- [x] B-096 Add adaptive nanosleep backoff in IPC barrier spin loops.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: barrier spin-wait loops avoid tight continuous polling under stragglers while preserving timeout/fail-fast behavior.
  - Status note: added periodic `__nanosleep(64)` backoff in BF16/blockscaled phase barriers and NVSHMEM dynamic IPC barrier loops.

- [x] B-097 Remove redundant pre-zero IPC barrier in distributed `dX` scatter paths.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: `scatter_dx_dist_bf16` and `scatter_dx_dist_from_pad_bf16` (BF16 + blockscaled IPC branches) launch with two cross-rank barriers instead of three, while keeping zero-before-send and send-before-reduce ordering.
  - Status note: removed the leading pre-zero barrier in both distributed `dX` scatter entrypoints; kept the post-zero and post-send barriers to preserve correctness contracts.

- [x] B-098 Collapse NVFP4 checkpoint missing-key warning spam to summary logging.
  - Files: `nmoe/checkpoint.py`
  - Acceptance: non-router checkpoint key mismatches no longer emit per-key warnings in the hot import loop; logs show one aggregated summary with bounded sample keys (with optional verbose override).
  - Status note: added aggregated missing-key accounting with `NMOE_NVFP4_MISSING_PARAM_VERBOSE=1` override for full per-key diagnostics.

## C) Router + Dispatch Kernel Work

- [x] C-001 Add 2-phase dispatch for blockscaled path (parity with BF16 fast path).
  - Files: `nmoe/csrc/rdep.cu` blockscaled dispatch kernels/state.
  - Acceptance: blockscaled path uses 2-phase kernels for multi-node; throughput improves with identical numerics.
  - Status note: added blockscaled 2-phase count/offset/deterministic-write kernels and enabled them by default for multi-rank IPC in both `dispatch_meta_blockscaled` and `dispatch_blockscaled`.

- [x] C-009 Tighten blockscaled dispatch ordering and remove extra index math in dispatch hot kernels.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: blockscaled dispatch kernels guarantee sys-visible remote writes before barrier consumers; per-token dispatch loops avoid redundant `tok*K+slot` address recompute for `eids/gates`; 2-phase count kernels take precomputed `M` (no per-launch `T/K` argument overhead); expert routing math uses loop-hoisted fast power-of-two `n_local` path to avoid integer divide/mod in hot loops.
  - Status note: additional pass applied: all remaining hot `i % K`/`rid % K` decode sites switched to one-divide arithmetic (`slot = i - tok*K` / `slot = rid - (rid/K)*K`), row-id decode also removed `% T` in favor of one-divide arithmetic (`tok = tmp - (tmp/T)*T`), dispatch/backward kernels now use power-of-two `K` fast-path (`>>`) when applicable, and hybrid kernels use fast power-of-two `local_world` rank split helpers instead of per-token `/` and `%`. Final decode tightening is now landed via `decode_rid_fast` + per-kernel cached `K/T` pow2 flags across return/backward hot kernels (including hybrid NVSHMEM gather), removing repeated per-row pow2 checks and duplicate decode work.

- [ ] C-002 Remove redundant metadata rebuild (sort/offset) when fused router metadata is available.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/fused_router.py`, `nmoe/model.py`.
  - Acceptance: fewer metadata kernels and launches; no routing correctness drift.

- [~] C-003 Router forward path heuristic: use TC GEMM path for large shapes, fused path for small shapes.
  - Files: `nmoe/fused_router.py`
  - Acceptance: speedup over current path across token ranges; no top-k mismatch.
  - Status note: added auto-selector (`NMOE_ROUTER_FORWARD_MODE=auto|dense|fused`) with default large-shape dense GEMM thresholds (`NMOE_ROUTER_DENSE_MIN_D/T/WORK`) and a cuBLAS-backed dense forward implementation; selector policy is cached once (no per-layer env parsing overhead). Forward strict mode is enabled by default (`NMOE_ROUTER_FWD_STRICT=1`) so dense fallback is disabled unless explicitly opted out; unknown mode values now fail fast instead of silently falling back to heuristic branch. Final threshold tuning still needs profiling on full 16xB200 workload.

- [x] C-014 Enforce fused-router-forward only (dense path disabled in production).
  - Files: `nmoe/fused_router.py`, `nmoe/model.py`
  - Acceptance: `NMOE_ROUTER_FORWARD_MODE` accepts fused/triton only; MoE construction/forward cannot route into dense fallback branches.
  - Status note: dense forward selector now hard-fails, MoE init rejects `use_fused_router=false`, and runtime route logic is fused-only.

- [x] C-015 Forbid standalone router backward shared-object fallback.
  - Files: `nmoe/fused_router.py`, `nmoe/csrc/router_bwd.cu`
  - Acceptance: loader cannot silently use standalone `.so`; router backward APIs fail fast off-target.
  - Status note: `NMOE_ROUTER_BWD_ALLOW_STANDALONE=1` now hard-errors and router backward C API adds explicit `sm_80+` runtime checks.

- [~] C-004 Router backward: remove extra cast + transpose chain.
  - Files: `nmoe/fused_router.py`, `nmoe/csrc/router_bwd.cu`
  - Acceptance: one fewer transformation step; gradients match baseline tolerance.
  - Status note: added BF16 fused-router backward entrypoint (`fused_router_backward_bf16`) and Python dispatch now uses it directly for BF16 `pre_probs/gates/grad_gates`, eliminating per-step Python-side BF16->FP32 cast kernels. Loader ordering now prefers the active interpreter extension suffix to avoid accidentally loading stale `cpython-313` binaries from Python 3.14. Runtime now hard-fails on any non-BF16 backward inputs and no longer uses the FP32 backward call path; loader checks also require BF16 symbol directly (no legacy FP32 symbol dependency). FP32 accumulation + transpose/cast stage is still present.

- [x] C-012 Remove extra fused-router gate memory round-trip in forward kernel.
  - Files: `nmoe/fused_router.py`
  - Acceptance: no write/read of temporary unnormalized `gates` values before normalization; normalized gates are written once from `pre_probs`.
  - Status note: Triton router kernel now stores only `pre_probs` during TopK selection and computes normalized `gates` from `pre_probs` in the normalization pass.

- [x] C-013 Emit `dGates` directly in BF16 in MoE backward (remove cast kernel).
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/bindings.cpp`, `nmoe/moe.py`, `nmoe/train.py`
  - Acceptance: no steady-state `dGates_tk.float32 -> bfloat16` cast in `moe.py` backward path; BF16 dGate collection kernels are required at launch.
  - Status note: added BF16-output dGate scatter/collect entrypoints (`scatter_gate_bf16_out_bf16`, `send_dgate_dist_bf16_out_bf16`) and switched `moe.py` backward hot path to consume BF16 directly.

- [x] C-011 Router backward: reuse forward pre-normalized probs to skip logit recompute pass.
  - Files: `nmoe/fused_router.py`, `nmoe/csrc/router_bwd.cu`
  - Acceptance: fused router backward no longer recomputes per-token TopK logits (`hidden @ router_weight` for selected experts) just to recover sigmoid derivative; derivative uses saved pre-normalized probs from forward.
  - Status note: forward now stores per-token selected pre-normalized probs (`pre_probs`) and backward C++ kernel consumes them directly; removed the backward logit-recompute phase and one full D-dimension pass from the fused router backward hot path.

- [x] C-010 Emit fused-router gates directly in target dtype (no FP16 staging cast).
  - Files: `nmoe/fused_router.py`
  - Acceptance: no unconditional `gates.to(dtype)` cast kernel after fused router forward.
  - Status note: Triton kernel now stores gates directly to output tensor dtype and forward returns `gates` directly; default BF16 path no longer stages through FP16 + cast.

- [x] C-005 Remove unused `dispatch_indices` payload if not consumed.
  - Files: `nmoe/fused_router.py`, `nmoe/model.py`
  - Acceptance: no allocation/write of dead tensor in hot path; no behavior change.
  - Status note: fused router Triton kernel no longer writes `dispatch_indices`, forward no longer allocates/returns it, and model path consumes `{eid, gates, expert_counts}` only.

- [~] C-006 Gate fused-router NVTX by env (off by default in production).
  - Files: `nmoe/fused_router.py`
  - Acceptance: no unconditional NVTX push/pop in production runs.
  - Status note: NVTX context now obeys `NMOE_NVTX`; runtime trace verification pending.

- [x] C-007 Initialize ctypes argtypes/restype once, not per call.
  - Files: `nmoe/fused_router.py`
  - Acceptance: one-time setup at import/init; reduced Python overhead in profile.
  - Status note: one-time signature init added in loader path; router backward now also reuses per-device FP32 accumulation scratch (`[E, D]`) and caches BF16 entrypoint + strict-fastpath/NVTX flags (no repeated `hasattr`/env parsing in backward hot path). Scratch buffers are stream-scoped and cache-bounded (`NMOE_ROUTER_STREAM_CACHE_LIMIT`) for both router backward accumulation and fused bias-count temporaries to avoid cross-stream races and unbounded growth.

- [x] C-008 Fuse blockscaled gather + swizzle to remove temporary SF path.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`, `nmoe/csrc/rdep_nvshmem.cuh`
  - Acceptance: remove temp memset + separate swizzle launch in hot route.
  - Status note: blockscaled gather now writes SF directly into CUTLASS MMA layout (`cutlass_sf_swizzle_offset`) while gathering rows; removed `sfa_gather_tmp` workspace and `swizzle_sf_mkl_to_mma` launches from IPC/single and hybrid NVSHMEM paths. Padding SF bytes are initialized by padding-only swizzled fill kernels (no full-buffer `cudaMemsetAsync` in gather paths).

- [x] C-016 Cache router arch capability checks (remove per-call device-prop queries).
  - Files: `nmoe/csrc/router_bwd.cu`
  - Acceptance: `fused_router_backward*`/transpose entrypoints do not re-query `cudaGetDeviceProperties` on every call for the same device/thread.
  - Status note: `require_sm80_or_newer(...)` now uses thread-local device/result cache.

- [x] C-017 Reduce per-lane scale decode overhead in strided quant kernels.
  - Files: `nmoe/csrc/quant.cu`
  - Acceptance: active FP8/NVFP4 strided MMA quant kernels decode E8M0 inverse scale once per warp chunk (lane 0 + warp broadcast), not once per lane.
  - Status note: updated `k_quantize_pack_tilewise_{fp8,nvfp4}_sf_strided_mma`, `k_swiglu_quantize_pack_tilewise_{fp8,nvfp4}_sf_strided_mma`, baseline SwiGLU quant kernels (`k_swiglu_quantize_pack_tilewise_{fp8,nvfp4}`), and CT conversion kernels (`k_ct_nvfp4_to_blockscaled_mma`, `k_ct_nvfp4x2_interleaved_to_blockscaled_mma`) to broadcast `inv_scale` directly.

- [x] C-017 Remove redundant fused-router backward loop guards and dead fused-forward branching.
  - Files: `nmoe/csrc/router_bwd.cu`, `nmoe/fused_router.py`
  - Acceptance: no `k < K && k < MAX_K` checks in router backward kernels after host validation; fused forward path has no dead dense-branch runtime branch in hot path.
  - Status note: router backward loops now iterate `k < K` only (host enforces `K<=MAX_K`), fused forward directly enforces fused policy once and launches without per-call dead branch checks; backward also avoids redundant `.contiguous()` on saved tensors.

- [x] C-018 Harden hybrid backward tok-slot kernels against stale/invalid row IDs.
  - Files: `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: gather/scatter tok-slot kernels validate decoded `src_rank/tok/slot` bounds before writing IPC/NVSHMEM targets; phase tagging is monotonic and non-racy under concurrent host calls.
  - Status note: added global-world and tok-slot bounds guards, upgraded same-node tok-gate remote writes to sys-scope stores, and replaced non-atomic phase counter with atomic phase generation + wraparound tag reset barrier.

- [x] C-042 Enforce strict NVFP4 cache refresh (no BF16 re-quant fallback path).
  - Files: `nmoe/model.py`
  - Acceptance: `dtype='nvfp4'` cache refresh cannot route through BF16 quantization path; model must use NVFP4 packed buffers.
  - Status note: `refresh_weight_cache()` now hard-errors for NVFP4 when `_nvfp4_primary` is false and validates full W1/W3/W2 packed triplets when primary mode is enabled; dead unreachable guard removed. `has_nvfp4_buffers()` now also checks complete W1/W3/W2 triplets (not only W1) to avoid partial-buffer false positives.

- [x] C-019 Remove avoidable NVFP4 W2 dequant relayout in backward grouped-mm path.
  - Files: `nmoe/moe.py`
  - Acceptance: NVFP4 staggered backward dequantizes W2 directly to grouped-mm consumption layout for `dA`, avoiding immediate transpose-after-dequant.
  - Status note: switched NVFP4 W2 transient dequant to `transpose=False` (`[E,H,Dff]`) and fed it directly into grouped MM; removed extra relayout step.

- [x] C-043 Vectorize CT NVFP4->BF16 dequant by decoding one packed byte per thread.
  - Files: `nmoe/csrc/quant.cu`
  - Acceptance: `ct_nvfp4_to_bf16` kernel no longer launches per BF16 element; each thread decodes low+high nibbles from one byte and writes two outputs.
  - Status note: updated `k_ct_nvfp4_to_bf16` and launcher to byte-granular threading (`M * (K/2)` work items), reducing packed-byte loads and index overhead across transpose modes `0/1/2`.

- [x] C-044 Harden CT NVFP4->BF16 API contract checks for expert/global-scale mode.
  - Files: `nmoe/csrc/quant.cu`
  - Acceptance: invalid `gs_stride` or inconsistent `expert_rows` for per-expert global-scale mode fails fast with `cudaErrorInvalidValue`.
  - Status note: C API now enforces `gs_stride in {0,1}` and validates `expert_rows` divisibility when `gs_stride==1`, preventing undefined expert-index/global-scale addressing.

- [x] C-045 Optimize CT NVFP4->BF16 hot kernel math/launch geometry.
  - Files: `nmoe/csrc/quant.cu`
  - Acceptance: remove dead output-pair bounds branches in dequant hot path, use packed BF16 vector stores in transpose=0 path, and launch a 2D fast path to reduce div/mod index overhead.
  - Status note: `k_ct_nvfp4_to_bf16` now computes reciprocal once (`__frcp_rn`), decodes/writes both nibbles without `k1<K` branches (K-even contract), uses `__nv_bfloat162` packed stores for non-transpose output, and `launch_ct_nvfp4_to_bf16` now uses 2D `dim3(BX=128,BY=2)` when grid.y fits, with a safe 1D fallback.

- [x] C-020 Remove dead non-blockscaled MoE runtime branch and duplicate cast pressure.
  - Files: `nmoe/model.py`
  - Acceptance: MoE runtime path is blockscaled-only with fail-fast dtype guard; no dead BF16 dispatch branch in forward.
  - Status note: constructor now hard-rejects non-`fp8|nvfp4` MoE dtype and forward uses blockscaled dispatch path only.

- [ ] C-021 Eliminate router backward FP32 transpose/cast tail.
  - Files: `nmoe/fused_router.py`, `nmoe/csrc/router_bwd.cu`
  - Acceptance: fused-router backward no longer materializes full-size FP32 accumulation + separate BF16 transpose/cast output pass.

- [x] C-022 Remove NVFP4->BF16->FP8 fallback quantization chain.
  - Files: `nmoe/blockscaled/grouped.py`, `nmoe/csrc/quant.cu`
  - Acceptance: production path has direct NVFP4->FP8 blockscaled conversion kernels; BF16 fallback chain is disabled/fail-fast.
  - Status note: `quantize_weights_from_nvfp4(...)` now hard-fails on non-`nvfp4` profile instead of silently running BF16 fallback conversion.

- [~] C-023 Replace scalar GEMV-style fused-router forward with TC-tiled logits path.
  - Files: `nmoe/fused_router.py`
  - Acceptance: router forward no longer uses one-token scalar accumulation loop; tiled tensor-core logits path + fused topk/gate/count path improves throughput on large token batches.
  - Status note: replaced per-dimension scalar accumulation with a tiled `D`-blocked accumulation in the fused Triton kernel (vectorized hidden/weight tile loads and reduction), removed duplicate `e_offs == max_idx` comparisons in TopK selection by reusing one selection mask, removed per-step probability-vector reduction for selected gates by deriving `gate_val = max_score - bias[max_idx]`, and tuned launch meta for large `D` (`BLOCK_D` and `num_warps/num_stages` scaling). Tensor-core/`tl.dot` path is still pending.

- [ ] C-024 Reduce router backward global atomic pressure.
  - Files: `nmoe/csrc/router_bwd.cu`, `nmoe/fused_router.py`
  - Acceptance: gradient accumulation no longer relies on fully dense `K*D` global atomics per token; CTA/expert-group reductions reduce contention and step latency.
  - Status note: in-progress kernel tightening: inner `K` loops are now predicated-unrolled (`MAX_K=16`) to reduce dynamic loop/control overhead on hot paths, and launch geometry now dynamically selects `4` vs `8` warps/block by `(D,K,N)` to improve occupancy on large-shape router backward calls; full CTA/expert-group reduction for atomic-pressure reduction is still pending.

- [x] C-025 Skip zero-work router backward tokens in no-grad-hidden mode.
  - Files: `nmoe/csrc/router_bwd.cu`
  - Acceptance: when `kComputeGradHidden=false` and all `d_logit[k]==0`, kernel exits before D-dimension atomic loops.
  - Status note: added `any_dk` precheck and early return for exact zero-gradient tokens; hidden-grad path now also fast-zeros the output row and exits when all `d_logit[k]==0`.

- [x] C-026 Remove eager int32->float cast in fused router bias-update counts path.
  - Files: `nmoe/fused_router.py`
  - Acceptance: `total!=None` path passes raw counts to Triton kernel and casts in-kernel; compatibility retained for `total=None` callers.
  - Status note: `_update_bias_fused_kernel` now casts loaded counts to FP32 in-kernel; host-side eager cast removed from `total!=None` branch.

- [x] C-027 Require fused BF16 router-backward symbol (drop compatibility fallback).
  - Files: `nmoe/fused_router.py`
  - Acceptance: router-backward loader/dispatcher fails fast when `fused_router_backward_bf16_fused` is missing; no legacy transpose compatibility path.
  - Status note: signature init now hard-requires `fused_router_backward_bf16_fused`, loader no longer requires legacy transpose symbol, and backward dispatch uses fused entrypoint only.

- [x] C-028 Enforce transpose mode-2 only for NVFP4 transient dequant.
  - Files: `nmoe/moe.py`
  - Acceptance: `dequant_nvfp4_to_bf16_transient(..., transpose=True)` always fails if mode-2 kernel support is unavailable.
  - Status note: env escape hatch removed; transpose fallback path is now hard-disabled.

- [x] C-029 Disable NVFP4->BF16->FP8 fallback chain in cache builder.
  - Files: `nmoe/blockscaled/grouped.py`
  - Acceptance: `quantize_weights_from_nvfp4(...)` refuses non-`nvfp4` profile instead of running BF16 fallback conversion.
  - Status note: non-NVFP4 profiles now hard-error with explicit no-fallback message.

- [x] C-030 Enforce expert-id-only router bias update path.
  - Files: `nmoe/model.py`, `nmoe/fused_router.py`, `nmoe/opt.py`
  - Acceptance: post-step router bias update always uses fused expert-id bincount path; counts-only compatibility path cannot be selected.
  - Status note: MoE now stores `last_expert_ids`, optimizer post-hook passes `expert_ids`, and `update_bias(...)` hard-fails if expert IDs are absent.

- [x] C-031 Remove normalized-count compatibility branch in fused bias update helper.
  - Files: `nmoe/fused_router.py`
  - Acceptance: `fused_update_bias_from_counts(...)` requires explicit `total`; no implicit normalized-load compatibility path.
  - Status note: helper now raises when `total` is omitted.

- [x] C-032 Trim fused-router forward overhead and tighten launch contract.
  - Files: `nmoe/fused_router.py`
  - Acceptance: no dead dense-policy call in hot path; per-token expert-count atomics avoid extra expert-id reload loop; launch specifies warps/stages and fails fast for unsupported `E>256`.
  - Status note: removed dead dense-forward policy stubs, moved `atomic_add(expert_counts, max_idx)` into TopK selection loop, and added explicit `num_warps/num_stages` launch metadata with `E<=256` guard.

- [x] C-033 Relax router-backward loader symbol contract to fused entrypoint only.
  - Files: `nmoe/fused_router.py`
  - Acceptance: loader/init requires only `fused_router_backward_bf16_fused` for production path.
  - Status note: dropped hard requirement on legacy `fused_router_backward_bf16` symbol.

- [x] C-034 Skip fused-router expert-count atomics when counts are not needed.
  - Files: `nmoe/fused_router.py`, `nmoe/model.py`
  - Acceptance: fused router forward can run with `need_counts=false`, avoiding per-token atomic count updates in hot path when router load tracking is disabled.
  - Status note: added `need_counts` forward contract and `COMPUTE_COUNTS` kernel constexpr gate; model now requests counts only when router-bias tracking is enabled.

- [x] C-035 Remove hidden contiguous-copy fallbacks in fused-router hot path.
  - Files: `nmoe/fused_router.py`
  - Acceptance: fused-router forward/backward and fused bias-update helpers fail fast on non-contiguous tensors instead of silently materializing `.contiguous()` copies.
  - Status note: replaced implicit `.contiguous()` materialization for `hidden`, `grad_gates`, `expert_ids`, `expert_counts`, and `bias` with explicit no-fallback runtime guards; added zero-token guard in expert-id fused bias update (`TK==0` early return).

- [x] C-036 Return hidden gradient from fused router backward and tighten launch contract.
  - Files: `nmoe/fused_router.py`
  - Acceptance: autograd returns `grad_hidden` when requested, backward launch supports hidden-only grad requests, and invalid `topk` fails before kernel launch.
  - Status note: fused router backward now optionally allocates/returns `grad_hidden`, launch condition keys off `ctx.needs_input_grad`, and forward validates `0 < topk <= min(n_experts, 16)`.

- [x] C-037 Remove no-count expert-count allocation churn in fused router forward.
  - Files: `nmoe/fused_router.py`
  - Acceptance: `need_counts=false` path does not allocate a fresh `[E]` int32 tensor per forward call.
  - Status note: added reusable per-device cached dummy int32 `expert_counts` buffer for no-count launches.

- [x] C-038 Add hidden-only fused router backward kernel entrypoint.
  - Files: `nmoe/csrc/router_bwd.cu`, `nmoe/fused_router.py`
  - Acceptance: when router weights are frozen (`need_hidden_grad=true`, `need_wgrad=false`), backward avoids allocating/producing full router-weight gradient output.
  - Status note: added `fused_router_backward_bf16_hidden_only` CUDA entrypoint (no `grad_rw` atomics/scratch/transpose), and Python backward now dispatches to it when `need_hidden_grad=true` and `need_wgrad=false`, with strict fail-fast if the symbol is unavailable.

- [x] C-039 Require fused BF16 router-backward symbol at launch preflight.
  - Files: `nmoe/train.py`
  - Acceptance: production launch validation fails early when `fused_router_backward_bf16_fused` is missing from the active extension.
  - Status note: startup router symbol checks now require `fused_router_backward_bf16_fused` in addition to legacy symbols.

- [x] C-040 Enforce fused-router expert-count limit in config validation.
  - Files: `nmoe/model.py`
  - Acceptance: model construction fails fast if `n_routed_experts > 256` (instead of runtime failure inside fused router forward).
  - Status note: `_validate_moe_config` now validates `n_routed_experts <= 256`.

- [x] C-041 Align FlashMLA preflight defaults with production runtime defaults.
  - Files: `nmoe/train.py`
  - Acceptance: launch-time CUDA binding validation checks FA4/FlashMLA requirements by default, matching MLA runtime default behavior.
  - Status note: `_validate_required_cuda_bindings` now defaults `NMOE_USE_FA4=1` and `NMOE_REQUIRE_FLASHMLA=1`.

## D) Attention + RoPE Kernel Work

- [ ] D-001 Force Flash SDPA backend first, explicit fallback only on failure.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: backend selection logged; default path avoids unintended slower SDPA modes.

- [ ] D-002 Replace packed SDPA mask path with FlashMLA varlen path as primary.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: packed mode avoids materialized SxS mask; faster packed throughput without NaNs.

- [~] D-003 Rewrite varlen packed path to flatten docs into one launch and remove host sync points.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: fewer launches and no per-doc sync in trace; numerics match.
  - Status note: merged-varlen path now avoids redundant `cu_seqlens` device->host->device moves in the per-sample loop and preallocates final output tensor (removes per-sample zero/cat/stack churn). Per-sample `cu.cpu()` syncs were replaced with batched starts/totals readback, max-seqlen now derives from device deltas with one scalar handoff, monotonicity checks are explicit opt-in (`NMOE_VALIDATE_PACKED_CU_SEQLENS=1`), and FlashMLA wrapper CPU `cu_seqlens` validation is now strict opt-in (`FLASH_MLA_VALIDATE_VARLEN_INPUTS=1`) instead of default per-call host sync.

- [x] D-016 Remove packed SDPA float-mask materialization in fallback path.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: packed SDPA path does not allocate additive float mask tensors each call.
  - Status note: `scaled_dot_product_attention` now receives the bool mask directly (`attn_mask=mask`) in packed fallback mode.

- [ ] D-004 Add FlashMLA dense forward binding and compile object in nmoe csrc build.
  - Files: `nmoe/csrc/flashmla_sm100_bindings.cpp`, `nmoe/csrc/Makefile`
  - Acceptance: dense forward callable from nmoe binding; no ABI break.

- [ ] D-005 Remove `torch.cat` K materialization by kernel path that consumes split K tensors.
  - Files: `nmoe/attention/mla.py` and related binding path.
  - Acceptance: less memory traffic; exact/near-exact output parity.

- [ ] D-006 RoPE fused kernel: 2D mapping to reduce integer index math.
  - Files: `nmoe/csrc/rope_fused.cu`
  - Acceptance: instruction reduction and lower latency in rope microbench.

- [ ] D-007 RoPE fused kernel: add aligned vec4 path with guarded fallback.
  - Files: `nmoe/csrc/rope_fused.cu`
  - Acceptance: higher effective bandwidth with no alignment-induced errors.

- [ ] D-008 Allow BF16 return-scatter output mode to avoid extra cast kernel.
  - Files: `nmoe/moe.py`, `nmoe/csrc/rdep.cu`
  - Acceptance: fewer casts in trace; no quality regression.

- [x] D-009 Remove scalar RoPE fallback from production C API wrappers.
  - Files: `nmoe/csrc/rope_fused.cu`
  - Acceptance: odd half-dim inputs fail fast instead of silently launching scalar fallback kernels.
  - Status note: `fused_rope_forward/backward` and partial variants are now vectorized-only on production path.

- [x] D-010 Enforce packed-attention no-fallback backend policy.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: packed MLA path has explicit backend policy (`NMOE_PACKED_ATTN_BACKEND=auto|flashmla|sdpa`) and `NMOE_REQUIRE_FLASHMLA=1` forbids SDPA fallback.
  - Status note: packed backend now defaults to `flashmla` and fails fast when unavailable (explicit `sdpa` is required to opt into slower fallback); non-packed route also fails fast under `NMOE_REQUIRE_FLASHMLA=1` if `NMOE_USE_FA4=1` is not set.

- [x] D-011 Replace per-sample packed FlashMLA loop/backward with merged varlen call path.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: packed FlashMLA executes one merged varlen call per forward, avoiding per-sample custom backward loops and repeated Python/kernel launch overhead.
  - Status note: packed FlashMLA now flattens `B*T` active tokens with merged `cu_seqlens`, runs one `flash_attn_varlen_func`, then re-pads outputs per sample.

- [x] D-012 Fail fast in SM100 FlashMLA wrappers on unsupported head dims.
  - Files: `third_party/flashmla/csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu`, `third_party/flashmla/csrc/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu`
  - Acceptance: unsupported `(head_dim_qk, head_dim_vo)` combinations throw explicit errors (no silent print-and-continue path).
  - Status note: replaced no-kernel `std::cout` branches with `TORCH_CHECK` hard failures.

- [x] D-013 Cache FlashMLA varlen workspaces to remove allocator churn.
  - Files: `third_party/flashmla/flash_mla/flash_mla_interface.py`
  - Acceptance: forward/backward varlen wrappers reuse per-device scratch buffers instead of allocating each call.
  - Status note: added stream-keyed, size-capped workspace cache for both forward and backward varlen entrypoints.

- [x] D-014 Validate FlashMLA varlen inputs and normalize `cu_seqlens` dtype.
  - Files: `third_party/flashmla/flash_mla/flash_mla_interface.py`, `nmoe/attention/mla.py`
  - Acceptance: varlen wrappers reject malformed shapes/bounds and always pass `int32` `cu_seqlens` to CUDA kernels.
  - Status note: added explicit runtime guards for tensor/device/shape contracts, monotonic `cu_seqlens` checks, and int32 normalization before CUDA launch.

- [x] D-015 Tighten fused RoPE C API shape guards and remove silent odd-dimension paths.
  - Files: `nmoe/csrc/rope_fused.cu`, `nmoe/attention/rope.py`
  - Acceptance: fused RoPE entrypoints return `cudaErrorInvalidValue` for invalid non-vectorizable shapes; Python wrapper avoids unnecessary contiguous conversions in hot path.
  - Status note: full/partial wrappers now enforce vectorized dimensionality constraints for real-work shapes (while preserving zero-work no-op behavior), and Python path fast-paths already-contiguous tensors while forcing contiguous BF16 cos/sin after cast.

- [x] D-021 Remove hidden non-contiguous K-RoPE copy in MLA forward.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: K rope rotation does not call `rotate_pe(...)` on a non-contiguous split view.
  - Status note: K rope is now rotated in-place via `rotate_pe_partial(..., nope_dim=kv_lora_rank)` before split, eliminating an implicit contiguous materialization.

- [x] D-022 Remove packed attention reassembly/mask allocation churn.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: packed SDPA mask path does not allocate/intermediate-stack per sample; packed FlashMLA reassembly does not allocate per-sample pad/cat tensors.
  - Status note: packed SDPA now writes mask directly into preallocated `[B,1,S,S]`; packed FlashMLA now writes merged output slices into one preallocated output tensor.

- [x] D-023 Stop repeated FlashMLA runtime retries in `auto` backend after first failure.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: when `NMOE_PACKED_AUTO_ALLOW_SDPA=1` and FlashMLA runtime fails once, subsequent calls do not re-attempt failing FlashMLA path each step.
  - Status note: added `_PACKED_AUTO_DISABLE_FLASHMLA` runtime latch after first auto-path failure so fallback mode avoids repeated exception overhead.

- [x] D-024 Keep SDPA packed block-causal mask build GPU-resident.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: `_build_block_causal_mask` avoids per-call GPU->CPU->GPU `cu_seqlens` round-trip and writes directly into preallocated output mask slices.
  - Status note: mask builder now normalizes `cu_seqlens` on-device and supports `out=` destination to avoid temporary copy in packed SDPA loop.

- [x] D-025 Enforce FlashMLA-first production defaults in MLA path.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: packed attention requires `flashmla`, non-packed requires FA4 path, and SDPA routes are hard-disabled.
  - Status note: defaults now set `NMOE_PACKED_ATTN_BACKEND=flashmla`, `NMOE_USE_FA4=1`; any packed or non-packed SDPA branch now hard-fails.

- [x] D-026 Remove hidden contiguous fallback in fused RoPE entrypoint.
  - Files: `nmoe/attention/rope.py`
  - Acceptance: `rotate_pe(...)` fails fast on non-contiguous `x` instead of silently materializing a copy.
  - Status note: implicit `x.contiguous()` was removed from the fused RoPE wrapper.

- [x] D-027 Disable eager `flex_attention` fallback in sparse attention modules.
  - Files: `nmoe/attention/dsa.py`, `nmoe/attention/nsa.py`
  - Acceptance: import fails fast when `torch.compile(flex_attention)` fails; no eager fallback branch remains.
  - Status note: eager fallback path was removed; compile failure is now a hard error.

- [x] D-028 Reduce non-packed MLA Python/allocator overhead.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: non-packed FA4 path reuses cached uniform `cu` tensors, avoids per-forward dynamic import overhead, and avoids forced output contiguity copy before reshape.
  - Status note: added module caches for FA4/FlashMLA modules and uniform `cu`; final projection path now uses `reshape`.

- [x] D-029 Correct packed FlashMLA max-seqlen tracking and bound workspace cache growth.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: packed varlen path computes `max_seqlen` from max document span (not sample total tokens) and MLA workspace cache is bounded by stream count per device.
  - Status note: packed path now uses `max(cu[i+1]-cu[i])`; workspace cache now prunes per-device stream entries via `NMOE_MLA_STREAM_CACHE_LIMIT`.

- [x] D-030 Remove per-sample CUDA scalar syncs in packed FlashMLA merge path.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: packed FlashMLA preprocessing avoids per-sample `.item()` host syncs and per-sample Python gather/scatter loops.
  - Status note: packed path now computes token masks/merged tensors in batched tensor operations and reassembles output via boolean mask scatter, with mandatory aggregate token-count `.item()` sync removed from default path (only optional validation keeps scalar sync checks).

- [x] D-031 Fix non-packed SM100 FlashMLA dense path varlen-mode contract.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: non-packed dense path (`[B,S,H,D]`) invokes SM100 FlashMLA forward/backward with fixed-length mode (`is_varlen=false`), eliminating varlen-mode misuse in step-loop training.
  - Status note: `_MlaFa4FwdFlashMlaBwd` now passes `is_varlen=false` in both `dense_prefill_fwd` and `dense_prefill_bwd` calls for non-packed execution.

- [x] D-032 Harden FlashMLA SM100 Python binding contracts with fail-fast checks.
  - Files: `nmoe/csrc/flashmla_sm100_bindings.cpp`
  - Acceptance: malformed tensor/device/dtype/layout inputs fail before kernel launch; no silent undefined behavior from invalid lse stride/shape contracts.
  - Status note: added strict CUDA/device/dtype/shape/contiguity checks (including `lse.stride(0)==1` and finite `softmax_scale`) for both dense forward and backward entrypoints.

- [x] D-017 Add fused packed-RoPE support for per-sample position-reset tables (`[B,S,half]`) without falling back.
  - Files: `nmoe/attention/rope.py`
  - Acceptance: packed-mode 3D cos/sin inputs use fused CUDA kernels safely (no undefined layout assumptions / no OOB behavior).
  - Status note: `rotate_pe` and `rotate_pe_partial` now support both 2D and 3D BF16 tables; 3D path flattens to a fused-safe `[1, B*S, H, D]` invocation with `[B*S, half]` cos/sin.

- [x] D-018 Normalize packed `cu_seqlens` validation controls and SDPA mask inputs.
  - Files: `nmoe/attention/mla.py`, `nmoe/model.py`
  - Acceptance: packed `cu_seqlens` contract checks remain available via env flag while hot path avoids mandatory per-step host sync; SDPA mask path normalizes `cu_seqlens` device/dtype.
  - Status note: strict packed validation is now opt-in (`NMOE_VALIDATE_PACKED_CU_SEQLENS=1`), keeping default FlashMLA packed path sync-light; model-level packed RoPE position reset path still validates boundaries before `searchsorted`.

- [x] D-031 Remove mandatory packed FlashMLA max-seqlen scalar host sync.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: packed FlashMLA forward no longer reads `max_seqlen` from a CUDA scalar via `.item()` on each step.
  - Status note: packed path now uses `max_seqlen=seqlen` bound for varlen launch and keeps scalar syncs behind optional validation checks.

- [x] D-019 Align packed SDPA fallback semantics with FlashMLA for padded tails.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: SDPA packed fallback does not produce non-zero outputs for padded tail tokens (`position >= total_tokens`).
  - Status note: block-causal mask now includes a valid-token mask (`q,k < total_tokens`) so padded positions are fully masked.

- [x] D-020 Bound fused-RoPE output cache growth.
  - Files: `nmoe/attention/rope.py`
  - Acceptance: varying sequence-shape runs do not grow `_rope_buf` unbounded across steps.
  - Status note: added `NMOE_ROPE_CACHE_LIMIT` (default `8`) and oldest-entry eviction for `_rope_buf`.

## E) ECO Optimizer Kernel Work

- [x] E-001 Fuse NVFP4 group-scale tighten into main ECO update kernel.
  - Files: `nmoe/eco.py`, `nmoe/csrc/eco_adam.cu`
  - Acceptance: remove post-pass tighten kernel(s); unchanged update numerics.
  - Status note: `k_eco_adam_nvfp4_update` now computes and writes true per-group E4M3 scales inline; Python no longer launches `_recompute_nvfp4_group_scales` in fused hot path.

- [~] E-002 Fold FP8 row-scale recompute for `m/v` into main write path.
  - Files: `nmoe/csrc/eco_adam.cu`
  - Acceptance: fewer kernels and DRAM reads/writes; stable loss.
  - Status note: non-factored path now fuses `m/v` row-scale recompute into one pair kernel launch; full in-main-kernel fold remains.

- [~] E-003 Collapse factored-v prepass kernels and remove `v_rms` atomic hotspot.
  - Files: `nmoe/csrc/eco_adam.cu`
  - Acceptance: reduced launch count and atomic stalls; parity maintained.
  - Status note: `k_factored_v_row` now does one `atomicAdd` per row block for `v_rms` accumulation (removed per-warp shuffle/atomic overhead); factored-v finalize kernel (`k_factored_v_rms_finalize`) was removed by accumulating normalized contributions in-row (one fewer launch per factored-v update). Full prepass collapse remains.

- [ ] E-004 Remove triple gradient reread in factored-v path.
  - Files: `nmoe/csrc/eco_adam.cu`
  - Acceptance: lower memory traffic in Nsight; same optimizer results.

- [x] E-005 Accept BF16 grad directly in ECO CUDA API and cast in-kernel.
  - Files: `nmoe/eco.py`, `nmoe/csrc/eco_adam.cu`, bindings.
  - Acceptance: cast kernel removal; step parity within tolerance.
  - Status note: added BF16 ECO entrypoints (`eco_adam_*_bf16`, `eco_mv_accumulate*_bf16`) and Python dispatch; fused path no longer forces BF16->FP32 cast for BF16-wire updates.

- [ ] E-006 Port vectorized state I/O patterns (`float4` where aligned).
  - Files: `nmoe/csrc/eco_adam.cu`
  - Acceptance: improved occupancy/bandwidth on aligned shapes, with safe fallback.

- [ ] E-007 Amortize Philox generation (4 elems per call path).
  - Files: `nmoe/csrc/eco_adam.cu`, `nmoe/csrc/ptx.cu`
  - Acceptance: fewer RNG ops, deterministic-seed contract retained.

- [ ] E-008 Emit blockscaled cache during ECO update instead of invalidating + reconverting.
  - Files: `nmoe/eco.py`, `nmoe/csrc/quant.cu`
  - Acceptance: lower backward->next-forward gap; cache parity verified.

- [ ] E-009 Optimize CT->MMA kernels to reduce redundant nibble/scale loads.
  - Files: `nmoe/csrc/quant.cu`
  - Acceptance: kernel-only microbench improvement, exact output parity.
  - Status note: CT->MMA kernels now broadcast per-group E4M3 scales via warp shuffles using group-index matching (`__match_any_sync`, safe for non-32-aligned group boundaries), use power-of-two fast paths for `group_idx`, and remove branchy nibble decode in `ct_nvfp4_to_bf16` by using `ptx::e2m1_nibble_to_f32`.

- [ ] E-010 Precompute row->expert mapping in strided quant kernels.
  - Files: `nmoe/csrc/quant.cu`
  - Acceptance: reduced branch/index overhead with identical results.
  - Status note: immediate improvement landed by replacing O(E) per-row expert boundary scans with O(log E) device binary search in strided quant/swiglu-quant kernels. Full precomputed mapping path is still pending.

- [x] E-011 Remove grouped-dense fallback host sync path in quant kernels.
  - Files: `nmoe/csrc/quant.cu`
  - Acceptance: no D2H metadata/sync in this path.
  - Status note: legacy `grouped_dense_nvfp4_gemm_bf16_strided` host-sync fallback is now fail-fast (`cudaErrorNotSupported`) instead of silently running D2H+sync.

- [ ] E-012 Add single binding entrypoint for full ECO update sequence.
  - Files: `nmoe/csrc/bindings.cpp`, `nmoe/eco.py`
  - Acceptance: lower Python dispatch overhead; graph-capture behavior improved.

- [~] E-013 Remove host-driven per-expert BF16 wgrad loop.
  - Files: `nmoe/csrc/gemm.cu`, `nmoe/csrc/bindings.cpp`, `nmoe/moe.py`
  - Acceptance: no host offset sync gate in MoE backward hot path (`copy_event.synchronize()` + host offset walk); grouped/device-driven matmul path replaces per-expert host loop launches.
  - Status note: partial improvement landed: BF16 wgrad kernels now use adaptive zeroing (full-buffer memset only when many experts are empty), sparse/empty expert regions are zeroed with coalesced contiguous memset runs, and paired `W1/W3` BF16 wgrad now runs through one combined path (`bf16_wgrad_w13_pair_cublaslt`). Host-driven offset handoff (`copy_event.synchronize()` + host offsets) still remains.

- [x] E-014 Reduce grouped BF16 wgrad host descriptor churn.
  - Files: `nmoe/csrc/gemm.cu`
  - Acceptance: grouped-GEMM scratch arrays avoid redundant `reserve()` calls on every launch.
  - Status note: launch scratch reset now reserves only on capacity growth, reducing per-step host overhead for repeated grouped wgrad calls.

- [x] E-014 Replace per-expert BF16 cublasLt loop with grouped matmul launch.
  - Files: `nmoe/csrc/gemm.cu`
  - Acceptance: one grouped launch per tensor family (`W2`, `W1/W3`) instead of host looping `E` launches; step-time and CPU overhead improve without regression.
  - Status note: BF16 wgrad paths now submit grouped batched GEMM (`cublasGemmGroupedBatchedEx`) instead of per-expert cuBLASLt launches for `W2`, `W13`, and paired `W1/W3`; grouped-call host coefficient/group arrays and full launch-metadata vectors (`op/m/n/k/ld*` + pointer arrays) are reused thread-locally to reduce allocator churn in the hot path.

- [x] E-015 Remove per-step tiny allocation churn in blockscaled expert path.
  - Files: `nmoe/blockscaled/grouped.py`
  - Acceptance: hot path no longer allocates per-call `offs` (`torch.cat`) and `dummy_c`; both are reused from cached expert scratch.
  - Status note: `_ExpertScratch` now caches `offs` (`[E+1]`) and `dummy_c` (`[1,1,1]` BF16), and `expert_blockscaled` updates `offs` in-place each call.

- [x] E-016 Remove unsupported ECO pseudo-fallback config path.
  - Files: `nmoe/eco.py`
  - Acceptance: `eco_require_cuda=False` fails fast at init instead of deferring to runtime fused-update failures.
  - Status note: `FusedBackwardECO` now hard-rejects `eco_require_cuda=False` and enforces CUDA-kernel-only contract at construction.

- [x] E-017 Avoid redundant cuBLAS stream rebinding in grouped wgrad launcher.
  - Files: `nmoe/csrc/gemm.cu`
  - Acceptance: grouped BF16 GEMM path does not call `cublasSetStream` when launch stream is unchanged.
  - Status note: thread-local cuBLAS state now caches last stream and only rebinds on stream transitions.

- [x] E-018 Fix grouped BF16 wgrad GEMM layout contract for column-major cuBLAS grouped API.
  - Files: `nmoe/csrc/gemm.cu`
  - Acceptance: grouped BF16 wgrad launch metadata (`opA/opB`, `m/n/k`, `ld*`) is consistent with row-major tensor storage under `cublasGemmGroupedBatchedEx`; no invalid leading-dimension combinations under large expert-token counts.
  - Status note: all BF16 grouped wgrad paths now use swapped-operand column-major mapping with explicit row-stride pointer arithmetic (`stride_*`), fixing potential invalid-value/wrong-math cases from prior metadata assumptions.

- [x] E-019 Expose blockscaled activation dequant CUDA entrypoints for backward recompute.
  - Files: `nmoe/csrc/quant.cu`, `nmoe/csrc/bindings.cpp`
  - Acceptance: Python path can invoke FP8/NVFP4 blockscaled activation dequantization to BF16 directly on GPU (`dequant_fp8_to_bf16`, `dequant_nvfp4_to_bf16`), with no host dequant fallback.
  - Status note: added C API wrappers and pybind exports for both dequant kernels and wired them into MoE hybrid backward path.

## F) Build and Compiler Tuning (SM100/B200)

- [x] F-001 Make `-rdc=true` conditional (required for NVSHMEM build only).
  - Files: `nmoe/csrc/Makefile`
  - Acceptance: default non-NVSHMEM build omits RDC/dlink; NVSHMEM build still works.
  - Status note: non-NVSHMEM path now builds with `RDC=0` (no `-rdc=true`, no device-link, no `-lcudadevrt`); NVSHMEM path retains RDC + dlink.

- [x] F-002 FlashMLA compile mode tuning (`-std=c++20` + `-DNDEBUG`) with compatibility guard.
  - Files: `nmoe/csrc/Makefile`
  - Acceptance: compiles cleanly; no correctness regressions; improved kernel stack/reg profile.
  - Status note: FlashMLA NVCC/C++ flags now use `-std=c++20 -DNDEBUG`; explicit rebuild of `flashmla_sm100` succeeded on current toolchain.

- [ ] F-003 Add ptxas diagnostic capture and spill/reg regression check.
  - Files: `nmoe/csrc/Makefile`, CI workflow
  - Acceptance: build logs expose reg/spill stats; CI can fail on defined thresholds.

- [ ] F-004 Use targeted register caps for ECO only (no global cap).
  - Files: `nmoe/csrc/Makefile`
  - Acceptance: no spills on tuned kernels and measurable speed gain.

- [ ] F-005 Ensure train image inherits exactly the built base/toolchain image.
  - Files: `docker/Dockerfile.train`, CI build workflow
  - Acceptance: no toolchain drift across stages.

- [ ] F-006 Unify arch strategy across native build paths.
  - Files: `nmoe/csrc/Makefile`, `third_party/flashmla/setup.py`
  - Acceptance: one canonical SM100 target path; consistent cubins.

- [ ] F-007 Optional PTX embedding mode for debug/forward-compat (off by default).
  - Files: `nmoe/csrc/Makefile`
  - Acceptance: deterministic default remains SASS-only; optional mode documented.

## G) Runtime and Launch Tuning (No Admin Changes)

- [~] G-001 Force `eco_allreduce_dtype=bf16` in override generation path.
  - Files: `../nmoe-multinode/unified_config.py` (default) and launch merge path.
  - Acceptance: `--set` launches cannot drift to fp32 wire unexpectedly.
  - Status note: default + launch guardrails patched; rollout verification pending.

- [~] G-002 Stop forcing `NMOE_NVTX=1` on every launch.
  - Files: `../nmoe-multinode/orchestrate.py`
  - Acceptance: profiling-only opt-in; production launch defaults to no NVTX overhead.
  - Status note: production default switched to off unless explicitly set in `nccl.extra_env`.

- [~] G-003 Default `NMOE_TIMERS=0` for production launch.
  - Files: `../nmoe-multinode/orchestrate.py`
  - Acceptance: timer instrumentation off by default in production.
  - Status note: production default switched to off unless explicitly set in `nccl.extra_env`.

- [~] G-004 Add socket NCCL tuning bundle.
  - Files: `../nmoe-multinode/cluster.yaml` (`nccl.extra_env`)
  - Candidate vars: `NCCL_NSOCKS_PERTHREAD=4`, `NCCL_SOCKET_NTHREADS=2`, `NCCL_BUFFSIZE=8388608`, `NCCL_MIN_NCHANNELS=4`.
  - Acceptance: improved collective throughput and stable run.
  - Status note: defaults added to `cluster.yaml`; cluster A/B validation pending.

- [~] G-005 Add `CUDA_DEVICE_MAX_CONNECTIONS=1` in production launch env.
  - Files: `../nmoe-multinode/cluster.yaml` and launch env path.
  - Acceptance: better overlap in traces with no stability penalty.
  - Status note: default added to `cluster.yaml`; overlap validation pending.

- [ ] G-006 Tune ZeRO-2 RS chunk size sweep.
  - Files: launch env plus `nmoe/zero2.py` default awareness.
  - Acceptance: selected chunk value improves p50 step-time over baseline.

- [ ] G-007 Tune ECO allreduce chunk from 16MB to 32MB (A/B with 16/32/64).
  - Files: `configs/dsv3_reap_sft_16node.toml`
  - Acceptance: improved throughput without increasing stalls/timeouts.

- [ ] G-008 Tune ECO async queue depth from 1 to 2 (and budget accordingly).
  - Files: `configs/dsv3_reap_sft_16node.toml`
  - Acceptance: overlap gain with stable memory footprint and no comm deadlocks.

- [ ] G-009 Evaluate `sft_packing_enabled=true` for actual dataset.
  - Files: `configs/dsv3_reap_sft_16node.toml`
  - Acceptance: effective tokens/s gain and no training quality regression.

## H) Benchmarking and Regression Gating

- [ ] H-001 Fix stale E2E benchmark script API mismatch.
  - Files: `bench_moe_e2e.py`
  - Acceptance: script runs cleanly against current `Rdep` API.

- [ ] H-002 Add attention/router/rdep/eco microbench suite into repo.
  - Source seed scripts currently available at `/tmp/nmoe_microbench_suite.py`.
  - Target location: `scripts/perf/` or `nmoe/tools/`.
  - Acceptance: outputs JSON with p50/p95/p99 for all target kernels.

- [ ] H-003 Add short E2E TPS suite into repo.
  - Source seed script currently available at `/tmp/nmoe_e2e_tps_suite.sh`.
  - Acceptance: emits run summary with gpu/node TPS metrics.

- [ ] H-004 Add baseline vs candidate comparator script.
  - Acceptance: automated pass/fail by threshold.

- [ ] H-005 Add CI perf gate for key metrics.
  - Files: perf workflow.
  - Acceptance: fails on >5% microbench regression or >3% E2E TPS regression.

## I) Verification Checklist (for each task)

- [ ] I-001 Correctness: output parity (or tolerance-backed parity for mixed precision).
- [ ] I-002 Stability: no NaN/Inf, no hangs, no watchdog regressions.
- [ ] I-003 Performance: capture before/after p50 and p95 step-time.
- [ ] I-004 Reproducibility: capture exact commit SHAs and config fingerprint.
- [ ] I-005 Rollback plan: one-command revert path if regressions appear.

## J) Execution Order (Recommended)

1. B-001, B-002, B-003
2. C-001, C-002
3. E-001, E-002
4. G-001, G-002, G-003, G-004
5. D-002, D-003, D-004
6. F-001, F-003
7. H-001, H-002, H-003, H-004, H-005

## K) Deployment, Rollout, and Repo Sync (nmoe + nmoe-multinode)

- [x] K-001 Default provisioning behavior uses latest remote `master` when `--repo-sha` is not provided.
  - Files: `../nmoe-multinode/orchestrate.py` (`resolve_repo_target_sha`)
  - Acceptance: `orchestrate.py provision --clone-only` resolves one immutable remote tip SHA and applies same SHA to all nodes.

- [x] K-002 Add an explicit "force full repo refresh" runbook and verify it for all 16 nodes.
  - Command target: `../nmoe-multinode/orchestrate.py provision --clone-only --force`
  - Acceptance: all nodes report provision success and SHA parity check passes.
  - Status note: live run completed successfully (16/16 provisioned) and `repo-parity --strict --expected-sha 662b162332bdf9117ee08507e41b11a7156daa07` returned 16/16 OK.

- [ ] K-003 Add an explicit "force full stack refresh" runbook (agent + torch + repo + csrc).
  - Commands: `deploy --force`, `provision --force`, then launch.
  - Acceptance: all nodes run the same nmoe commit and csrc build hash.

- [x] K-004 Add rollout precheck task that blocks when local `nmoe` or `nmoe-multinode` has uncommitted changes.
  - Files: `../nmoe-multinode/orchestrate.py` preflight/provision path.
  - Acceptance: rollout prints actionable remediation and exits before mutating nodes.
  - Status note: `deploy` and `provision` now enforce dual-repo clean checks (`nmoe` + `nmoe-multinode`) unless `--allow-dirty`.

- [x] K-005 Add a one-command "repo parity report" across all training nodes.
  - Output: node -> nmoe HEAD SHA, dirty state, submodule SHAs.
  - Acceptance: report can be used as launch gate.
  - Status note: `repo-parity` command added and exercised against 16 nodes with `--strict`.

- [ ] K-006 Add a one-command "force provisioning from exact SHAs in both repos" mode.
  - Inputs: nmoe SHA + nmoe-multinode SHA manifest.
  - Acceptance: immutable rollout with reproducible provenance.

- [x] K-007 Add a documented command matrix for post-commit update workflows.
  - Cases: latest-tip rollout, pinned SHA rollout, forced rollout.
  - Acceptance: single docs page with copy/paste-safe commands.
  - Status note: added `../nmoe-multinode/ROLLOUT_COMMAND_MATRIX.md`.

## L) Crash Forensics and Per-Node Logging

- [x] L-001 Expose per-node training log discovery/listing.
  - Files: `../nmoe-multinode/orchestrate.py` (`logs-nodes`, `logs --list`)
  - Acceptance: user can list all nodes and see log availability.

- [x] L-002 Expose per-node agent journald retrieval.
  - Files: `../nmoe-multinode/orchestrate.py` (`agent-logs --all`)
  - Acceptance: user can fetch `nmoe-agent` logs from any/all nodes.

- [x] L-003 Ensure training launch always emits per-rank log files on each node.
  - Files: `../nmoe-multinode/agent.py`, launcher command assembly.
  - Acceptance: every node has `train_rank*.log` for the active run.
  - Status note: `agent.py launch_training()` now bootstraps/truncates `train_rank{node_rank}.log` before preflight checks and appends explicit launch success/failure markers; `orchestrate.py launch_training()` performs best-effort per-rank log inventory checks with warnings.

- [~] L-004 Add automatic crash bundle collection on failed launch.
  - Bundle: last N training log lines, last N agent journal lines, `nvidia-smi`, `dmesg`, socket stats.
  - Acceptance: one command exports all-node crash evidence.
  - Status note: `crash-bundle` command added and launch failure path now auto-captures bundles; full 16-node failure-path drill pending.

- [~] L-005 Add "hang snapshot" command for NCCL incidents.
  - Commands per node: `ss`, `nstat`, `ethtool -S`, GPU power/util, kernel tail.
  - Acceptance: snapshot artifacts are timestamped and grouped by node.
  - Status note: `hang-snapshot` command added and exercised across the fleet; one host-key trust fix still required for 16/16 coverage.

- [ ] L-006 Add log retention/rotation policy for long multi-week runs.
  - Acceptance: no disk fill failures while preserving enough history for root-cause analysis.

## M) No-Admin Fastest Path (Current Constraint: No GCP Admin Changes)

- [~] M-001 Implement chunked DP all-reduce in ECO for large tensors (size-thresholded path).
  - Files: `nmoe/eco.py`
  - Acceptance: large all-reduce payload is split into bounded chunks with maintained numerics.
  - Status note: chunk-threshold logic patched; cluster perf/stability validation pending.

- [ ] M-002 Add async-vs-sync DP all-reduce mode switch for controlled A/B.
  - Files: `nmoe/eco.py`, config surface
  - Acceptance: reproducible benchmark table for async and sync at 16 nodes.

- [~] M-003 Sweep chunk size + async queue depth jointly (16/32/64MB x depth 1/2/3).
  - Files: `configs/dsv3_reap_sft_16node.toml`, perf scripts
  - Acceptance: select fastest stable point with no hangs for >=200 steps.
  - Status note: `orchestrate.py sweep-eco-comm` implemented with scoring JSON output; full 200-step matrix run pending.

- [ ] M-004 Add comm-stream priority and explicit wait discipline to avoid hidden host sync.
  - Files: `nmoe/eco.py`, `nmoe/zero2.py`
  - Acceptance: no extra host sync points in trace; overlap improves.

- [~] M-005 Disable default overhead instrumentation in production.
  - Files: `../nmoe-multinode/orchestrate.py`
  - Acceptance: `NMOE_NVTX=0` and `NMOE_TIMERS=0` unless explicitly enabled.
  - Status note: production defaults patched; rollout-level verification pending.

- [~] M-006 Add progressive scale stress test command (2/4/8/16 nodes at fixed large tensor sizes).
  - Acceptance: exact failure threshold and transport behavior are captured before full launch.
  - Status note: `orchestrate.py progressive-scale` implemented with DP-size correctness checks, fatal-log detection, and JSON report output; full 2/4/8/16 live run pending.

- [ ] M-007 Add no-admin emergency fallback policy.
  - Options: temporary sync mode, smaller chunks, reduced overlap budget.
  - Acceptance: cluster can complete steps without infra changes while root cause work continues.

## N) Admin-Side Infra Migration (Optional Track, Highest Ceiling)

- [ ] N-001 TCPXO plugin/daemon installation checklist for GCP A4 (Rocky 9) with verification commands.
  - Verification: `libnccl-net.so` discovered, service healthy, NCCL plugin selected.
  - Acceptance: documented, repeatable install/verify flow per node.

- [ ] N-002 MTU migration checklist from 1460 to 8896 (VPC + NIC + end-to-end validation).
  - Acceptance: jumbo ping and NCCL traffic validated post-change.

- [ ] N-003 Placement and topology validation checklist (compact placement/resource policy).
  - Acceptance: node placement metadata captured and archived with run metadata.

- [ ] N-004 NCCL plugin activation policy (`NCCL_NET_PLUGIN`, `NCCL_NET`, library path hygiene).
  - Acceptance: launch logs confirm expected plugin path, no silent fallback to plain sockets.

- [ ] N-005 Pre/post migration benchmarking protocol.
  - Metrics: all-reduce latency curve by tensor size, p50/p95 step-time, link saturation.
  - Acceptance: quantitative evidence of migration impact.

- [ ] N-006 Rollback checklist for failed infra migration.
  - Acceptance: one documented path back to known-good socket mode with minimal downtime.

## O) CUDA-Only Path Validation (No Silent Python/PyTorch Fallback)

- [~] O-001 Add launch-time CUDA-kernel presence checks (router, ECO, RDEP, quant).
  - Files: `nmoe/train.py`, `nmoe/eco.py`, `nmoe/moe.py`
  - Acceptance: run fails fast if expected custom CUDA bindings are unavailable.
  - Status note: validator now runs from both `main()` and direct `train(cfg)` entry, with optional FA4+FlashMLA import checks; end-to-end launch validation pending.

- [x] O-002 Add runtime counters proving fused kernel path usage every step.
  - Output: step-level counters for fused-router, ECO fused update, RDEP dispatch kernel variant.
  - Acceptance: counters stay at 100% expected path for production config.
  - Status note: added per-step `runtime/*` metrics in `train.py` sourced from new counters in `model.py`, `moe.py`, and `eco.py` (`fused_router_calls`, ECO fused calls, and RDEP IPC/hybrid dispatch variants).

- [x] O-003 Add CI/static audits for hot-path anti-patterns.
  - Checks: `.item()` in forward hot path, host spin waits, host metadata sync in dispatch path.
  - Acceptance: CI fails if regressions reintroduce these patterns.
  - Status note: added `scripts/check_hotpath_regressions.py` and wired it into `.github/workflows/ci.yml` lint job.

- [ ] O-004 Add optional debug mode that logs actual chosen kernel backend per subsystem.
  - Acceptance: one-run audit can prove no fallback decisions occurred.

- [ ] O-005 Add unit/integration tests for fallback guardrails.
  - Acceptance: misconfigured fallback toggles are rejected by tests and launch path.

## P) Supply-Chain and Reproducibility Hardening

- [ ] P-001 Audit and pin all runtime dependencies used during node provisioning with hashes.
  - Files: `../nmoe-multinode/locks/*.lock`
  - Acceptance: no unpinned runtime install path remains.

- [x] P-002 Pin external FlashMLA and CUTLASS refs to immutable commits.
  - Files: `../nmoe-multinode/orchestrate.py`
  - Acceptance: provision/build enforces pinned SHAs.

- [ ] P-003 Replace destructive repo sync primitives with safer, reversible flow where possible.
  - Files: `../nmoe-multinode/orchestrate.py`
  - Acceptance: failed sync cannot strand nodes in partial state.

- [x] P-004 Capture run provenance manifest.
  - Manifest: nmoe SHA, nmoe-multinode SHA, torch lock hash, FlashMLA SHA, config hash.
  - Acceptance: each launch writes immutable provenance artifact.
  - Status note: `orchestrate.py launch` now creates immutable `artifacts/provenance/<run_id>.json` + `.sha256`, includes run ID propagation (`NMOE_RUN`), repo/dependency/config fingerprints, and redacted forwarded env.

- [ ] P-005 Rotate and re-issue any exposed third-party API credentials before production runs.
  - Acceptance: revoked old tokens and new secrets distributed via secure env management.

## Q) Immediate Next Queue (Execution Priority)

1. [~] M-001 Chunk large ECO DP all-reduce.
2. [~] M-003 Sweep chunk size + queue depth and lock best stable pair.
3. [~] O-001 Add launch-time custom CUDA binding presence checks.
4. [~] L-004 Add automatic crash bundle collection on launch failure.
5. [x] K-005 Add one-command repo parity report across 16 nodes.
6. [~] M-006 Add progressive scale stress test command (2/4/8/16 nodes).
7. [~] G-001 Force `eco_allreduce_dtype=bf16` in override generation path.
8. [~] G-002 and G-003 disable default NVTX/timers in production launches.
9. [x] C-001 Implement blockscaled 2-phase dispatch.
10. [~] E-002 Fold FP8 row-scale recompute for `m/v` into main write path.

## R) Latest Deep Audit Batch (Current Session)

- [x] R-001 Add direct expert-major NVFP4 transpose dequant path in CUDA.
  - Files: `nmoe/csrc/quant.cu`, `nmoe/csrc/bindings.cpp`, `nmoe/moe.py`, `nmoe/checkpoint.py`
  - Acceptance: transpose mode `2` writes `[E, K, M]` directly and removes mandatory post-kernel permute copy on supported builds.

- [x] R-002 Add strict capability/ABI guard for NVFP4 transpose mode.
  - Files: `nmoe/moe.py`, `nmoe/checkpoint.py`, `nmoe/train.py`
  - Acceptance: production path fails fast when `ct_nvfp4_to_bf16_max_transpose_mode < 2` (`NMOE_REQUIRE_CT_TRANSPOSE_MODE2=1`).

- [x] R-003 Remove hidden per-refresh contiguous copies for NVFP4 cache build.
  - Files: `nmoe/model.py`, `nmoe/blockscaled/grouped.py`
  - Acceptance: canonicalize NVFP4 buffers once at set-time; cache refresh no longer issues hidden `.contiguous()` copies on packed tensors.

- [x] R-004 Fix padded output initialization correctness after tail-memset optimization.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: multi-expert layouts (`n_local>1`) use full safe prefill; single-expert retains tail-only fast path.

- [x] R-005 Add gather-path bounds guardrails for `M_recv/M_pad`.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: gather APIs reject out-of-capacity `M_recv` and oversized `M_pad` before pointer arithmetic/scratch access.

- [x] R-006 Remove hard-fail regression for distributed hybrid backward in MoE.
  - Files: `nmoe/moe.py`
  - Acceptance: hybrid mode routes through `gather_dy_dist_bf16` path instead of IPC-only split-dGate abort path.

- [x] R-007 Eliminate packed MLA GPU↔CPU tensor round-trip in FlashMLA packed path.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: no explicit `.to(cpu)` / `.to(cuda)` round-trip in `_mla_flashmla_packed_forward`; requires CUDA int32 contiguous `cu_seqlens`.

- [x] R-008 Remove silent checkpoint layout corruption path on mismatch.
  - Files: `nmoe/checkpoint.py`
  - Acceptance: no blind `reshape(target_shape)` fallback; only exact shape or explicit transpose-compatible copies are permitted.

- [x] R-009 Fuse router backward host call path (zero + backward + transpose) into one C API.
  - Files: `nmoe/csrc/router_bwd.cu`, `nmoe/fused_router.py`
  - Acceptance: BF16 router backward uses `fused_router_backward_bf16_fused` when available, removing one Python call boundary and redundant host-side orchestration.

- [x] R-010 Remove redundant per-lane scalar setup in router backward kernel.
  - Files: `nmoe/csrc/router_bwd.cu`
  - Acceptance: lane 0 computes token-scalar terms (`gate_sum`, `dot_gs`, `d_logit`) once and broadcasts to warp lanes.

- [x] R-011 Add warp-stride CTA capping in RDEP dispatch/gather hot kernels.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: launch CTA count is capped by SM-aware helper for warp-stride kernels while preserving full work coverage.

- [x] R-012 Make IPC handle open/sync paths fail-closed.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: IPC handle open/sync errors are fatal (no "error + return" silent no-op path), and IPC barriers reject use before handle-open readiness.

- [x] R-013 Make async `M_recv` prefetch stream-safe.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: async host counter slots are keyed by `(device counter ptr, stream)` and cannot leak pending reads across streams.

- [x] R-014 Harden IPC barrier phase discipline.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: IPC barrier path enforces single-stream usage per profile and checks barrier kernel launch errors immediately.

- [x] R-015 Add BF16 dispatch alignment fail-fast.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: `rdep_dispatch_meta_bf16` rejects invalid `align` (`<=0` or non-multiple-of-8) before padded-mapping kernels.

- [x] R-016 Remove additional fail-open return sites in MoE dispatch/return/backward comm wrappers.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: gather/return/distributed backward wrappers abort on invalid capacity/world/tok-slot states instead of silently returning.

- [x] R-017 Disable dropped-token polling by default on production hot path.
  - Files: `nmoe/moe.py`
  - Acceptance: dropped-token polling is opt-in (`NMOE_CHECK_DROPPED_TOKENS=1`) and no longer enabled by default.

- [x] R-018 Replace assert-only quantization guardrails with runtime exceptions.
  - Files: `nmoe/blockscaled/grouped.py`
  - Acceptance: shape/stride invariants raise `RuntimeError` under all Python modes (including `-O`).

- [x] R-019 Add strict BF16 C-API guardrails for fused AdamW/RoPE.
  - Files: `nmoe/csrc/adamw_fused_step.cu`, `nmoe/csrc/rope_fused.cu`
  - Acceptance: APIs fail fast on invalid sizes/null pointers/unsupported arch; capability probe returns real CUDA errors; RoPE launch rejects index-overflow shapes.

- [x] R-020 Remove forward-path host scalar sync for blockscaled `M_pad`.
  - Files: `nmoe/moe.py`
  - Acceptance: no `.item()` read on `M_host` in `_MoEBlockscaledFused.forward`; `M_pad` is computed from deterministic `M_recv` and bounded by capacity formula.

- [x] R-021 Fix CT NVFP4->blockscaled MMA kernels to handle 2D launch indexing.
  - Files: `nmoe/csrc/quant.cu`
  - Acceptance: `k_ct_nvfp4_to_blockscaled_mma` and `_interleaved` use `(blockIdx.x, blockIdx.y)` directly for 2D launches and keep linearized fallback only for `gridDim.y==1`.

- [x] R-022 Tighten legacy BF16 return-scatter dynamic launch bound.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: `k_scatter_received_bf16_dynamic` launch uses `min(capacity, T*K)` work bound instead of raw `capacity`.

- [x] R-023 Remove hybrid blockscaled dispatch scalar counter-add micro-kernel.
  - Files: `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: `k_merge_nvshmem_into_ipc_blockscaled_dynamic` updates `ipc_counter` in-kernel; `k_add_counter_sum<<<1,1>>>` launch is removed.

- [x] R-024 Topology-gate hybrid forwarding stages and tighten launch work bounds.
  - Files: `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: forwarding kernels are skipped for `num_nodes==1`; dispatch forwarding/merge launch sizing uses deterministic receive upper bounds; return forwarding launch sizing uses `min(capacity, T*K)`.

- [x] R-025 Remove dropped-counter double counting in dynamic NVSHMEM clamp stages.
  - Files: `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: dynamic forward/merge/return clamp paths no longer re-add `nv_count-capacity` drops already counted at producer-side overflow.

- [x] R-026 Clamp hybrid blockscaled IPC counter on all merge early-return paths.
  - Files: `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: `k_merge_nvshmem_into_ipc_blockscaled_dynamic` writes bounded `ipc_counter` before any early return, preventing stale `counter>capacity` overflow aborts.

- [x] R-027 Harden distributed dX scatter contracts for vectorized kernels.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: `rdep_scatter_dx_dist_bf16` now enforces `H % 8 == 0` and BF16/blockscaled state-`H` equality, matching `_from_pad` guardrails and preventing vectorized OOB risk.

- [x] R-028 Add blockscaled IPC buffer-pointer fail-fast checks in distributed dX scatter.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: `rdep_scatter_dx_dist_bf16` validates all `g_block.buffer_ptrs[r]` before tok-slot kernels, mirroring BF16 branch safety.

- [x] R-029 Enforce single-stream hybrid backward tok-slot phase ownership.
  - Files: `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: hybrid backward phase allocator fails fast when called from multiple CUDA streams, preventing shared tok-slot buffer clobber races.

- [x] R-030 Reset hybrid backward phase/stream latches on NVSHMEM lifecycle transitions.
  - Files: `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: `g_bwd_phase` and `g_bwd_stream_slot` are reset on `init()` and `finalize()` paths, preventing stale stream ownership after re-init.

- [x] R-031 Fix distributed blockscaled backward dequant SFA layout contract.
  - Files: `nmoe/csrc/quant.cu`, `nmoe/csrc/bindings.cpp`, `nmoe/moe.py`, `nmoe/train.py`
  - Acceptance: distributed `gather_xe_blockscaled` backward path dequants with CUTLASS MMA-swizzled SFA (`dequant_*_to_bf16_mma_sf`) and no longer applies row-major SFA indexing to swizzled buffers.
  - Status note: added dedicated FP8/NVFP4 MMA-SF dequant kernels + bindings, switched distributed backward callsites in `moe.py`, and made preflight require the new bindings for blockscaled runs.

- [x] R-032 Remove env-gated NVFP4 transpose downshift in checkpoint load path.
  - Files: `nmoe/checkpoint.py`, `nmoe/train.py`
  - Acceptance: NVFP4 checkpoint transpose mode 2 is mandatory for expert-major load path; mode-1 fallback cannot be re-enabled via environment override.
  - Status note: removed `NMOE_REQUIRE_CT_TRANSPOSE_MODE2` runtime gate from load/preflight logic and enforced strict `ct_nvfp4_to_bf16_max_transpose_mode >= 2` for `dtype=nvfp4`.

- [x] R-033 Enforce blockscaled distributed dispatch purity at runtime.
  - Files: `nmoe/train.py`
  - Acceptance: blockscaled distributed training aborts if BF16 distributed dispatch counters are non-zero.
  - Status note: added hard runtime check on `rdep_dispatch_bf16_ipc_calls + rdep_dispatch_bf16_hybrid_calls` in training step loop.

- [x] R-034 Add aligned dequant fast paths for FP8/NVFP4 row-major and MMA-SF kernels.
  - Files: `nmoe/csrc/quant.cu`
  - Acceptance: dequant launchers select branch-free aligned kernels when `K % 32 == 0`, reducing tail-branch overhead in hot dequant loops.
  - Status note: added `k_dequantize_fp8_to_bf16_aligned32`, `k_dequantize_fp8_to_bf16_mma_sf_aligned32`, `k_dequantize_nvfp4_to_bf16_aligned32`, and `k_dequantize_nvfp4_to_bf16_mma_sf_aligned32`, with launcher dispatch fallback to generic kernels for non-aligned shapes.

- [x] R-035 Trim swizzle and grouped-wgrad host overhead in quant/gemm setup paths.
  - Files: `nmoe/csrc/quant.cu`, `nmoe/csrc/gemm.cu`
  - Acceptance: swizzle path avoids redundant memset when output is fully overwritten; grouped wgrad setup exits early on all-empty expert batches and skips zero-run branching when no zero experts exist.
  - Status note: `launch_swizzle_sf_strided` now conditionally skips full memset; grouped wgrad setup now uses `do_zero_run_memset` gate and early return when all experts are empty.

- [x] R-036 Cache 2-phase dispatch receive offsets in shared memory.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: 2-phase BF16 and blockscaled dispatch write kernels do not repeatedly load `recv_offsets[dest]` from global memory for every token-slot assignment.
  - Status note: `k_dispatch_2phase_bf16` and `k_dispatch_2phase_blockscaled` now preload per-rank receive offsets into CTA shared memory and reuse via lane-0 broadcast.

- [x] R-037 Account dynamic NVSHMEM forward/return clamp overflow in dropped counters.
  - Files: `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: dynamic clamp sites (`nv_count > capacity`) update dropped counters before clamping to prevent hidden overload undercount.
  - Status note: added one-time atomic dropped accounting in dynamic forward and return forwarding kernels prior to `nv_count = capacity`.

- [x] R-038 Remove per-step training loss `.item()` sync and fix runtime RDEP mode lookup.
  - Files: `nmoe/train.py`
  - Acceptance: step loop no longer forces host sync via `accumulated_loss_gpu.item()` each iteration; BF16-dispatch purity guard reads runtime RDEP mode from MoE layer RDEP state instead of a missing model attribute.
  - Status note: `last_loss` now stays device-resident until logging cadence requires scalarization, and hybrid BF16-dispatch fail-fast uses cached runtime mode from `model._moe_layers[0]._rdep._mode`.

- [x] R-039 Allow fresh NVFP4 runs without checkpoint-load false positives.
  - Files: `nmoe/train.py`
  - Acceptance: direct NVFP4 checkpoint-load requirement is enforced only on real resume (`start_step > 0`), not on fresh launches with default `resume=true`.
  - Status note: startup guard now keys on `start_step` returned by checkpoint loader, preventing pre-step aborts on intentional from-scratch runs.

- [x] R-040 Harden runtime RDEP mode contract and trim repeated host-side runtime probes.
  - Files: `nmoe/train.py`, `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`, `nmoe/zero2.py`, `nmoe/opt.py`
  - Acceptance: runtime BF16-dispatch purity guard cannot silently down-classify missing mode to IPC; warp-stride launch cap helper avoids per-call CUDA device queries; NVTX checks in optimizer/ZeRO paths are import-time cached.
  - Status note: train startup now validates each MoE layer mode is explicitly one of `ipc|hybrid` and consistent across layers; launch-cap helpers now use thread-local one-time cap initialization; optimizer and ZeRO NVTX wrappers now use cached capability booleans.

- [x] R-041 Reduce per-step parameter iteration overhead in training loop.
  - Files: `nmoe/train.py`
  - Acceptance: optimizer-step hot path avoids repeated full `model.parameters()` iteration and avoids rebuilding dense clip candidate sets from scratch each step.
  - Status note: train loop now precomputes `grad_norm_params` once and deduplicated `dense_clip_params` from dense optimizer groups; per-step grad filtering operates on those cached lists.

- [x] R-042 Bypass CUB sort and sort-temp setup on trivial local routing.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: dispatch paths skip radix sort when `n_local == 1` or `M_recv <= 1`, and alloc paths do not allocate sort temp storage when `n_local == 1`.
  - Status note: BF16/blockscaled IPC and hybrid dispatch now gate `DeviceRadixSort::SortPairs` on `n_local > 1 && M_recv > 1`; alloc paths keep `sort_temp=nullptr` unless sort can execute.

- [ ] R-043 Remove pre-zero tok-slot IPC barriers without correctness risk.
  - Files: `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: hybrid backward no longer needs pre-memset IPC barrier while preserving phase isolation under overlap; local-path gate accumulation is phase-safe without relying on mailbox zero timing.
  - Status note: current pre-zero barriers are still required because local-path collection reads `ipc_tok_gate` when `tok_tag != phase`; removing them safely needs a phase-tagged local path.

- [x] R-044 Skip empty-work tok-slot send/gather kernel launches in hybrid backward.
  - Files: `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: hybrid backward avoids launching `M`-scaled tok-slot kernels when `M == 0`, while preserving barrier/collective ordering.
  - Status note: guarded `k_gather_dy_from_stage_and_send_gate_hybrid`, `k_gather_dy_from_stage_nogate_hybrid`, `k_send_dgate_tokslot_hybrid`, `k_send_dx_tokslot_hybrid`, and `k_send_dx_tokslot_hybrid_from_pad` behind `if (M > 0)`.

- [x] R-045 Pre-arm async `M_recv` handoff before blocking reads in IPC/hybrid dispatch.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: dispatch paths issue stream-ordered async D2H for receive counters before `read_device_int_blocking`, reducing blocking wait tail without changing semantics.
  - Status note: added `poll_device_int_async` helper parity in NVSHMEM and inserted pre-arm callsites after final visibility barriers in BF16/blockscaled IPC and hybrid dispatch paths (including restored hybrid NVSHMEM BF16/blockscaled callsites before blocking `M_recv` reads).

- [x] R-046 Add single-local-expert metadata fast path for dispatch sorting pipeline.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: when `n_local == 1`, dispatch metadata path bypasses `extract/sort/binary-search-offset/fill-dest` kernels and initializes `order/dest/offsets/offs_pad` directly.
  - Status note: wired a single-kernel `n_local==1` fast path that initializes `order`, `dest`, `offsets`, and `offs_pad` directly in BF16/blockscaled IPC and hybrid dispatch, removing the multi-kernel sort/offset pipeline for this case.

- [x] R-047 Use contiguous tail memset for `n_local==1` padding in BF16/blockscaled gather paths.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: single-local-expert path avoids per-expert padding kernels and zeros only the contiguous tail rows via `cudaMemsetAsync`.
  - Status note: BF16 and blockscaled IPC/hybrid gather paths now branch to direct tail memset when `n_local==1 && M_pad > M_recv`.

- [x] R-048 Trim metadata handoff micro-overhead in dispatch helpers.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: async counter-slot ring indexing avoids modulo ops in hot helper paths, and 32-thread offset-prefix kernels use constant full-warp mask.
  - Status note: replaced `% kMaxSlots` with bitmask indexing under power-of-two static assertions and switched 2-phase offset kernels to `0xFFFFFFFFu` warp masks.

- [x] R-049 Cache barrier CTA sizing and trim extra branch/launch overhead in dispatch-side helpers.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: IPC/NVSHMEM barrier launch thread sizing avoids recomputing power-of-two sizing each call, `M==0` skips no-op IPC dYe gather launch, and 2-phase count kernels use unsigned single-bound destination validation.
  - Status note: added cached barrier thread sizing in IPC and NVSHMEM barrier helpers, guarded `k_gather_dy_from_stage_nogate_ipc_bf16` behind `if (M > 0)`, and converted `dest` validity checks in count kernels to unsigned bound form.

- [x] R-050 Remove Python-side `offs_pad` host-copy synchronization before grouped BF16 wgrad.
  - Files: `nmoe/moe.py`, `nmoe/csrc/gemm.cu`, `nmoe/csrc/bindings.cpp`
  - Acceptance: backward path does not allocate/copy pinned `offs_pad` on Python side and does not call `copy_event.synchronize()` before `bf16_wgrad_w2_cublaslt` / `bf16_wgrad_w13_pair_cublaslt`.
  - Status note: backward now prepares host-visible `offs_pad` once via `bf16_prepare_offs_pad_host(...)` and reuses that pointer across grouped wgrad calls; Python-side copy-stream/event path was removed and no `copy_event.synchronize()` remains.

- [x] R-051 Add device-backed `offs_pad` host-view resolver for grouped BF16 wgrad entrypoints.
  - Files: `nmoe/csrc/gemm.cu`
  - Acceptance: grouped BF16 wgrad entrypoints accept CUDA `offs_pad` pointers and resolve to pinned host scratch internally without changing grouped GEMM numerics.
  - Status note: added `resolve_offs_pad_host_view(...)` and reused it across `bf16_wgrad_w2_cublaslt`, `bf16_wgrad_w13_cublaslt`, and `bf16_wgrad_w13_pair_cublaslt`.

- [x] R-052 Fuse tiny metadata offset+prefix kernels in dispatch setup.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: single launch computes local offset mark + prefix in IPC/hybrid dispatch fast paths, removing one tiny kernel launch from each path.
  - Status note: introduced fused offset-prefix helper kernels and switched BF16/blockscaled IPC and hybrid fast paths to the fused launch sequence.

- [x] R-053 Remove redundant immediate async-poll calls in non-overlap dispatch paths.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: dispatch code does not enqueue `poll_device_int_async` immediately before guaranteed blocking `read_device_int_blocking` when no overlap work exists between them.
  - Status note: pruned redundant poll calls in IPC and hybrid dispatch callsites; retained only overlap-relevant pre-arm locations.

- [x] R-054 Skip no-op barrier kernels on single-rank local paths.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: IPC barrier launches are skipped when `local_world <= 1` in dispatch hot paths.
  - Status note: dispatch BF16/blockscaled paths now gate no-op barrier helper launches on local-rank count to avoid empty kernel overhead.

- [x] R-055 Remove dead BF16 scalar tail branches under `H % 8 == 0` contract in dispatch hot loops.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: BF16 dispatch loops no longer carry tail-scalar branch paths that are unreachable with enforced `H % 8 == 0`.
  - Status note: removed dead scalar cleanup branches from IPC BF16 dispatch kernels while preserving vectorized copy semantics.

- [x] R-056 Cache one-time runtime contracts and launch config decisions in Python hot path.
  - Files: `nmoe/model.py`, `nmoe/moe.py`, `nmoe/fused_router.py`, `nmoe/blockscaled/grouped.py`
  - Acceptance: repeated per-step contract checks and device-property queries are replaced by one-time/cache-keyed validation in NVFP4 production path.
  - Status note: added one-shot router/NVFP4 contract checks, cached fused-router launch config and block sizing, cached backward binding contract checks, and cached SM/capability queries in blockscaled grouped path.

- [x] R-057 Fold BF16 AdamW scalar tail update into vector kernel launch.
  - Files: `nmoe/csrc/adamw_fused_step.cu`
  - Acceptance: BF16 AdamW launch path no longer emits extra `<<<1,1>>>` scalar tail kernel when `n` is odd.
  - Status note: extended `k_fused_adamw_bf16_vec2` with tail handling and switched launcher to single-kernel execution with unchanged update equations.

- [ ] R-058 Remove remaining grouped-wgrad stream sync by moving grouped launch metadata build to device.
  - Files: `nmoe/csrc/gemm.cu`
  - Acceptance: grouped BF16 wgrad path avoids host-side `cudaStreamSynchronize(stream)` while preserving exact grouped GEMM partitioning.
  - Status note: hidden `cudaStreamSynchronize(stream)` fallback was removed from grouped-wgrad call path; host still waits on small D2H readiness event because grouped metadata is host-built (device-side metadata build still open work).

- [ ] R-059 Eliminate blocking host `M_recv` handoff in dispatch by device-resident launch control.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: dispatch paths no longer block on `cudaEventSynchronize` in `read_device_int_blocking` for steady-state routing.
  - Status note: remaining blocking reads are still required to size host-launched sort/gather stages; full removal needs device-driven control flow or fixed-shape overprovisioning design.

- [x] R-060 Cap per-stream runtime cache growth in MoE backward helpers.
  - Files: `nmoe/moe.py`
  - Acceptance: stream-keyed runtime caches used by backward metadata helpers cannot grow unbounded under stream churn.
  - Status note: added bounded insertion helper with configurable cap (`NMOE_PER_STREAM_CACHE_MAX`, default 32) and applied it to `offs_pad` device cache and pinned scalar host cache.

- [x] R-061 Convert grouped-wgrad host-offset prep to event-backed async handoff.
  - Files: `nmoe/csrc/gemm.cu`, `nmoe/moe.py`
  - Acceptance: `bf16_prepare_offs_pad_host(...)` no longer forces immediate full stream sync and grouped-wgrad entrypoints can consume prepared host scratch with event-based readiness.
  - Status note: `resolve_offs_pad_host_view(...)` now supports async prepare (`block_for_device_copy=false`) with event tracking and fast-path reuse of prepared host scratch pointers.

- [x] R-062 Trim hybrid/IPC dispatch inner-loop and no-work launch overhead.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: remote/local IPC-handoff branches are hoisted out of inner `h` loops and guaranteed no-work launch sites are host-guarded.
  - Status note: hoisted constant remote/local branches in stage-push kernels and added `M==0` host launch guards across hybrid dispatch and return/backward helper launchers.

- [x] R-063 Reduce fused router backward tail/transpose overhead in BF16 path.
  - Files: `nmoe/csrc/router_bwd.cu`
  - Acceptance: BF16 fused router backward uses tighter vector-lane coverage and interior transpose tiles skip per-element bounds checks.
  - Status note: switched runtime branches to compile-time branches where possible, tightened scalar-tail start to `D_vec`, and added full-tile fast path in transpose/cast kernel.

- [x] R-064 Enforce aux-loss single-kernel no-fallback contract and move reset to CUDA.
  - Files: `nmoe/csrc/aux_loss.cu`, `nmoe/fused_aux_loss.py`
  - Acceptance: active aux-loss path no longer silently downshifts to legacy two-kernel APIs and Python side avoids per-call accumulator/counter reset writes.
  - Status note: added runtime capability handshake (`fused_aux_loss_single_caps`), strict no-fallback checks in Python loader, CUDA-side reset kernel, and input-contract guards that forbid implicit copy paths.

- [x] R-065 Remove FP32 AdamW tail launch and cap ECO grid-stride launch overhead.
  - Files: `nmoe/csrc/adamw_fused_step.cu`, `nmoe/csrc/eco_adam.cu`
  - Acceptance: FP32 AdamW no longer emits extra scalar tail launch when `n % 4 != 0`, and ECO AdamA accumulation launch sizes are bounded by cached SM-aware caps.
  - Status note: folded FP32 tail into vector kernel launch and added capped launch helper + zero-work early exits in ECO update/recompute launchers.

- [x] R-066 Cache optimizer post-hook module list to remove per-step full block scans.
  - Files: `nmoe/opt.py`
  - Acceptance: optimizer post-hook path iterates cached FFN modules instead of scanning `model.blocks` each step.
  - Status note: added `_get_post_hook_ffns(...)` cache keyed on model instance and switched post-hook loop to the cached module list.

- [x] R-067 Enforce strict no-fallback grouped-wgrad offset staging contract.
  - Files: `nmoe/csrc/gemm.cu`, `nmoe/moe.py`
  - Acceptance: grouped BF16 wgrad entrypoints cannot silently downshift to full-stream sync when given device `offs_pad` without prior staging.
  - Status note: `resolve_offs_pad_host_view(...)` now hard-fails device-pointer calls on blocking path; production backward pre-stages via `bf16_prepare_offs_pad_host(...)` and reuses staged host scratch pointer.

- [x] R-068 Add per-stream grouped-wgrad host-offset scratch slots to prevent cross-stream staging races.
  - Files: `nmoe/csrc/gemm.cu`
  - Acceptance: host offset staging for grouped BF16 wgrad is keyed by CUDA stream and does not reuse a single thread-local scratch slot across overlapping streams.
  - Status note: replaced singleton `OffsHostScratch` with stream-keyed table (`OffsHostScratchTable`), plus pointer/stream lookup helpers for staged offset readiness.

- [x] R-069 Add host-offs grouped-wgrad entrypoints to skip resolver overhead on hot path.
  - Files: `nmoe/csrc/gemm.cu`, `nmoe/csrc/bindings.cpp`, `nmoe/moe.py`
  - Acceptance: hot grouped-wgrad calls in MoE backward use explicit host-offs entrypoints and avoid repeated pointer-introspection resolver traversal.
  - Status note: added `bf16_wgrad_w2_cublaslt_host_offs` + `bf16_wgrad_w13_pair_cublaslt_host_offs` and switched MoE backward callsites to them behind strict capability checks.

- [x] R-070 Specialize blockscaled dispatch kernels on profile at compile time.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: blockscaled dispatch hot kernels do not branch on runtime `profile` inside inner quantization loops.
  - Status note: converted IPC/hybrid blockscaled dispatch kernels to `template<bool kFP8>` and switched launch callsites to explicit profile-specialized kernels with invalid-profile fail-fast guards.

- [x] R-071 Fold aux-loss accumulator reset into fused single-kernel path.
  - Files: `nmoe/csrc/aux_loss.cu`, `nmoe/fused_aux_loss.py`
  - Acceptance: active fused aux-loss path no longer launches a separate reset kernel per call.
  - Status note: moved reset into single-kernel epilogue (`is_last_block`) for all dtypes, removed reset-launch sequence, and trimmed Python-side per-call stream bookkeeping/ctypes wrapper overhead.

- [x] R-072 Harden hybrid NVSHMEM gather/return paths against silent fail-open returns.
  - Files: `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: invalid init/profile/pointer/state in hybrid blockscaled gather and hybrid return-scatter paths fails loudly instead of silently returning.
  - Status note: upgraded invalid-state return sites to explicit fatal aborts and tightened `M_pad` validation (`M_recv>0 && M_pad<=0` now hard-fail).

- [x] R-073 Restore hybrid pre-arm callsites before blocking `M_recv` reads.
  - Files: `nmoe/csrc/rdep_nvshmem.cu`
  - Acceptance: hybrid BF16 and blockscaled dispatch paths issue `poll_device_int_async(local_counter, stream)` immediately before blocking counter handoff.
  - Status note: restored pre-arm callsites after final visibility barriers in both hybrid dispatch paths to reduce D2H wait tail.

- [x] R-074 Cache optimizer post-hook module iteration set.
  - Files: `nmoe/opt.py`
  - Acceptance: optimizer post-hook step path no longer scans all `model.blocks` every iteration.
  - Status note: added `_get_post_hook_ffns(...)` with model-instance cache and switched post-hook loop to cached FFN module list.

- [x] R-075 Avoid redundant MoE fused-path tensor canonicalization copies.
  - Files: `nmoe/moe.py`
  - Acceptance: forward/backward canonicalization does not allocate/copy when tensors are already contiguous and in target dtype.
  - Status note: added `_as_contiguous_dtype(...)` helper and replaced unconditional `.contiguous().<cast>()` chains for `x`, `dOut`, `eid`, and `gates_fp32`.

- [x] R-076 Cache aux-loss dynamic shared-memory configuration per kernel/device.
  - Files: `nmoe/csrc/aux_loss.cu`
  - Acceptance: fused aux-loss launch path avoids repeating `cudaDeviceGetAttribute` and `cudaFuncSetAttribute` when `(device, smem)` configuration is unchanged.
  - Status note: `configure_dynamic_smem_if_needed(...)` now keeps template-local thread caches for max opt-in shared memory and last configured dynamic-smem setting.

- [x] R-077 Remove per-call counts zeroing pass in fused router bias update.
  - Files: `nmoe/fused_router.py`
  - Acceptance: fused router bias update does not issue separate `counts.zero_()` pass before bincount.
  - Status note: `_update_bias_fused_kernel` now clears `counts` after consuming them, scratch allocation initializes with zeros once, and Python-side explicit zeroing was removed.
