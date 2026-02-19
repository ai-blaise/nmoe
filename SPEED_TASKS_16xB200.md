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

- [ ] A-004 Propagate the new `nmoe` commit to all nodes and verify file-level parity.
  - Files: `../nmoe-multinode/orchestrate.py` flow (`provision`)
  - Acceptance: all nodes show same `git rev-parse HEAD` and expected guardrail code strings.
  - Status note: live `progressive-scale` run failed on stale node code (`Config.__init__` missing `use_fused_router`), confirming rollout parity is still required.

## B) Highest ROI Hot-Path Sync Removal

- [x] B-001 Remove per-layer GPU->CPU sync from MoE load CV metric.
  - Files: `nmoe/model.py` around load CV calculation.
  - Change: remove `.item()` in hot path; keep tensor on device and convert only in low-frequency logging.
  - Acceptance: no `.item()` in MoE forward hot path; p50 step-time improves; no metric correctness loss.
  - Status note: `MoE.forward()` now keeps CV tensor on device and `mean_expert_load_cv` converts once at logging boundary.

- [x] B-002 Remove Python spin wait in MoE forward for dispatch event.
  - Files: `nmoe/moe.py` around `while not evt.query()`.
  - Change: avoid host polling if C path already synchronizes/guarantees readiness.
  - Acceptance: no busy-wait loop in forward; same outputs and no race failures.
  - Status note: busy-spin query loops replaced with event synchronize waits (no Python spin loops).

- [ ] B-003 Remove host sync and D2H metadata reads in RDEP dispatch.
  - Files: `nmoe/csrc/rdep.cu` dispatch metadata path.
  - Change: keep `M_recv/M_pad` on device; remove forced stream sync + host reads from hot path.
  - Acceptance: no D2H metadata transfer/sync in trace during steady-state; p50 and p95 step-time improve.

## C) Router + Dispatch Kernel Work

- [ ] C-001 Add 2-phase dispatch for blockscaled path (parity with BF16 fast path).
  - Files: `nmoe/csrc/rdep.cu` blockscaled dispatch kernels/state.
  - Acceptance: blockscaled path uses 2-phase kernels for multi-node; throughput improves with identical numerics.

- [ ] C-002 Remove redundant metadata rebuild (sort/offset) when fused router metadata is available.
  - Files: `nmoe/csrc/rdep.cu`, `nmoe/fused_router.py`, `nmoe/model.py`.
  - Acceptance: fewer metadata kernels and launches; no routing correctness drift.

- [ ] C-003 Router forward path heuristic: use TC GEMM path for large shapes, fused path for small shapes.
  - Files: `nmoe/fused_router.py`
  - Acceptance: speedup over current path across token ranges; no top-k mismatch.

- [ ] C-004 Router backward: remove extra cast + transpose chain.
  - Files: `nmoe/fused_router.py`, `nmoe/csrc/router_bwd.cu`
  - Acceptance: one fewer transformation step; gradients match baseline tolerance.

- [ ] C-005 Remove unused `dispatch_indices` payload if not consumed.
  - Files: `nmoe/fused_router.py`, `nmoe/model.py`
  - Acceptance: no allocation/write of dead tensor in hot path; no behavior change.

- [~] C-006 Gate fused-router NVTX by env (off by default in production).
  - Files: `nmoe/fused_router.py`
  - Acceptance: no unconditional NVTX push/pop in production runs.
  - Status note: NVTX context now obeys `NMOE_NVTX`; runtime trace verification pending.

- [~] C-007 Initialize ctypes argtypes/restype once, not per call.
  - Files: `nmoe/fused_router.py`
  - Acceptance: one-time setup at import/init; reduced Python overhead in profile.
  - Status note: one-time signature init added in loader path; profile validation pending.

- [ ] C-008 Fuse blockscaled gather + swizzle to remove temporary SF path.
  - Files: `nmoe/csrc/rdep.cu`
  - Acceptance: remove temp memset + separate swizzle launch in hot route.

## D) Attention + RoPE Kernel Work

- [ ] D-001 Force Flash SDPA backend first, explicit fallback only on failure.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: backend selection logged; default path avoids unintended slower SDPA modes.

- [ ] D-002 Replace packed SDPA mask path with FlashMLA varlen path as primary.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: packed mode avoids materialized SxS mask; faster packed throughput without NaNs.

- [ ] D-003 Rewrite varlen packed path to flatten docs into one launch and remove host sync points.
  - Files: `nmoe/attention/mla.py`
  - Acceptance: fewer launches and no per-doc sync in trace; numerics match.

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

## E) ECO Optimizer Kernel Work

- [ ] E-001 Fuse NVFP4 group-scale tighten into main ECO update kernel.
  - Files: `nmoe/eco.py`, `nmoe/csrc/eco_adam.cu`
  - Acceptance: remove post-pass tighten kernel(s); unchanged update numerics.

- [ ] E-002 Fold FP8 row-scale recompute for `m/v` into main write path.
  - Files: `nmoe/csrc/eco_adam.cu`
  - Acceptance: fewer kernels and DRAM reads/writes; stable loss.

- [ ] E-003 Collapse factored-v prepass kernels and remove `v_rms` atomic hotspot.
  - Files: `nmoe/csrc/eco_adam.cu`
  - Acceptance: reduced launch count and atomic stalls; parity maintained.

- [ ] E-004 Remove triple gradient reread in factored-v path.
  - Files: `nmoe/csrc/eco_adam.cu`
  - Acceptance: lower memory traffic in Nsight; same optimizer results.

- [ ] E-005 Accept BF16 grad directly in ECO CUDA API and cast in-kernel.
  - Files: `nmoe/eco.py`, `nmoe/csrc/eco_adam.cu`, bindings.
  - Acceptance: cast kernel removal; step parity within tolerance.

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

- [ ] E-010 Precompute row->expert mapping in strided quant kernels.
  - Files: `nmoe/csrc/quant.cu`
  - Acceptance: reduced branch/index overhead with identical results.

- [ ] E-011 Remove grouped-dense fallback host sync path in quant kernels.
  - Files: `nmoe/csrc/quant.cu`
  - Acceptance: no D2H metadata/sync in this path.

- [ ] E-012 Add single binding entrypoint for full ECO update sequence.
  - Files: `nmoe/csrc/bindings.cpp`, `nmoe/eco.py`
  - Acceptance: lower Python dispatch overhead; graph-capture behavior improved.

## F) Build and Compiler Tuning (SM100/B200)

- [ ] F-001 Make `-rdc=true` conditional (required for NVSHMEM build only).
  - Files: `nmoe/csrc/Makefile`
  - Acceptance: default non-NVSHMEM build omits RDC/dlink; NVSHMEM build still works.

- [ ] F-002 FlashMLA compile mode tuning (`-std=c++20` + `-DNDEBUG`) with compatibility guard.
  - Files: `nmoe/csrc/Makefile`
  - Acceptance: compiles cleanly; no correctness regressions; improved kernel stack/reg profile.

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

- [ ] K-002 Add an explicit "force full repo refresh" runbook and verify it for all 16 nodes.
  - Command target: `../nmoe-multinode/orchestrate.py provision --clone-only --force`
  - Acceptance: all nodes report provision success and SHA parity check passes.

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
  - Status note: `agent.py launch_training()` now bootstraps `train_rank{node_rank}.log` before preflight checks and appends explicit launch success/failure markers; `orchestrate.py launch_training()` now validates expected rank log presence/freshness and fails launch if missing/stale.

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
9. [ ] C-001 Implement blockscaled 2-phase dispatch.
10. [ ] E-001 Fuse NVFP4 tighten into main ECO update kernel.
