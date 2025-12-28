## PR Attestation (YAML, required)
This block is intentionally machine-readable and public-safe. Tier‑B is filled by the maintainer-run workflow.

```yaml
nmoe_pr:
  issue: 03a
  baseline_sha: <sha>     # required if hot-path or perf is claimed
  pr_sha: <sha>
  risk: hot               # hot|warm|cold

gates_passed:
  tier_a:
    - cpu:import
    - cpu:compileall
  tier_b: []              # filled after Tier‑B run

perf:                     # optional
  gate: perf:baseline_delta
  compare: b200:moonlight_8x20
  metric: node_tps_p50
  baseline: <number>
  result: <number>
  delta_pct: <number>
  budget_pct: -10
  status: pass|fail
```

## Summary
- What changed (1–5 bullets):
- Why:

## Links
- Fixes: #

## Public-safety (required)
- [ ] No internal hostnames, cluster identifiers, private URLs, credentials, or runbooks in this PR, commits, or code.
- [ ] No hardcoded internal paths/locations; paths are config-driven (TOML) and container-first.

## Reviewer focus (AI-friendly)
- Files/symbols to focus on:
- Invariants preserved (B200-only, no NCCL all-to-all on MoE path, deterministic resume if applicable):
- Risky edges / fast validation:
