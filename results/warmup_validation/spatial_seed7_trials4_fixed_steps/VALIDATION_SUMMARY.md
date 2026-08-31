# Latent warmup validation summary

Date: 2026-09-01 (UTC)

Checkpoint:
`runs/paper_per_suite_corrected/paper_spatial_seed0/stage_checkpoints/latent_warmup_complete`

## Stage-boundary validation

- Stage counter: 50,000 / 50,000.
- The archive is independently resumable and contains all 15 required files.
- The BC and warmup action-head hashes are identical.
- AVA-VLA parameters outside `reasoning_policy` and `latent_transition` are identical.
- The complete AVA-VLA hash changed, proving that the warmup trainable modules were updated.
- Validator report: `runs/paper_per_suite_corrected/paper_spatial_seed0/stage_checkpoints/STAGE_VALIDATION.json`.

## Paired LIBERO Spatial canary

Both conditions use the same warmup checkpoint, seed 7, tasks, trial indices,
initial states, image processing, proprioception, wrist images, history state,
and eight-action open-loop execution. Four shards cover one paired trial for
each of the ten Spatial tasks, for 40 episodes per condition.

| Condition | Successes | Rate | Mean query latency | P90 query latency |
|---|---:|---:|---:|---:|
| Fixed 0 latent-transition steps (`z_0`) | 38 / 40 | 95.0% | 137.16 ms | 163.27 ms |
| Fixed 5 latent-transition steps | 37 / 40 | 92.5% | 153.28 ms | 185.04 ms |

Paired outcomes contain one 5-step improvement, two 5-step regressions, and
one shared failure. The net difference is one episode and is not statistically
distinguishable at this canary size. The result supports preservation of the
BC-level policy after warmup, but does not establish an environment-success
improvement from warmup alone.

## PPO integration canary

The same warmup archive completed 10 formal PPO updates and 39,839 environment
steps. Across these updates:

- PPO ratio stayed between approximately 0.985 and 1.010.
- Approximate KL stayed between approximately 0.0038 and 0.0099.
- Global maximum KL stayed between approximately 0.0100 and 0.0212, below the 0.03 safety bound.
- Online rewards, terminal outcomes, successes, value loss, and policy loss were finite.
- The action head remained frozen and action PPO remained disabled.
- Update 10 was atomically checkpointed and is independently resumable.

Conclusion: the warmup checkpoint satisfies its freeze and resumability
contracts, preserves high LIBERO Spatial success in this paired canary, and
integrates safely into online PPO. Formal PPO can resume from update 10.
