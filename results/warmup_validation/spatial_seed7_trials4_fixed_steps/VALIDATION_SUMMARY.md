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

## Offline warmup metrics

All 50,000 warmup metric rows are present with a continuous stage-step sequence.
The action head and all modules outside the reasoning policy and transition are
frozen, while demonstration-action L1 is back-propagated through the final
latent state. Comparing equal 500-update windows:

| Metric | First 500 updates | Last 500 updates |
|---|---:|---:|
| Demonstration action L1 | 0.031973 | 0.029701 |
| Current-action L1 | 0.034053 | 0.031844 |
| Future-action L1 | 0.031674 | 0.029396 |
| Latent step distance | 0.002960 | 0.004647 |
| Trainable gradient norm | 0.010201 | 0.008328 |

The action-alignment loss fell by 7.1%.  The final-window latent distance was
strictly non-zero for every update (minimum 0.003462), as were the trainable
gradient norms (minimum 0.005602).  Together with the parameter-scope hashes,
this rules out a frozen or identity/no-op warmup and verifies that the learned
latent path receives an action-related training signal.  It does not, by
itself, prove that additional latent steps improve environment success.

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
one shared failure. The three discordant outcomes under otherwise identical
initial conditions confirm that latent transitions affect executed behavior.
The net success difference is one episode and is not statistically
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

The update-10 checkpoint was then stopped and resumed.  All eight ranks
restored the AVA-VLA components, action head, RNG state, and online-PPO
optimizer/scheduler.  Updates 11 through 15 completed continuously, reaching
59,743 environment steps.  Their PPO ratios remained between 0.9853 and
1.0095, approximate KL between 0.0052 and 0.0125, and global maximum KL between
0.0123 and 0.0232, below the 0.03 safety bound.  Completed episodes returned
after the first post-resume fixed-horizon rollout, and no counter reset, NaN,
worker restart, or checkpoint rollback occurred.

Conclusion: the warmup checkpoint satisfies its freeze and resumability
contracts, contains non-zero action-coupled latent updates, preserves high
LIBERO Spatial success in this paired canary, and integrates safely into
resumable online PPO.  These tests validate implementation behavior; they are
not a strict numerical reproduction of the paper's 384px Table 1 result.
