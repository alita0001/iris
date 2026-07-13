# IRIS Dataset Card

## Dataset summary

**Release status:** not ready for model training, benchmark claims, or public release.

**Snapshot date:** 2026-07-13.

IRIS is intended to contain point-level web-agent transitions grounded by an execute–then–undo protocol. The intended unit is one uniquely identified state and one legal action instance, including the observed pre/post transition, executed recovery attempts, recovery cost, residual state difference, solver budget, and full provenance.

The canonical formal files are currently empty:

```text
data/grounded/probe_points.jsonl       0 rows
data/grounded/POINT_MANIFEST.jsonl     0 rows
```

The repository therefore does **not** currently provide a formal IRIS training or evaluation dataset. Existing non-empty files are frozen historical/development assets and are excluded from formal export by default.

## Current asset inventory

| Path | Rows | Scope | Formal eligibility |
|---|---:|---|---|
| `data/grounded/reversibility.jsonl` | 32 | legacy/class-level probe smoke body | excluded |
| `data/grounded/MANIFEST.jsonl` | 30 | legacy manifest | excluded |
| `data/grounded/quarantine/legacy_rows.jsonl` | 2 | rows without probe provenance | quarantined |
| `data/grounded/quarantine/class_probe_smoke_index.jsonl` | 30 | class smoke inventory | excluded |
| `data/grounded/probe_points.jsonl` | 0 | canonical formal point body | eligible schema; no rows |
| `data/grounded/POINT_MANIFEST.jsonl` | 0 | canonical formal manifest | eligible schema; no rows |
| `data/train/sft/revact_sft.jsonl` | 92 | historical single-step Magento SFT | excluded |
| `data/train/sft/revact_sft_multiturn.jsonl` | 62 | historical; 12 mock / 50 WebArena inferred from `trajectory_id` prefixes; all 62 lack explicit origin/mock/success/run fields | excluded |
| `data/train/sft/revact_sft_distilled.jsonl` | 0 | teacher-distilled output | no rows |
| `data/train/dpo/revact_dpo.jsonl` | 230 | synthetic flips by generation mechanism; 230/230 lack `negative_source` | excluded as `legacy_unspecified` |
| `data/train/dpo/revact_dpo_multiturn.jsonl` | 155 | synthetic flips by generation mechanism; 155/155 lack `negative_source` | excluded as `legacy_unspecified` |
| `data/raw/trajectories_meta.jsonl` | 93 | historical collection attempts; 31 logical trajectory IDs | raw audit only |
| `data/raw/candidates/iris_candidates.v3.jsonl` | 300 | `iris.candidate.v2` legal proposals across 50 states; no decoy/on-policy rows | awaiting point grounding |
| `data/raw/candidates/CANDIDATE_MANIFEST.jsonl` | 300 | 1:1 content hashes for the v3 candidate body | formal integrity gate |
| `data/raw/quarantine/legacy_lineage_rows.jsonl` | 93 | non-destructive index of ambiguous historical attempts | excluded |

Counts can be reproduced with:

```bash
wc -l data/grounded/{reversibility,MANIFEST,probe_points,POINT_MANIFEST}.jsonl
wc -l data/grounded/quarantine/*.jsonl
wc -l data/train/sft/*.jsonl data/train/dpo/*.jsonl
wc -l data/raw/trajectories_meta.jsonl
```

## Canonical schema

Schema version: `iris.grounding.point.v1`, defined by `revact/grounding/schema.py`.

### Label ontology

The dataset separates whether the measured state changed from whether the recorded solvers recovered it:

```text
effect_status:
  CHANGED
  NO_EFFECT
  UNKNOWN

recovery_status:
  RECOVERED
  PARTIALLY_RECOVERED
  NOT_RECOVERED_WITHIN_BUDGET
  UNKNOWN

undo_cost_steps: nullable non-negative integer
```

`NO_EFFECT` is not a recovery class. `IRREVERSIBLE` is accepted only by the legacy reader and is mapped to `NOT_RECOVERED_WITHIN_BUDGET`; new formal records never emit it.

### Required point provenance

A formal point includes at least:

- schema, point, run, state, action-instance, task, trajectory, and environment identities;
- raw and canonical actions, site, account/privilege, seed, URL, and mock flag;
- pre/post observation hashes and pre/post signals;
- solver set, controller version, budget, executed undo actions and intermediate hashes;
- final signal, effect/recovery statuses, recovery cost, residual diff, and budget exhaustion;
- timestamp, code version, and evidence packet.

Every point must have exactly one manifest row keyed by `probe_point_id`, with a content hash. Formal samples must reference this exact point; action-class lookup is not an admissible foreign key.

## Measurement protocol and label meaning

The intended protocol is:

1. identify a recoverable environment snapshot and a legal action instance;
2. record pre-observation and structured signal;
3. execute the action and record the measured transition;
4. run the declared recovery solvers under budget `k`;
5. record every attempted recovery action, final signal, residual diff, and termination reason;
6. admit the point only if schema and manifest validation pass.

Evidence is asymmetric:

- `RECOVERED` has constructive support from an executed recovery sequence.
- `PARTIALLY_RECOVERED` has an executed compensating trace plus a non-empty residual.
- `NOT_RECOVERED_WITHIN_BUDGET` means only that the declared solvers did not recover the measured state within the declared budget. It is not a proof that no recovery sequence exists.
- `UNKNOWN` is required when execution, signals, or provenance are insufficient.

All recovery claims are relative to the agent action space, account privilege, environment instance, observation signal σ, solver set, and budget. Controller–human agreement is supplementary quality control, not an irreversibility proof.

## Recoverability is not normative safety

The grounding schema does not contain a safety label. A separate evaluation truth record must state policy/constraint truth, whether the action is required for the goal, whether it violates a constraint, and the expected decision.

A recoverable action may still be unauthorized or harmful. A hard-to-recover action may be explicitly requested and permitted. Restoring a UI/API/DB projection also does not retract email delivery, payment settlement, third-party observation, notification, audit logs, caches, or reputational effects.

## Source environments and coverage

The code registers WebArena mirrors for Magento shopping, Magento admin, and Postmill reddit. Registration and probe code do not imply that formal data exist for those sites.

- Historical single SFT is 92/92 Magento shopping.
- Historical multi SFT has `site=shopping` in all 62 rows. The 12 mock / 50
  WebArena split is inferred from `trajectory_id` prefixes, not recorded in
  `environment_origin` or `is_mock`; all rows also lack `collector_success` and
  immutable `run_id`.
- There are no formal point-level rows from any site.
- There are no formal GitLab, WorkArena/ServiceNow, VisualWebArena, Mind2Web, or real external-service sandbox rows.

Consequently, this snapshot cannot support web-wide, cross-site, cross-privilege, or cross-action generalization claims. Magento shopping templates and the legacy action mix are strong confounders.

## Historical state and action bias

The legacy key-state miner uses a shallow English keyword mapping for `add_to_cart`, `place_order`, and `delete_address`. Its 241 historical shopping key states are distributed 210/29/2 across those classes. This is a collector/detector distribution, not an estimate of the natural web action distribution.

The fixture-safe mutation detector now enumerates legal interactive controls and verifies measured state changes; its 240-control result is an offline fixture test, not a live recall estimate. Keywords, language models, URL patterns, and page-template clusters may prioritize candidates, but must not generate ground-truth labels. The live mutation census remains blocked pending reset-safe execution approval.

## Candidate and preference data

The formal consumer reads `iris_candidates.v3.jsonl`: 300 `iris.candidate.v2`
records across 50 states (250 `a11y_enumeration`, 50 with producer-declared
`source=expert`). The latter means only that the producer supplied the target
bid; candidate v2 does not carry `trajectory_id`, `run_id`, or
`collector_success`, so these rows are not evidence of successful expert
provenance. Their bids and snapshot hashes are validated, but their categories
are proposal-side coverage hypotheses, not effect, recovery, or safety labels.
This artifact contains no decoy or on-policy rows and is not yet joined to formal
points. Earlier v1/v2 files are development assets and are not consumed by the
formal DPO path.

Existing DPO pairs are synthetic label/text flips by generation mechanism and do
not represent the deployed policy's error distribution. Their historical rows do
not record `negative_source`, so audits classify them as `legacy_unspecified`
rather than inventing row-level `synthetic_flip` provenance. Formal candidate
records contain snapshot identity, legal bid, canonical action, category, source,
proposer version, and legality evidence.

Formal DPO export is required to contain a preregistered proportion of legal snapshot candidates or on-policy model errors, with results stratified by negative source. Until that material exists, historical preference files remain development fixtures.

## Training sample admission

A formal training sample must satisfy all of the following:

- exact `probe_point_id` join;
- `prediction_source=probe_transition`;
- `undo_source=probe_point_id`;
- every supervised `click(bid)` appears in the adjacent serialized observation;
- non-mock origin;
- an explicit successful-collector declaration for expert SFT, plus an exact
  join to the immutable collection run that proves it (a boolean copied into a
  sample is insufficient);
- true observation-derived history, not canonical/mock plan text;
- the shared train/deploy message builder;
- explicit handling of overlength samples with zero silent tokenizer drops;
- immutable `prompts_fp` resolvable to its full prompt bundle and a distinct
  `prompt_generation_fp` resolving producer/model/decode configuration.

The current historical SFT/DPO files fail one or more of these requirements and are excluded. The teacher-distilled output currently contains zero rows; teacher rewriting must not be described as deployed.

## Splits and leakage

No non-empty, usable formal train/dev/test split exists because there are no formal points. Empty files are materialized deliberately. When data exists, a single joint component assignment covers both single and multiturn families and keeps each state group, canonical entity, environment instance, and page template within one partition. Request and constraint variants of the same state remain together.

The split audit must report overlap for state, entity/product, template,
environment, action class, and privilege. Challenge shard paths such as
`cross_site`, `cross_action`, and `cross_privilege` may be materialized as empty
stale-clearing sentinels. Only a corresponding
`SPLIT_REPORT.challenges.<axis>.available=true` together with non-empty train and
test shards constitutes an available split; neither an empty filename nor split
code is evidence of an experiment.

## Evaluation

Formal evaluation must preserve numerator and denominator sample IDs for:

- FSR-declaration, FSR-attempt, and FSR-commit;
- FBR, IER, and UCR;
- recovery macro-F1 and per-class results;
- undo execution success and undo-cost MAE;
- Brier score, ECE, risk–coverage, task success, and Completion-under-Policy.

The historical `9.1% FSR` value is retracted because its denominator did not represent true-danger points. It must not be used as a model result. Cells with fewer than 30 points may report counts and intervals only, not effect conclusions.

## Prompt and model provenance

New assembly separates two immutable identities: `prompts_fp` addresses the full prompt text, parent, author, and timestamp, while `prompt_generation_fp` addresses the prompt fingerprint together with producer, model identity, and decoding configuration. Thus two runs with identical text but different temperature/model settings remain distinguishable and recoverable. Formal export copies both bundle families and rejects missing or mismatched formal generation provenance. The historical SFT fingerprint `259fceac4e0c` has no corresponding bundle under `data/manifests/prompts`; it must be reported as unresolved rather than substituted with the current registry.

Secrets must never be stored in data files, prompts, logs, manifests, or documentation. API credentials are environment-only and should be rotated immediately if exposed.

## Safety, privacy, and destructive scope

- Live order placement, payment, posting, deletion, refund, email, and other externally visible actions require explicit approval and destructive gates.
- Formal online RL requires cloned accounts or transactional rollback; real orders must not be generated as training rollouts.
- Mock data must remain in a separate dataset configuration and is rejected by formal export.
- This snapshot has not completed a PII audit. Public release is blocked until URLs, account identifiers, free text, order identifiers, and screenshots/AXTrees are reviewed and redacted as needed.
- A repository/data license has not been established in the current materials. Redistribution is blocked until site terms, benchmark licenses, and derived-data rights are documented.

## Known limitations and blocked items

At this snapshot:

- the normal live probe CLI writes class-smoke rows to a separate `grounded/smoke/`
  namespace and cannot modify the frozen legacy body; a fail-closed `--formal-spec`
  and validated import boundary exist, but no live registered probe yet emits a
  complete admissible point;
- formal points and formal training samples are zero;
- solver-union schemas and adversarial fixtures exist, but no live negative point or attack break-rate report exists;
- DB/API upper-bound signals are not available for every action class;
- S4 legal candidate materialization and a fail-closed reviewed/on-policy DPO
  materializer exist, but no candidate is point-grounded and the formal DPO file is empty;
- teacher output has zero rows; because the formal source also has zero rows, coverage is undefined (`null`), not 0%;
- the historical prompt fingerprint is unresolved; new assembly records model/
  decoding provenance, but no non-empty formal sample currently exercises it;
- no formal cross-site, cross-action, cross-privilege, coupled/decoupled, or model-training result exists;
- GRPO/GSPO code is offline RLVR, not environment-in-the-loop RL.

See [docs/Limitations.md](docs/Limitations.md) and [docs/plan/IRIS项目计划书.md](docs/plan/IRIS项目计划书.md) for the research boundary and remediation gates.

## Versioning and rollback

Legacy migration is non-destructive. It creates quarantine inventories and empty canonical files without modifying `reversibility.jsonl` or `MANIFEST.jsonl`. The migration report records input hashes and generated paths. Generated migration artifacts can be removed to roll back the inventory; the frozen source remains unchanged.

Any future dataset release must publish:

- schema and data version;
- code/controller versions and immutable run IDs;
- per-split site/action/effect/recovery/decision/history/environment distributions;
- mock and failed-run exclusion counts;
- solver sets, budgets, signal coverage, UNKNOWN policy, and attack break rate;
- split leakage audit and prompt bundles;
- license, PII review, destructive-action scope, and known failure modes.
