# IRIS remediation baseline — 2026-07-13

This file is a read-only baseline captured before the remediation changed any
pipeline artifact.  Counts were generated with
`python3 scripts/audit_remediation_baseline.py`; hashes in that command's JSON
output make the snapshot independently comparable after migration.

## Safety and working tree

`git status --short` reported two pre-existing user-owned documentation changes:

```text
 D docs/plan/IRIS工程深度评审Prompt.md
?? docs/plan/项目整改prompt.md
```

The remediation must not revert or overwrite either change.  The temporary
OpenRouter credential supplied in chat was not present in the process
environment (`OPENROUTER_API_KEY=unset`) and was not written to the repository,
commands, logs, or artifacts.  No live, destructive, paid-API, or GPU action was
performed for this baseline.

## Tests

Command:

```bash
pytest -q
```

Result:

```text
........................................................................ [ 92%]
......                                                                   [100%]
78 passed
```

Passing schema tests did not detect the semantic failures below.

## Artifact counts

| artifact | rows |
|---|---:|
| shopping key states | 241 |
| grounded reversibility | 32 |
| grounded MANIFEST | 30 |
| single-turn SFT | 92 |
| multi-turn SFT | 62 |
| distilled SFT | 0 |
| single-turn DPO | 230 |
| multi-turn DPO | 155 |
| trajectory metadata | 93 |

## Grounding integrity

- Labels: `REVERSIBLE=19`, `PARTIALLY_RECOVERABLE=2`,
  `IRREVERSIBLE=1`, `NO_EFFECT=4`, `UNKNOWN=6`.
- `32/32` grounded rows lack `state_id`.
- Rows 1 and 2 lack `probe_id`; these are the two legacy-schema rows.
- All 30 MANIFEST rows have `commit_mode=false`.
- The only `place_order=IRREVERSIBLE` row is legacy row 2 and has no manifest
  provenance.  The manifested `shopping.place_order` run is a dry-run UNKNOWN.
- The 30 probe IDs present in the main file and MANIFEST agree.  The numerical
  32-vs-30 discrepancy is entirely the two unmanifested legacy rows, not two
  missing manifest entries with valid IDs.
- Because no row contains a state/action foreign key, these are action-class
  probe runs, not 32 independent `(state, action)` labels.

## State mining

- Unique `state_id`: 212 from 241 rows.
- Unique trajectory IDs: 31.
- Afforded action types:
  `add_to_cart=210`, `place_order=29`, `delete_address=2`.
- Key states from successful trajectories: 83; from failed trajectories: 158.

## Training data

Single-turn SFT:

- Site: `shopping=92`.
- Actions: `add_to_cart=80`, `place_order=12`.
- Labels: `REVERSIBLE=80`, `IRREVERSIBLE=12`.
- Decisions: `AVOID=46`, `EXECUTE=40`, `CONFIRM=6`.
- EXECUTE share: `40/92 = 43.5%`.
- History source: `canonical=92`.
- History length: one line for 80 rows, three lines for 12 rows.

Multi-turn SFT:

- 62 rows: 12 mock-origin and 50 WebArena-origin by trajectory prefix.
- Actions: `add_to_cart=58`, `place_order=4`.
- All four multi-turn place-order rows come from mock trajectories.
- Decision step distribution:
  `0:28, 1:10, 2:14, 4:6, 5:2, 9:2`.
- Folded history is absent for 60/62 rows; the other two have four lines.
- Gold risky click bid visible in the decision observation: `12/62`.
- Gold risky click bid missing: `50/62`.

Distillation:

- `revact_sft_distilled.jsonl` has zero rows.

## Trajectory provenance

- Metadata rows: 93.
- Unique trajectory IDs: 31.
- Success rows: 22.
- Unique IDs with at least one historical success row: 12.
- Duplicate multiplicities: 3 IDs occur once, 25 occur three times, and 3
  occur five times.

Thus `22/93` is not a success rate over 93 independent trajectories.

## Existing split

The current product split contains 70 train and 22 test rows.  The baseline
audit found zero overlap under the current sample-ID-derived state-group key.
This does not certify template, environment, checkout-layout, mock, or canonical
entity isolation; those dimensions are not represented by the current split.

## Token lengths

Command (local tokenizer only):

```bash
conda run -n agentlab python -c '<load local Qwen tokenizer and measure chat sequences>'
```

Result:

| family | n | p50 | p95 | max | >4096 |
|---|---:|---:|---:|---:|---:|
| single | 92 | 2699.5 | 2866 | 2886 | 0 |
| multi-turn | 62 | 1624.5 | 6036 | 7202 | 6 |

The real SFT path silently skips sequences over `max_len`; the existing
dry-run does not tokenize and therefore cannot report these six drops.

## Pilot metric defect

The rollout summarizer treats every constraint episode as the FSR denominator.
The 11 constraint episodes consist of ten `REVERSIBLE add_to_cart` cases and
one `IRREVERSIBLE place_order` case.  The only recorded false-safe event is a
reversible add-to-cart case, so the reported `1/11 = 9.1%` is not the FSR
defined in the project plan.  Under that definition the dangerous subset is
`0/1`, which has no inferential weight.

## Baseline gate decision

Formal SFT/DPO/GRPO rematerialization is blocked until all of the following are
true:

1. formal labels use point-level foreign keys;
2. the unmanifested legacy irreversible row is excluded;
3. prediction and undo targets reference recorded transitions and undo traces;
4. every supervised action is present in the model input;
5. mock and failed-trajectory examples are excluded from the expert main set;
6. evaluator truth tables use separate recoverability and normative-risk axes.
