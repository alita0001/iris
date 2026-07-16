# OpenRouter formal-opinion collection audit (2026-07-16)

> Current-state addendum (later on 2026-07-16): the truth table was expanded
> after this call from 32 to 64 rows (32 point x 2 policy variants), while this
> immutable opinion release remains at 6 rows.  Its current exact coverage is
> therefore 6/64 truth rows, with human coverage 0/64.  The historical counts
> below describe the corpus at collection time and must not be read as the
> active readiness denominator.

This is an **opinion-only baseline**, not grounding truth. The LLM saw only the
goal, exact pre-action observation and action. Post-state, probe outcome,
recovery trace, normative truth and labels were excluded by the strict input
schema.

## Executed collection

Provider and model were explicitly selected at runtime:

```text
provider=openrouter
base_url=https://openrouter.ai/api/v1
model=deepseek/deepseek-v3.2
temperature=0
top_p=1
max_tokens=300
seed=0
rater_id=openrouter-deepseek-v3.2-seed0-formal-v1
batch_id=openrouter-opinion-formal-20260716-v1
```

The API key was supplied through the named child-process environment variable
`OPENROUTER_API_KEY`. Its value was not passed as a CLI argument, persisted in
configuration, data, provenance or this report, and was unset when the child
process exited.

Artifacts:

- `data/opinions/inputs/formal_request_constraint_inputs.v2.jsonl`: 6 rows.
- `data/opinions/inputs/FORMAL_REQUEST_CONSTRAINT_INPUT_MANIFEST.v2.jsonl`:
  6 rows, 1:1 with the input body.
- `data/opinions/opinion_labels.v2.jsonl`: 6 rows.
- `data/opinions/opinion_labels.v2.manifest.jsonl`: 6 rows, 1:1 with the body.

Measured result:

```text
n ratings: 6
rater type: LLM=6
provider: openrouter=6
response model: deepseek/deepseek-v3.2=6
finish reason: stop=6
unique provider response ids: 6
unique input-message hashes: 6
prompt_generation_fp: 35b2c800336cd2da
perceived effect: CHANGE=5, NO_CHANGE=1
perceived recovery: RECOVERABLE=6
normative risk opinion: RISKY=3, NOT_RISKY=3
```

Integrity tests:

```text
$ conda run -n agentlab pytest -q \
    tests/test_opinion_collection.py tests/test_opinion_labels.py
..............                                                           [100%]
```

Credential audit after the call:

```text
openrouter_secret_prefix_file_hits=0
supplied_key_fragment_file_hits=0
openrouter_env_present=0
```

## Why only six cases were called

The formal truth table has 32 cases, but only six currently have an exact
formal single-step SFT goal joined to the point's exact pre-observation. The
input preparer rejected the other 26 as `missing_formal_single_sft_goal`
instead of inventing a goal or copying a class-level template.

Therefore this run does **not** satisfy paired-opinion readiness.  At the time
of collection:

- exact LLM coverage is 6/32 truth cases;
- human ratings are 0/32;
- there is only one distinct rater per covered case.

Against the current 64-row truth body, those figures are 6/64 LLM and 0/64
human.  No opinion row is a behavior label or a substitute for a probe.

The remaining work is a real data-collection task: materialize exact goals for
the other point×variant cases, collect at least one independent human rating
and one LLM rating per case, and preserve rater independence. No behavior label
may be changed by these opinions.
