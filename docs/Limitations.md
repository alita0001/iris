# IRIS Limitations

## Paper-ready limitations paragraph

> IRIS measures agent-relative operational recoverability rather than universal action safety. Each result is conditional on the agent's available actions and privilege, a finite recovery budget \(k\), the tested solver set, and the observed state signal \(\sigma\). A successful recovery trace is constructive evidence that the measured state projection was restored, whereas an unsuccessful trace only means that the tested solvers did not find a recovery within budget; it is not a proof of irreversibility. Moreover, restoring UI, API, or database state cannot necessarily retract real-world effects such as delivered messages, settled payments, third-party observations, audit records, or reputational harm. Our present implementation and historical assets are concentrated on a Magento shopping mirror; code-level cross-action and cross-privilege splits do not substitute for environmental diversity, and no formal cross-site result is currently available. These dependencies limit both the external validity of the labels and any safety conclusions drawn from them.

## 1. Construct validity

IRIS observes:

`recoverability(s, a | agent_action_space, privilege, k, σ, solver_set)`

It does not observe an environment-independent property of an action name. A label can change when the account privilege, current entity state, interface affordances, recovery budget, solver quality, or measured signal changes. Formal records therefore require those conditions as data rather than burying them in prose.

Recoverability and normative safety are different constructs. A reversible upvote may violate a user's instruction; a purchase not recovered within the declared budget may still be explicitly authorized. A safety decision must additionally use task necessity, policy constraints, authorization, consent, and expected real-world harm.

## 2. Positive and negative evidence are asymmetric

`RECOVERED` has constructive support only when the recorded recovery actions were actually executed and the declared final signal matched the pre-action signal. `PARTIALLY_RECOVERED` has an executed compensating trace but retains a declared residual.

`NOT_RECOVERED_WITHIN_BUDGET` is one-sided evidence. It says that the recorded solver set failed under budget `k`; it does not falsify the existence of an untested sequence. A weak deterministic controller can create systematic false negatives. Formal negative points therefore require a solver union and must disclose explored states/actions, termination reasons, attack attempts, budget exhaustion, and the fraction of initially negative points overturned by stronger attackers.

Controller–human agreement can reveal implementation defects, but even perfect agreement between people and the same controller does not prove that no recovery path exists.

## 3. Signal σ is incomplete

Current probes primarily use structured interface signals. Even an API or database diff covers only the systems to which the benchmark has access. Signals can miss asynchronous jobs, caches, webhooks, notifications, emails, external payment networks, replicas, audit logs, third-party reads, and concurrent updates.

DB/API comparisons improve the upper bound on whether the measured state was restored; they do not make undo search complete and do not erase external consequences. Each action class needs a published signal-sufficiency statement and known blind spots.

## 4. Environment and task coverage

The repository registers Magento shopping/admin and Postmill reddit sites, but formal point-level grounding is currently empty. Historical training assets carry `site=shopping`; the historical multi file's 12 mock / 50 WebArena split is inferred from `trajectory_id` prefixes because all 62 rows lack explicit environment-origin, mock, collector-success, and immutable-run provenance. Site registration and probe code are not evidence of a cross-site experiment.

Cross-action, cross-privilege, and page-template splits test narrower generalization axes and cannot replace training and evaluation on different applications, interaction conventions, permission models, and side-effect systems. Before making web-wide claims, IRIS needs multiple non-Magento formal environments and a held-out environment whose probes do not reuse the same interface templates.

Static corpora such as Mind2Web cannot directly validate execute–then–undo labels because their environments cannot be mutated and restored. They are suitable only for candidate discovery, opinion baselines, or out-of-distribution analysis unless paired with an executable reconstruction.

## 5. Current data volume and provenance

As of 2026-07-13, `probe_points.jsonl` and `POINT_MANIFEST.jsonl` each contain zero rows. The 32-row legacy grounding body and 30-row legacy manifest are quarantined/class-smoke assets: two rows lack probe provenance and the remaining 30 lack point-level state, action-instance, and transition identity. No legacy row is silently upgraded.

Historical SFT and DPO files do not establish the proposed method. The single SFT file is Magento-only. In the 62-row multi file, 12 mock and 50 WebArena origins are inferred from trajectory-name prefixes rather than explicit row-level provenance, and collector success is not recorded in the samples. Existing preferences are synthetic flips by generation mechanism, but their historical rows do not carry `negative_source`; and the distilled teacher file is empty. The legacy/mock assets are indexed in quarantine or isolated dataset tiers rather than silently rewritten. All must remain outside formal exports until exact point joins and admission gates are satisfied.

The historical trajectory metadata contains repeated logical trajectory IDs. Collection attempts cannot be counted as independent trajectories without immutable run-level identity and transactional raw/meta/key-state linkage.

## 6. State and candidate selection bias

The legacy key-state miner is an English keyword detector covering only three shopping action classes. Its 210/29/2 state distribution is strongly shaped by the Magento interface and collector trajectories. It cannot estimate recall over state-changing web actions, and it fails by construction on unenumerated domains.

The fixture-safe mutation miner can enumerate legal controls and verify observed changes, but it has not met the preregistered live coverage target. Keywords and LLMs may rank proposals but must not determine labels. The formal S4 consumer currently reads 300 snapshot-legal `iris.candidate.v2` proposals across 50 states (250 accessibility-enumerated and 50 producer-declared `source=expert` target bids), protected by a 300-row content-hash manifest. Candidate v2 lacks trajectory/run/collector-success provenance, so the latter 50 are not evidence of successful expert demonstrations. It contains no decoy or on-policy row; earlier development files with a decoy category are not admissible evidence and are not consumed by the formal DPO path. No candidate is point-grounded. Until candidate→probe→DPO lineage is complete, DPO negatives remain unlike deployment errors and cross-risk tests are vulnerable to lexical shortcuts such as treating every `delete` string as dangerous.

## 7. Input observability and rationale validity

A supervised action is learnable only if its bid and necessary context survive serialization. Prefix-only AXTree truncation can remove the target or undo affordance. The remediated builder uses action-anchored pruning and hard bid-visibility gates, but these safeguards need validation on newly collected long pages and multiple sites. The multiturn builder recomputes history deltas from consecutive observations; the single reached-state path currently trusts a declared `history_source=trajectory` and does not yet bind every history entry to immutable raw-run hashes.

`<prediction>`, `<undo>`, and `<rev_check>` can create false confidence if generated from action templates. Formal targets must use measured post-state diffs and executed undo traces, and must distinguish evidence visible in the current input from evidence observed only after executing the probe. Teacher prose does not strengthen a label; it can only verbalize pinned evidence. Teacher output has zero rows; with zero formal source rows, coverage is undefined rather than 0%.

## 8. Training limitations

SFT and DPO code paths do not compensate for missing ground truth. Existing synthetic DPO flips are useful adversarial fixtures but need not match model errors encountered during deployment. Formal DPO requires legal candidates or on-policy negatives and must report results by negative source.

The implemented GRPO/GSPO path is offline RL with verifiable rewards on fixed examples. Because the environment is not in the training loop, it cannot establish that the policy avoids repeated risk exposure across a multi-step rollout. Environment-in-the-loop training would require cloned accounts or transactional rollback, safe episode reset, and step/episode credit assignment; it must not create real orders or other persistent effects.

Reward weights and sparse penalties require ablation. Raw string matching is not a sufficient oracle for action legality, risky attempts, or backend commits.

## 9. Evaluation limitations

The historical `9.1% FSR` is withdrawn because the denominator did not represent true-danger points. Recoverability alone cannot define that denominator. Formal evaluation needs an independent policy/constraint truth table and separate declaration, attempted-action, and backend-commit rates.

Small cells are unstable. Any cell with fewer than 30 points should report raw numerator/denominator IDs and an interval, not an effectiveness claim. Label noise is especially consequential for FSR because finite-search negative labels can enter its denominator; reports must include UNKNOWN bounds, synthetic noise sensitivity, and clustered intervals by state/site.

The code implementation of an indicator is not an empirical result. No cross-site, cross-action, cross-privilege, calibration, task-success, or coupled/decoupled result currently exists.

## 10. Coupled versus decoupled comparison

The proposed comparison is not yet implemented as a complete, fair experiment. A verifier that sees an action and predicted delta may receive more tokens or different evidence than a coupled policy. A valid comparison must hold base checkpoint, point labels, candidate/evidence packet, and evaluation examples fixed; it should include capacity-matched and compute-matched variants plus a single-backbone multi-head baseline.

If paired inputs, calls, tokens, latency, parameter counts, training FLOPs, and labels cannot be audited, the coupling study must be removed from the contribution list rather than presented as landed.

## 11. Relationship to prior work

IRIS does not claim the first internalized world-model trace, the first predictive web guardrail, or the first use of reversibility as a safety signal. VAGEN is a direct precedent for explicit internalized world-model reasoning and RL credit assignment. WMA/WAC/SafePred cover forward or decoupled predictive use. WebGuard's normative risk and IRIS's operational recoverability are complementary axes, not competing definitions. Leave No Trace and Grinsztajn et al. predate IRIS on reversibility as a safety signal. Revisable by Design already contains a reversibility taxonomy, algorithm, StreamBench, and empirical agent evaluation; any IRIS delta must be demonstrated through point-level behavioral measurement and learned consumption, not asserted from terminology.

## 12. Reproducibility, privacy, and release

Prompt fingerprints are useful only if they resolve to immutable full text, parent/diff provenance, model identifier, and decoding parameters. New materialization separates a content `prompts_fp` from a `prompt_generation_fp` over producer/model/decode settings and formal gates validate both. The historical SFT fingerprint still has no bundle, and no non-empty formal sample currently demonstrates the new lineage. Model, opinion-label, or judge prompts added later must enter the same registry. Secrets must remain environment-only and be rotated if exposed.

The current assets have not completed a public-release PII and licensing audit. URLs, account information, order identifiers, user-generated text, and observations may contain sensitive or licensed material. Dataset release is blocked until redaction, benchmark/site terms, derived-data rights, destructive-action scope, and retention policy are documented.

## 13. Claims gate

Until formal point data and preregistered experiments exist, the paper may claim only that the repository implements a versioned ontology/schema, non-destructive legacy quarantine, shared input serialization, evidence and provenance gates, and a pilot execute–then–undo probe framework.

The paper must not claim:

- universal web-agent safety or universal irreversibility;
- a completed point-grounded dataset;
- true next-state supervision in current historical SFT;
- effective teacher distillation;
- cross-site/cross-action/cross-privilege generalization;
- coupled-vs-decoupled empirical findings;
- online multi-step RL results;
- the withdrawn 9.1% FSR as a safety result.
