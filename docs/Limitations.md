# IRIS Limitations

## Paper-ready limitations paragraph

> IRIS measures agent-relative operational recoverability rather than universal action safety. Each result is conditional on the agent's available actions and privilege, a finite recovery budget \(k\), the tested solver set, and the observed state signal \(\sigma\). A successful recovery trace is constructive evidence that the measured state projection was restored, whereas an unsuccessful trace only means that the tested solvers did not find a recovery within budget; it is not a proof of irreversibility. Moreover, restoring UI, API, or database state cannot necessarily retract real-world effects such as delivered messages, settled payments, third-party observations, audit records, or reputational harm. The present canonical smoke contains only 32 points across Magento shopping and Postmill reddit, and only 19 have complete transition bodies admitted to active forward supervision. Strict base train/dev/test and cross-site partitions remain unavailable; a cross-action diagnostic is not a model experiment and cannot substitute for a valid base split, a third environment, or long episode evidence. These dependencies limit both the external validity of the labels and any safety conclusions drawn from them.

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

Current formal probes use structured UI signals; no qualifying API/DB signal evidence is present in the readiness corpus. Even a future API or database diff would cover only the systems to which the benchmark has access. Signals can miss asynchronous jobs, caches, webhooks, notifications, emails, external payment networks, replicas, audit logs, third-party reads, and concurrent updates.

DB/API comparisons improve the upper bound on whether the measured state was restored; they do not make undo search complete and do not erase external consequences. Each action class needs a published signal-sufficiency statement and known blind spots.

## 4. Environment and task coverage

The repository registers Magento shopping/admin and Postmill reddit sites. The 32 current point-level smoke rows contain 21 Magento shopping/customer points and 11 Postmill reddit/customer points; no shopping-admin point exists. Historical training assets still carry `site=shopping`; the historical multi file's 12 mock / 50 WebArena split is inferred from `trajectory_id` prefixes because all 62 rows lack explicit environment-origin, mock, collector-success, and immutable-run provenance. Two-technology-family smoke data and challenge shard generation are not evidence of cross-site model generalization.

Cross-action, cross-privilege, and page-template splits test narrower generalization axes and cannot replace training and evaluation on different applications, interaction conventions, permission models, and side-effect systems. Before making web-wide claims, IRIS needs multiple non-Magento formal environments and a held-out environment whose probes do not reuse the same interface templates.

Static corpora such as Mind2Web cannot directly validate execute–then–undo labels because their environments cannot be mutated and restored. They are suitable only for candidate discovery, opinion baselines, or out-of-distribution analysis unless paired with an executable reconstruction.

## 5. Current data volume and provenance

As of 2026-07-16, `probe_points.jsonl` and `POINT_MANIFEST.jsonl` each contain 32 rows produced by reviewed, non-destructive live probes: 30 `CHANGED/RECOVERED` transitions and two `NO_EFFECT/UNKNOWN`. They cover add-to-cart, wishlist-add, compare-add, reddit vote, and reddit subscribe, but contain no negative-recovery point. Nineteen points have complete, manifest-pinned transition bodies; the other 13 points remain measurement evidence and are excluded from active forward supervision. The 32-row legacy grounding body and 30-row legacy manifest remain quarantined/class-smoke assets: two rows lack probe provenance and the remaining 30 lack point-level state, action-instance, and transition identity. No legacy row was silently upgraded.

Historical SFT and DPO files do not establish the proposed method. The single legacy SFT file is Magento-only. In the 62-row legacy multi file, 12 mock and 50 WebArena origins are inferred from trajectory-name prefixes rather than explicit row-level provenance, and collector success is not recorded in the samples. The active transition-v3 path materializes 38 single views over 19 transition-backed points and 24 trajectory-conditioned views over 12 exact-trajectory points, with exact lineage, no mock/failed collectors, and full bid visibility. Every serialized history has one step, no canonical long episode trace joins the formal SFT, and no valid base split exists. Active DPO contains 28 pairs: six reviewed legal-candidate counterfactuals and 22 manifest-pinned on-policy model errors. Two new 12-case partial preflights admitted 9 records and quarantined 3 provenance-incomplete calls each; the 18 admitted traces contain zero target-action attempts and do not establish legal deployment-action-error coverage. Active teacher output passes evidence QC for 62/62 views (38 single, 24 trajectory-conditioned). Earlier materialization, teacher, and transition-v2 stages are superseded audit evidence. Neither materialization establishes model effectiveness, and legacy/mock assets remain outside formal exports. The machine-readable audit remains `ready=false` with 13 blockers, including five split/DPO-release blockers and eight missing research-evidence gates.

The historical trajectory metadata contains repeated logical trajectory IDs. Collection attempts cannot be counted as independent trajectories without immutable run-level identity and transactional raw/meta/key-state linkage.

## 6. State and candidate selection bias

The legacy key-state miner is an English keyword detector covering only three shopping action classes. Its 210/29/2 state distribution is strongly shaped by the Magento interface and collector trajectories. It cannot estimate recall over state-changing web actions, and it fails by construction on unenumerated domains.

The fixture-safe mutation miner can enumerate legal controls and verify observed changes, but it has not met the preregistered live coverage target. Keywords and LLMs may rank proposals but must not determine labels. The active S4 v4 body contains 150 snapshot-legal candidates over 25 states (25 expert, 118 ordinary, and 7 constraint-trigger proposals), protected by a content-hash manifest. Candidate-role evidence is still incomplete: the 278-row v6 sidecar contains 228 proposals and only 50 evidence-backed roles, covering 3/8 roles: expert (16), constraint-trigger (17), and goal-violating (17). Evidence for ordinary, decoy, safe-alternative, uncertain/VERIFY, and policy-generated-error roles remains absent. Active DPO has 28 pairs: six reviewed legal-candidate counterfactuals and 22 on-policy errors. The new admitted traces contain no target-action attempts and do not provide an exact candidate-role evidence join, so they neither establish legal deployment-action-error coverage nor satisfy the `policy_generated_error` role. Cross-risk tests therefore remain vulnerable to lexical shortcuts such as treating every `delete` string as dangerous.

## 7. Input observability and rationale validity

A supervised action is learnable only if its bid and necessary context survive serialization. Prefix-only AXTree truncation can remove the target or undo affordance. The remediated builder uses action-anchored pruning and hard bid-visibility gates; all 62 active pinned risky-action bids and all 31 assistant click bids are visible. These checks now cover 19 transition-backed points across two technology families, but still need validation on additional sites and page families. Both materializers recompute history deltas from exact raw trajectories and pass collection-closure audit. Every current serialized history is only one step, and the canonical long-episode evidence set is empty, so long-history robustness remains untested.

`<prediction>`, `<undo>`, and `<rev_check>` can create false confidence if generated from action templates. Formal targets must use measured post-state diffs and executed undo traces, and must distinguish evidence visible in the current input from evidence observed only after executing the probe. Teacher prose does not strengthen a label; it can only verbalize pinned evidence. Teacher output passes evidence QC for all 62 active source views (38/38 single and 24/24 trajectory-conditioned). Earlier teacher and transition-v2 stages are retained as superseded audit evidence. Passing data QC does not establish that distillation improves a model.

## 8. Training limitations

SFT and DPO code paths do not compensate for missing ground truth. Existing synthetic DPO flips are useful adversarial fixtures but need not match model errors encountered during deployment. Formal DPO requires legal candidates or on-policy negatives and must report results by negative source.

The implemented GRPO/GSPO path is offline RL with verifiable rewards on fixed examples. Because the environment is not in the training loop, it cannot establish that the policy avoids repeated risk exposure across a multi-step rollout. Environment-in-the-loop training would require cloned accounts or transactional rollback, safe episode reset, and step/episode credit assignment; it must not create real orders or other persistent effects.

Reward weights and sparse penalties require ablation. Raw string matching is not a sufficient oracle for action legality, risky attempts, or backend commits.

## 9. Evaluation limitations

The historical `9.1% FSR` is withdrawn because the denominator did not represent true-danger points. Recoverability alone cannot define that denominator. Formal evaluation needs an independent policy/constraint truth table and separate declaration, attempted-action, and backend-commit rates.

Small cells are unstable. Any cell with fewer than 30 points should report raw numerator/denominator IDs and an interval, not an effectiveness claim. Label noise is especially consequential for FSR because finite-search negative labels can enter its denominator; reports must include UNKNOWN bounds, synthetic noise sensitivity, and clustered intervals by state/site.

The code implementation of an indicator is not an empirical result. A cross-action held-out diagnostic can be materialized, but the strict base split and cross-site split are unavailable; historical shards are superseded diagnostics. No cross-site, cross-action, cross-privilege, calibration, task-success, or coupled/decoupled model result currently exists.

The preserved OpenRouter formal rollout smokes are failure-analysis artifacts. The shopping smoke has two pre-observation-hash `reach_error` cases and zero evaluable episodes. An earlier reddit smoke reaches one point under both variants: its constraint case contains one declaration error but no risky action attempt, while its request case emits an illegal textual bid. The historical strict transition-v2 run reaches both variants but yields no executable action; its exact-hash errors are reused in v3 DPO. The active technology transition-v3 smoke also reaches both variants and yields no executable action, target execution zero, no established backend commit, and a constraint `EXECUTE` declaration (raw FSR-declaration 1/1). Every metric cell has only one independent point and `claimable=false`. These rows cannot support an effectiveness conclusion. OpenRouter was selected for these runs through the configurable provider interface and is not a repository-wide default.

## 10. Coupled versus decoupled comparison

The proposed comparison is not yet implemented as a complete, fair experiment. A verifier that sees an action and predicted delta may receive more tokens or different evidence than a coupled policy. A valid comparison must hold base checkpoint, point labels, candidate/evidence packet, and evaluation examples fixed; it should include capacity-matched and compute-matched variants plus a single-backbone multi-head baseline.

If paired inputs, calls, tokens, latency, parameter counts, training FLOPs, and labels cannot be audited, the coupling study must be removed from the contribution list rather than presented as landed.

## 11. Relationship to prior work

IRIS does not claim the first internalized world-model trace, the first predictive web guardrail, or the first use of reversibility as a safety signal. VAGEN is a direct precedent for explicit internalized world-model reasoning and RL credit assignment. WMA/WAC/SafePred cover forward or decoupled predictive use. WebGuard's normative risk and IRIS's operational recoverability are complementary axes, not competing definitions. Leave No Trace and Grinsztajn et al. predate IRIS on reversibility as a safety signal. Revisable by Design already contains a reversibility taxonomy, algorithm, StreamBench, and empirical agent evaluation; any IRIS delta must be demonstrated through point-level behavioral measurement and learned consumption, not asserted from terminology.

## 12. Reproducibility, privacy, and release

Prompt fingerprints are useful only if they resolve to immutable full text, parent/diff provenance, model identifier, and decoding parameters. New materialization separates a content `prompts_fp` from a `prompt_generation_fp` over producer/model/decode settings and formal gates validate both; the 62 active views exercise this lineage. The historical SFT fingerprint still has no bundle, and superseded bodies are not active evidence. Agent, collector, teacher, goal-template, LLM undo-attacker, and opinion-rater prompts are registry-managed; an external judge prompt added later must follow the same rule. Opinion records are keyed by point × goal variant × rater and retain exact input/raw-response hashes, but the canonical OpenRouter opinion run has only six LLM rows over 64 truth cases, no human rater, and no inferential value. Secrets must remain environment-only and be rotated if exposed. Provider, base URL, model, and key environment remain user-selectable in the frontend; the use of OpenRouter for the current live execution does not change that governance boundary.

The current assets have not completed a public-release PII and licensing audit. URLs, account information, order identifiers, user-generated text, and observations may contain sensitive or licensed material. Dataset release is blocked until redaction, benchmark/site terms, derived-data rights, destructive-action scope, and retention policy are documented.

## 13. Claims gate

Until statistically adequate multi-environment point data and preregistered experiments exist, the paper may claim only that the repository implements a versioned ontology/schema, non-destructive legacy quarantine, shared input serialization, evidence/provenance gates, and a non-empty 32-point, two-technology-family execute–then–undo pipeline smoke with 30 constructive recovery traces. Nineteen points currently have complete transition bodies admitted to active forward supervision; the resulting 62 source views and 62 teacher rewrites remain blocked from release by the strict split and research-evidence gates.

The paper must not claim:

- universal web-agent safety or universal irreversibility;
- a completed point-grounded dataset;
- true next-state supervision in current historical SFT;
- effective teacher distillation merely because active 62/62 rewrites—or superseded stages—passed data QC;
- cross-site/cross-action/cross-privilege generalization;
- coupled-vs-decoupled empirical findings;
- online multi-step RL results;
- the withdrawn 9.1% FSR as a safety result.
