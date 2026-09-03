# System Card - SOC AI Platform v10

Generated: 2026-09-03 11:10

## Intended use
Decision-support system for SOC analysts: vulnerability triage, compliance
mapping, remediation suggestion, and prioritization of security scanner
findings. The system RANKS and EXPLAINS; the analyst DECIDES. It is not
an autonomous remediation system and must not gate production deployments
without human review.

## Models and data sources
- LLM: microsoft/Phi-3.5-mini-instruct (4-bit NF4)
  + LoRA adapter (base) entraine uniquement sur des exemples approuves par l analyste
- Risk scoring: XGBoost + LightGBM ensemble, isotonic-calibrated (Layer 5B).
  Cible d entrainement: labels analystes uniquement (aucun score de regle en
  proxy). Evaluation: out_of_fold, f1=0.9157 (IC95 0.847-0.9677) sur n=79 cas.
  Les findings sans label sont scores mais ne servent jamais de cible.
- Retrieval: FAISS over 8 compliance frameworks (ISO 27001, NIST 800-53,
  NIST CSF 2.0, PCI DSS, GDPR, OWASP Top 10, CIS v8, SOC 2),
  similarity floor 0.45.
- Threat intel: CISA KEV catalog, EPSS scores, MITRE ATT&CK (P5 embeddings).
- Scanner inputs: SAST / SCA / secrets / container / DAST reports uploaded
  by the operator. No external customer data is used.

## Hallucination controls
- Closed-world control citation: compliance.controls restricted to the
  retrieved whitelist (allowed_control_ids); off-list ids are discarded
  and logged in rag_quality.
- Applicability filter (P1A): framework controls outside the asset scope
  (e.g. PCI on a dev asset) are removed before prompting.
- Deterministic verifiers (P1B/P1C): remediation version claims must match
  scanner fix_version; compliance evidence must anchor to the actual
  finding. Violations downgrade the field to manual_review_required.
- Enum guards on every structured field (priority, treatment, residual risk).
- Evidence tagging: every output field carries its origin (scanner, rag,
  llm, verifier) for auditability.

## Abstention policy
The system prefers silence over confident error. It abstains
(manual_review_required) when retrieval quality is insufficient, when the
verifier flags unsupported claims, and when the multi-agent debate (P7)
does not converge (epsilon = 0.02) or its consensus margin is below
0.08 or entropy above 1.3.

## Training-data firewall
Only examples with review_status=approved can enter fine-tuning
(safe_append_training_example, P1D). Files are audited before every
training run; contaminated records are quarantined. Retraining uses EWC +
experience replay (P6) and is followed by a rollback guard: a model that
regresses on the held-out set is not promoted.

## Human-in-the-loop
Analyst feedback (false-positive marks, severity adjustments, approvals)
is the only source of training labels and the only mechanism that changes
agent reliability weights. Demo feedback simulation is disabled by default
(demo_feedback_simulation = False).

## Limitations
- Trained and evaluated on a limited finding corpus; metrics with
  pseudo-labels are heuristic and partially circular (stated in Layer 12/13).
- Phi-3.5-mini (3.8B) on a T4: outputs are schema-validated but reasoning
  depth is bounded; the verifier layer compensates, not replaces, review.
- Compliance mappings are advisory and do not constitute audit evidence.
- KEV/EPSS coverage depends on refresh recency (Layer 3).

## Ethical and security boundaries
- The system analyzes defensive scanner output only; it does not generate
  exploit code, attack payloads, or offensive tooling.
- The red-team agent (7B) challenges the SYSTEM's own conclusions; it does
  not probe external targets.
- RBAC (Layer 10/11): destructive actions restricted to soc_lead role.
- All artifacts remain in the operator-controlled environment.
