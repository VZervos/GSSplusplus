# Hallucination detection with GSS (FactEval / T-REx pilot)

This document describes the post-generation hallucination-detection experiment built **around** GSS without modifying the GSS algorithm. GSS is treated as a black box: experiment scripts only choose the **query string** and apply **post-hoc** checks on the returned ACE subgraph.

**Current primary study:** n=30 stratified run — see [`RESEARCH_REPORT_N30.md`](RESEARCH_REPORT_N30.md).  
**Final research proposal (n=10 + n=30):** [`RESEARCH_PROPOSAL.md`](RESEARCH_PROPOSAL.md).  
**n=10 pilot artifacts:** archived at `out/hallucination/archive_n10_pilot/` (tables below document that earlier run).

---

## 1. Goal

Pivot from query-driven summarization to **post-generation claim validation**:

1. An LLM answers a factual question with no KG access.
2. Stock GSS builds an ACE subgraph from a mode-specific query.
3. A validator checks whether the ACE supports the claimed answer under the FactEval relation.
4. Labels vs gold are for analysis — **unsupported / not_validated ≠ proven hallucination**.

---

## 2. Experiment setup

### Data

- Source: LAMA T-REx (FactEval `trex` KG family), 10 WH-questions (`dataset/facteval_trex/sample_10.json`).
- LLM answers: local Ollama `llama3.1:8b` (`out/hallucination/llm_answers.json`), natural mistakes only.
- LLM accuracy on this sample: **4/10** (exact gold match).

| # | Rel | Focus | LLM answer | Gold | Exact match |
|---|-----|--------|------------|------|-------------|
| 1 | P19 | Catherine Tasca born? | France | Lyon | ✗ |
| 2 | P20 | Fredrik Idestam die? | Tampere | Helsinki | ✗ |
| 3 | P27 | Bahr al-Ulloum citizenship? | Sudan | Iraq | ✗ |
| 4 | P36 | Capital of Albania? | Tirana | Tirana | ✓ |
| 5 | P106 | Don Durant occupation? | Football player | actor | ✗ |
| 6 | P17 | Yuquot country? | Canada | Canada | ✓ |
| 7 | P159 | Roger Dubuis HQ? | Geneva | Geneva | ✓ |
| 8 | P495 | Milanesa created? | Argentina | Italy | ✗ |
| 9 | P740 | Pink Fairies formed? | London | London | ✓ |
| 10 | P937 | Newman work? | Oxford University | Rome | ✗ |

### Stack (unchanged GSS)

| Role | Choice |
|------|--------|
| GSS entry point | `pipeline.pipeline(query)` only |
| Entity extraction LLM | Ollama `llama3.1:8b` @ `http://127.0.0.1:11434` |
| Similarity | `all-MiniLM-L6-v2` |
| KG | DBpedia SPARQL `https://dbpedia.org/sparql` |
| Scoring (inside GSS) | \(w = 0.6\,w_{\mathrm{imp}} + 0.4\,w_{\mathrm{sim}}\) |

### Modes (query string only)

| Mode | GSS query example |
|------|-------------------|
| **answer** | `France` |
| **subject** | `Catherine Tasca` |
| **claim** | `Catherine Tasca was born in France.` |

### Post-hoc validation rule (current)

**Decisions**

| Decision | Meaning |
|----------|---------|
| **supported** | Subject + answer co-occur on a triple whose predicate is in the FactEval relation allowlist (e.g. P19 → `birthPlace`) |
| **weak_support** | Subject + answer co-occur, but predicate is **not** in the allowlist (reported separately; not counted as full support) |
| **unsupported** | ACE non-empty, no subject–answer co-occurrence |
| **uncertain** | Empty ACE / empty answer / pipeline error |

**Gold acceptance**

- `gold_match`: exact/fuzzy label match to T-REx gold.
- `granularity_match`: coarser/multi-valued agreement with gold **in the ACE**, e.g.:
  - **multi_valued_relation**: subject `-[rel]->` LLM answer **and** subject `-[rel]->` gold (Tasca birthPlace France **and** Lyon)
  - **hierarchical_containment**: gold and LLM answer linked by a containment-like predicate in the ACE (e.g. city–country), with relation support for the claim
- `gold_acceptable` = `gold_match` ∨ `granularity_match`

**Detection labels (analysis vs gold)**

| Detection | Condition |
|-----------|-----------|
| `true_support` | exact gold + supported |
| `true_support_granularity` | granularity only + supported |
| `weak_true_support` | gold_acceptable + weak_support |
| `missed_support` | gold_acceptable + unsupported |
| `false_support` | not gold_acceptable + supported |
| `weak_false_support` | not gold_acceptable + weak_support |
| `not_validated` | not gold_acceptable + unsupported |
| `uncertain` | uncertain decision |

Scripts: `answer_facteval.py`, `validate_with_gss.py`, `validation.py`, `pipeline/claim_validation.py`.

### How to run

```powershell
$env:LLM_PROVIDER='ollama'; $env:DATASET='dbpedia'; $env:PYTHONUNBUFFERED='1'; $env:PYTHONIOENCODING='utf-8'
python -u src/hallucination/answer_facteval.py -n 10
python -u src/hallucination/validate_with_gss.py --modes answer subject claim -n 10
```

Outputs: `out/hallucination/validation_results_{answer,subject,claim}.json`.

Latest full 3×10 rerun (with weak_support + granularity): ~**94 minutes**.

---

## 3. Results (rerun with fixes 2 & 3)

### Aggregate decisions / detections

| Metric | answer | subject | claim |
|--------|--------|---------|-------|
| supported | 1 | 3 | 3 |
| weak_support | 1 | 1 | 0 |
| unsupported | 8 | 5 | 6 |
| uncertain | 0 | 1 | 1 |
| true_support | 1 | 2 | 2 |
| true_support_granularity | 0 | **1** | **1** |
| weak_true_support | 0 | **1** | 0 |
| missed_support | 3 | 1 | 2 |
| false_support | **0** | **0** | **0** |
| weak_false_support | 1 | 0 | 0 |
| not_validated | 5 | 4 | 4 |
| gold_match (exact) | 4 | 4 | 4 |
| gold_acceptable | 4 | **5** | **5** |
| granularity_match | 0 | 2 | 2 |

### Per-item snapshot

| Rel | LLM → gold | answer | subject | claim |
|-----|------------|--------|---------|-------|
| P19 | France → Lyon | not_validated | **true_support_granularity** | **true_support_granularity** |
| P20 | Tampere → Helsinki | weak_false_support | not_validated | not_validated |
| P27 | Sudan → Iraq | not_validated | uncertain | not_validated |
| P36 | Tirana → Tirana | true_support | true_support | true_support |
| P106 | Football → actor | not_validated | not_validated | uncertain |
| P17 | Canada → Canada | missed_support | **weak_true_support** | missed_support |
| P159 | Geneva → Geneva | missed_support | true_support | true_support |
| P495 | Argentina → Italy | not_validated | not_validated | not_validated |
| P740 | London → London | missed_support | missed_support | missed_support |
| P937 | Oxford → Rome | not_validated | not_validated | not_validated |

### Detector as binary classifier

Treat **supported** only as accept (weak_support = reject for strict F1). Gold positive = `gold_acceptable`.

| Mode | Strict accepts that are gold_acceptable | Strict false accepts | Notes |
|------|------------------------------------------|----------------------|-------|
| answer | 1 (Tirana) | 0 | Still high-precision / low-recall |
| subject | 3 (Tirana, Geneva, France/Lyon gran.) | **0** | Best balance after fixes |
| claim | 3 (same as subject on strict support) | **0** | Same strict accepts as subject |

Compared to the previous run (before fixes): subject/claim **false_support dropped from 1 → 0** because France vs Lyon is now `true_support_granularity`. Yuquot–Canada under subject is now **weak_true_support** instead of hard `missed_support`.

---

## 4. Discussion

### Effect of the two fixes

1. **weak_support**  
   - Separates “linked but wrong/near predicate” from full support and from total miss.  
   - Example: subject mode Yuquot–Canada via `dbo:location` (not in strict P17 allowlist) → `weak_support` / `weak_true_support`.  
   - Example: answer mode Tampere linked to Idestam without deathPlace → `weak_false_support` (not treated as full support).

2. **Granularity / multi-valued gold**  
   - ACE contains both `Tasca birthPlace France` and `Tasca birthPlace Lyon`.  
   - LLM “France” is no longer `false_support` vs T-REx; it is **KG-faithful coarser answer** → `true_support_granularity`.  
   - `gold_acceptable` rises to 5/10 on subject/claim (exact 4 + granularity 1 on the France case; Geneva also flags multi-valued metadata when present).

### What still holds from the earlier analysis

- **Answer mode** almost always keeps the answer entity and rarely the subject → many `missed_support` on correct hub answers (Canada, Geneva, London).
- **Subject / claim** remain the useful detection modes; they agree on most strict supports.
- **Pink Fairies → London** remains `missed_support` in all modes: ACE stays band-centric; no founding/hometown–London triple kept.
- Empty ACE → **uncertain** is honest (Bahr subject; Durant claim).
- Bottleneck remains **joint subject–predicate–answer** evidence in the ACE, not empty summaries in general.

### Mode recommendation (unchanged direction, cleaner metrics)

| Goal | Prefer |
|------|--------|
| Main detection protocol | **subject** (or claim) |
| Ablation / high precision | **answer** |
| Reporting | Report strict `supported` as primary; show `weak_support` as secondary diagnostic |

### Limits

- n=10, one LLM, one KG snapshot.
- Allowlist still incomplete for some geo relations (hence weak tier).
- Hierarchical containment beyond multi-valued same-predicate cases is conservative.
- Runtime dominated by public DBpedia + local Llama entity extraction (~1.5 h for 3×10).

### Highest-value next steps (still outside GSS)

1. Optionally expand P17 allowlist (`location`, …) if weak_true cases should become full support.
2. Scale to ~30 items after these scoring rules.
3. Keep gold-oracle mode out of the main protocol (no test-time use case).

---

## 5. Design history (short)

- Custom answer-centered seeding wrappers were dropped: **GSS stays unmodified**; modes only change the query string.
- Validation evolved: strict relation link → plus **weak_support** and **granularity-aware gold** so evaluation matches KG reality (multi-valued birthPlace, near-synonym predicates).

---

## 6. Artifacts

| Path | Role |
|------|------|
| `dataset/facteval_trex/sample_10.json` | 10 questions + gold |
| `out/hallucination/llm_answers.json` | Stage-1 Llama answers |
| `out/hallucination/validation_results_answer.json` | Answer-mode results (latest) |
| `out/hallucination/validation_results_subject.json` | Subject-mode results (latest) |
| `out/hallucination/validation_results_claim.json` | Claim-mode results (latest) |
