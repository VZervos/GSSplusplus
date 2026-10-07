# CoLoTa demo — archived main experiment (n=20)

**Date:** 2026-09-20 (rerun artifacts frozen 2026-09-23)  
**Code snapshot:** `out/colota_demo/archive_n20_main/code/`  
**Live code (may have moved on):** `src/colota_demo/`  
**Raw outputs:** `results.csv`, `results.json`, `metrics.json` in this folder

This archive freezes the first CoLoTa RAG demo: same 20 yes/no questions, three answering conditions, and retrieval-depth / ranking diagnostics.

**Follow-up (summary quality of each top-k list):** [`out/colota_summary/REPORT.md`](../../colota_summary/REPORT.md) and code in `src/colota_summary/`. That experiment is not part of this archive.

**Later answering runs (same 20 IDs):** GSS live Wikidata [`../gss/REPORT.md`](../gss/REPORT.md), GSS on the 37-triple corpus [`../gss_corpus/REPORT.md`](../gss_corpus/REPORT.md), iSummary official JAR [`../isummary/REPORT.md`](../isummary/REPORT.md).

---

## 1. Motivation

CoLoTa is a long-tail commonsense QA benchmark grounded in Wikidata. Each item has a yes/no question, a gold answer, anchor entities, a tiny gold subgraph (`kg_triples`), an inference rule, and annotated reasoning steps.

The research question for this demo was **Problem 2 from the CoLoTa notes**: does giving the LLM question-specific KG context help on obscure entities, compared with parametric memory alone?

We wanted a *minimal* comparison, not a graph database:

| Condition | What it isolates |
|-----------|------------------|
| **LLM-only** | Baseline parametric knowledge |
| **Gold triples** | Reasoning ceiling given oracle evidence |
| **RAG@k** | A realistic system: retrieve then answer |
| **Recall@k / MRR** | Retrieval quality, separate from answering |

CoLoTa’s gold **inference rule** was tried and then **dropped** from the main protocol. The rule is a diagnostic/oracle annotation, not something a real RAG system would retrieve. In a realistic pipeline the model must infer the relationship from facts.

---

## 2. Setup

### 2.1 Model and retrieval

- **LLM:** local Ollama `llama3.1:8b`, temperature 0, JSON constrained to `TRUE` / `FALSE` (RAG may also return `INSUFFICIENT`)
- **Retriever:** `sentence-transformers/all-MiniLM-L6-v2`, cosine similarity
- **Corpus:** all unique gold triples from the 20 sampled questions pooled together (37 documents). For each question the retriever searches this pool, so distractors are facts that belong to *other* questions.
- **Depths:** k ∈ {1, 3, 5, 10}; primary reporting k = 5

### 2.2 Item selection (not random)

Items were **hand-picked**, not drawn at random. Two filters:

1. Boolean gold answer and non-empty `kg_triples`.
2. **Entity alignment:** names that appear in the question also appear in the gold triples. A large slice of CoLoTa rewrites the query to a long-tail entity but leaves triples about someone else; gold-context is meaningless on those rows.

The 20 IDs are balanced **10 True / 10 False** and mix reasoning types (geo, occupation, temporal, set inclusion, medical, cultural, …):

| ID | Gold | Strategy (short) | Question |
|----|------|------------------|----------|
| S1 | True | number comparison | Horsens vs Ikast population to 60k |
| S2 | False | set inclusion | Yui Yuigahama’s pet a corgi? |
| S3 | True | occupation | Leopold Lanner’s wife make a living with her voice? |
| S5 | True | political | Ulf Kristersson’s PM predecessor a woman? |
| S13 | False | geographical | Taxi from Vaughan to Jamjuree Art Gallery? |
| S37 | False | geo / physical | Travel Gujan → Aousserd only by car? |
| S51 | True | temporal / sports | Could Ianis Hagi’s dad see his Serie A debut? |
| S87 | False | biological | Paolo Cannavaro middle-child syndrome? |
| S95 | False | technological | Can an IKCO Samand make a vlog on its own? |
| S131 | False | occupation | Would CEO Pinar Abay typically clean toilets? |
| S142 | True | language / geo | Can Arimasa Mori talk to New Brunswick without a translator? |
| S160 | True | geo / citizenship | Can Arsen Minasian go to Tehran without a passport? |
| S164 | True | education | Would a math professor explain Pythagoras in class? |
| S175 | False | music | Is Lee Hyori a major disco singer? |
| S184 | False | historical / medical | Did Bernard Gohienetxe die peacefully? |
| S4 | False | temporal / tech | Could Maria de Ventadorn phone 100 miles away? |
| S10 | False | cultural | Tinker Tailor Soldier Spy for preschool? |
| S34 | True | literature | Dowlatabadi work in the genre of *The Makioka Sisters*? |
| S55 | True | medical | Ala al-Din Muhammad III mental disorder? |
| S124 | True | entity comparison | Same fate: Ivan Shuisky and Benny Frey? |

### 2.3 Prompts (no inference rule)

- **LLM-only:** question → JSON `{ "answer": "TRUE"|"FALSE" }`
- **Gold triples:** question + the item’s full `kg_triples`; “use only this knowledge”
- **RAG:** question + top-k retrieved triples; may abstain with `INSUFFICIENT`

### 2.4 Metrics

**Answering**

- Accuracy of LLM-only, gold triples, RAG@k
- `INSUFFICIENT` counts as incorrect

**Retrieval (independent of the LLM answer)**

- **Recall@k:** fraction of that item’s gold triples appearing in top-k
- **MRR:** reciprocal rank of the *first* gold triple in the full corpus ranking
- **Mean RR:** average of 1/rank over *all* gold triples (misses count as 0)
- First-relevant rank; share of items with first gold at rank 1 / ≤3 / ≤5

**Failure tags (at k=5)**

| Tag | Meaning |
|-----|---------|
| `knowledge` | LLM-only wrong, gold triples right → missing parametric facts |
| `oracle_reasoning` | Even gold triples are wrong → hard inference |
| `retrieval` | Recall@5 < 1 → incomplete evidence in the prompt |
| `reasoning` | Recall@5 = 1 but RAG wrong → facts present, inference failed |
| `insufficient` | RAG abstained |

---

## 3. Results

### 3.1 Main conditions (k=5)

| Condition | Accuracy | Role |
|-----------|----------|------|
| LLM-only | **50.0%** | baseline |
| Gold triples | **90.0%** | reasoning ceiling |
| RAG@5 | **70.0%** | realistic system |

LLM-only is chance on a balanced binary set. Oracle evidence almost solves the task. RAG sits in between.

### 3.2 Retrieval depth

| k | Recall | RAG accuracy |
|---|--------|----------------|
| 1 | 64.6% | 65.0% |
| 3 | 95.8% | **75.0%** |
| 5 | 97.5% | 70.0% |
| 10 | 100.0% | 75.0% |

Recall saturates by k=3–10. Answer accuracy does **not** keep rising. RAG@5 is worse than RAG@3: extra triples add distractors (e.g. S142 is correct at k=1 and k=3, wrong at k=5, correct again at k=10).

### 3.3 Ranking

| Metric | Value |
|--------|-------|
| MRR (first relevant) | **1.000** |
| Mean RR (all gold triples) | 0.802 |
| Mean first-relevant rank | 1.00 |
| First relevant @1 / @3 / @5 | 100% / 100% / 100% |

On this 37-document pool, the retriever always puts **at least one** gold triple at rank 1. Later supporting facts sit lower (mean RR 0.80). Retrieval of *a* relevant fact is not the bottleneck; assembling the *full* gold subgraph and reasoning over a mixed list is.

### 3.4 Failure tags at k=5

| Tag | Count | Reading |
|-----|-------|---------|
| knowledge | 8 | LLM-only fails, oracle triples would have sufficed |
| reasoning | 6 | Full gold set in top-5, RAG still wrong |
| oracle_reasoning | 2 | S1 (numeric comparison) and S34 (genre matching) fail even with gold triples |
| retrieval | 1 | S37: continents only appear by k=10 |
| insufficient | 1 | S4: model abstains despite having birth/death dates |

### 3.5 Per-item snapshot

See `results.csv`. Notable rows:

- **S37 (Gujan–Aousserd):** Recall@5 = 0.5 (countries only); continents arrive at k=10. Classic incomplete multi-hop subgraph.
- **S142 (Arimasa Mori / French):** both gold facts by rank 2, but RAG@5 flips to False — distractor effect.
- **S51, S160:** gold triples correct, RAG wrong at every k — reasoning under retrieved (slightly noisy) context.
- **S1, S34:** gold-triples condition itself is wrong — llama3.1:8b does not always execute the numeric / set-inclusion step even with a perfect fact list.

---

## 4. Findings

1. **Long-tail parametric knowledge is weak** (50%). Context helps.
2. **Oracle triples are almost enough** (90%). The intended commonsense step is usually within reach of this model when the fact list is clean and complete.
3. **RAG’s gap to the ceiling is not “missed the first fact.”** MRR = 1 and Recall@3 ≈ 96%. The remaining errors are incomplete multi-hop sets, distractors in top-k, and failed inference.
4. **Bigger k is not automatically better.** That is the GSS-shaped result: a larger neighborhood is a worse *summary* even when coverage rises.
5. **The inference rule is the wrong main condition** for a RAG story. It was removed from this protocol.

### Limitations

- n=20, one local 8B model, binary QA only.
- Corpus is tiny (37 triples from the same 20 items). High MRR is partly an artifact of that pool; a full CoLoTa or live Wikidata neighborhood would stress retrieval much harder.
- No summary-quality metrics (precision, F1, redundancy) — those are the follow-up experiment.
- Frozen CoLoTa triples, not live Wikidata.

---

## 5. How to reproduce

From the repo root, with Ollama serving `llama3.1:8b`:

```text
python src/colota_demo/run.py --k 5 --ks 1,3,5,10 --model llama3.1:8b
```

The hand-picked IDs live in `src/colota_demo/scripts/dataset.py` (`DEMO_IDS`).
