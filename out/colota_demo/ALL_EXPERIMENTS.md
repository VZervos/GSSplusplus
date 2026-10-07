# CoLoTa demo — all approaches (n=20)

**Date compiled:** 2026-10-07  
**Live code:** `src/colota_demo/approaches/`  
**Suite driver:** `src/colota_demo/run_suite.py` + `src/colota_demo/config.json`  
**Shared items:** `DEMO_IDS` in `src/colota_demo/utils/dataset.py` (10 True / 10 False)

This document is the combined write-up of every answering approach under `src/colota_demo/approaches/`. Per-run artifacts stay in their experiment folders. **2026-10-07** added PPR and extract_embed with **GSS llama NER** seeds (not a rerun of the rest). LLM-only, gold triples, MiniLM RAG, GSS (Wikidata + corpus), iSummary, and gold-seed PPR/extract_embed numbers stay from the archives listed below.

| Approach | Code | Experiment folder | Per-run report |
|----------|------|-------------------|----------------|
| `llm_only` | `approaches/llm_only.py` | `out/colota_demo/first_experiment/` | [`first_experiment/REPORT.md`](first_experiment/REPORT.md) |
| `gold_triples` | `approaches/gold_triples.py` | same | same |
| `rag_transformers` | `approaches/rag_transformers/` | same | same |
| `gss` (live Wikidata) | `approaches/gss/` | `out/colota_demo/gss/` | [`gss/REPORT.md`](gss/REPORT.md) |
| `gss` (37-triple corpus) | same, `gss.kg=corpus` | `out/colota_demo/gss_corpus/` | [`gss_corpus/REPORT.md`](gss_corpus/REPORT.md) |
| `isummary` | `approaches/isummary/` | `out/colota_demo/isummary/` | [`isummary/REPORT.md`](isummary/REPORT.md) |
| `ppr` (gold `kg_entities`) | `approaches/ppr/` | `out/colota_demo/ppr_extract_embed/` | [`ppr_extract_embed/REPORT_ppr.md`](ppr_extract_embed/REPORT_ppr.md) |
| `extract_embed` (gold `kg_entities`) | `approaches/extract_embed/` | same | [`ppr_extract_embed/REPORT_extract_embed.md`](ppr_extract_embed/REPORT_extract_embed.md) |
| `ppr` (GSS NER) | same, `--seeds gss_ner` | `out/colota_demo/ppr_ner/` | [`ppr_ner/REPORT.md`](ppr_ner/REPORT.md) |
| `extract_embed` (GSS NER) | same, shared NER cache | `out/colota_demo/extract_embed_ner/` | [`extract_embed_ner/REPORT.md`](extract_embed_ner/REPORT.md) |

GSS appears twice because it is one approach with two knowledge sources. iSummary is the official JAR, not a Python reimplementation. PPR and extract_embed are **toy** rankers (not HippoRAG / not G-Retriever).

---

## 1. Research question

CoLoTa is a long-tail commonsense QA benchmark grounded in Wikidata. Each item has a yes/no question, a gold boolean, Wikidata `kg_entities`, a tiny English gold subgraph (`kg_triples`), and (unused here) an inference-rule annotation.

The shared question across these demos is **Problem 2 from the CoLoTa notes**: does **question-specific KG context** help a small local LLM on obscure entities, compared with parametric memory alone?

That splits into three layers:

1. **Ceiling** — if the model is handed the gold subgraph, can it do the commonsense step?
2. **Retrieval** — can a realistic ranker recover those gold triples (Recall@k, MRR, Hits@k)?
3. **Answering under noise** — given a top-k list that may be incomplete or mixed with other questions’ facts, does accuracy rise with k or fall?

The CoLoTa **inference rule** is **not** in the protocol. It is an oracle annotation, not something a deployed retriever would return.

---

## 2. Shared protocol

### 2.1 Items (not random)

Twenty CoLoTa QA rows, **hand-picked**, not a random sample:

1. Boolean gold answer and non-empty `kg_triples`.
2. **Entity alignment:** names that appear in the question also appear in the gold triples. A large slice of CoLoTa rewrites the query to a long-tail entity but leaves triples about someone else; gold context is meaningless on those rows.

Balanced **10 True / 10 False**. Mix of reasoning types (numeric, set inclusion, occupation, geo, temporal, medical, cultural, …).

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

### 2.2 Model, depths, corpus

- **Answering LLM:** local Ollama `llama3.1:8b`, temperature 0, JSON `TRUE` / `FALSE`. Retrieval approaches may also return `INSUFFICIENT`.
- **Depths:** k ∈ {1, 3, 5, 10}. **Primary reporting k = 5.**
- **Closed corpus (when used):** all unique gold triples from these 20 items pooled together → **37 documents**. For a given question the ranker searches this pool, so distractors are facts that belong to *other* questions.
- **Embedding model (RAG / extract_embed):** `sentence-transformers/all-MiniLM-L6-v2`.

### 2.3 Metrics

**Answering**

- Accuracy at each k. `INSUFFICIENT` counts as **incorrect**.
- LLM-only and gold triples have a single prediction (no k).

**Retrieval** (independent of the LLM answer), vs that item’s English `kg_triples` (exact string match):

- **Recall@k:** fraction of gold triples in the top-k list.
- **MRR:** reciprocal rank of the *first* gold triple in the full ranking.
- **Mean RR:** average of 1/rank over *all* gold triples (misses = 0).
- **Hits@k:** share of items whose first gold triple has rank ≤ k.
- Mean first-relevant rank (among items that find any gold).

Live Wikidata GSS did not compute Recall/MRR against English gold triples (URI neighborhood ≠ 37-document pool). iSummary Recall is 0 because PARTI paths are SPARQL URIs, not `(Horsens, population, …)` strings.

### 2.4 What was *not* rerun

LLM-only, gold triples, RAG, both GSS runs, and iSummary were **included from archives**, not re-executed with ppr/extract_embed. Answering numbers for those conditions are therefore from earlier llama passes (first experiment frozen 2026-09-23; GSS Wikidata 2026-10-05; GSS corpus and iSummary 2026-10-06). PPR vs extract_embed comparisons at the same k are same-process; PPR vs RAG answering gaps mix two days.

---

## 3. Headline comparison

### 3.1 Accuracy at primary k=5

| Rank | Condition | Acc@5 | Knowledge source | Linking / ranking |
|------|-----------|-------|------------------|-------------------|
| 1 | Gold triples | **90.0%** | that item’s `kg_triples` | oracle (no ranking) |
| 2 | PPR@5 gold entities | **75.0%** | 37-triple pool | gold `kg_entities` + PageRank α=0.85 |
| 2 | extract_embed@5 gold entities | **75.0%** | 37-triple pool | gold `kg_entities` filter + MiniLM |
| 4 | MiniLM RAG@5 | **70.0%** | 37-triple pool | question embedding, no entity filter |
| 5 | LLM-only | **50.0%** | none | parametric |
| 5 | GSS@5 live Wikidata | **50.0%** | SPARQL 1-hop | llama NER + a=0.75 mix |
| 7 | GSS@5 corpus | **40.0%** | 37-triple pool | llama NER + a=0.75 mix |
| 7 | PPR@5 GSS NER | **40.0%** | 37-triple pool | GSS NER resources + PageRank |
| 9 | extract_embed@5 GSS NER | **35.0%** | 37-triple pool | GSS NER resources + MiniLM filter |
| 10 | iSummary@5 | **10.0%** | Wikidata SPARQL log | official JAR, log seeds |

Oracle facts almost solve the task. Toy rankers with **gold entity names** beat MiniLM RAG. The same toys with **GSS NER** collapse to GSS-corpus territory (40% / 35%). Full-question MiniLM still beats parametric memory **without** a linker. Original GSS and iSummary do **not** beat LLM-only once linking / log coverage fail.

### 3.2 Retrieval at k=5 (closed 37-pool, English gold strings)

| Condition | Recall@5 | MRR | Mean RR | Hits@1 | Hits@5 |
|-----------|----------|-----|---------|--------|--------|
| MiniLM RAG | 97.5% | 1.000 | 0.802 | 100% | 100% |
| PPR (gold entities) | 97.5% | 1.000 | 0.800 | 100% | 100% |
| extract_embed (gold entities) | 86.7% | 1.000 | 0.753 | 100% | 100% |
| PPR (GSS NER) | 40.0% | 0.450 | 0.320 | 45% | 45% |
| GSS corpus | 38.3% | 0.450 | 0.319 | 45% | 45% |
| extract_embed (GSS NER) | 35.8% | 0.450 | 0.306 | 45% | 45% |
| iSummary | 0.0% | 0.000 | 0.000 | 0% | 0% |
| GSS Wikidata | — | — | — | — | — |

On this pool, **finding the first gold fact is trivial** for MiniLM and gold-seed PPR/extract_embed (Hits@1 = 100%). GSS NER PPR/extract_embed match corpus GSS: **no gold triple on 11/20 items**. iSummary never emits an English gold string.

### 3.3 Depth: answering vs coverage

| k | RAG acc | RAG R | PPR acc | PPR R | extract_embed acc | extract_embed R | GSS corpus acc | GSS corpus R | GSS Wiki acc | iSummary acc |
|---|---------|-------|---------|-------|-------------------|-----------------|----------------|--------------|--------------|--------------|
| 1 | 65% | 64.6% | 70% | 64.6% | **75%** | 64.6% | 35% | 25.4% | **55%** | 10% |
| 3 | **75%** | 95.8% | **85%** | 96.2% | **80%** | 86.7% | 40% | 38.3% | 40% | 10% |
| 5 | 70% | 97.5% | 75% | 97.5% | 75% | 86.7% | 40% | 38.3% | 50% | 10% |
| 10 | 75% | 100% | 80% | 97.5% | 75% | 86.7% | 40% | 38.3% | 30% | 10% |

**k=3 is the best answering depth** for RAG (75%), PPR (85%), and extract_embed (80%). Extra triples add distractors. Live Wikidata GSS is worst at k=10 (30%). Corpus GSS and iSummary are flat because the summary is already empty or tiny by k=3.

---

## 4. `llm_only` — parametric baseline

**Code:** `approaches/llm_only.py`  
**Run:** first experiment (archived). No corpus, no k.

### Theory

A frozen 8B chat model stores some facts in weights. CoLoTa is built so many entities are **long-tail**: not Wikipedia-famous, not likely to be memorized cleanly. If parametric memory were enough, retrieval would be unnecessary. The baseline measures that.

### Algorithm

Prompt: “Answer using only your own knowledge.” JSON `TRUE`/`FALSE` only (no abstain). No triples.

### Results

**Accuracy 50.0%** (chance on a balanced binary set).

True items: **1/10** correct (S55 only). Almost every Gold-True long-tail fact is missing. Most of the 50% is Gold-False items that the model also labels False (default “no” on obscure claims).

Failure tag `knowledge` (LLM wrong, gold triples right) hits **8** items in the first experiment: the intended facts exist in CoLoTa and the model can use them when they are written in the prompt.

---

## 5. `gold_triples` — oracle ceiling

**Code:** `approaches/gold_triples.py`  
**Run:** first experiment (archived). No ranking.

### Theory

This is the **reasoning ceiling** given a complete, clean, English fact list. It isolates “can llama3.1:8b execute the commonsense step?” from “can we retrieve the facts?” If this condition is near-perfect, remaining RAG errors are retrieval/distractors, not an impossible inference.

### Algorithm

The item’s full `kg_triples` are pasted into the prompt (“use only this knowledge”). JSON `TRUE`/`FALSE`. The CoLoTa inference rule is **not** included.

### Results

**Accuracy 90.0%.** Misses:

- **S1** (True → False): both population numbers are in the prompt; the numeric “who hits 60k first” comparison fails.
- **S34** (True → False): genre / set-inclusion over literary works fails even with the gold list.

Those two are tagged `oracle_reasoning`. The intended CoLoTa step is usually within reach of this 8B model when the fact list is short, complete, and in English.

---

## 6. `rag_transformers` — MiniLM cosine RAG

**Code:** `approaches/rag_transformers/` (`retriever.py` + approach wrapper)  
**Run:** first experiment (archived).

### Theory

Standard retrieve-then-generate. Encode the question and every corpus document with a sentence embedding model; rank by cosine similarity; pass top-k to the LLM. No entity linking, no graph walk. On a 37-document pool this is a **strong** retriever because gold triples share lexical overlap with the question (entity names). Distractors are other items’ triples that happen to be close in embedding space.

### Algorithm

1. Pool 37 unique gold triples; encode with MiniLM once.
2. Encode the question; cosine vs all 37; sort.
3. Top-k → llama with permission to answer `INSUFFICIENT`.
4. Ranking metrics vs that item’s `kg_triples`.

### Results (k=5)

- **Accuracy 70.0%** (between LLM-only 50% and gold 90%).
- **Recall@5 97.5%, MRR 1.000, Hits@1 100%.** At least one gold triple is always rank 1. Full-subgraph recall saturates by k=3–10 (100% at k=10).
- **RAG@3 = 75% > RAG@5 = 70%.** S142 is correct at k=1 and k=3, wrong at k=5, correct again at k=10 (distractor list, not missing facts).
- **S4:** model abstains at every k despite having birth/death dates (`insufficient`).
- **S37:** Recall@5 = 50% (countries only); continents arrive at k=10 (`retrieval`).
- **S51, S160:** gold triples correct, RAG wrong at every k (`reasoning` under noisy context).

Failure tags at k=5: knowledge 8, reasoning 6, oracle_reasoning 2, retrieval 1, insufficient 1.

**Finding:** the RAG gap to 90% is **not** “missed the first fact.” It is incomplete multi-hop sets, distractors in top-k, and failed inference (including two items the oracle itself cannot solve).

---

## 7. `gss` — Graph Summarization Scoring (two KGs)

**Code:** `approaches/gss/` (original pipeline copied in; ranking/SPARQL/scoring **not** modified for these demos).  
**Runs:** live Wikidata (`out/colota_demo/gss/`), then local 37-triple corpus (`out/colota_demo/gss_corpus/`).

### Theory

GSS builds a **query-focused subgraph** rather than embedding a closed document list:

1. **NER** on the question (here: same local llama3.1:8b, original GSS few-shot prompt).
2. Map entity strings to KG nodes (Wikidata URI lookup, or name match on the local graph).
3. **1-hop expansion** around those nodes.
4. Prune, then rank with  
   \(\mathrm{final\_score} = a \times \mathrm{importance} + (1-a) \times \mathrm{similarity}\),  
   paper mixing weight **a = 0.75**.

Importance is graph-structural (degree / hubness in the expanded neighborhood). Similarity is question–triple relatedness in the original GSS scorer. The intended product is a short, important, on-topic summary — not the full 1-hop dump.

This is a **different** theory from MiniLM RAG: linking quality dominates. If NER emits the wrong QIDs, SPARQL (or local lookup) walks the wrong graph, and ranking cannot recover.

### Algorithm (as run)

- NER → URI / local node → 1-hop → prune → mix a=0.75 → top-k English or URI triples → separate answering prompt (`TRUE`/`FALSE`/`INSUFFICIENT`).
- Empty subgraph if NER matches nothing (no question-word fallback).
- Wikidata: live `query.wikidata.org`, retries on 504. Context lines are mostly **URI triples** (`s_label` empty in this pipeline).
- Corpus: lookup/fetch/degree restricted to the 37 triples. Context is the local `(s, p, o)` strings.

JSON peel / `//` strip on NER replies is glue so llama’s wrapped output still parses; it does not change ranking math.

### 7.1 Live Wikidata results

**GSS@5 = 50.0% = LLM-only.** Depth: 55% / 40% / 50% / **30%** at k=1,3,5,10. More triples make answers worse.

True items correct at k=5: **2/10** (S5, S51 — both k=5-only flips with GMT leak in the subgraph). S160 ranked subgraph size **0** → `INSUFFICIENT` at every k (`RANKING_MIN_SCORE = 0.1`).

Typical SPARQL pattern:

1. **Few-shot NER leakage.** Prompt examples (GMT / Q30 / P36 / Q11424 / P57) are copied into unrelated questions; those QIDs dominate hops (and 504 on high-degree nodes).
2. **URI-only answering context** even when NER hits the right people (S3, S34, S87, S124, S175).
3. Extra neighborhood triples are statement nodes and leaked entities, not a cleaner summary.

No Recall@k vs English `kg_triples` (live neighborhood ≠ 37-pool). First SPARQL process died after S124’s cache; answering resumed from `gss_cache/{id}_a0.75.json`.

### 7.2 Corpus (37 triples) results

**GSS@5 = 40.0%**, below LLM-only. Recall@5 **38.3%**, MRR **0.450**, Hits@k **45%** at every depth (rankings often 0–2 triples; k=5 and k=10 add nothing).

When GSS finds a gold triple, it is at rank 1. It finds **none** on 11/20 items. Nine `INSUFFICIENT` at k=5. True items correct: **S124 only (1/10)**.

Leaked QIDs (Q30, Q11424, GMT, …) are **not nodes** in `(Horsens, population, …)`, so fetch returns empty. On the items where NER hits real names (S1, S13, S87, S131, S175, S124, …), Recall@5 is often 100% — ranking math works; **linking is the bottleneck**.

**Finding:** putting GSS on the same 37 triples as RAG does **not** make it match RAG. RAG does not need NER. GSS does.

---

## 8. `isummary` — official (λ, κ)-selective summary

**Code:** `approaches/isummary/` (JAR wrapper only).  
**System:** `third_party/iSummary/isummary.jar` (Vassiliou, Papadakis, Kondylakis, SWJ 2025).  
**Run:** `out/colota_demo/isummary/`.

### Theory

iSummary does **not** walk a KG from the question. It builds a **(λ, κ)-selective summary from a SPARQL query log**: frequent query patterns around a seed entity become the summary. Popular entities in the log (Berlin Q64, humans Q5, films Q11424, …) get rich summaries. Long-tail CoLoTa QIDs that never appear in the log cannot be summarized from that workload.

The JAR is a **coverage evaluation driver**, not a “summarize this seed” API. Size gate: abort (`BROKER FOR`) if the weighted node set is smaller than `4 × top_k / 5`. `choose_from` must be ≥ 2 (`nextInt(1, choose_from)`).

### Algorithm (as run)

1. Map CoLoTa `kg_entities` to `http://www.wikidata.org/entity/Q…`.
2. Seed = first QID that appears in the authors’ Wikidata `train.txt` (~154k queries). If none: skip JAR, empty subgraph.
3. `java -jar isummary.jar <dummy_test.tsv> <train.txt> <2-line nodes, seed first> <top_k=12> <choose_from=2>`.
4. Parse `---------- PARTI` paths. Variable resolution is log-only (no SPARQL endpoint).
5. Empty → answering `INSUFFICIENT`. Python does **not** run Algorithm 1.

### Results

**iSummary@5 = 10.0%.** Recall/MRR/Hits = **0** at every k.

- **18/20** DEMO QIDs never appear in `train.txt`.
- **S51** (Serie A `Q15804`): 13 log queries, weighted node set size 2 < 9 → original `BROKER FOR`. Cache `[]`.
- **S160** (Tehran `Q3616`): 1 log hit, Jena parse success 0. Cache `[]`.
- Correct at k=5: **S95, S10 only** — both Gold-False guesses on empty context.

Berlin Q64 (not in `DEMO_IDS`) does produce PARTI paths; this slice is long-tail relative to the paper log.

**Finding:** this is a **workload mismatch**, not a JAR bug. Ranking vs English gold triples would be 0 even on a non-empty URI path. 10% is not useful context.

---

## 9. `ppr` — toy Personalized PageRank

**Code:** `approaches/ppr/`  
**Run:** 2026-10-06, same process as extract_embed. Full theory/algorithm: [`ppr_extract_embed/REPORT_ppr.md`](ppr_extract_embed/REPORT_ppr.md).

### Theory

Personalized PageRank (Haveliwala, 2003) is a biased random walk: with probability \(1-\alpha\) the walker **teleports to seed nodes**, so stationary mass is importance *from the query entities*. KGQA systems (GRAFT-Net, PullNet, HippoRAG) use this to grow a query-focused subgraph.

On 37 triples the walk cannot invent facts. It **re-ranks** existing edges. A gold triple that does not mention a seed by name can still rank high if it is 2-hop from a seed (pet → breed). Isolated components without seeds stay near 0.

This is **not HippoRAG** (no OpenIE, no passage index).

### Algorithm

1. Parse the 37 triples as undirected **s—o** edges (predicates are not nodes).
2. Seeds = graph nodes matching gold `kg_entities` labels (lowercase, `_` → space). **Not** llama NER.
3. No seed match → empty ranking → `INSUFFICIENT`.
4. NetworkX `pagerank` α=0.85, equal personalization on seeds, dangling=personalization, max_iter=100.
5. Triple score = \(\frac{1}{2}(\mathrm{PPR}(s)+\mathrm{PPR}(o))\); sort all 37; top-k → llama.

### Results

| k | Acc | Recall | Hits@k |
|---|-----|--------|--------|
| 1 | 70% | 64.6% | 100% |
| 3 | **85%** | 96.2% | 100% |
| 5 | 75% | 97.5% | 100% |
| 10 | 80% | 97.5% | 100% |

MRR = 1.000, Mean RR = 0.800. True items @k=5: **8/10** (miss S1, S34 — same oracle_reasoning pair). Wrong @k=5: S1, S2, S37, S4, S34.

S2 is the walk illustration: `(Sablé, animal breed, dachshund)` is 2-hop from Yui Yuigahama; PPR Recall@5 = 100%. extract_embed drops that triple (R@5 = 50%). Both models still answer True (wrong).

S5 stays at Recall 50% at every k: the second gold fact is never pulled into the top 10.

**Finding:** oracle entity linking + PPR on this pool is as strong a retriever as MiniLM (same Recall@5 / MRR) and the **best answering depth in the whole suite is PPR@3 = 85%**. Treat the 85% vs RAG@3 75% gap cautiously (different llama pass). Seeds are gold labels — an **upper bound** on PPR-if-linking-is-perfect, not a deployed linker.

---

## 10. `extract_embed` — toy entity filter + MiniLM

**Code:** `approaches/extract_embed/`  
**Run:** 2026-10-06, same process as PPR. Full theory/algorithm: [`ppr_extract_embed/REPORT_extract_embed.md`](ppr_extract_embed/REPORT_extract_embed.md).

### Theory

Many KG-RAG pipelines are two-stage: **extract/link topic entities**, then **embed-rank** only triples incident to those nodes. The hope is to drop other questions’ distractors. The cost is a hard ceiling: any gold fact whose subject and object are **not** listed entities is deleted (typical 2-hop evidence).

This is **not G-Retriever** (no PCST, no connectivity prize) and **not HippoRAG**.

### Algorithm

1. Encode the full 37-triple corpus with the same MiniLM as RAG (once).
2. Keep a triple iff its **subject or object** matches a gold `kg_entities` name (same matcher as PPR). Predicates are not matched.
3. Empty subset → empty ranking → `INSUFFICIENT`.
4. Cosine(question, kept rows only); sort that subset; top-k → llama.
5. Gold triples the filter dropped count as recall misses. Ranks are 1-indexed **inside the subset**.

### Results

| k | Acc | Recall | Hits@k |
|---|-----|--------|--------|
| 1 | 75% | 64.6% | 100% |
| 3 | **80%** | 86.7% | 100% |
| 5 | 75% | 86.7% | 100% |
| 10 | 75% | 86.7% | 100% |

MRR = 1.000, Mean RR = 0.753. Recall **saturates at 86.7% by k=3** — the missing ~13% is the gate, not ranking depth. k=10 cannot recover dropped 2-hop facts. Hits@1 is still 100%: every item has *some* gold triple that mentions a listed entity.

True items @k=5: **8/10** (miss S1, S34). Wrong @k=5: S1, S2, S37, S175, S34.

Filter vs PPR on the same items:

| ID | extract_embed R@5 | PPR R@5 | Why |
|----|-------------------|---------|-----|
| S2 | 50% | 100% | dachshund triple not incident to listed names |
| S3 | 50% | 100% | 2-hop occupation evidence |
| S37 | 50% | 100% | extra geo hops |
| S34 | 33% | 100% | 1 of 3 gold triples incident |
| S5 | 50% | 50% | both miss the same second fact |

**Finding:** answering **beats RAG@5 (75% vs 70%)** even though Recall@5 is **worse** (86.7% vs 97.5%). Fewer cross-question distractors can outweigh missing hops. The gate *is* the retrieval ceiling.

---

## 11. Per-item snapshot at k=5

`INSUFFICIENT` abbreviated `INS`. RAG/LLM/gold from the first experiment; GSS-W from live Wikidata; GSS-C from corpus; iSum / PPR / EE from their runs.

| ID | Gold | LLM | Gold triples | RAG | GSS-W | GSS-C | iSum | PPR | EE |
|----|------|-----|--------------|-----|-------|-------|------|-----|-----|
| S1 | T | F | F | F | F | F | INS | F | F |
| S2 | F | F | F | F | F | INS | INS | T | T |
| S3 | T | F | T | T | F | F | INS | T | T |
| S5 | T | F | T | T | T | INS | INS | T | T |
| S13 | F | T | F | F | F | F | INS | F | F |
| S37 | F | F | F | F | T | F | INS | T | T |
| S51 | T | F | T | F | T | INS | INS | T | T |
| S87 | F | F | F | F | F | F | INS | F | F |
| S95 | F | F | F | F | F | F | **F** | F | F |
| S131 | F | F | F | F | F | F | INS | F | F |
| S142 | T | F | T | F | F | INS | INS | T | T |
| S160 | T | F | T | F | INS | INS | INS | T | T |
| S164 | T | F | T | T | F | INS | INS | T | T |
| S175 | F | F | F | F | T | F | INS | F | T |
| S184 | F | F | F | F | F | INS | INS | F | F |
| S4 | F | F | F | INS | F | INS | INS | T | F |
| S10 | F | F | F | F | F | F | **F** | F | F |
| S34 | T | F | F | F | F | F | INS | F | F |
| S55 | T | T | T | T | F | INS | INS | T | T |
| S124 | T | F | T | T | F | **T** | INS | T | T |

**Always-hard True items:** S1 and S34 fail gold triples, RAG, PPR, extract_embed, and both GSS runs. Context does not fix those two llama inferences.

**Where gold entities help RAG:** S51, S142, S160 are wrong under MiniLM RAG@5 but **correct** under PPR@5 and extract_embed@5 (oracle seeds suppress other questions’ distractors / assemble the subgraph).

**Where the 2-hop gate hurts extract_embed answering:** S175 is correct under PPR (False) and wrong under extract_embed (True) even though extract_embed Recall@5 is 100% on that item — a reasoning flip, not the dachshund pattern.

**Empty-context False hits:** iSummary S95/S10 and several GSS-C rows match Gold-False by guessing False / `INS` counted wrong. That inflates “accuracy” without supporting the commonsense step.

---

## 12. Cross-cutting findings

1. **Parametric memory is chance (50%).** Long-tail CoLoTa names are not in llama3.1:8b in a usable way. Context is the point of the demo.

2. **Oracle English triples are almost enough (90%).** Two failures are model inference (numeric, literary genre), not missing KG facts.

3. **On a 37-triple pool, MiniLM already has MRR = 1.** The interesting RAG errors are distractors and multi-hop completeness, not “the retriever missed the entity.”

4. **Gold entity names are a stronger handle than the question embedding alone — and they were confounding the toys vs GSS.** Gold-seed PPR/extract_embed reach **75% @k=5**. The same rankers with **GSS NER** fall to **40% / 35%**, tying or undercutting corpus GSS (40%). The 75% was oracle linking, not a better summary algorithm. Unfiltered MiniLM RAG (70%) still wins among methods that do not get gold labels.

5. **Graph walk vs entity filter.** PPR can surface 2-hop gold (S2 dachshund); extract_embed cannot. extract_embed’s Recall ceiling is 86.7%. Answering can still rise vs unfiltered RAG because the list is cleaner.

6. **Bigger k is not a better summary.** RAG, PPR, extract_embed, and live GSS all show non-monotone or falling accuracy as k grows. Corpus GSS/iSummary are flat because the list is already empty.

7. **Original published systems on this slice are workload-limited.** Live GSS is URI noise + NER leakage. iSummary’s Wikidata log does not contain these QIDs and aborts on the two that appear. Trustworthiness of those two runs comes from **not reimplementing** the algorithms; the measured outcome is “this CoLoTa slice is not their intended workload.”

8. **True vs False.** LLM-only, GSS, iSummary, and **GSS-NER toys** mostly “succeed” on Gold-False (PPR-NER True items 2/10). Gold-seed PPR/extract_embed/gold triples are the only conditions that systematically get Gold-True right (8–9/10).

9. **Future cited systems (not run).** HippoRAG (PPR over an OpenIE index, pip-installable) and G-Retriever (embeddings + PCST) remain notes for later. ReFinED is the setup-able linker if NER should stop being gold labels or llama few-shot.

---

## 13. Limitations (all experiments)

- **n=20**, one local 8B model, binary QA only, hand-picked entity-aligned IDs.
- Closed corpus is **tiny** (37 triples from the same 20 items). High MiniLM/PPR MRR is partly an artifact; live Wikidata or full CoLoTa would stress retrieval.
- PPR and extract_embed have **two seed conditions**: gold `kg_entities` (archived) and GSS NER resource names (2026-10-07). The NER pass is the same cache for both toys; it is **not** bit-identical to the GSS-corpus NER pass (S5 differs).
- Answering passes are **not all contemporaneous**; gold-seed ppr vs extract_embed share a process; NER-seed ppr vs extract_embed share a later process and one NER cache.
- GSS Wikidata context is URI triples; paper figures show labels.
- iSummary dummy testdata (coverage unused); seed via 2-line ranking file; English gold vs SPARQL paths.
- No summary-quality metrics (precision, F1, redundancy) in this suite (`out/colota_summary/` is a separate follow-up).
- Inference-rule condition was dropped on purpose.

---

## 14. How to reproduce

Ollama must serve `llama3.1:8b`. From the repo root:

```text
python src/colota_demo/run_suite.py --approaches llm_only,gold_triples,rag_transformers --experiment first_experiment
python src/colota_demo/run_suite.py --approaches gss --experiment gss
python src/colota_demo/run_suite.py --approaches gss --experiment gss_corpus
python src/colota_demo/run_suite.py --approaches isummary --experiment isummary
python src/colota_demo/run_suite.py --approaches ppr,extract_embed --experiment ppr_extract_embed
python src/colota_demo/run_suite.py --approaches ppr --experiment ppr_ner --seeds gss_ner
python src/colota_demo/run_suite.py --approaches extract_embed --experiment extract_embed_ner --seeds gss_ner
```

`config.json`: `gss.a = 0.75`, `gss.kg` is `wikidata` or `corpus`, `ppr.alpha = 0.85`, `entity_seeds` is `gold` or `gss_ner`, `isummary.kappa = 12` with the Wikidata `train.txt` dump under `third_party/iSummary/data/wikidata/`. IDs: `DEMO_IDS`. Caches: `gss_cache/`, `isummary_cache/`, `gss_ner_cache/`.

This compilation did **not** re-run llm/gold/RAG/GSS/iSummary or gold-seed PPR/extract_embed; it reused those archives and added the 2026-10-07 NER-seed runs.

---

## 15. GSS NER seeds for PPR and extract_embed (2026-10-07)

**Question:** does the 75% of the toy rankers survive if seeds come from **exact GSS NER** (`extract_entities`, resource names only) instead of gold `kg_entities`?

**Setup.** Same `DEMO_IDS`, 37-triple pool, α=0.85, MiniLM, answering prompt. Ranking math **not** changed. One NER cache shared by both folders. Empty name-match → `INSUFFICIENT`. Properties skipped. Q-ids kept if NER emitted them as resources (they almost never match English nodes).

**Results @k=5**

| Condition | Acc | Recall | MRR | Hits@5 |
|-----------|-----|--------|-----|--------|
| PPR gold entities | 75% | 97.5% | 1.000 | 100% |
| extract_embed gold entities | 75% | 86.7% | 1.000 | 100% |
| MiniLM RAG | 70% | 97.5% | 1.000 | 100% |
| GSS corpus | 40% | 38.3% | 0.450 | 45% |
| **PPR GSS NER** | **40%** | 40.0% | 0.450 | 45% |
| **extract_embed GSS NER** | **35%** | 35.8% | 0.450 | 45% |

Nine items are `INSUFFICIENT` on both NER toys (S2, S3, S51, S142, S160, S164, S184, S4, S55). S2’s cache is `Q2513` + `Q11424` (Nolan/film few-shot), not Yui Yuigahama. True-item hits: **2/10** (S5, S124) for both.

**Read:** with a shared noisy linker, PPR ≈ GSS corpus and extract_embed is slightly worse (hard 1-hop gate). Unfiltered RAG remains the strongest method that does not receive gold labels. Full reports: [`ppr_ner/REPORT.md`](ppr_ner/REPORT.md), [`extract_embed_ner/REPORT.md`](extract_embed_ner/REPORT.md).
