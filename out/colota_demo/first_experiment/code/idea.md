# Minimal CoLoTa Demo

Use **10–20 CoLoTa questions** and compare just three conditions:

```text
                    Same 10–20 questions
                            │
             ┌──────────────┼──────────────┐
             ↓              ↓              ↓
         LLM-only       Gold-context       RAG
             │              │              │
             ↓              ↓              ↓
          answer         answer          answer
             │              │              │
             └──────────────┼──────────────┘
                            ↓
                     Compare results
```

## Condition 1 — LLM only

Just send:

```text
Question: Could you travel from X to Y only by car?

Answer only TRUE or FALSE.
```

Save the prediction.

---

## Condition 2 — Gold context

Take the **KG triples already provided by CoLoTa** and put them in the prompt:

```text
Question:
Could you travel from X to Y only by car?

Relevant knowledge:
- X → country → A
- A → continent → Europe
- Y → country → B
- B → continent → Asia

Use only the provided knowledge.
Answer TRUE or FALSE.
```

This is extremely easy because you don't even need a retriever yet.

---

## Condition 3 — Actual RAG

Put the triples into a tiny vector store or even a simple retrieval function.

For the first prototype, you don't need Neo4j, a graph database, or anything complicated.

You could literally have:

```python
documents = [
    "X is located in country A.",
    "A is located in Europe.",
    "Y is located in country B.",
    "B is located in Asia.",
    ...
]
```

Then:

```text
question
   ↓
similarity search
   ↓
top 3–5 documents
   ↓
LLM
   ↓
TRUE/FALSE
```

That is enough to demonstrate the concept.

---

## Your evaluation can be tiny

Create a CSV like:

```text
question_id,gold,llm,gold_context,rag
1,True,False,True,True
2,False,False,False,True
3,True,False,True,False
...
```

Then calculate:

```text
LLM accuracy
Gold-context accuracy
RAG accuracy
```

And for RAG:

```text
Recall@k
```

You can manually inspect the retrieved facts for the first 10–20 questions.

---

## The really useful part

You can classify every failure:

```text
LLM wrong
Gold-context correct
→ likely knowledge/context problem

RAG retrieves wrong/missing facts
→ retrieval problem

RAG retrieves correct facts
but answer is wrong
→ reasoning problem
```
