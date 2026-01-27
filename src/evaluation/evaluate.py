import json
from collections import defaultdict
from typing import List, Tuple, Set, Dict

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

Triple = Tuple[str, str, str]

# =========================
# CONFIGURATION
# =========================
TOP_K_INITIAL = 15
RELATIVE_THRESHOLD = 0.7


# =========================
# URI & Text Utilities
# =========================
def is_dbpedia_resource(uri: str) -> bool:
    return isinstance(uri, str) and uri.startswith("http://dbpedia.org/resource/")


def triple_to_text(triple: Triple) -> str:
    return " ".join(x.split("/")[-1].replace("_", " ") for x in triple)


def extract_candidate_answers(triples: List[Triple]) -> Set[str]:
    entities = set()
    for s, _, o in triples:
        if is_dbpedia_resource(s): entities.add(s)
        if is_dbpedia_resource(o): entities.add(o)
    return entities


# =========================
# ADVANCED METRICS (Path & Density)
# =========================
def check_path_consistency(triples: List[Triple], gold_answers: Set[str]) -> float:
    """
    Checks if the summary forms a connected component that leads to the answer.
    """
    if not gold_answers or not triples:
        return 0.0

    # Build adjacency graph
    graph = defaultdict(list)
    nodes = set()
    for s, p, o in triples:
        graph[s].append(o)
        graph[o].append(s)
        nodes.add(s)
        nodes.add(o)

    # Start nodes: Entities in the summary that are NOT the gold answer
    start_nodes = nodes - gold_answers
    if not start_nodes: return 0.0

    # BFS to find if any gold answer is reachable from non-answer nodes in the summary
    found_path = 0
    for start in start_nodes:
        visited = {start}
        queue = [start]
        while queue:
            curr = queue.pop(0)
            if curr in gold_answers:
                found_path = 1
                break
            for neighbor in graph[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if found_path: break

    return float(found_path)


def calculate_graph_density(triples: List[Triple]) -> float:
    entities = extract_candidate_answers(triples)
    if len(entities) < 2: return 0.0
    # Ratio of actual edges to possible edges
    possible = len(entities) * (len(entities) - 1) / 2
    return len(triples) / possible


# =========================
# RANKING & INTERPRETABILITY
# =========================
def calculate_ranking_metrics(triples: List[Triple], gold: Set[str]) -> Dict[str, float]:
    mrr, h1, h3 = 0.0, 0.0, 0.0
    for i, (s, p, o) in enumerate(triples):
        rank = i + 1
        if s in gold or o in gold:
            if mrr == 0:
                mrr = 1.0 / rank
                if rank == 1: h1 = 1.0
                if rank <= 3: h3 = 1.0
    return {"mrr": mrr, "hits@1": h1, "hits@3": h3}


def get_bridge_entities(triples: List[Triple], gold: Set[str]) -> float:
    all_entities = extract_candidate_answers(triples)
    non_gold = all_entities - gold
    bridges = set()
    for s, p, o in triples:
        if (s in gold and o in non_gold): bridges.add(o)
        if (o in gold and s in non_gold): bridges.add(s)
    return float(len(bridges))


# =========================
# CORE EVALUATOR
# =========================
class RerankingEvaluator:
    def __init__(self):
        self.bi_encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device="cpu")

    def refine_triples(self, question: str, sys_answers: List[Dict]) -> List[Triple]:
        valid_entries = [a for a in sys_answers if len(a.get("triple", "").split()) == 3]
        if not valid_entries: return []

        candidate_triples = [tuple(e["triple"].split()) for e in valid_entries[:TOP_K_INITIAL]]
        texts = [triple_to_text(t) for t in candidate_triples]
        rerank_scores = self.cross_encoder.predict([[question, txt] for txt in texts])

        # Pair scores with triples and sort
        scored_triples = sorted(zip(rerank_scores, candidate_triples), key=lambda x: x[0], reverse=True)

        max_score = scored_triples[0][0] if scored_triples else 0
        return [t for s, t in scored_triples if s >= (max_score * RELATIVE_THRESHOLD)]


# =========================
# DATA LOADING
# =========================
def load_gt(path: str):
    with open(path) as f:
        data = json.load(f)
    return {qid: {"question": item["question"],
                  "answers": set(a for a in item["answers"] if is_dbpedia_resource(a))}
            for qid, item in data.items()}


def load_sys(path: str):
    with open(path) as f:
        return json.load(f)


# =========================
# MAIN
# =========================
def main():
    # Update these paths to your actual files
    try:
        gt = load_gt("../data/gt_dbpedia_answers.json")
        sys_data = load_sys("../data/gss_dbpedia_answers.json")
    except FileNotFoundError:
        print("Error: Ensure your data files exist at the specified paths.")
        return

    evaluator = RerankingEvaluator()
    results = []

    print(f"Starting evaluation on {len(gt)} questions...")

    for qid, gt_item in gt.items():
        gold = gt_item["answers"]
        if not gold: continue

        sys_item = sys_data.get(qid)
        if not sys_item or "answers" not in sys_item: continue

        # Refine/Summarize
        triples = evaluator.refine_triples(gt_item["question"], sys_item["answers"])
        if not triples: continue

        # Calculate everything
        predicted_entities = extract_candidate_answers(triples)

        # 1. PRF
        tp = len(predicted_entities & gold)
        p = tp / len(predicted_entities) if predicted_entities else 0
        r = tp / len(gold) if gold else 0
        f1 = (2 * p * r) / (p + r) if (p + r) else 0

        # 2. Ranking & Graph Logic
        ranking = calculate_ranking_metrics(triples, gold)
        path_score = check_path_consistency(triples, gold)
        density = calculate_graph_density(triples)
        bridges = get_bridge_entities(triples, gold)

        results.append({
            "precision": p, "recall": r, "f1": f1,
            **ranking,
            "path_consistency": path_score,
            "graph_density": density,
            "bridge_entities": bridges,
            "num_triples": len(triples)
        })

    # Aggregation
    if not results:
        print("No results to aggregate.")
        return

    print("\n=== FINAL ADVANCED SUMMARY ===")
    metrics_to_show = results[0].keys()
    for m in metrics_to_show:
        vals = [r[m] for r in results]
        print(f"{m:20}: {np.mean(vals):.6f} (std: {np.std(vals):.4f})")


if __name__ == "__main__":
    main()
