"""Knowledge graph visualization utilities."""

import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils.dbpedia import extract_entity_name_from_uri
from config.settings import (
    VIS_MAX_NODES,
    VIS_FIGSIZE,
    VIS_DPI,
    VIS_NODE_SIZE_MIN,
    VIS_NODE_SIZE_MULTIPLIER,
    VIS_HIGH_IMPORTANCE_THRESHOLD,
    VIS_MEDIUM_IMPORTANCE_THRESHOLD,
    VIS_COLOR_HIGH_IMPORTANCE,
    VIS_COLOR_MEDIUM_IMPORTANCE,
    VIS_COLOR_LOW_IMPORTANCE,
    VIS_LABEL_TRUNCATE_LENGTH,
    VIS_MAX_EDGE_LABELS,
    VIS_MAX_EDGE_LABEL_LENGTH,
    VIS_FONT_SIZE,
    VIS_EDGE_FONT_SIZE,
    VIS_LEGEND_FONT_SIZE,
    VIS_TITLE_FONT_SIZE,
    VIS_SPRING_LAYOUT_K,
    VIS_SPRING_LAYOUT_ITERATIONS,
    VIS_SPRING_LAYOUT_SEED,
    VIS_NODE_ALPHA,
    VIS_EDGE_ALPHA,
    VIS_EDGE_LABEL_ALPHA,
    VIS_ARROW_SIZE,
    VIS_EDGE_WIDTH,
    VIS_EDGE_COLOR,
    VIS_NODE_FILTER_IMPORTANCE_MULTIPLIER,
    VIS_NODE_FILTER_DEGREE_WEIGHT
)


def visualize_knowledge_graph(
        triples: list,
        output_path: str,
        uri_importance_map: dict = None,
        max_nodes: int = None,
        figsize: tuple = None
) -> None:
    """
    Visualizes a knowledge graph from triples and saves it as an image.
    
    Args:
        triples: List of triples, each with 's', 'p', 'o' keys
        output_path: Path to save the visualization image
        uri_importance_map: Optional dict mapping URIs to importance scores for node sizing
        max_nodes: Maximum number of nodes to display (default: from settings)
        figsize: Figure size (width, height) in inches (default: from settings)
    """
    if max_nodes is None:
        max_nodes = VIS_MAX_NODES
    if figsize is None:
        figsize = VIS_FIGSIZE

    if not triples:
        print("  No triples to visualize")
        return

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes and edges from triples
    node_importance = {}
    for triple in triples:
        s = triple.get('s', '')
        p = triple.get('p', '')
        o = triple.get('o', '')

        if not s or not p or not o:
            continue

        s_name = extract_entity_name_from_uri(s) or s
        o_name = extract_entity_name_from_uri(o) or o
        p_name = extract_entity_name_from_uri(p) or p

        if s not in G:
            G.add_node(s, label=s_name, uri=s)
            if uri_importance_map and s in uri_importance_map:
                node_importance[s] = uri_importance_map[s].get('score', 0.0)
            else:
                node_importance[s] = 0.0

        if o not in G:
            G.add_node(o, label=o_name, uri=o)
            if uri_importance_map and o in uri_importance_map:
                node_importance[o] = uri_importance_map[o].get('score', 0.0)
            else:
                node_importance[o] = 0.0

        if not G.has_edge(s, o):
            G.add_edge(s, o, label=p_name, predicate=p)

    if len(G.nodes()) > max_nodes:
        node_scores = {}
        for node in G.nodes():
            importance = node_importance.get(node, 0.0)
            degree = G.degree(node)
            node_scores[
                node] = importance * VIS_NODE_FILTER_IMPORTANCE_MULTIPLIER + degree * VIS_NODE_FILTER_DEGREE_WEIGHT

        top_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_set = {node for node, _ in top_nodes}
        G = G.subgraph(top_node_set).copy()
        print(f"  Limited graph to {len(G.nodes())} nodes (from {len(node_scores)} total)")

    if len(G.nodes()) == 0:
        print("  No nodes to visualize after filtering")
        return

    plt.figure(figsize=figsize)

    try:
        pos = nx.spring_layout(G, k=VIS_SPRING_LAYOUT_K, iterations=VIS_SPRING_LAYOUT_ITERATIONS,
                               seed=VIS_SPRING_LAYOUT_SEED)
    except:
        pos = nx.circular_layout(G)

    node_sizes = []
    for node in G.nodes():
        importance = node_importance.get(node, 0.0)
        size = VIS_NODE_SIZE_MIN + importance * VIS_NODE_SIZE_MULTIPLIER
        node_sizes.append(size)

    node_colors = []
    for node in G.nodes():
        importance = node_importance.get(node, 0.0)
        if importance > VIS_HIGH_IMPORTANCE_THRESHOLD:
            node_colors.append(VIS_COLOR_HIGH_IMPORTANCE)
        elif importance > VIS_MEDIUM_IMPORTANCE_THRESHOLD:
            node_colors.append(VIS_COLOR_MEDIUM_IMPORTANCE)
        else:
            node_colors.append(VIS_COLOR_LOW_IMPORTANCE)

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=VIS_NODE_ALPHA,
        node_shape='o'
    )

    nx.draw_networkx_edges(
        G, pos,
        edge_color=VIS_EDGE_COLOR,
        alpha=VIS_EDGE_ALPHA,
        arrows=True,
        arrowsize=VIS_ARROW_SIZE,
        arrowstyle='->',
        width=VIS_EDGE_WIDTH
    )

    labels = {}
    for node in G.nodes():
        label = G.nodes[node].get('label', '')
        if len(label) > VIS_LABEL_TRUNCATE_LENGTH:
            label = label[:VIS_LABEL_TRUNCATE_LENGTH - 3] + '...'
        labels[node] = label

    nx.draw_networkx_labels(
        G, pos,
        labels,
        font_size=VIS_FONT_SIZE,
        font_weight='bold',
        font_family='sans-serif'
    )

    edge_labels = {}
    edge_count = 0
    for edge in G.edges():
        if edge_count < VIS_MAX_EDGE_LABELS:
            predicate = G.edges[edge].get('label', '')
            if len(predicate) <= VIS_MAX_EDGE_LABEL_LENGTH:
                edge_labels[edge] = predicate
                edge_count += 1

    if edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels,
            font_size=VIS_EDGE_FONT_SIZE,
            font_color='darkblue',
            alpha=VIS_EDGE_LABEL_ALPHA
        )

    legend_elements = [
        mpatches.Patch(color=VIS_COLOR_HIGH_IMPORTANCE,
                       label=f'High Importance (score > {VIS_HIGH_IMPORTANCE_THRESHOLD})'),
        mpatches.Patch(color=VIS_COLOR_MEDIUM_IMPORTANCE,
                       label=f'Medium Importance ({VIS_MEDIUM_IMPORTANCE_THRESHOLD} < score ≤ {VIS_HIGH_IMPORTANCE_THRESHOLD})'),
        mpatches.Patch(color=VIS_COLOR_LOW_IMPORTANCE,
                       label=f'Low Importance (score ≤ {VIS_MEDIUM_IMPORTANCE_THRESHOLD})')
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=VIS_LEGEND_FONT_SIZE)

    plt.title(f'Knowledge Graph Subgraph\n{len(G.nodes())} nodes, {len(G.edges())} edges',
              fontsize=VIS_TITLE_FONT_SIZE, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(output_path, dpi=VIS_DPI, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Graph visualization saved to: {output_path}")
    print(f"    Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")
