"""
Explanation Visualization Module
=================================

Generate visual explanations for GNN predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch_geometric.data import HeteroData

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


def plot_explanation_subgraph(
    data: HeteroData,
    employee_id: int,
    embeddings: Dict[str, torch.Tensor],
    turnover_prob: float,
    save_path: Path,
    top_k_neighbors: int = 5,
) -> None:
    """
    Generate a visual subgraph showing employee's important connections.

    Args:
        data: Full graph data
        employee_id: Target employee
        embeddings: Node embeddings
        turnover_prob: Predicted turnover probability
        save_path: Where to save the plot
        top_k_neighbors: Number of neighbors to show
    """
    if not HAS_NETWORKX:
        print("Warning: networkx not available, skipping subgraph visualization")
        return

    # Create directed graph
    G = nx.DiGraph()

    # Add employee node
    G.add_node(
        f"EMP_{employee_id}",
        node_type="employee",
        risk="High" if turnover_prob > 0.5 else "Low",
    )

    # Add current job
    if ("employee", "assigned_to", "current_job") in data.edge_index_dict:
        edge_index = data.edge_index_dict[("employee", "assigned_to", "current_job")]
        src, dst = edge_index

        emp_mask = (src == employee_id)
        if emp_mask.sum() > 0:
            job_id = dst[emp_mask][0].item()

            # Compute job importance (embedding similarity)
            emp_vec = embeddings["employee"][employee_id]
            job_vec = embeddings["current_job"][job_id]
            similarity = torch.cosine_similarity(
                emp_vec.unsqueeze(0),
                job_vec.unsqueeze(0)
            ).item()

            G.add_node(f"JOB_{job_id}", node_type="current_job")
            G.add_edge(
                f"EMP_{employee_id}",
                f"JOB_{job_id}",
                relation="assigned_to",
                weight=similarity,
            )

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 8))

    # Layout
    pos = nx.spring_layout(G, seed=42, k=2)

    # Node colors
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node.startswith("EMP"):
            risk = G.nodes[node].get("risk", "Low")
            node_colors.append("red" if risk == "High" else "lightgreen")
            node_sizes.append(2000)
        elif node.startswith("JOB"):
            node_colors.append("lightblue")
            node_sizes.append(1500)
        else:
            node_colors.append("lightgray")
            node_sizes.append(1000)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        ax=ax,
    )

    # Draw edges with weights
    edges = G.edges()
    weights = [G[u][v].get('weight', 0.5) for u, v in edges]

    nx.draw_networkx_edges(
        G, pos,
        width=[w * 3 for w in weights],
        alpha=0.6,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        ax=ax,
    )

    # Labels
    labels = {}
    for node in G.nodes():
        if node.startswith("EMP"):
            labels[node] = f"Employee\n{employee_id}"
        elif node.startswith("JOB"):
            job_id = node.split("_")[1]
            labels[node] = f"Job\n{job_id}"

    nx.draw_networkx_labels(
        G, pos,
        labels,
        font_size=10,
        font_weight='bold',
        ax=ax,
    )

    # Legend
    legend_elements = [
        mpatches.Patch(color='red', alpha=0.9, label='High Risk Employee'),
        mpatches.Patch(color='lightgreen', alpha=0.9, label='Low Risk Employee'),
        mpatches.Patch(color='lightblue', alpha=0.9, label='Current Job'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    ax.set_title(
        f'Employee {employee_id} - Important Connections\n'
        f'Turnover Risk: {turnover_prob:.2%} ({"High" if turnover_prob > 0.5 else "Low"})',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   Saved subgraph: {save_path}")


def plot_preference_comparison(
    employee_id: int,
    preferred_post: int,
    dispreferred_post: int,
    pref_score: float,
    disp_score: float,
    margin: float,
    save_path: Path,
) -> None:
    """
    Visualize preference comparison between two posts.

    Args:
        employee_id: Employee ID
        preferred_post: Preferred post ID
        dispreferred_post: Dispreferred post ID
        pref_score: Score for preferred post
        disp_score: Score for dispreferred post
        margin: Score margin
        save_path: Save location
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot
    posts = [f'Post {preferred_post}\n(Preferred)', f'Post {dispreferred_post}\n(Dispreferred)']
    scores = [pref_score, disp_score]
    colors = ['green', 'red']

    bars = ax.bar(posts, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add score labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{score:.3f}',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold'
        )

    # Add margin annotation
    ax.annotate(
        f'Margin: {margin:.3f}',
        xy=(0.5, max(scores)),
        xytext=(0.5, max(scores) + 0.3),
        ha='center',
        fontsize=13,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
        arrowprops=dict(arrowstyle='->', lw=2)
    )

    ax.set_ylabel('Preference Score', fontsize=12)
    ax.set_title(
        f'Employee {employee_id} - Job Preference Comparison',
        fontsize=14, fontweight='bold'
    )
    ax.set_ylim([min(scores) - 0.5, max(scores) + 0.8])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   Saved preference plot: {save_path}")
