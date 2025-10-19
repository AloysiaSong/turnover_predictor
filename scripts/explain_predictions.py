"""
Prediction Explanation Script
==============================

Generate explanations for GNN predictions on specific employees.

Usage:
    python scripts/explain_predictions.py \
        --run-dir outputs/dual_head/dual_head_main \
        --explain-ids 0 5 10 15 20

Output:
    - explanations/employee_XXXX_features.json
    - explanations/employee_XXXX_neighbors.json
    - explanations/employee_XXXX_importance.png
    - explanations/employee_XXXX_subgraph.png
    - explanations/preference_explanations.json
    - explanations/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch_geometric.data import HeteroData

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.dual_head_gnn import DualHeadGNN, DualHeadConfig
from src.models.hetero_gnn import HeteroGNN, HeteroGNNConfig
from src.models.multitask_heads import (
    TurnoverHead,
    TurnoverHeadConfig,
    PreferencePairwiseHead,
    PreferenceHeadConfig,
)
from src.models.layers.feature_normalizer import FeatureNormalizer
from src.models.explanations import (
    generate_explanation_report,
    visualize_feature_importance,
)
from src.models.visualization import (
    plot_explanation_subgraph,
    plot_preference_comparison,
)

try:
    import yaml
except ImportError:
    raise ImportError("Please install PyYAML")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain GNN predictions")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing trained model (e.g., outputs/dual_head/dual_head_main)"
    )
    parser.add_argument(
        "--explain-ids",
        nargs="+",
        type=int,
        required=True,
        help="Employee IDs to explain (e.g., 0 5 10 15)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: {run-dir}/explanations)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top features/neighbors to show"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots"
    )
    return parser.parse_args()


def load_model_and_data(run_dir: Path, device: torch.device):
    """Load trained model, config, and data."""

    # Load config
    config_path = run_dir / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    data_path = Path(config["data"]["graph_path"])
    data: HeteroData = torch.load(data_path)
    data = data.to(device)

    # Normalize features
    scaler_path = Path(config["data"]["scaler_path"])
    normalizer = FeatureNormalizer(scaler_path)
    data["employee"].x = normalizer(data["employee"].x.float())

    # Load triples
    triples_path = Path(config["data"]["triples_path"])
    triples = torch.load(triples_path).long().to(device)

    # Build model based on config type
    metadata = data.metadata()
    input_dims = {ntype: data[ntype].x.size(-1) for ntype in data.node_types}

    is_dual_head = "dual_head" in config["model"]

    if is_dual_head:
        # Dual-Head model
        gnn_config = HeteroGNNConfig(**config["model"]["gnn"])
        dual_head_config = DualHeadConfig(**config["model"]["dual_head"])
        model = DualHeadGNN(metadata, input_dims, gnn_config, dual_head_config).to(device)

        turnover_cfg = TurnoverHeadConfig(hidden_dim=config["model"]["gnn"]["hidden_dim"])
        turnover_head = TurnoverHead(
            input_dim=dual_head_config.turnover_proj_dim * 2,
            config=turnover_cfg,
        ).to(device)

        pref_head_cfg = PreferenceHeadConfig(mode="dot")
        preference_head = PreferencePairwiseHead(
            embedding_dim=dual_head_config.preference_proj_dim,
            config=pref_head_cfg,
        ).to(device)

    else:
        # Single-head model (HeteroGNN)
        gnn_config = HeteroGNNConfig(**config["model"])
        model = HeteroGNN(metadata, input_dims, gnn_config).to(device)

        turnover_cfg = TurnoverHeadConfig(hidden_dim=config["model"]["hidden_dim"])
        turnover_head = TurnoverHead(
            input_dim=config["model"]["hidden_dim"] * 2,
            config=turnover_cfg,
        ).to(device)

        pref_cfg_dict = config["loss"].get("preference_head", {})
        preference_head = PreferencePairwiseHead(
            embedding_dim=config["model"]["hidden_dim"],
            config=PreferenceHeadConfig(**pref_cfg_dict),
        ).to(device)

    # Load trained weights
    checkpoint_path = run_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model"])
    turnover_head.load_state_dict(checkpoint["turnover_head"])
    preference_head.load_state_dict(checkpoint["preference_head"])

    model.eval()
    turnover_head.eval()
    preference_head.eval()

    return model, turnover_head, preference_head, data, triples, config


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup output directory
    output_dir = args.output_dir or (args.run_dir / "explanations")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"EXPLAINING PREDICTIONS")
    print(f"{'='*70}")
    print(f"Run directory: {args.run_dir}")
    print(f"Employee IDs: {args.explain_ids}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")

    # Load model and data
    print("Loading model and data...")
    model, turnover_head, preference_head, data, triples, config = load_model_and_data(
        args.run_dir, device
    )
    print("✓ Model loaded successfully\n")

    # Generate explanations
    scaler_path = Path(config["data"]["scaler_path"])
    feature_names_path = Path(config["data"].get(
        "feature_names_path",
        "data/processed/feature_names.txt"
    ))

    generate_explanation_report(
        model=model,
        turnover_head=turnover_head,
        preference_head=preference_head,
        data=data,
        employee_ids=args.explain_ids,
        triples=triples,
        scaler_path=scaler_path,
        feature_names_path=feature_names_path,
        save_dir=output_dir,
    )

    # Generate visualizations if requested
    if args.visualize:
        print(f"\n{'='*70}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*70}\n")

        # Get embeddings
        with torch.no_grad():
            if hasattr(model, 'get_embeddings_for_tasks'):
                turnover_emb, preference_emb = model.get_embeddings_for_tasks(data)
            else:
                turnover_emb = model(data)
                preference_emb = turnover_emb

        current_job_idx = data["employee", "assigned_to", "current_job"].edge_index[1]

        # Load feature explanations for visualization
        for emp_id in args.explain_ids:
            # Feature importance plot
            explanation_file = output_dir / f"employee_{emp_id:04d}_features.json"
            if explanation_file.exists():
                with open(explanation_file, 'r') as f:
                    exp = json.load(f)

                visualize_feature_importance([exp], output_dir)

                # Subgraph plot
                turnover_prob = exp["turnover_probability"]
                plot_explanation_subgraph(
                    data=data,
                    employee_id=emp_id,
                    embeddings=turnover_emb,
                    turnover_prob=turnover_prob,
                    save_path=output_dir / f"employee_{emp_id:04d}_subgraph.png",
                )

        # Preference visualizations
        pref_explanation_file = output_dir / "preference_explanations.json"
        if pref_explanation_file.exists():
            with open(pref_explanation_file, 'r') as f:
                pref_exps = json.load(f)

            for i, pref_exp in enumerate(pref_exps[:5]):  # Top 5
                plot_preference_comparison(
                    employee_id=pref_exp["employee_id"],
                    preferred_post=pref_exp["preferred_post"],
                    dispreferred_post=pref_exp["dispreferred_post"],
                    pref_score=pref_exp["preference_score"],
                    disp_score=pref_exp["dispreference_score"],
                    margin=pref_exp["margin"],
                    save_path=output_dir / f"preference_comparison_{i}.png",
                )

    # Print summary
    print(f"\n{'='*70}")
    print("EXPLANATION SUMMARY")
    print(f"{'='*70}\n")

    summary_file = output_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)

        print(f"Employees analyzed: {summary['num_employees_analyzed']}")
        print(f"High risk: {len(summary['high_risk_employees'])}")
        print(f"Low risk: {len(summary['low_risk_employees'])}")

        if summary['high_risk_employees']:
            print(f"\nHigh risk employees: {summary['high_risk_employees']}")
        if summary['low_risk_employees']:
            print(f"Low risk employees: {summary['low_risk_employees']}")

    print(f"\n{'='*70}")
    print(f"All explanations saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
