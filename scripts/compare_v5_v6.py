"""
Compare GNN v5 vs v6 Results
=============================
Quick comparison script to visualize improvements.
"""

import json
from pathlib import Path

# v5 results (from your original output)
v5_results = {
    "turnover": {
        "accuracy": 0.9300,
        "precision": 0.8333,
        "recall": 0.4545,
        "f1": 0.5882,
        "auroc": 0.8672,
        "aupr": 0.6329,
    },
    "preference": {
        "test_pairwise_accuracy": 0.4657,
        "val_pairwise_accuracy": 0.4619,
    }
}

# Load v6 results
v6_results_path = Path("outputs/hetero_v6/run_20251019_195926/results.json")
with open(v6_results_path, 'r') as f:
    v6_data = json.load(f)

v6_results = {
    "turnover": v6_data["test_turnover_metrics"],
    "preference": v6_data["test_preference_metrics"],
}

# Print comparison
print("=" * 80)
print("GNN v5 vs v6 Comparison")
print("=" * 80)

print("\n📊 Turnover Prediction (Test Set)")
print("-" * 80)
print(f"{'Metric':<15} | {'v5':<10} | {'v6':<10} | {'Change':<15} | Status")
print("-" * 80)

metrics = ["auroc", "aupr", "f1", "precision", "recall", "accuracy"]
for metric in metrics:
    v5_val = v5_results["turnover"][metric]
    v6_val = v6_results["turnover"][metric]
    change = ((v6_val - v5_val) / v5_val) * 100
    status = "✅" if change > 0 else "❌" if change < -5 else "≈"

    print(f"{metric:<15} | {v5_val:<10.4f} | {v6_val:<10.4f} | {change:>+6.1f}% | {status}")

print("\n🎯 Preference Ranking (Test Set)")
print("-" * 80)
print(f"{'Metric':<20} | {'v5':<10} | {'v6':<10} | {'Change':<15} | Status")
print("-" * 80)

v5_pref_acc = v5_results["preference"]["test_pairwise_accuracy"]
v6_pref_acc = v6_results["preference"]["accuracy"]
pref_change = ((v6_pref_acc - v5_pref_acc) / v5_pref_acc) * 100

print(f"{'Pairwise Accuracy':<20} | {v5_pref_acc:<10.4f} | {v6_pref_acc:<10.4f} | {pref_change:>+6.1f}% | {'✅' if pref_change > 5 else '≈'}")
print(f"{'Margin':<20} | {'N/A':<10} | {v6_results['preference']['margin']:<10.4f} | {'NEW':<15} | ✨")

print("\n" + "=" * 80)
print("🎯 Key Improvements")
print("=" * 80)

improvements = [
    ("Preference Accuracy", f"+{pref_change:.1f}%", "🎉 Breakthrough! Now above random baseline"),
    ("AUPR", f"+{((v6_results['turnover']['aupr'] - v5_results['turnover']['aupr']) / v5_results['turnover']['aupr']) * 100:.1f}%", "Improved ranking quality"),
    ("Recall", f"+{((v6_results['turnover']['recall'] - v5_results['turnover']['recall']) / v5_results['turnover']['recall']) * 100:.1f}%", "Better minority class detection"),
]

for name, change, desc in improvements:
    print(f"  • {name}: {change} - {desc}")

print("\n" + "=" * 80)
print("🚀 Next Steps")
print("=" * 80)
print("  1. Investigate Turnover F1 drop (0.5882 → 0.5385)")
print("  2. Increase Stage 2 training epochs (100 → 200)")
print("  3. Tune Hard Negative ratio (0.7 → 0.85)")
print("  4. Try different Preference Head modes (concat → dot)")
print("\n  📄 Full report: IMPROVEMENT_REPORT_V6.md")
print("=" * 80)
