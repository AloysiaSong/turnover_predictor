"""
Comprehensive Comparison: v5 vs v6 vs v6_optimized vs v6_balanced vs v6_best
=============================================================================
"""

import json
from pathlib import Path
from typing import Dict, Any

def load_results(path: Path) -> Dict[str, Any]:
    """Load results.json from a run directory."""
    with open(path, 'r') as f:
        return json.load(f)

# Define all versions
versions = {
    "v5 (Baseline)": {
        "turnover": {
            "accuracy": 0.9300,
            "precision": 0.8333,
            "recall": 0.4545,
            "f1": 0.5882,
            "auroc": 0.8672,
            "aupr": 0.6329,
        },
        "preference": {
            "accuracy": 0.4657,
        },
        "config": "Random negative sampling"
    },
    "v6 (Initial)": {
        "turnover": load_results(Path("outputs/hetero_v6/run_20251019_195926/results.json"))["test_turnover_metrics"],
        "preference": load_results(Path("outputs/hetero_v6/run_20251019_195926/results.json"))["test_preference_metrics"],
        "config": "Hard neg (0.7), Stage2: 100ep"
    },
    "v6_optimized (P0+P1)": {
        "turnover": load_results(Path("outputs/hetero_v6_optimized/optimized_p0p1/results.json"))["test_turnover_metrics"],
        "preference": load_results(Path("outputs/hetero_v6_optimized/optimized_p0p1/results.json"))["test_preference_metrics"],
        "config": "Hard neg (0.85), Dot mode, Stage2: 200ep"
    },
    "v6_balanced": {
        "turnover": load_results(Path("outputs/hetero_v6_balanced/balanced/results.json"))["test_turnover_metrics"],
        "preference": load_results(Path("outputs/hetero_v6_balanced/balanced/results.json"))["test_preference_metrics"],
        "config": "Alpha=0.4, Beta=0.6, Concat mode"
    },
    "v6_best": {
        "turnover": load_results(Path("outputs/hetero_v6_best/best/results.json"))["test_turnover_metrics"],
        "preference": load_results(Path("outputs/hetero_v6_best/best/results.json"))["test_preference_metrics"],
        "config": "Alpha=0.5, Beta=0.5, Dot mode, LR=3e-4"
    },
}

# Print header
print("=" * 120)
print("COMPREHENSIVE MODEL COMPARISON - All Versions")
print("=" * 120)

# Turnover Metrics
print("\n📊 TURNOVER PREDICTION (Test Set)")
print("-" * 120)
header = f"{'Version':<25} | {'AUPR':<8} | {'AUROC':<8} | {'F1':<8} | {'Prec':<8} | {'Recall':<8} | {'Acc':<8}"
print(header)
print("-" * 120)

for version_name, data in versions.items():
    t = data["turnover"]
    print(
        f"{version_name:<25} | "
        f"{t['aupr']:<8.4f} | "
        f"{t['auroc']:<8.4f} | "
        f"{t['f1']:<8.4f} | "
        f"{t['precision']:<8.4f} | "
        f"{t['recall']:<8.4f} | "
        f"{t['accuracy']:<8.4f}"
    )

# Preference Metrics
print("\n🎯 PREFERENCE RANKING (Test Set)")
print("-" * 120)
print(f"{'Version':<25} | {'Pairwise Acc':<15} | {'vs Baseline':<15} | Status")
print("-" * 120)

baseline_pref = versions["v5 (Baseline)"]["preference"]["accuracy"]

for version_name, data in versions.items():
    p = data["preference"]
    pref_acc = p["accuracy"]
    improvement = ((pref_acc - baseline_pref) / baseline_pref) * 100

    if pref_acc > 0.65:
        status = "🌟 Excellent"
    elif pref_acc > 0.55:
        status = "✅ Good"
    elif pref_acc > 0.50:
        status = "⚠️ Marginal"
    else:
        status = "❌ Poor"

    print(
        f"{version_name:<25} | "
        f"{pref_acc:<15.4f} | "
        f"{improvement:>+6.1f}% | "
        f"{status}"
    )

# Configuration summary
print("\n📋 CONFIGURATION SUMMARY")
print("-" * 120)
print(f"{'Version':<25} | Configuration")
print("-" * 120)
for version_name, data in versions.items():
    print(f"{version_name:<25} | {data['config']}")

# Find best versions
print("\n" + "=" * 120)
print("🏆 BEST PERFORMERS")
print("=" * 120)

# Best F1
best_f1_version = max(versions.items(), key=lambda x: x[1]["turnover"]["f1"])
print(f"Best Turnover F1:        {best_f1_version[0]:<25} (F1={best_f1_version[1]['turnover']['f1']:.4f})")

# Best AUPR
best_aupr_version = max(versions.items(), key=lambda x: x[1]["turnover"]["aupr"])
print(f"Best Turnover AUPR:      {best_aupr_version[0]:<25} (AUPR={best_aupr_version[1]['turnover']['aupr']:.4f})")

# Best Preference
best_pref_version = max(versions.items(), key=lambda x: x[1]["preference"]["accuracy"])
print(f"Best Preference Accuracy: {best_pref_version[0]:<25} (Acc={best_pref_version[1]['preference']['accuracy']:.4f})")

# Balanced performance (harmonic mean)
import math

def harmonic_mean(f1, pref_acc):
    """Compute harmonic mean of F1 and Preference Accuracy."""
    return 2 * (f1 * pref_acc) / (f1 + pref_acc) if (f1 + pref_acc) > 0 else 0

best_balanced = max(
    versions.items(),
    key=lambda x: harmonic_mean(x[1]["turnover"]["f1"], x[1]["preference"]["accuracy"])
)
hm = harmonic_mean(best_balanced[1]["turnover"]["f1"], best_balanced[1]["preference"]["accuracy"])
print(f"Best Balanced (HM):       {best_balanced[0]:<25} (HM={hm:.4f})")

# Comparison with baselines
print("\n" + "=" * 120)
print("📈 IMPROVEMENT OVER NON-GRAPH BASELINES")
print("=" * 120)

mlp_f1 = 0.5714
xgb_f1 = 0.5926

print(f"{'Version':<25} | vs MLP (F1=0.5714) | vs XGB (F1=0.5926)")
print("-" * 120)

for version_name, data in versions.items():
    f1 = data["turnover"]["f1"]
    vs_mlp = f1 - mlp_f1
    vs_xgb = f1 - xgb_f1

    mlp_status = "✅" if vs_mlp > 0 else "❌"
    xgb_status = "✅" if vs_xgb > 0 else "❌"

    print(
        f"{version_name:<25} | "
        f"{vs_mlp:>+7.4f} {mlp_status:<3} | "
        f"{vs_xgb:>+7.4f} {xgb_status}"
    )

# Recommendations
print("\n" + "=" * 120)
print("💡 RECOMMENDATIONS")
print("=" * 120)

print("\n1. For PUBLICATION (need both high F1 AND high Pref Acc):")
print(f"   ⚠️ Current best balanced: {best_balanced[0]} (F1={best_balanced[1]['turnover']['f1']:.4f}, Pref={best_balanced[1]['preference']['accuracy']:.4f})")
print(f"   ❌ Still below publication threshold (F1 > 0.60, Pref > 0.60)")
print()

print("2. For PREFERENCE TASK specifically:")
print(f"   🌟 Use: {best_pref_version[0]} (Pref Acc = {best_pref_version[1]['preference']['accuracy']:.4f})")
print(f"   ✅ Significantly above random baseline (0.50)")
print()

print("3. For TURNOVER TASK specifically:")
print(f"   🌟 Use: {best_f1_version[0]} (F1 = {best_f1_version[1]['turnover']['f1']:.4f})")
print()

print("4. NEXT STEPS to improve:")
print("   • Investigate why concat mode fails for preference")
print("   • Try intermediate alpha values (0.3-0.4)")
print("   • Add neighbor aggregation regularization")
print("   • Implement curriculum learning for negative sampling")
print("   • Consider ensemble of v6_balanced + v6_optimized")

print("\n" + "=" * 120)
