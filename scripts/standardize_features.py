"""
标准化员工特征
==============
对 `employee_features.npy` 进行标准化，并保存 scaler 以供推理阶段复用。
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardize employee features.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/employee_features.npy"),
        help="原始特征文件 (.npy)",
    )
    parser.add_argument(
        "--train-mask",
        type=Path,
        default=Path("data/splits/train_mask.pt"),
        help="训练集 mask 文件",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/employee_features_standardized.npy"),
        help="标准化后的特征输出路径",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=Path("data/processed/feature_scaler.pkl"),
        help="保存 StandardScaler 的路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    features = np.load(args.input)
    if features.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {features.shape}")

    train_mask = torch.load(args.train_mask)
    if train_mask.dtype != torch.bool:
        train_mask = train_mask.bool()

    if train_mask.numel() != len(features):
        raise ValueError("train mask shape does not match number of feature rows.")

    scaler = StandardScaler()
    scaler.fit(features[train_mask.numpy()])

    transformed = scaler.transform(features)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, transformed)

    with open(args.scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print("✅ 标准化完成")
    print(f"   输入特征: {args.input}")
    print(f"   输出特征: {args.output}")
    print(f"   Scaler 保存至: {args.scaler_path}")


if __name__ == "__main__":
    main()
