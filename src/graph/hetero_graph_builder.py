"""
异构图构建器
=============
生成包含员工、当前岗位、目标岗位以及公司属性的 HeteroData。
同时导出岗位偏好三元组供 pairwise loss 使用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import re

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def _load_mask(path: Path) -> torch.Tensor:
    mask = torch.load(path)
    if mask.dtype != torch.bool:
        mask = mask.bool()
    return mask


def _default_job_columns() -> List[Tuple[str, str]]:
    """返回 (列名, 简短别名) 列表。"""
    return [
        ("Q5_1", "data"),
        ("Q5_2", "algorithm"),
        ("Q5_3", "analytics"),
        ("Q5_4", "product"),
        ("Q5_5", "operations"),
        ("Q5_6", "sales"),
        ("Q5_7", "hr"),
        ("Q5_8", "finance"),
        ("Q5_9", "legal"),
        ("Q5_10", "admin"),
        ("Q5_11", "rd"),
        ("Q5_12", "manufacture"),
        ("Q5_13", "other"),
    ]


@dataclass
class HeteroGraphBuilder:
    raw_csv_path: Path = Path("data/raw/originaldata.csv")
    features_path: Path = Path("data/processed/employee_features.npy")
    labels_path: Path = Path("data/processed/y_turnover_binary.npy")
    preference_pairs_path: Path = Path("data/processed/preference_pairs.csv")
    splits_dir: Path = Path("data/splits")
    output_dir: Path = Path("data/processed")
    encoding: str = "gbk"
    job_columns: List[Tuple[str, str]] = field(default_factory=_default_job_columns)
    company_type_col: str = "Q3"
    company_size_col: str = "Q4"
    unknown_current_job_name: str = "unknown"

    def build(self) -> HeteroData:
        df, original_header = self._load_raw_dataframe()
        features = self._load_employee_features()
        preference_pairs = self._load_preference_pairs(len(features))
        max_pref_post = max((max(pair[1], pair[2]) for pair in preference_pairs), default=0)

        if len(df) > len(features):
            df = df.iloc[: len(features)].reset_index(drop=True)

        # 构建映射
        job_cols, job_alias = zip(*self.job_columns)
        job_alias = list(job_alias)
        num_post_types = max(len(self.job_columns), max_pref_post + 1)
        post_type_names = [f"post_{i:02d}" for i in range(num_post_types)]
        company_size_map = self._create_category_mapping(df[self.company_size_col])
        company_type_map = self._create_category_mapping(df[self.company_type_col])

        job_values = self._extract_job_matrix(df, job_cols)
        primary_job_ids = self._determine_primary_job(job_values)

        employee_post_edges = self._build_employee_post_edges(job_values)
        employee_current_job_edges = self._build_employee_current_job_edges(primary_job_ids)
        employee_company_size_edges = self._build_employee_company_edges(df, company_size_map, self.company_size_col)
        employee_company_type_edges = self._build_employee_company_edges(df, company_type_map, self.company_type_col)

        current_job_size_edges = self._build_current_job_attribute_edges(primary_job_ids, employee_company_size_edges, len(job_alias))
        current_job_type_edges = self._build_current_job_attribute_edges(primary_job_ids, employee_company_type_edges, len(job_alias))

        prefer_edges, disprefer_edges, preference_triples = self._build_preference_edges(preference_pairs)

        hetero = HeteroData()
        hetero["employee"].x = torch.from_numpy(features).float()
        hetero["employee"].y = torch.from_numpy(self._load_labels(len(features))).long()
        hetero["employee"].train_mask = _load_mask(self.splits_dir / "train_mask.pt")
        hetero["employee"].val_mask = _load_mask(self.splits_dir / "val_mask.pt")
        hetero["employee"].test_mask = _load_mask(self.splits_dir / "test_mask.pt")

        hetero["current_job"].x = torch.eye(len(job_alias) + 1, dtype=torch.float32)  # +1 for unknown
        hetero["post_type"].x = torch.eye(num_post_types, dtype=torch.float32)
        hetero["company_size"].x = torch.eye(len(company_size_map), dtype=torch.float32)
        hetero["company_type"].x = torch.eye(len(company_type_map), dtype=torch.float32)

        hetero["employee", "assigned_to", "current_job"].edge_index = employee_current_job_edges
        hetero["current_job", "rev_assigned_to", "employee"].edge_index = employee_current_job_edges.flip(0)

        hetero["employee", "interested_in", "post_type"].edge_index = employee_post_edges
        hetero["post_type", "rev_interested_in", "employee"].edge_index = employee_post_edges.flip(0)

        hetero["employee", "belongs_to_size", "company_size"].edge_index = employee_company_size_edges
        hetero["company_size", "rev_belongs_to_size", "employee"].edge_index = employee_company_size_edges.flip(0)

        hetero["employee", "belongs_to_type", "company_type"].edge_index = employee_company_type_edges
        hetero["company_type", "rev_belongs_to_type", "employee"].edge_index = employee_company_type_edges.flip(0)

        hetero["current_job", "has_size", "company_size"].edge_index = current_job_size_edges
        hetero["company_size", "rev_has_size", "current_job"].edge_index = current_job_size_edges.flip(0)

        hetero["current_job", "has_type", "company_type"].edge_index = current_job_type_edges
        hetero["company_type", "rev_has_type", "current_job"].edge_index = current_job_type_edges.flip(0)

        hetero["employee", "prefer", "post_type"].edge_index = prefer_edges
        hetero["post_type", "rev_prefer", "employee"].edge_index = prefer_edges.flip(0)
        hetero["employee", "disprefer", "post_type"].edge_index = disprefer_edges
        hetero["post_type", "rev_disprefer", "employee"].edge_index = disprefer_edges.flip(0)

        self._save_outputs(
            hetero,
            preference_triples,
            {
                "job_alias": job_alias + [self.unknown_current_job_name],
                "post_type_names": post_type_names,
                "company_size_map": company_size_map,
                "company_type_map": company_type_map,
                "original_header": original_header,
            },
        )

        return hetero

    # ------------------------------------------------------------------ #
    # 加载辅助
    # ------------------------------------------------------------------ #
    def _load_raw_dataframe(self) -> Tuple[pd.DataFrame, List[str]]:
        df = pd.read_csv(self.raw_csv_path, encoding=self.encoding)
        original_header = list(df.columns)

        if not df.empty:
            second_row = df.iloc[0].astype(str)
            pattern = re.compile(r"^Q\d+(?:_\d+)?$")
            mask = second_row.str.match(pattern)
            if mask.sum() > len(second_row) * 0.1:
                new_columns = [
                    second_row.iloc[i] if mask.iloc[i] else original_header[i]
                    for i in range(len(original_header))
                ]
                df = df.iloc[1:].reset_index(drop=True)
                df.columns = new_columns

        return df.reset_index(drop=True), original_header

    def _load_employee_features(self) -> np.ndarray:
        features = np.load(self.features_path)
        if features.ndim != 2:
            raise ValueError(f"employee_features expected 2D array, got shape {features.shape}")
        return features

    def _load_labels(self, expected_len: int) -> np.ndarray:
        labels = np.load(self.labels_path)
        if labels.ndim != 1:
            labels = labels.reshape(-1)
        if len(labels) < expected_len:
            raise ValueError("labels shorter than number of employees.")
        return labels[:expected_len]

    def _load_preference_pairs(self, num_employees: int) -> List[Tuple[int, int, int, int]]:
        pairs: List[Tuple[int, int, int, int]] = []
        with open(self.preference_pairs_path, "r", newline="") as f:
            import csv

            reader = csv.DictReader(f)
            for row in reader:
                emp = int(row["employee_idx"])
                if emp >= num_employees:
                    continue
                pairs.append(
                    (
                        emp,
                        int(row["post_A_id"]),
                        int(row["post_B_id"]),
                        int(row["choice"]),
                    )
                )
        return pairs

    # ------------------------------------------------------------------ #
    # 构建边
    # ------------------------------------------------------------------ #
    def _extract_job_matrix(self, df: pd.DataFrame, job_cols: Iterable[str]) -> np.ndarray:
        job_df = df.loc[:, job_cols].applymap(lambda v: str(v).strip())
        job_df = job_df.replace(
            {
                "1": 1,
                "是": 1,
                "yes": 1,
                "Yes": 1,
                "0": 0,
                "否": 0,
                "": 0,
                "nan": 0,
            }
        )
        job_values = job_df.fillna(0).astype(float).values
        return job_values

    def _determine_primary_job(self, job_matrix: np.ndarray) -> np.ndarray:
        primary = np.full(job_matrix.shape[0], job_matrix.shape[1], dtype=np.int64)  # default unknown
        for i, row in enumerate(job_matrix):
            ones = np.where(row > 0)[0]
            if ones.size == 0:
                continue
            # 多选时，取最先出现的岗位类型
            primary[i] = int(ones[0])
        return primary

    def _build_employee_post_edges(self, job_matrix: np.ndarray) -> torch.Tensor:
        src, dst = [], []
        for emp_idx, row in enumerate(job_matrix):
            job_ids = np.where(row > 0)[0]
            for job_id in job_ids:
                src.append(emp_idx)
                dst.append(job_id)
        if not src:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor([src, dst], dtype=torch.long)

    def _build_employee_current_job_edges(self, primary_job_ids: np.ndarray) -> torch.Tensor:
        src = torch.arange(len(primary_job_ids), dtype=torch.long)
        dst = torch.from_numpy(primary_job_ids).long()
        return torch.stack([src, dst], dim=0)

    def _build_employee_company_edges(
        self,
        df: pd.DataFrame,
        mapping: Dict[str, int],
        column: str,
    ) -> torch.Tensor:
        src, dst = [], []
        for emp_idx, value in enumerate(df[column].astype(str)):
            value = value.strip()
            if value == "" or value not in mapping:
                continue
            src.append(emp_idx)
            dst.append(mapping[value])
        if not src:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor([src, dst], dtype=torch.long)

    def _build_current_job_attribute_edges(
        self,
        primary_job_ids: np.ndarray,
        employee_attr_edges: torch.Tensor,
        num_job_categories: int,
    ) -> torch.Tensor:
        if employee_attr_edges.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long)
        job_attr_pairs = set()
        attr_by_employee = dict(zip(employee_attr_edges[0].tolist(), employee_attr_edges[1].tolist()))
        for emp_idx, job_id in enumerate(primary_job_ids):
            attr_id = attr_by_employee.get(emp_idx)
            if attr_id is None:
                continue
            job_attr_pairs.add((int(job_id), int(attr_id)))
        if not job_attr_pairs:
            return torch.empty((2, 0), dtype=torch.long)
        src, dst = zip(*sorted(job_attr_pairs))
        return torch.tensor([src, dst], dtype=torch.long)

    def _build_preference_edges(
        self,
        preference_pairs: List[Tuple[int, int, int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefer_src, prefer_dst = [], []
        disprefer_src, disprefer_dst = [], []
        triples = []
        for emp, post_a, post_b, choice in preference_pairs:
            if choice == 0:
                prefer, disprefer = post_a, post_b
            else:
                prefer, disprefer = post_b, post_a
            prefer_src.append(emp)
            prefer_dst.append(prefer)
            disprefer_src.append(emp)
            disprefer_dst.append(disprefer)
            triples.append((emp, prefer, disprefer))
        prefer_edges = torch.tensor([prefer_src, prefer_dst], dtype=torch.long) if prefer_src else torch.empty((2, 0), dtype=torch.long)
        disprefer_edges = torch.tensor([disprefer_src, disprefer_dst], dtype=torch.long) if disprefer_src else torch.empty((2, 0), dtype=torch.long)
        preference_triples = torch.tensor(triples, dtype=torch.long) if triples else torch.empty((0, 3), dtype=torch.long)
        return prefer_edges, disprefer_edges, preference_triples

    # ------------------------------------------------------------------ #
    # 工具函数
    # ------------------------------------------------------------------ #
    def _create_category_mapping(self, series: pd.Series) -> Dict[str, int]:
        unique_values = [v for v in series.astype(str).str.strip().unique() if v and v != "nan"]
        return {value: idx for idx, value in enumerate(sorted(unique_values))}

    def _save_outputs(
        self,
        data: HeteroData,
        preference_triples: torch.Tensor,
        metadata: Dict[str, object],
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        graph_path = self.output_dir / "hetero_graph.pt"
        triples_path = self.output_dir / "preference_triples.pt"
        meta_path = self.output_dir / "hetero_graph_meta.json"

        torch.save(data, graph_path)
        torch.save(preference_triples, triples_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 异构图已保存到: {graph_path}")
        print(f"✅ 偏好三元组已保存到: {triples_path}")
        print(f"✅ 元数据已保存到: {meta_path}")


if __name__ == "__main__":
    builder = HeteroGraphBuilder()
    builder.build()
