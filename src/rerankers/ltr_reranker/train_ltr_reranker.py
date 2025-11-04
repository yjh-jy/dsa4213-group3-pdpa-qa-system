#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_ltr_reranker.py — Train a LightGBM LambdaMART (LambdaRank) model for reranking
- Reads:
    data/ltr_processed/train_ltr_data.jsonl
    data/ltr_processed/val_ltr_data.jsonl
- Writes:
    artefacts/ltr_reranker/model/lgbm_model_ltr_reranker.txt
    artefacts/ltr_reranker/model/lgbm_feature_importance_ltr_reranker.csv
    artefacts/ltr_reranker/model/best_params_ltr_reranker.json
"""

import pandas as pd
import lightgbm as lgb
from pathlib import Path
import json

# --- Paths ---
ROOT = Path(__file__).resolve().parents[3]      # repo root 
TRAIN_FILE = ROOT / "data" / "ltr_processed" / "train_ltr_data.jsonl"
VAL_FILE = ROOT / "data" / "ltr_processed" / "val_ltr_data.jsonl"
MODEL_DIR = ROOT / "artefacts" / "ltr_reranker" / "model"

class LTRTrainer:
    def __init__(self, train_path: Path, val_path: Path, model_dir: Path):
        self.train_path = train_path
        self.val_path = val_path
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = self.model_dir / "lgbm_model_ltr_reranker.txt"
        self.feature_imp_path = self.model_dir / "lgbm_feature_importance_ltr_reranker.csv"
        self.best_params_path = self.model_dir / "best_params_ltr_reranker.json"

        # Define feature, target, grouping, and weight columns
        self.feature_cols = ["bm25_score", "dense_score", "section_len", "is_pos", "is_neg"]
        self.target_col = "label"
        self.group_col = "qid"
        self.weight_col = "weight"

    # Load datasets 
    def load_data(self, path: Path):
        df = pd.read_json(path, lines=True)
        df = df.sort_values(self.group_col).reset_index(drop=True)
        return df

    # Prepare LightGBM datasets 
    def prepare_lgb_dataset(self, df):
        X = df[self.feature_cols]
        y = df[self.target_col]
        w = df[self.weight_col]
        groups = df.groupby(self.group_col, sort=False)[self.group_col].count().to_list()
        return lgb.Dataset(X, label=y, weight=w, group=groups)

    # Train LightGBM model
    def train_model(self, train_ds, val_ds, params=None, num_boost_round=100):
        if params is None:
            # Default hyperparameters
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [5, 10, 20],
                "num_leaves": 75,
                "learning_rate": 0.03,
                "min_data_in_leaf": 30,
                "feature_fraction": 0.975,
                "bagging_fraction": 0.8,
                "bagging_freq": 2,
                "lambda_l1": 1e-5,
                "lambda_l2": 5e-5,
                "device": "gpu",
                "random_state": 42,
                "verbose": -1,
                "feature_pre_filter": False
            }

        model = lgb.train(params, train_ds, valid_sets=[val_ds], num_boost_round=num_boost_round)
        # Save params with best iteration
        best_metadata = params.copy()
        best_metadata["best_iteration"] = model.best_iteration
        with open(self.best_params_path, "w") as f:
            json.dump(best_metadata, f, indent=2)
        print(f"Saved hyperparameters to {self.best_params_path}")
        return model

    # Save model & feature importance 
    def save_model(self, model):
        model.save_model(str(self.model_path))
        print(f"Model saved to {self.model_path}")
        fi = pd.DataFrame({"feature": self.feature_cols,
                           "importance": model.feature_importance(importance_type="gain")})
        fi.to_csv(self.feature_imp_path, index=False)
        print(f"Feature importance saved to {self.feature_imp_path}")


if __name__ == "__main__":
    trainer = LTRTrainer(TRAIN_FILE, VAL_FILE, MODEL_DIR)

    train_df = trainer.load_data(TRAIN_FILE)
    val_df = trainer.load_data(VAL_FILE)

    train_ds = trainer.prepare_lgb_dataset(train_df)
    val_ds = trainer.prepare_lgb_dataset(val_df)

    model = trainer.train_model(train_ds, val_ds, num_boost_round=100)

    trainer.save_model(model)
