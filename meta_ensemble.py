#!/usr/bin/env python3
"""
================================================================================
NCAA College Basketball Predictive Model - Meta-Ensemble Model
================================================================================

Production meta-ensemble with 3-layer stacking:
1. Base models (XGBoost, LightGBM, CatBoost)
2. Meta-learner (Ridge/Logistic)
3. Probability calibration (Isotonic)

================================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pickle
import warnings

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

from config import model_config, MODELS_DIR

warnings.filterwarnings('ignore')


class MetaEnsemble:
    """
    Production meta-ensemble for CBB prediction.

    Architecture:
    - Layer 1: 3 base models (XGBoost, LightGBM, CatBoost)
    - Layer 2: Meta-learner combining base predictions
    - Layer 3: Isotonic calibration for probabilities

    Outputs:
    - Win probability (calibrated)
    - Point margin (with std)
    - Total points (with std)
    """

    def __init__(self, config=None):
        self.config = config or model_config

        # Base models
        self.win_models = {}
        self.margin_models = {}
        self.total_models = {}

        # Meta-learners
        self.win_meta = None
        self.margin_meta = None
        self.total_meta = None

        # Calibration
        self.win_calibrator = None

        # Preprocessing
        self.scaler = StandardScaler()

        # Variance estimation
        self.margin_std = self.config.default_margin_std
        self.total_std = self.config.default_total_std

        # Metadata
        self.is_trained = False
        self.version = None
        self.feature_names = []
        self.feature_importance = {}
        self.training_metrics = {}

    def _create_base_models(self, task: str = 'classification') -> Dict:
        """Create base models for ensemble"""
        if task == 'classification':
            return {
                'xgb': xgb.XGBClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    subsample=self.config.subsample,
                    colsample_bytree=self.config.colsample_bytree,
                    random_state=42,
                    verbosity=0,
                    use_label_encoder=False,
                    eval_metric='logloss',
                ),
                'lgb': lgb.LGBMClassifier(
                    n_estimators=self.config.n_estimators,
                    num_leaves=2 ** self.config.max_depth - 1,
                    learning_rate=self.config.learning_rate,
                    feature_fraction=self.config.colsample_bytree,
                    bagging_fraction=self.config.subsample,
                    bagging_freq=1,
                    random_state=42,
                    verbose=-1,
                ),
                'cat': CatBoostClassifier(
                    iterations=self.config.n_estimators,
                    depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    random_seed=42,
                    verbose=False,
                ),
            }
        else:
            return {
                'xgb': xgb.XGBRegressor(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    subsample=self.config.subsample,
                    colsample_bytree=self.config.colsample_bytree,
                    random_state=42,
                    verbosity=0,
                ),
                'lgb': lgb.LGBMRegressor(
                    n_estimators=self.config.n_estimators,
                    num_leaves=2 ** self.config.max_depth - 1,
                    learning_rate=self.config.learning_rate,
                    feature_fraction=self.config.colsample_bytree,
                    bagging_fraction=self.config.subsample,
                    bagging_freq=1,
                    random_state=42,
                    verbose=-1,
                ),
                'cat': CatBoostRegressor(
                    iterations=self.config.n_estimators,
                    depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    random_seed=42,
                    verbose=False,
                ),
            }

    def train(self,
              X: np.ndarray,
              y_win: np.ndarray,
              y_margin: np.ndarray,
              y_total: np.ndarray,
              sample_weights: np.ndarray = None,
              feature_names: List[str] = None) -> Dict:
        """
        Train the full meta-ensemble.

        Args:
            X: Feature matrix
            y_win: Win labels (0/1)
            y_margin: Point margin targets
            y_total: Total points targets
            sample_weights: Optional recency weights
            feature_names: Feature names for importance tracking

        Returns:
            Training metrics dictionary
        """
        print("\n" + "=" * 60)
        print("TRAINING META-ENSEMBLE")
        print("=" * 60)
        print(f"Samples: {len(y_win)} | Features: {X.shape[1]}")

        self.feature_names = feature_names or []

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Time series cross-validation for OOF predictions
        tscv = TimeSeriesSplit(n_splits=self.config.cv_splits)

        # Initialize OOF arrays
        oof_win = {name: np.zeros(len(y_win)) for name in ['xgb', 'lgb', 'cat']}
        oof_margin = {name: np.zeros(len(y_margin)) for name in ['xgb', 'lgb', 'cat']}
        oof_total = {name: np.zeros(len(y_total)) for name in ['xgb', 'lgb', 'cat']}

        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            print(f"\nFold {fold + 1}/{self.config.cv_splits}...")

            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_win_train = y_win[train_idx]
            y_margin_train = y_margin[train_idx]
            y_total_train = y_total[train_idx]

            weights_train = sample_weights[train_idx] if sample_weights is not None else None

            # Train fold models
            win_models = self._create_base_models('classification')
            margin_models = self._create_base_models('regression')
            total_models = self._create_base_models('regression')

            for name in ['xgb', 'lgb', 'cat']:
                # Win models
                if weights_train is not None and name != 'cat':
                    win_models[name].fit(X_train, y_win_train, sample_weight=weights_train)
                    margin_models[name].fit(X_train, y_margin_train, sample_weight=weights_train)
                    total_models[name].fit(X_train, y_total_train, sample_weight=weights_train)
                else:
                    win_models[name].fit(X_train, y_win_train)
                    margin_models[name].fit(X_train, y_margin_train)
                    total_models[name].fit(X_train, y_total_train)

                # Get OOF predictions
                oof_win[name][val_idx] = win_models[name].predict_proba(X_val)[:, 1]
                oof_margin[name][val_idx] = margin_models[name].predict(X_val)
                oof_total[name][val_idx] = total_models[name].predict(X_val)

            # Fold metrics
            fold_preds = np.mean([oof_win[n][val_idx] for n in ['xgb', 'lgb', 'cat']], axis=0)
            fold_brier = brier_score_loss(y_win[val_idx], fold_preds)
            fold_metrics.append({'fold': fold + 1, 'brier': fold_brier})
            print(f"  Fold Brier Score: {fold_brier:.4f}")

        # Stack OOF predictions for meta-learner
        X_meta_win = np.column_stack([oof_win[name] for name in ['xgb', 'lgb', 'cat']])
        X_meta_margin = np.column_stack([oof_margin[name] for name in ['xgb', 'lgb', 'cat']])
        X_meta_total = np.column_stack([oof_total[name] for name in ['xgb', 'lgb', 'cat']])

        # Valid indices (skip first folds with no predictions)
        valid_mask = X_meta_win.sum(axis=1) != 0

        print("\nTraining meta-learners...")

        # Win meta-learner (logistic)
        self.win_meta = LogisticRegression(C=1.0 / self.config.l2_alpha, max_iter=1000)
        self.win_meta.fit(X_meta_win[valid_mask], y_win[valid_mask])

        # Margin meta-learner (ridge)
        self.margin_meta = Ridge(alpha=self.config.l2_alpha)
        self.margin_meta.fit(X_meta_margin[valid_mask], y_margin[valid_mask])

        # Total meta-learner (ridge)
        self.total_meta = Ridge(alpha=self.config.l2_alpha)
        self.total_meta.fit(X_meta_total[valid_mask], y_total[valid_mask])

        # Probability calibration
        if self.config.calibrate_probabilities:
            print("Calibrating probabilities...")
            meta_win_preds = self.win_meta.predict_proba(X_meta_win[valid_mask])[:, 1]
            self.win_calibrator = IsotonicRegression(out_of_bounds='clip')
            self.win_calibrator.fit(meta_win_preds, y_win[valid_mask])

        # Calculate residual standard deviations
        meta_margin_preds = self.margin_meta.predict(X_meta_margin[valid_mask])
        meta_total_preds = self.total_meta.predict(X_meta_total[valid_mask])
        self.margin_std = np.std(y_margin[valid_mask] - meta_margin_preds)
        self.total_std = np.std(y_total[valid_mask] - meta_total_preds)

        # Train final models on all data
        print("\nTraining final models...")

        self.win_models = self._create_base_models('classification')
        self.margin_models = self._create_base_models('regression')
        self.total_models = self._create_base_models('regression')

        for name in ['xgb', 'lgb', 'cat']:
            if sample_weights is not None and name != 'cat':
                self.win_models[name].fit(X_scaled, y_win, sample_weight=sample_weights)
                self.margin_models[name].fit(X_scaled, y_margin, sample_weight=sample_weights)
                self.total_models[name].fit(X_scaled, y_total, sample_weight=sample_weights)
            else:
                self.win_models[name].fit(X_scaled, y_win)
                self.margin_models[name].fit(X_scaled, y_margin)
                self.total_models[name].fit(X_scaled, y_total)

        # Calculate feature importance
        self._calculate_feature_importance()

        # Final metrics
        final_preds = self.predict_all(X)
        final_brier = brier_score_loss(y_win, final_preds['win_prob'])
        final_margin_rmse = np.sqrt(np.mean((y_margin - final_preds['margin']) ** 2))
        final_total_rmse = np.sqrt(np.mean((y_total - final_preds['total']) ** 2))

        self.training_metrics = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(y_win),
            'n_features': X.shape[1],
            'brier_score': float(final_brier),
            'margin_rmse': float(final_margin_rmse),
            'total_rmse': float(final_total_rmse),
            'margin_std': float(self.margin_std),
            'total_std': float(self.total_std),
            'cv_brier_scores': fold_metrics,
            'home_win_rate': float(y_win.mean()),
        }

        self.version = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.is_trained = True

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Brier Score: {final_brier:.4f}")
        print(f"Margin RMSE: {final_margin_rmse:.2f}")
        print(f"Total RMSE: {final_total_rmse:.2f}")
        print(f"Margin Std: {self.margin_std:.2f}")
        print(f"Total Std: {self.total_std:.2f}")

        return self.training_metrics

    def _calculate_feature_importance(self):
        """Calculate feature importance from base models"""
        if not self.feature_names:
            return

        importance = {name: 0.0 for name in self.feature_names}

        # Average importance across models
        for name in ['xgb', 'lgb']:
            if name == 'xgb':
                model_imp = self.win_models[name].feature_importances_
            else:
                model_imp = self.win_models[name].feature_importances_

            for i, feat in enumerate(self.feature_names):
                if i < len(model_imp):
                    importance[feat] += model_imp[i] / 2

        self.feature_importance = importance

    def predict_all(self, X: np.ndarray) -> Dict:
        """
        Generate all predictions.

        Returns:
            Dict with win_prob, margin, margin_std, total, total_std
        """
        X_scaled = self.scaler.transform(X)

        # Base predictions
        win_base = np.column_stack([
            self.win_models[name].predict_proba(X_scaled)[:, 1]
            for name in ['xgb', 'lgb', 'cat']
        ])
        margin_base = np.column_stack([
            self.margin_models[name].predict(X_scaled)
            for name in ['xgb', 'lgb', 'cat']
        ])
        total_base = np.column_stack([
            self.total_models[name].predict(X_scaled)
            for name in ['xgb', 'lgb', 'cat']
        ])

        # Meta predictions
        win_meta = self.win_meta.predict_proba(win_base)[:, 1]
        margin = self.margin_meta.predict(margin_base)
        total = self.total_meta.predict(total_base)

        # Calibrate
        if self.win_calibrator:
            win_prob = np.clip(self.win_calibrator.predict(win_meta), 0.01, 0.99)
        else:
            win_prob = np.clip(win_meta, 0.01, 0.99)

        return {
            'win_prob': win_prob,
            'margin': margin,
            'margin_std': np.full(len(X), self.margin_std),
            'total': total,
            'total_std': np.full(len(X), self.total_std),
        }

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N features by importance"""
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n]

    def save(self, filepath: str = None):
        """Save model to disk"""
        if filepath is None:
            filepath = MODELS_DIR / f"model_{self.version}.pkl"

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

        # Also save as latest
        latest_path = MODELS_DIR / "model_latest.pkl"
        with open(latest_path, 'wb') as f:
            pickle.dump(self, f)

        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str = None) -> 'MetaEnsemble':
        """Load model from disk"""
        if filepath is None:
            filepath = MODELS_DIR / "model_latest.pkl"

        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        print(f"Model loaded from {filepath}")
        return model
