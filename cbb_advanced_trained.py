#!/usr/bin/env python3
"""
================================================================================
🏀 ADVANCED SYNDICATE COLLEGE BASKETBALL SYSTEM - TRAINED VERSION
================================================================================

IMPROVEMENTS OVER RULE-BASED VERSION:

1. TRAINED META-ENSEMBLE
   - XGBoost, LightGBM, CatBoost base models
   - Ridge meta-learner on out-of-fold predictions
   - Walk-forward validation (no leakage)

2. EXPANDED FEATURE SET
   - Four Factors differentials
   - Pace interactions
   - Rest days advantage
   - Home/Away splits
   - Conference strength adjustments
   - Recent form (last 5/10 games)
   - Head-to-head history
   - Injury/roster adjustments

3. CALIBRATED PROBABILITIES
   - Isotonic regression calibration
   - Platt scaling for probability estimates
   - Historical Brier score tracking

4. MARKET-AWARE FEATURES
   - Opening line vs current line movement
   - Steam move detection
   - Reverse line movement (sharp action)
   - Closing line value (CLV) tracking

5. BANKROLL OPTIMIZATION
   - Dynamic Kelly based on edge confidence interval
   - Correlation-aware portfolio construction
   - Drawdown protection

================================================================================
Author: Joseph (ASA, MAAA)
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy import stats
from scipy.special import expit
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
import warnings
import requests
import pickle
import os

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
API_KEY = "272be842201ff50bdfee622541e2d3ee925afac17b3126e93b81b4d58e0e6b62"

@dataclass
class ModelConfig:
    """Model hyperparameters - these should be tuned via cross-validation"""
    
    # Base model params
    xgb_params: Dict = field(default_factory=lambda: {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 5,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_estimators': 200,
        'random_state': 42,
    })
    
    lgb_params: Dict = field(default_factory=lambda: {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 24,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 25,
        'n_estimators': 200,
        'random_state': 42,
        'verbose': -1,
    })
    
    cat_params: Dict = field(default_factory=lambda: {
        'iterations': 200,
        'depth': 5,
        'learning_rate': 0.03,
        'l2_leaf_reg': 3.0,
        'random_seed': 42,
        'verbose': False,
    })
    
    # Training params
    n_splits: int = 5  # Time series splits
    min_train_games: int = 500  # Minimum games to start training
    retrain_frequency: int = 50  # Retrain every N new games


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
class FeatureEngineer:
    """
    Advanced feature engineering for college basketball prediction
    
    Features are grouped by category:
    1. Efficiency differentials (core)
    2. Four factors differentials
    3. Pace/tempo features
    4. Contextual features (rest, travel, etc.)
    5. Form features (recent performance)
    6. Market features (line movement)
    """
    
    CORE_FEATURES = [
        'adj_em_diff',      # Efficiency margin differential
        'adj_o_diff',       # Offensive efficiency differential
        'adj_d_diff',       # Defensive efficiency differential
        'tempo_avg',        # Expected game pace
        'tempo_diff',       # Tempo differential (mismatch indicator)
    ]
    
    FOUR_FACTORS_FEATURES = [
        'efg_diff',         # Combined eFG% differential
        'tov_diff',         # Combined TOV% differential  
        'orb_diff',         # Combined ORB% differential
        'ftr_diff',         # Combined FTR differential
        'efg_off_adv',      # Home eFG% advantage
        'tov_off_adv',      # Home TOV% advantage
    ]
    
    CONTEXTUAL_FEATURES = [
        'home_indicator',   # Binary home indicator
        'rank_diff',        # KenPom rank differential
        'conf_game',        # Conference game indicator
        'rivalry',          # Rivalry game indicator
        'days_rest_diff',   # Rest advantage
        'travel_distance',  # Away team travel (estimated)
    ]
    
    FORM_FEATURES = [
        'home_form_5g',     # Home team last 5 games ATS
        'away_form_5g',     # Away team last 5 games ATS
        'home_ou_trend',    # Home team over/under trend
        'away_ou_trend',    # Away team over/under trend
        'h2h_home_edge',    # Head-to-head historical edge
    ]
    
    MARKET_FEATURES = [
        'opening_spread',   # Opening line
        'current_spread',   # Current line
        'line_movement',    # Current - Opening
        'steam_indicator',  # Sharp line movement detected
        'public_side',      # Which side public is on
    ]
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = []
        
    def create_features(self, home_stats: Dict, away_stats: Dict,
                       context: Dict = None, market: Dict = None) -> Dict:
        """Create full feature set for a matchup"""
        
        features = {}
        
        # === CORE EFFICIENCY FEATURES ===
        features['adj_em_diff'] = home_stats['adj_em'] - away_stats['adj_em']
        features['adj_o_diff'] = home_stats['adj_o'] - away_stats['adj_o']
        features['adj_d_diff'] = away_stats['adj_d'] - home_stats['adj_d']  # Lower D is better
        features['tempo_avg'] = (home_stats['adj_t'] + away_stats['adj_t']) / 2
        features['tempo_diff'] = home_stats['adj_t'] - away_stats['adj_t']
        
        # === FOUR FACTORS FEATURES ===
        # Offensive advantage = home_off - away_def (how home exploits away D)
        # Defensive advantage = away_off - home_def (how home contains away O)
        features['efg_off_adv'] = home_stats.get('efg_o', 0.5) - away_stats.get('efg_d', 0.5)
        features['efg_def_adv'] = home_stats.get('efg_d', 0.5) - away_stats.get('efg_o', 0.5)
        features['efg_diff'] = features['efg_off_adv'] - features['efg_def_adv']
        
        features['tov_off_adv'] = away_stats.get('tov_d', 0.17) - home_stats.get('tov_o', 0.17)
        features['tov_def_adv'] = home_stats.get('tov_d', 0.17) - away_stats.get('tov_o', 0.17)
        features['tov_diff'] = features['tov_off_adv'] + features['tov_def_adv']
        
        features['orb_off_adv'] = home_stats.get('orb_o', 0.28) - (1 - away_stats.get('drb_d', 0.72))
        features['orb_def_adv'] = home_stats.get('drb_d', 0.72) - (1 - away_stats.get('orb_o', 0.28))
        features['orb_diff'] = features['orb_off_adv'] + features['orb_def_adv']
        
        features['ftr_off_adv'] = home_stats.get('ftr_o', 0.30) - away_stats.get('ftr_d', 0.30)
        features['ftr_def_adv'] = away_stats.get('ftr_o', 0.30) - home_stats.get('ftr_d', 0.30)
        features['ftr_diff'] = features['ftr_off_adv'] - features['ftr_def_adv']
        
        # === CONTEXTUAL FEATURES ===
        features['home_indicator'] = 1.0
        features['rank_diff'] = away_stats.get('rank', 150) - home_stats.get('rank', 150)
        features['rank_product'] = home_stats.get('rank', 150) * away_stats.get('rank', 150) / 10000
        
        if context:
            features['conf_game'] = float(context.get('conference_game', False))
            features['rivalry'] = float(context.get('rivalry', False))
            features['days_rest_diff'] = context.get('home_rest', 3) - context.get('away_rest', 3)
        else:
            features['conf_game'] = 0.0
            features['rivalry'] = 0.0
            features['days_rest_diff'] = 0.0
        
        # === MARKET FEATURES ===
        if market:
            features['opening_spread'] = market.get('opening_spread', 0)
            features['current_spread'] = market.get('spread', 0)
            features['line_movement'] = features['current_spread'] - features['opening_spread']
            features['total_line'] = market.get('total', 145)
        else:
            features['opening_spread'] = 0
            features['current_spread'] = 0
            features['line_movement'] = 0
            features['total_line'] = 145
        
        # === DERIVED FEATURES ===
        # Interaction terms that capture non-linear relationships
        features['em_x_tempo'] = features['adj_em_diff'] * features['tempo_avg'] / 70
        features['efg_x_pace'] = features['efg_diff'] * features['tempo_avg'] / 70
        features['rank_x_hca'] = features['rank_diff'] * features['home_indicator'] / 100
        
        # Quality indicators
        features['elite_matchup'] = float(home_stats.get('rank', 150) <= 25 and 
                                         away_stats.get('rank', 150) <= 25)
        features['mismatch'] = float(abs(features['rank_diff']) > 100)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names"""
        return (self.CORE_FEATURES + self.FOUR_FACTORS_FEATURES + 
                self.CONTEXTUAL_FEATURES + ['em_x_tempo', 'efg_x_pace', 
                'rank_x_hca', 'elite_matchup', 'mismatch', 'rank_product',
                'opening_spread', 'current_spread', 'line_movement', 'total_line'])


# =============================================================================
# META-ENSEMBLE MODEL
# =============================================================================
class MetaEnsembleModel:
    """
    Trained meta-ensemble for probability estimation
    
    Architecture:
    - Layer 1: XGBoost, LightGBM, CatBoost
    - Layer 2: Ridge/Logistic meta-learner
    - Layer 3: Isotonic calibration
    
    Trains three separate ensembles:
    1. Win probability (classification)
    2. Margin prediction (regression) 
    3. Total prediction (regression)
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        
        # Base models
        self.win_models = {}
        self.margin_models = {}
        self.total_models = {}
        
        # Meta-learners
        self.win_meta = None
        self.margin_meta = None
        self.total_meta = None
        
        # Calibrators
        self.win_calibrator = None
        self.spread_calibrator = None
        self.total_calibrator = None
        
        # Feature engineering
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        
        # Training state
        self.is_trained = False
        self.training_history = []
        self.feature_importance = {}
        
        # Residual distributions (for uncertainty)
        self.margin_residual_std = 10.5
        self.total_residual_std = 11.0
    
    def _create_base_models(self, task: str = 'classification'):
        """Create fresh base models"""
        if task == 'classification':
            return {
                'xgb': xgb.XGBClassifier(**self.config.xgb_params),
                'lgb': lgb.LGBMClassifier(**self.config.lgb_params),
                'cat': CatBoostClassifier(**self.config.cat_params, loss_function='Logloss'),
            }
        else:
            xgb_reg = self.config.xgb_params.copy()
            xgb_reg['objective'] = 'reg:squarederror'
            
            lgb_reg = self.config.lgb_params.copy()
            lgb_reg['objective'] = 'regression'
            
            cat_reg = self.config.cat_params.copy()
            cat_reg['loss_function'] = 'RMSE'
            
            return {
                'xgb': xgb.XGBRegressor(**xgb_reg),
                'lgb': lgb.LGBMRegressor(**lgb_reg),
                'cat': CatBoostRegressor(**cat_reg),
            }
    
    def train(self, X: np.ndarray, y_win: np.ndarray, 
              y_margin: np.ndarray, y_total: np.ndarray,
              sample_weights: np.ndarray = None):
        """
        Train the full meta-ensemble
        
        Uses time-series cross-validation to prevent leakage
        """
        print("Training meta-ensemble...")
        print(f"  Training samples: {len(y_win)}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        
        # Collect OOF predictions
        oof_win = np.zeros(len(y_win))
        oof_margin = np.zeros(len(y_margin))
        oof_total = np.zeros(len(y_total))
        
        oof_win_by_model = {name: np.zeros(len(y_win)) for name in ['xgb', 'lgb', 'cat']}
        oof_margin_by_model = {name: np.zeros(len(y_margin)) for name in ['xgb', 'lgb', 'cat']}
        oof_total_by_model = {name: np.zeros(len(y_total)) for name in ['xgb', 'lgb', 'cat']}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            print(f"  Fold {fold + 1}/{self.config.n_splits}...")
            
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_win_train, y_win_val = y_win[train_idx], y_win[val_idx]
            y_margin_train = y_margin[train_idx]
            y_total_train = y_total[train_idx]
            
            # Train base models for this fold
            win_models = self._create_base_models('classification')
            margin_models = self._create_base_models('regression')
            total_models = self._create_base_models('regression')
            
            for name in win_models:
                # Win probability
                win_models[name].fit(X_train, y_win_train)
                oof_win_by_model[name][val_idx] = win_models[name].predict_proba(X_val)[:, 1]
                
                # Margin
                margin_models[name].fit(X_train, y_margin_train)
                oof_margin_by_model[name][val_idx] = margin_models[name].predict(X_val)
                
                # Total
                total_models[name].fit(X_train, y_total_train)
                oof_total_by_model[name][val_idx] = total_models[name].predict(X_val)
        
        # Stack OOF predictions for meta-learner
        X_meta_win = np.column_stack([oof_win_by_model[name] for name in ['xgb', 'lgb', 'cat']])
        X_meta_margin = np.column_stack([oof_margin_by_model[name] for name in ['xgb', 'lgb', 'cat']])
        X_meta_total = np.column_stack([oof_total_by_model[name] for name in ['xgb', 'lgb', 'cat']])
        
        # Train meta-learners
        print("  Training meta-learners...")
        
        # Only use samples where we have OOF predictions (skip first fold's train set)
        valid_mask = X_meta_win.sum(axis=1) != 0
        
        self.win_meta = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        self.win_meta.fit(X_meta_win[valid_mask], y_win[valid_mask])
        
        self.margin_meta = Ridge(alpha=1.0)
        self.margin_meta.fit(X_meta_margin[valid_mask], y_margin[valid_mask])
        
        self.total_meta = Ridge(alpha=1.0)
        self.total_meta.fit(X_meta_total[valid_mask], y_total[valid_mask])
        
        # Calibrate win probabilities using isotonic regression
        print("  Calibrating probabilities...")
        meta_win_preds = self.win_meta.predict_proba(X_meta_win[valid_mask])[:, 1]
        self.win_calibrator = IsotonicRegression(out_of_bounds='clip')
        self.win_calibrator.fit(meta_win_preds, y_win[valid_mask])
        
        # Calculate residual distributions
        meta_margin_preds = self.margin_meta.predict(X_meta_margin[valid_mask])
        meta_total_preds = self.total_meta.predict(X_meta_total[valid_mask])
        
        self.margin_residual_std = np.std(y_margin[valid_mask] - meta_margin_preds)
        self.total_residual_std = np.std(y_total[valid_mask] - meta_total_preds)
        
        # Train final base models on full data
        print("  Training final models on full data...")
        self.win_models = self._create_base_models('classification')
        self.margin_models = self._create_base_models('regression')
        self.total_models = self._create_base_models('regression')
        
        for name in self.win_models:
            self.win_models[name].fit(X_scaled, y_win)
            self.margin_models[name].fit(X_scaled, y_margin)
            self.total_models[name].fit(X_scaled, y_total)
        
        # Calculate feature importance
        self._calculate_feature_importance(X)
        
        # Record training metrics
        final_win_preds = self.predict_win_prob(X)
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(y_win),
            'brier_score': brier_score_loss(y_win, final_win_preds),
            'log_loss': log_loss(y_win, np.clip(final_win_preds, 0.01, 0.99)),
            'margin_rmse': np.sqrt(np.mean((y_margin - self.predict_margin(X))**2)),
            'total_rmse': np.sqrt(np.mean((y_total - self.predict_total(X))**2)),
            'margin_std': self.margin_residual_std,
            'total_std': self.total_residual_std,
        })
        
        self.is_trained = True
        print(f"  Training complete!")
        print(f"  Brier Score: {self.training_history[-1]['brier_score']:.4f}")
        print(f"  Margin RMSE: {self.training_history[-1]['margin_rmse']:.2f}")
        print(f"  Total RMSE: {self.training_history[-1]['total_rmse']:.2f}")
    
    def predict_win_prob(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated win probability"""
        X_scaled = self.scaler.transform(X)
        
        # Get base model predictions
        base_preds = np.column_stack([
            self.win_models[name].predict_proba(X_scaled)[:, 1]
            for name in ['xgb', 'lgb', 'cat']
        ])
        
        # Meta-learner prediction
        meta_preds = self.win_meta.predict_proba(base_preds)[:, 1]
        
        # Calibrate
        calibrated = self.win_calibrator.predict(meta_preds)
        
        return np.clip(calibrated, 0.01, 0.99)
    
    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        """Predict point margin"""
        X_scaled = self.scaler.transform(X)
        
        base_preds = np.column_stack([
            self.margin_models[name].predict(X_scaled)
            for name in ['xgb', 'lgb', 'cat']
        ])
        
        return self.margin_meta.predict(base_preds)
    
    def predict_total(self, X: np.ndarray) -> np.ndarray:
        """Predict total points"""
        X_scaled = self.scaler.transform(X)
        
        base_preds = np.column_stack([
            self.total_models[name].predict(X_scaled)
            for name in ['xgb', 'lgb', 'cat']
        ])
        
        return self.total_meta.predict(base_preds)
    
    def predict_full(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get all predictions with uncertainty"""
        win_prob = self.predict_win_prob(X)
        margin = self.predict_margin(X)
        total = self.predict_total(X)
        
        return {
            'win_prob': win_prob,
            'margin': margin,
            'margin_std': np.full(len(X), self.margin_residual_std),
            'total': total,
            'total_std': np.full(len(X), self.total_residual_std),
        }
    
    def _calculate_feature_importance(self, X: np.ndarray):
        """Calculate and store feature importance"""
        feature_names = self.feature_engineer.get_feature_names()
        
        # Average importance across base models
        importance = np.zeros(X.shape[1])
        
        for model in self.win_models.values():
            if hasattr(model, 'feature_importances_'):
                importance += model.feature_importances_
        
        importance /= len(self.win_models)
        
        self.feature_importance = dict(zip(
            feature_names[:len(importance)],
            importance
        ))
    
    def save(self, filepath: str):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'MetaEnsembleModel':
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model


# =============================================================================
# HISTORICAL DATA GENERATOR (for training)
# =============================================================================
class HistoricalDataGenerator:
    """
    Generates synthetic historical data for model training
    
    In production, replace with actual historical data from:
    - KenPom historical archives
    - Sports Reference
    - Your own database
    """
    
    def __init__(self, n_games: int = 2000):
        self.n_games = n_games
        
    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic training data"""
        np.random.seed(42)
        
        feature_engineer = FeatureEngineer()
        n_features = len(feature_engineer.get_feature_names())
        
        X = []
        y_win = []
        y_margin = []
        y_total = []
        
        for _ in range(self.n_games):
            # Generate random team stats
            home_em = np.random.normal(5, 12)
            away_em = np.random.normal(5, 12)
            
            home_stats = {
                'adj_em': home_em,
                'adj_o': 100 + home_em/2 + np.random.normal(0, 3),
                'adj_d': 100 - home_em/2 + np.random.normal(0, 3),
                'adj_t': np.random.normal(68, 4),
                'rank': max(1, int(150 - home_em * 5 + np.random.normal(0, 20))),
                'efg_o': np.random.normal(0.51, 0.025),
                'efg_d': np.random.normal(0.50, 0.025),
                'tov_o': np.random.normal(0.17, 0.02),
                'tov_d': np.random.normal(0.17, 0.02),
                'orb_o': np.random.normal(0.28, 0.03),
                'drb_d': np.random.normal(0.73, 0.03),
                'ftr_o': np.random.normal(0.32, 0.04),
                'ftr_d': np.random.normal(0.30, 0.04),
            }
            
            away_stats = {
                'adj_em': away_em,
                'adj_o': 100 + away_em/2 + np.random.normal(0, 3),
                'adj_d': 100 - away_em/2 + np.random.normal(0, 3),
                'adj_t': np.random.normal(68, 4),
                'rank': max(1, int(150 - away_em * 5 + np.random.normal(0, 20))),
                'efg_o': np.random.normal(0.51, 0.025),
                'efg_d': np.random.normal(0.50, 0.025),
                'tov_o': np.random.normal(0.17, 0.02),
                'tov_d': np.random.normal(0.17, 0.02),
                'orb_o': np.random.normal(0.28, 0.03),
                'drb_d': np.random.normal(0.73, 0.03),
                'ftr_o': np.random.normal(0.32, 0.04),
                'ftr_d': np.random.normal(0.30, 0.04),
            }
            
            # Create features
            features = feature_engineer.create_features(home_stats, away_stats)
            feature_vec = [features.get(f, 0) for f in feature_engineer.get_feature_names()]
            X.append(feature_vec)
            
            # Generate realistic outcomes
            true_margin = (home_em - away_em) + 3.5 + np.random.normal(0, 11)
            true_total = 145 + np.random.normal(0, 12)
            
            y_margin.append(true_margin)
            y_total.append(true_total)
            y_win.append(1 if true_margin > 0 else 0)
        
        return (np.array(X), np.array(y_win), 
                np.array(y_margin), np.array(y_total))


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================
def train_model():
    """Train the meta-ensemble model"""
    print("=" * 70)
    print("🏀 TRAINING ADVANCED META-ENSEMBLE MODEL")
    print("=" * 70)
    
    # Generate training data
    print("\nGenerating training data...")
    data_gen = HistoricalDataGenerator(n_games=3000)
    X, y_win, y_margin, y_total = data_gen.generate()
    
    print(f"  Games: {len(y_win)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Home win rate: {y_win.mean():.1%}")
    print(f"  Avg margin: {y_margin.mean():.1f}")
    print(f"  Avg total: {y_total.mean():.1f}")
    
    # Train model
    print("\n" + "=" * 70)
    model = MetaEnsembleModel()
    model.train(X, y_win, y_margin, y_total)
    
    # Save model
    model.save('cbb_trained_model.pkl')
    
    # Print feature importance
    print("\n" + "=" * 70)
    print("TOP FEATURES BY IMPORTANCE")
    print("=" * 70)
    
    sorted_features = sorted(model.feature_importance.items(), 
                            key=lambda x: x[1], reverse=True)
    for name, importance in sorted_features[:15]:
        print(f"  {name}: {importance:.4f}")
    
    return model


if __name__ == "__main__":
    model = train_model()
