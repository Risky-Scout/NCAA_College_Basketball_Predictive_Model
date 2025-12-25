#!/usr/bin/env python3
"""
================================================================================
NCAA College Basketball Predictive Model - Training Script
================================================================================

SYNDICATE-GRADE MODEL TRAINING

Enhanced with:
- Full syndicate feature set (47 features)
- Time-series cross-validation with leakage protection
- Exponential recency weighting
- Walk-forward validation

This script is run daily by GitHub Actions at 7:00 AM EST.

================================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import model_config, DATA_DIR, MODELS_DIR, ALL_FEATURES
from data_pipeline import TeamDatabase, HistoricalDataManager
from feature_engineering import FeatureEngineer
from meta_ensemble import MetaEnsemble


def generate_training_data(n_games: int = 5000) -> tuple:
    """
    Generate training data with full syndicate feature set.

    In production, this would load from historical database.
    For now, generates high-quality synthetic data.

    CRITICAL: All features are computed with proper temporal ordering
    to prevent data leakage.
    """
    print("\nGenerating training data with syndicate features...")
    print(f"  Total features: {len(ALL_FEATURES)}")

    np.random.seed(42)

    team_db = TeamDatabase()
    feature_eng = FeatureEngineer()

    X, y_win, y_margin, y_total = [], [], [], []
    dates = []  # For temporal ordering

    # Generate games with realistic distributions
    teams = list(team_db.teams.keys())

    for i in range(n_games):
        # Random matchup
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])

        home_stats = team_db.get(home_team)
        away_stats = team_db.get(away_team)

        if not home_stats or not away_stats:
            continue

        # Generate game date (for temporal ordering)
        days_ago = np.random.randint(0, 365 * 3)  # 3 seasons
        game_date = datetime.now() - timedelta(days=days_ago)
        dates.append(game_date)

        # Situational context
        home_rest = np.random.choice([1, 2, 3, 4, 5, 6, 7], p=[0.05, 0.15, 0.35, 0.25, 0.10, 0.05, 0.05])
        away_rest = np.random.choice([1, 2, 3, 4, 5, 6, 7], p=[0.05, 0.15, 0.35, 0.25, 0.10, 0.05, 0.05])
        neutral_site = np.random.choice([0, 1], p=[0.92, 0.08])
        conf_game = np.random.choice([0, 1], p=[0.4, 0.6])

        context = {
            'home_rest': home_rest,
            'away_rest': away_rest,
            'neutral_site': neutral_site,
            'conference_game': conf_game,
            'rivalry': np.random.choice([0, 1], p=[0.90, 0.10]),
            'away_travel_miles': np.random.uniform(100, 2000),
        }

        # True margin (what we're trying to predict)
        hca = 3.5 * (1 - neutral_site)
        rest_adj = (home_rest - away_rest) * 0.3
        true_margin = (home_stats['adj_em'] - away_stats['adj_em']) + hca + rest_adj

        # Market simulation (imperfect estimate of true margin)
        market_spread = true_margin + np.random.normal(0, 2)

        # Simulate syndicate signals
        has_steam = np.random.choice([0, 1], p=[0.92, 0.08])
        has_rlm = np.random.choice([0, 1], p=[0.88, 0.12])
        has_injury = np.random.choice([0, 1], p=[0.85, 0.15])

        market = {
            'spread': round(market_spread * 2) / 2,
            'total': 145 + np.random.normal(0, 5),
            'opening_spread': market_spread + np.random.normal(0, 1.5),
            'line_movement': np.random.normal(0, 1),
            'steam_move': has_steam,
            'public_pct': np.clip(50 + (market_spread * 2) + np.random.normal(0, 10), 20, 80),
            'reverse_line_movement': has_rlm,
            'sharp_signal': np.random.normal(0, 0.5) if has_rlm else 0,
        }

        # Simulate syndicate features
        syndicate = {
            'spread_velocity': np.random.normal(0, 0.1),
            'spread_acceleration': np.random.normal(0, 0.05),
            'sharp_divergence': np.random.normal(0, 0.3),
            'pinnacle_divergence': np.random.normal(0, 0.2),
            'steam_move': has_steam,
            'steam_confidence': np.random.uniform(0.6, 0.95) if has_steam else 0,
            'steam_home': 1 if has_steam and np.random.random() > 0.5 else 0,
            'steam_away': 1 if has_steam and np.random.random() <= 0.5 else 0,
            'rlm_strength': np.random.uniform(0.2, 0.8) if has_rlm else 0,
            'rlm_signal': np.random.uniform(0.1, 0.4) if has_rlm else 0,
            'public_confidence': np.random.uniform(0.4, 0.7),
            'injury_spread_adj': np.random.normal(0, 1.5) if has_injury else 0,
            'injury_total_adj': np.random.normal(0, 1.0) if has_injury else 0,
            'has_injury_edge': has_injury,
        }

        # Create features
        features = feature_eng.create_features(home_stats, away_stats, context, market, syndicate)
        feature_vec = [features.get(f, 0) for f in feature_eng.feature_names]
        X.append(feature_vec)

        # Generate outcomes with realistic variance
        # Apply syndicate adjustments to true outcome
        injury_effect = syndicate['injury_spread_adj'] * 0.5  # Partial realization
        actual_margin = true_margin + injury_effect + np.random.normal(0, 11)
        expected_total = 145 + (home_stats['adj_t'] + away_stats['adj_t'] - 136) * 0.5
        actual_total = expected_total + syndicate['injury_total_adj'] * 0.5 + np.random.normal(0, 12)

        y_margin.append(actual_margin)
        y_total.append(actual_total)
        y_win.append(1 if actual_margin > 0 else 0)

    # Sort by date (temporal ordering for proper CV)
    sort_idx = np.argsort(dates)
    X = np.array(X)[sort_idx]
    y_win = np.array(y_win)[sort_idx]
    y_margin = np.array(y_margin)[sort_idx]
    y_total = np.array(y_total)[sort_idx]

    print(f"  Generated {len(y_win)} games")
    print(f"  Features per game: {X.shape[1]}")
    print(f"  Home win rate: {y_win.mean():.1%}")
    print(f"  Mean margin: {y_margin.mean():.1f}")
    print(f"  Mean total: {y_total.mean():.1f}")

    return X, y_win, y_margin, y_total, feature_eng.feature_names


def calculate_recency_weights(n_samples: int, decay: float = 2.0) -> np.ndarray:
    """
    Calculate exponential recency weights.
    
    More recent games get higher weight in training.
    
    Args:
        n_samples: Number of samples
        decay: Decay factor (higher = more weight on recent)
    
    Returns:
        Normalized weights array
    """
    weights = np.exp(np.linspace(-decay, 0, n_samples))
    weights /= weights.sum()  # Normalize
    weights *= n_samples  # Scale back up
    return weights


def train():
    """Main training function"""
    print("=" * 60)
    print("NCAA CBB MODEL - DAILY TRAINING")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Generate training data
    X, y_win, y_margin, y_total, feature_names = generate_training_data(n_games=5000)
    
    # Calculate recency weights
    print("\nCalculating recency weights...")
    weights = calculate_recency_weights(len(y_win), decay=model_config.recency_decay)
    print(f"  Oldest sample weight: {weights[0]:.4f}")
    print(f"  Newest sample weight: {weights[-1]:.4f}")
    
    # Train model
    model = MetaEnsemble(config=model_config)
    metrics = model.train(
        X, y_win, y_margin, y_total,
        sample_weights=weights,
        feature_names=feature_names
    )
    
    # Print top features
    print("\nTop 10 Features by Importance:")
    for name, importance in model.get_top_features(10):
        print(f"  {name}: {importance:.4f}")
    
    # Save model
    model.save()
    
    # Save training metrics
    metrics_path = DATA_DIR / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Training metrics saved to {metrics_path}")
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = train()
