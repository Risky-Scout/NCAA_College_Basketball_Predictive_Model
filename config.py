#!/usr/bin/env python3
"""
================================================================================
NCAA College Basketball Predictive Model - Configuration
================================================================================

Centralized configuration for the betting syndicate model.

================================================================================
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# =============================================================================
# DIRECTORY PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PREDICTIONS_DIR = BASE_DIR / "predictions"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
PREDICTIONS_DIR.mkdir(exist_ok=True)


# =============================================================================
# API CONFIGURATION
# =============================================================================
ODDS_API_KEY = "272be842201ff50bdfee622541e2d3ee925afac17b3126e93b81b4d58e0e6b62"


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
@dataclass
class ModelConfig:
    """Training and model hyperparameters"""
    # Base model settings
    n_estimators: int = 200
    max_depth: int = 5
    learning_rate: float = 0.03

    # Training settings
    cv_splits: int = 5
    recency_decay: float = 2.0  # Exponential decay for sample weighting
    min_samples: int = 100

    # Regularization
    l2_alpha: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8

    # Calibration
    calibrate_probabilities: bool = True

    # Variance estimation
    default_margin_std: float = 10.5
    default_total_std: float = 11.0


model_config = ModelConfig()


# =============================================================================
# BETTING CONFIGURATION
# =============================================================================
@dataclass
class BettingConfig:
    """Betting strategy parameters"""
    # Edge thresholds
    min_edge: float = 0.025  # 2.5% minimum edge
    high_confidence_edge: float = 0.07  # 7%+ for HIGH confidence
    medium_confidence_edge: float = 0.04  # 4%+ for MEDIUM confidence

    # Kelly sizing
    kelly_fraction: float = 0.25  # Use 25% of full Kelly
    max_kelly: float = 0.05  # Cap at 5% of bankroll per bet

    # Risk management
    max_daily_exposure: float = 0.20  # 20% max daily exposure
    max_correlated_bets: int = 3
    stop_loss_daily: float = -0.10  # -10% daily stop loss

    # Value bet requirements
    min_ev_percent: float = 2.0  # Minimum 2% expected value


betting_config = BettingConfig()


# =============================================================================
# GAME CONFIGURATION
# =============================================================================
@dataclass
class GameConfig:
    """Game-level constants"""
    # Home court advantage
    base_hca: float = 3.5
    neutral_site_hca: float = 0.0

    # Rest impact (points per day differential)
    rest_impact_per_day: float = 0.3
    b2b_penalty: float = 1.5

    # Travel impact
    travel_impact_per_1000_miles: float = 0.5

    # Conference game adjustment
    conference_game_margin_adj: float = -1.5

    # Rivalry adjustment
    rivalry_margin_adj: float = -2.0


game_config = GameConfig()


# =============================================================================
# SYNDICATE DATA CONFIGURATION (NEW)
# =============================================================================
@dataclass
class SyndicateConfig:
    """Configuration for syndicate-grade data features"""
    # Sharp book identification
    sharp_books: List[str] = field(default_factory=lambda: [
        'pinnacle', 'circa', 'bookmaker', 'betcris', 'heritage'
    ])

    # Steam detection thresholds
    steam_velocity_threshold: float = 0.25  # Points per minute
    steam_magnitude_threshold: float = 0.5  # Points minimum movement
    steam_book_count: int = 2  # Minimum books moving together

    # RLM thresholds
    rlm_public_threshold: float = 55.0  # Public % for RLM signal
    rlm_movement_threshold: float = 0.25  # Minimum movement for RLM

    # Injury impact weights
    starter_impact_base: float = 2.5
    rotation_impact_base: float = 0.8

    # Cache durations (minutes)
    injury_cache_duration: int = 30
    odds_cache_duration: int = 5

    # Line movement tracking
    line_history_max_hours: int = 24

    # Public betting estimation
    public_darlings_bonus: int = 5  # Extra public % for popular teams


syndicate_config = SyndicateConfig()


# =============================================================================
# FEATURE SETS
# =============================================================================
CORE_FEATURES = [
    'adj_em_diff', 'adj_o_diff', 'adj_d_diff', 'tempo_avg', 'tempo_diff',
]

FOUR_FACTORS_FEATURES = [
    'efg_diff', 'tov_diff', 'orb_diff', 'ftr_diff',
]

RANKING_FEATURES = [
    'rank_diff', 'rank_product', 'elite_matchup', 'mismatch',
]

SITUATIONAL_FEATURES = [
    'rest_diff', 'b2b_home', 'b2b_away', 'conf_game', 'neutral_site',
    'rivalry', 'travel_factor',
]

MARKET_FEATURES = [
    'spread', 'total_line', 'opening_spread', 'line_movement',
    'steam_move', 'public_pct', 'rlm', 'sharp_signal',
]

# NEW: Enhanced syndicate features
SYNDICATE_FEATURES = [
    'spread_velocity', 'spread_acceleration', 'sharp_divergence',
    'pinnacle_divergence', 'steam_confidence', 'steam_home', 'steam_away',
    'rlm_strength', 'rlm_signal', 'public_confidence',
    'injury_spread_adj', 'injury_total_adj', 'has_injury_edge',
]

INTERACTION_FEATURES = [
    'em_x_tempo', 'efg_x_pace', 'rank_x_rest',
]

ADVANCED_FEATURES = [
    'sos_diff', 'luck_diff',
]

# Full feature list (order matters for model input)
ALL_FEATURES = (
    CORE_FEATURES +
    FOUR_FACTORS_FEATURES +
    RANKING_FEATURES +
    SITUATIONAL_FEATURES +
    MARKET_FEATURES +
    SYNDICATE_FEATURES +
    INTERACTION_FEATURES +
    ADVANCED_FEATURES
)
