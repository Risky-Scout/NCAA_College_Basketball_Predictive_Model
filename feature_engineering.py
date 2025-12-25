#!/usr/bin/env python3
"""
================================================================================
NCAA College Basketball Predictive Model - Feature Engineering
================================================================================

Complete feature engineering with syndicate-grade market features.

================================================================================
"""

import numpy as np
from typing import Dict, List, Optional

from config import (
    CORE_FEATURES, FOUR_FACTORS_FEATURES, RANKING_FEATURES,
    SITUATIONAL_FEATURES, MARKET_FEATURES, SYNDICATE_FEATURES,
    INTERACTION_FEATURES, ADVANCED_FEATURES, ALL_FEATURES,
    game_config
)


class FeatureEngineer:
    """
    Complete feature engineering with enhanced syndicate features.

    Combines:
    - Core efficiency metrics
    - Four factors
    - Situational factors
    - Market features (basic + syndicate)
    - Interactions
    """

    def __init__(self):
        self.feature_names = ALL_FEATURES

    def create_features(self,
                        home_stats: Dict,
                        away_stats: Dict,
                        context: Dict = None,
                        market: Dict = None,
                        syndicate: Dict = None) -> Dict:
        """
        Create comprehensive feature set.

        Args:
            home_stats: Home team statistics
            away_stats: Away team statistics
            context: Situational context (rest, travel, etc.)
            market: Basic market data (spread, total, movement)
            syndicate: Enhanced syndicate data (steam, RLM, injuries)

        Returns:
            Dictionary of all features
        """
        features = {}

        # Initialize defaults
        context = context or {}
        market = market or {}
        syndicate = syndicate or {}

        # =====================================================================
        # CORE EFFICIENCY METRICS
        # =====================================================================
        features['adj_em_diff'] = home_stats['adj_em'] - away_stats['adj_em']
        features['adj_o_diff'] = home_stats['adj_o'] - away_stats['adj_o']
        features['adj_d_diff'] = away_stats['adj_d'] - home_stats['adj_d']  # Lower is better for defense
        features['tempo_avg'] = (home_stats['adj_t'] + away_stats['adj_t']) / 2
        features['tempo_diff'] = home_stats['adj_t'] - away_stats['adj_t']

        # =====================================================================
        # FOUR FACTORS
        # =====================================================================
        # eFG% differential (shooting efficiency)
        features['efg_diff'] = (
            (home_stats.get('efg_o', 0.50) - away_stats.get('efg_d', 0.50)) -
            (away_stats.get('efg_o', 0.50) - home_stats.get('efg_d', 0.50))
        )

        # Turnover differential
        features['tov_diff'] = (
            (away_stats.get('tov_d', 0.17) - home_stats.get('tov_o', 0.17)) +
            (home_stats.get('tov_d', 0.17) - away_stats.get('tov_o', 0.17))
        )

        # Rebounding differential
        features['orb_diff'] = (
            (home_stats.get('orb_o', 0.28) - (1 - away_stats.get('drb_d', 0.72))) +
            (home_stats.get('drb_d', 0.72) - (1 - away_stats.get('orb_o', 0.28)))
        )

        # Free throw rate differential
        features['ftr_diff'] = (
            (home_stats.get('ftr_o', 0.30) - away_stats.get('ftr_d', 0.30)) -
            (away_stats.get('ftr_o', 0.30) - home_stats.get('ftr_d', 0.30))
        )

        # =====================================================================
        # RANKINGS
        # =====================================================================
        home_rank = home_stats.get('rank', 150)
        away_rank = away_stats.get('rank', 150)

        features['rank_diff'] = away_rank - home_rank  # Positive = home is better
        features['rank_product'] = (home_rank * away_rank) / 10000  # Quality of matchup
        features['elite_matchup'] = float(home_rank <= 25 and away_rank <= 25)
        features['mismatch'] = float(abs(features['rank_diff']) > 100)

        # =====================================================================
        # SITUATIONAL FACTORS
        # =====================================================================
        home_rest = context.get('home_rest', 3)
        away_rest = context.get('away_rest', 3)

        features['rest_diff'] = home_rest - away_rest
        features['b2b_home'] = float(home_rest <= 1)
        features['b2b_away'] = float(away_rest <= 1)
        features['conf_game'] = float(context.get('conference_game', 0))
        features['neutral_site'] = float(context.get('neutral_site', 0))
        features['rivalry'] = float(context.get('rivalry', 0))
        features['travel_factor'] = context.get('away_travel_miles', 500) / 1000

        # =====================================================================
        # BASIC MARKET FEATURES
        # =====================================================================
        features['spread'] = market.get('spread', 0) or 0
        features['total_line'] = market.get('total', 145) or 145
        features['opening_spread'] = market.get('opening_spread', features['spread']) or features['spread']
        features['line_movement'] = features['spread'] - features['opening_spread']
        features['steam_move'] = float(market.get('steam_move', 0))

        # Public betting (centered at 0)
        public_pct = market.get('public_pct', 50)
        features['public_pct'] = (public_pct / 100 - 0.5) if public_pct else 0

        features['rlm'] = float(market.get('reverse_line_movement', 0))

        # Basic sharp signal
        if abs(features['line_movement']) > 0.5 and abs(features['public_pct']) > 0.1:
            features['sharp_signal'] = features['line_movement'] * (-features['public_pct'])
        else:
            features['sharp_signal'] = market.get('sharp_signal', 0)

        # =====================================================================
        # ENHANCED SYNDICATE FEATURES
        # =====================================================================
        # Velocity-based movement
        features['spread_velocity'] = syndicate.get('spread_velocity', 0)
        features['spread_acceleration'] = syndicate.get('spread_acceleration', 0)

        # Sharp book divergence
        features['sharp_divergence'] = syndicate.get('sharp_divergence', 0)
        features['pinnacle_divergence'] = syndicate.get('pinnacle_divergence', 0)

        # Enhanced steam detection
        features['steam_confidence'] = syndicate.get('steam_confidence', 0)
        features['steam_home'] = syndicate.get('steam_home', 0)
        features['steam_away'] = syndicate.get('steam_away', 0)

        # Enhanced RLM
        features['rlm_strength'] = syndicate.get('rlm_strength', 0)
        features['rlm_signal'] = syndicate.get('rlm_signal', 0)

        # Public betting confidence
        features['public_confidence'] = syndicate.get('public_confidence', 0.5)

        # Injury adjustments
        features['injury_spread_adj'] = syndicate.get('injury_spread_adj', 0)
        features['injury_total_adj'] = syndicate.get('injury_total_adj', 0)
        features['has_injury_edge'] = syndicate.get('has_injury_edge', 0)

        # =====================================================================
        # INTERACTIONS
        # =====================================================================
        features['em_x_tempo'] = features['adj_em_diff'] * features['tempo_avg'] / 70
        features['efg_x_pace'] = features['efg_diff'] * features['tempo_avg'] / 70
        features['rank_x_rest'] = features['rank_diff'] * features['rest_diff'] / 100

        # =====================================================================
        # ADVANCED STATS
        # =====================================================================
        features['sos_diff'] = home_stats.get('sos', 0) - away_stats.get('sos', 0)
        features['luck_diff'] = home_stats.get('luck', 0) - away_stats.get('luck', 0)

        return features

    def create_feature_vector(self, features: Dict) -> List[float]:
        """Convert feature dict to ordered vector for model input"""
        return [features.get(f, 0) for f in self.feature_names]


class EnhancedFeatureEngineer(FeatureEngineer):
    """
    Extended feature engineer with syndicate data integration.

    Automatically pulls from SyndicateDataManager when available.
    """

    def __init__(self, syndicate_manager=None):
        super().__init__()
        self.syndicate_manager = syndicate_manager

    def create_features_with_syndicate(self,
                                       home_stats: Dict,
                                       away_stats: Dict,
                                       game_data: Dict,
                                       context: Dict = None) -> Dict:
        """
        Create features with automatic syndicate data integration.

        Args:
            home_stats: Home team statistics
            away_stats: Away team statistics
            game_data: Raw game data from odds API
            context: Situational context

        Returns:
            Complete feature dictionary including syndicate features
        """
        # Get syndicate analysis if manager available
        if self.syndicate_manager:
            analysis = self.syndicate_manager.get_full_game_analysis(game_data)
            syndicate = self.syndicate_manager.get_enhanced_features(analysis)

            market = {
                'spread': analysis.get('spread'),
                'total': analysis.get('total'),
                'opening_spread': analysis.get('opening_spread'),
                'line_movement': analysis.get('spread_movement', 0),
                'steam_move': analysis.get('has_steam', 0),
                'public_pct': analysis.get('public_home_pct', 50),
                'reverse_line_movement': analysis.get('is_rlm', False),
                'sharp_signal': syndicate.get('sharp_signal', 0),
            }
        else:
            syndicate = {}
            market = {
                'spread': game_data.get('spread'),
                'total': game_data.get('total'),
            }

        return self.create_features(home_stats, away_stats, context, market, syndicate)


# =============================================================================
# FEATURE IMPORTANCE TRACKING
# =============================================================================
class FeatureImportanceTracker:
    """Track and analyze feature importance over time"""

    def __init__(self):
        self.importance_history = []

    def record(self, importance_dict: Dict):
        """Record feature importance from training"""
        self.importance_history.append({
            'timestamp': str(np.datetime64('now')),
            'importance': importance_dict,
        })

    def get_top_features(self, n: int = 10) -> List[tuple]:
        """Get top N features by average importance"""
        if not self.importance_history:
            return []

        # Average across history
        all_features = set()
        for h in self.importance_history:
            all_features.update(h['importance'].keys())

        avg_importance = {}
        for feature in all_features:
            values = [
                h['importance'].get(feature, 0)
                for h in self.importance_history
            ]
            avg_importance[feature] = np.mean(values)

        sorted_features = sorted(
            avg_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_features[:n]

    def get_stable_features(self, min_importance: float = 0.01) -> List[str]:
        """Get features that are consistently important"""
        if len(self.importance_history) < 3:
            return []

        stable = []
        for feature in ALL_FEATURES:
            values = [
                h['importance'].get(feature, 0)
                for h in self.importance_history[-10:]  # Last 10 trainings
            ]
            if np.mean(values) > min_importance and np.std(values) < 0.02:
                stable.append(feature)

        return stable
