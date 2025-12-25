#!/usr/bin/env python3
"""
================================================================================
NCAA College Basketball Predictive Model - Prediction Script
================================================================================

SYNDICATE-GRADE DAILY PREDICTIONS

Enhanced with:
- Real-time injury tracking
- Multi-book sharp line detection
- Time-weighted line movement (velocity/acceleration)
- Steam move detection
- Reverse line movement magnitude scoring
- Sharp money flow classification

This script is run daily by GitHub Actions at 4:30 PM EST.

================================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from scipy import stats
import json
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import betting_config, game_config, MODELS_DIR, PREDICTIONS_DIR
from data_pipeline import TeamDatabase, OddsAPIClient
from feature_engineering import FeatureEngineer, EnhancedFeatureEngineer
from meta_ensemble import MetaEnsemble

# Import syndicate data module
try:
    from syndicate_data import SyndicateDataManager
    SYNDICATE_ENABLED = True
except ImportError:
    SYNDICATE_ENABLED = False
    print("Warning: Syndicate data module not available")


def american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability"""
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds"""
    return odds / 100 + 1 if odds > 0 else 100 / abs(odds) + 1


def spread_probability(pred_margin: float, margin_std: float, spread: float) -> float:
    """P(home covers spread)"""
    z = (pred_margin - (-spread)) / margin_std
    return float(stats.norm.cdf(z))


def total_probability(pred_total: float, total_std: float, line: float, over: bool = True) -> float:
    """P(over/under)"""
    z = (line - pred_total) / total_std
    p_under = float(stats.norm.cdf(z))
    return (1 - p_under) if over else p_under


def analyze_betting_value(pred: dict, odds: dict) -> list:
    """Analyze all betting opportunities for a game."""
    value_bets = []
    
    # MONEYLINE
    if odds.get('ml_home') and odds.get('ml_away'):
        h_imp = american_to_implied_prob(odds['ml_home'])
        a_imp = american_to_implied_prob(odds['ml_away'])
        fair_h = h_imp / (h_imp + a_imp)
        fair_a = a_imp / (h_imp + a_imp)
        
        # Home ML
        edge_h = pred['win_prob'] - fair_h
        dec_h = american_to_decimal(odds['ml_home'])
        ev_h = pred['win_prob'] * (dec_h - 1) - (1 - pred['win_prob'])
        kelly_h = max(0, ev_h / (dec_h - 1) * betting_config.kelly_fraction) if ev_h > 0 else 0
        
        if edge_h > betting_config.min_edge and ev_h > 0:
            conf = 'HIGH' if edge_h > betting_config.high_confidence_edge else \
                   ('MEDIUM' if edge_h > betting_config.medium_confidence_edge else 'LOW')
            value_bets.append({
                'bet_type': 'ML', 'side': 'HOME', 'bet': f"{odds['home']} ML",
                'odds': odds['ml_home'], 'model_prob': round(pred['win_prob'], 4),
                'fair_prob': round(fair_h, 4), 'edge': round(edge_h, 4),
                'ev_pct': round(ev_h * 100, 2), 'kelly': round(min(kelly_h, betting_config.max_kelly), 4),
                'confidence': conf,
            })
        
        # Away ML
        away_prob = 1 - pred['win_prob']
        edge_a = away_prob - fair_a
        dec_a = american_to_decimal(odds['ml_away'])
        ev_a = away_prob * (dec_a - 1) - (1 - away_prob)
        kelly_a = max(0, ev_a / (dec_a - 1) * betting_config.kelly_fraction) if ev_a > 0 else 0
        
        if edge_a > betting_config.min_edge and ev_a > 0:
            conf = 'HIGH' if edge_a > betting_config.high_confidence_edge else \
                   ('MEDIUM' if edge_a > betting_config.medium_confidence_edge else 'LOW')
            value_bets.append({
                'bet_type': 'ML', 'side': 'AWAY', 'bet': f"{odds['away']} ML",
                'odds': odds['ml_away'], 'model_prob': round(away_prob, 4),
                'fair_prob': round(fair_a, 4), 'edge': round(edge_a, 4),
                'ev_pct': round(ev_a * 100, 2), 'kelly': round(min(kelly_a, betting_config.max_kelly), 4),
                'confidence': conf,
            })
    
    # SPREAD
    if odds.get('spread') is not None:
        h_cov = spread_probability(pred['margin'], pred['margin_std'], odds['spread'])
        h_juice_imp = american_to_implied_prob(odds.get('spread_juice_home', -110))
        a_juice_imp = american_to_implied_prob(odds.get('spread_juice_away', -110))
        fair_h = h_juice_imp / (h_juice_imp + a_juice_imp)
        fair_a = a_juice_imp / (h_juice_imp + a_juice_imp)
        
        # Home spread
        edge_h = h_cov - fair_h
        dec_h = american_to_decimal(odds.get('spread_juice_home', -110))
        ev_h = h_cov * (dec_h - 1) - (1 - h_cov)
        kelly_h = max(0, ev_h / (dec_h - 1) * betting_config.kelly_fraction) if ev_h > 0 else 0
        
        if edge_h > betting_config.min_edge and ev_h > 0:
            conf = 'HIGH' if edge_h > betting_config.high_confidence_edge else \
                   ('MEDIUM' if edge_h > betting_config.medium_confidence_edge else 'LOW')
            value_bets.append({
                'bet_type': 'SPREAD', 'side': 'HOME', 'bet': f"{odds['home']} {odds['spread']:+.1f}",
                'odds': odds.get('spread_juice_home', -110), 'line': odds['spread'],
                'model_prob': round(h_cov, 4), 'fair_prob': round(fair_h, 4),
                'edge': round(edge_h, 4), 'ev_pct': round(ev_h * 100, 2),
                'kelly': round(min(kelly_h, betting_config.max_kelly), 4), 'confidence': conf,
            })
        
        # Away spread
        edge_a = (1 - h_cov) - fair_a
        dec_a = american_to_decimal(odds.get('spread_juice_away', -110))
        ev_a = (1 - h_cov) * (dec_a - 1) - h_cov
        kelly_a = max(0, ev_a / (dec_a - 1) * betting_config.kelly_fraction) if ev_a > 0 else 0
        
        if edge_a > betting_config.min_edge and ev_a > 0:
            conf = 'HIGH' if edge_a > betting_config.high_confidence_edge else \
                   ('MEDIUM' if edge_a > betting_config.medium_confidence_edge else 'LOW')
            value_bets.append({
                'bet_type': 'SPREAD', 'side': 'AWAY', 'bet': f"{odds['away']} {-odds['spread']:+.1f}",
                'odds': odds.get('spread_juice_away', -110), 'line': -odds['spread'],
                'model_prob': round(1 - h_cov, 4), 'fair_prob': round(fair_a, 4),
                'edge': round(edge_a, 4), 'ev_pct': round(ev_a * 100, 2),
                'kelly': round(min(kelly_a, betting_config.max_kelly), 4), 'confidence': conf,
            })
    
    # TOTAL
    if odds.get('total') is not None:
        ov_prob = total_probability(pred['total'], pred['total_std'], odds['total'], over=True)
        o_juice_imp = american_to_implied_prob(odds.get('over_juice', -110))
        u_juice_imp = american_to_implied_prob(odds.get('under_juice', -110))
        fair_o = o_juice_imp / (o_juice_imp + u_juice_imp)
        fair_u = u_juice_imp / (o_juice_imp + u_juice_imp)
        
        # Over
        edge_o = ov_prob - fair_o
        dec_o = american_to_decimal(odds.get('over_juice', -110))
        ev_o = ov_prob * (dec_o - 1) - (1 - ov_prob)
        kelly_o = max(0, ev_o / (dec_o - 1) * betting_config.kelly_fraction) if ev_o > 0 else 0
        
        if edge_o > betting_config.min_edge and ev_o > 0:
            conf = 'HIGH' if edge_o > betting_config.high_confidence_edge else \
                   ('MEDIUM' if edge_o > betting_config.medium_confidence_edge else 'LOW')
            value_bets.append({
                'bet_type': 'TOTAL', 'side': 'OVER', 'bet': f"Over {odds['total']}",
                'odds': odds.get('over_juice', -110), 'line': odds['total'],
                'model_prob': round(ov_prob, 4), 'fair_prob': round(fair_o, 4),
                'edge': round(edge_o, 4), 'ev_pct': round(ev_o * 100, 2),
                'kelly': round(min(kelly_o, betting_config.max_kelly), 4), 'confidence': conf,
            })
        
        # Under
        edge_u = (1 - ov_prob) - fair_u
        dec_u = american_to_decimal(odds.get('under_juice', -110))
        ev_u = (1 - ov_prob) * (dec_u - 1) - ov_prob
        kelly_u = max(0, ev_u / (dec_u - 1) * betting_config.kelly_fraction) if ev_u > 0 else 0
        
        if edge_u > betting_config.min_edge and ev_u > 0:
            conf = 'HIGH' if edge_u > betting_config.high_confidence_edge else \
                   ('MEDIUM' if edge_u > betting_config.medium_confidence_edge else 'LOW')
            value_bets.append({
                'bet_type': 'TOTAL', 'side': 'UNDER', 'bet': f"Under {odds['total']}",
                'odds': odds.get('under_juice', -110), 'line': odds['total'],
                'model_prob': round(1 - ov_prob, 4), 'fair_prob': round(fair_u, 4),
                'edge': round(edge_u, 4), 'ev_pct': round(ev_u * 100, 2),
                'kelly': round(min(kelly_u, betting_config.max_kelly), 4), 'confidence': conf,
            })
    
    return value_bets


def get_demo_games() -> list:
    """Demo games for testing"""
    return [
        {'game_id': 'd1', 'home': 'Auburn', 'away': 'Ohio State', 'commence_time': '2024-12-24T19:00:00Z',
         'ml_home': -450, 'ml_away': 350, 'spread': -10.5, 'spread_juice_home': -110, 'spread_juice_away': -110,
         'total': 148.5, 'over_juice': -110, 'under_juice': -110},
        {'game_id': 'd2', 'home': 'Duke', 'away': 'NC State', 'commence_time': '2024-12-24T19:00:00Z',
         'ml_home': -650, 'ml_away': 475, 'spread': -13.5, 'spread_juice_home': -108, 'spread_juice_away': -112,
         'total': 152.5, 'over_juice': -105, 'under_juice': -115},
        {'game_id': 'd3', 'home': 'Kansas', 'away': 'Missouri', 'commence_time': '2024-12-24T21:00:00Z',
         'ml_home': -700, 'ml_away': 500, 'spread': -14.5, 'spread_juice_home': -110, 'spread_juice_away': -110,
         'total': 150.5, 'over_juice': -110, 'under_juice': -110},
        {'game_id': 'd4', 'home': 'Florida', 'away': 'North Carolina', 'commence_time': '2024-12-24T18:00:00Z',
         'ml_home': -225, 'ml_away': 185, 'spread': -5.5, 'spread_juice_home': -110, 'spread_juice_away': -110,
         'total': 155.5, 'over_juice': -110, 'under_juice': -110},
        {'game_id': 'd5', 'home': 'Oregon', 'away': 'UCLA', 'commence_time': '2024-12-24T22:00:00Z',
         'ml_home': 125, 'ml_away': -145, 'spread': 2.5, 'spread_juice_home': -110, 'spread_juice_away': -110,
         'total': 146.5, 'over_juice': -110, 'under_juice': -110},
    ]


def predict():
    """Main prediction function with syndicate-grade enhancements"""
    today = datetime.now().strftime('%Y-%m-%d')

    print("=" * 70)
    print("NCAA CBB MODEL - SYNDICATE-GRADE DAILY PREDICTIONS")
    print("=" * 70)
    print(f"Date: {today}")
    print(f"Syndicate Data: {'ENABLED' if SYNDICATE_ENABLED else 'DISABLED'}")

    # Load model
    print("\nLoading model...")
    try:
        model = MetaEnsemble.load()
    except FileNotFoundError:
        print("No trained model found. Running training first...")
        from train import train
        model, _ = train()

    team_db = TeamDatabase()

    # Initialize syndicate data manager if available
    syndicate_manager = None
    if SYNDICATE_ENABLED:
        print("\nInitializing syndicate data sources...")
        syndicate_manager = SyndicateDataManager()
        syndicate_manager.refresh_all_data()
        print("  - Injury tracker: ACTIVE")
        print("  - Line movement tracker: ACTIVE")
        print("  - Multi-book aggregator: ACTIVE")
        print("  - RLM analyzer: ACTIVE")

    # Use enhanced feature engineer if syndicate data available
    if syndicate_manager:
        feature_eng = EnhancedFeatureEngineer(syndicate_manager)
    else:
        feature_eng = FeatureEngineer()

    # Fetch games
    print("\nFetching today's games...")
    odds_client = OddsAPIClient()
    raw_games = odds_client.fetch_games()

    if not raw_games:
        print("API unavailable - using demo games")
        games = get_demo_games()
        raw_games = []
    else:
        games = [odds_client.parse_game(g) for g in raw_games]

    print(f"Found {len(games)} games")

    # Process games
    output = {
        'generated_at': datetime.now().isoformat(),
        'model_version': model.version,
        'syndicate_enabled': SYNDICATE_ENABLED,
        'n_games': len(games),
        'games': [],
        'summary': {
            'total_value_bets': 0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0,
            'steam_moves_detected': 0,
            'rlm_signals': 0,
            'injury_edges': 0,
        }
    }

    all_value_bets = []

    for i, odds in enumerate(games):
        home_stats = team_db.get(odds['home'])
        away_stats = team_db.get(odds['away'])

        if not home_stats or not away_stats:
            continue

        context = {'home_rest': 3, 'away_rest': 3, 'neutral_site': 0, 'conference_game': 0}

        # Get syndicate analysis if available
        syndicate_analysis = None
        syndicate_features = {}

        if syndicate_manager and i < len(raw_games):
            syndicate_analysis = syndicate_manager.get_full_game_analysis(raw_games[i])
            syndicate_features = syndicate_manager.get_enhanced_features(syndicate_analysis)

            # Track syndicate signals
            if syndicate_analysis.get('has_steam'):
                output['summary']['steam_moves_detected'] += 1
            if syndicate_analysis.get('is_rlm'):
                output['summary']['rlm_signals'] += 1
            if syndicate_analysis.get('significant_injury_edge'):
                output['summary']['injury_edges'] += 1

        # Create market features
        market = {
            'spread': odds.get('spread'),
            'total': odds.get('total'),
            'opening_spread': odds.get('opening_spread', odds.get('spread')),
            'line_movement': odds.get('line_movement', 0),
            'steam_move': syndicate_features.get('steam_move', 0),
            'public_pct': syndicate_analysis.get('public_home_pct', 50) if syndicate_analysis else 50,
            'reverse_line_movement': syndicate_analysis.get('is_rlm', False) if syndicate_analysis else False,
            'sharp_signal': syndicate_features.get('sharp_signal', 0),
        }

        # Create features
        features = feature_eng.create_features(home_stats, away_stats, context, market, syndicate_features)
        feature_vec = [features.get(f, 0) for f in feature_eng.feature_names]
        X = np.array([feature_vec])

        # Get predictions
        preds = model.predict_all(X)

        # Apply injury adjustments if available
        margin_adj = syndicate_features.get('injury_spread_adj', 0)
        total_adj = syndicate_features.get('injury_total_adj', 0)

        pred = {
            'win_prob': float(preds['win_prob'][0]),
            'margin': float(preds['margin'][0]) + margin_adj,
            'margin_std': float(preds['margin_std'][0]),
            'total': float(preds['total'][0]) + total_adj,
            'total_std': float(preds['total_std'][0]),
        }

        value_bets = analyze_betting_value(pred, odds)

        # Build game output
        game_output = {
            'game_id': odds['game_id'],
            'home': odds['home'],
            'away': odds['away'],
            'predictions': {
                'win_prob': round(pred['win_prob'], 4),
                'margin': round(pred['margin'], 1),
                'total': round(pred['total'], 1),
            },
            'market': {
                'spread': odds.get('spread'),
                'total': odds.get('total'),
                'sharp_spread': odds.get('sharp_spread'),
            },
            'value_bets': value_bets,
        }

        # Add syndicate signals if available
        if syndicate_analysis:
            game_output['syndicate_signals'] = {
                'steam_detected': syndicate_analysis.get('has_steam', False),
                'steam_direction': syndicate_analysis.get('sharp_side') if syndicate_analysis.get('has_steam') else None,
                'steam_confidence': round(syndicate_analysis.get('steam_confidence', 0), 2),
                'is_rlm': syndicate_analysis.get('is_rlm', False),
                'rlm_strength': round(syndicate_analysis.get('rlm_strength', 0), 3),
                'sharp_side': syndicate_analysis.get('sharp_side'),
                'public_home_pct': round(syndicate_analysis.get('public_home_pct', 50), 1),
                'pinnacle_divergence': round(syndicate_analysis.get('pinnacle_divergence', 0), 2),
                'injury_edge': syndicate_analysis.get('significant_injury_edge', False),
                'injury_adj': round(margin_adj, 1) if margin_adj else 0,
            }

        output['games'].append(game_output)

        all_value_bets.extend(value_bets)
        for vb in value_bets:
            output['summary']['total_value_bets'] += 1
            output['summary'][f"{vb['confidence'].lower()}_confidence"] += 1

    # Sort and get top plays
    all_value_bets.sort(key=lambda x: x['edge'], reverse=True)
    output['top_plays'] = all_value_bets[:15]

    # Save predictions
    output_path = PREDICTIONS_DIR / f"{today}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    latest_path = PREDICTIONS_DIR / "latest.json"
    with open(latest_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    print(f"Predictions saved to {output_path}")
    print(f"Latest updated at {latest_path}")
    print(f"\nValue bets found: {output['summary']['total_value_bets']}")
    print(f"  - HIGH confidence: {output['summary']['high_confidence']}")
    print(f"  - MEDIUM confidence: {output['summary']['medium_confidence']}")
    print(f"  - LOW confidence: {output['summary']['low_confidence']}")

    if SYNDICATE_ENABLED:
        print(f"\nSyndicate Signals:")
        print(f"  - Steam moves detected: {output['summary']['steam_moves_detected']}")
        print(f"  - RLM signals: {output['summary']['rlm_signals']}")
        print(f"  - Significant injury edges: {output['summary']['injury_edges']}")

    return output


if __name__ == "__main__":
    predictions = predict()
