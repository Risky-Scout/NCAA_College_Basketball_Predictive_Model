#!/usr/bin/env python3
"""
================================================================================
NCAA College Basketball Predictive Model - Prediction Script
================================================================================

This script is run daily by GitHub Actions at 4:30 PM EST.

It:
1. Loads the trained model
2. Fetches today's games and odds
3. Generates predictions
4. Calculates edges and Kelly sizing
5. Outputs to predictions/YYYY-MM-DD.json

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
from feature_engineering import FeatureEngineer
from meta_ensemble import MetaEnsemble


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
    """Main prediction function"""
    today = datetime.now().strftime('%Y-%m-%d')
    
    print("=" * 60)
    print("NCAA CBB MODEL - DAILY PREDICTIONS")
    print("=" * 60)
    print(f"Date: {today}")
    
    # Load model
    print("\nLoading model...")
    try:
        model = MetaEnsemble.load()
    except FileNotFoundError:
        print("⚠️ No trained model found. Running training first...")
        from train import train
        model, _ = train()
    
    team_db = TeamDatabase()
    feature_eng = FeatureEngineer()
    
    # Fetch games
    print("\nFetching today's games...")
    odds_client = OddsAPIClient()
    raw_games = odds_client.fetch_games()
    
    if not raw_games:
        print("⚠️ API unavailable - using demo games")
        games = get_demo_games()
    else:
        games = [odds_client.parse_game(g) for g in raw_games]
    
    print(f"Found {len(games)} games")
    
    # Process games
    output = {
        'generated_at': datetime.now().isoformat(),
        'model_version': model.version,
        'n_games': len(games),
        'games': [],
        'summary': {'total_value_bets': 0, 'high_confidence': 0, 'medium_confidence': 0, 'low_confidence': 0}
    }
    
    all_value_bets = []
    
    for odds in games:
        home_stats = team_db.get(odds['home'])
        away_stats = team_db.get(odds['away'])
        
        if not home_stats or not away_stats:
            continue
        
        context = {'home_rest': 3, 'away_rest': 3, 'neutral_site': 0, 'conference_game': 0}
        market = {'spread': odds.get('spread'), 'total': odds.get('total'), 'opening_spread': odds.get('spread')}
        
        features = feature_eng.create_features(home_stats, away_stats, context, market)
        feature_vec = [features.get(f, 0) for f in feature_eng.feature_names]
        X = np.array([feature_vec])
        
        preds = model.predict_all(X)
        pred = {
            'win_prob': float(preds['win_prob'][0]),
            'margin': float(preds['margin'][0]),
            'margin_std': float(preds['margin_std'][0]),
            'total': float(preds['total'][0]),
            'total_std': float(preds['total_std'][0]),
        }
        
        value_bets = analyze_betting_value(pred, odds)
        
        output['games'].append({
            'game_id': odds['game_id'], 'home': odds['home'], 'away': odds['away'],
            'predictions': {'win_prob': round(pred['win_prob'], 4), 'margin': round(pred['margin'], 1), 'total': round(pred['total'], 1)},
            'market': {'spread': odds.get('spread'), 'total': odds.get('total')},
            'value_bets': value_bets,
        })
        
        all_value_bets.extend(value_bets)
        for vb in value_bets:
            output['summary']['total_value_bets'] += 1
            output['summary'][f"{vb['confidence'].lower()}_confidence"] += 1
    
    all_value_bets.sort(key=lambda x: x['edge'], reverse=True)
    output['top_plays'] = all_value_bets[:15]
    
    # Save
    output_path = PREDICTIONS_DIR / f"{today}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    latest_path = PREDICTIONS_DIR / "latest.json"
    with open(latest_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Predictions saved to {output_path}")
    print(f"✓ Latest updated at {latest_path}")
    print(f"\nValue bets found: {output['summary']['total_value_bets']}")
    
    return output


if __name__ == "__main__":
    predictions = predict()
