#!/usr/bin/env python3
"""
================================================================================
🏀 COMPLETE SYNDICATE SYSTEM - TRAINED META-ENSEMBLE + LIVE PREDICTIONS
================================================================================

This integrates:
1. Trained XGBoost/LightGBM/CatBoost meta-ensemble
2. Live odds from The-Odds-API
3. KenPom-style team database
4. Edge detection and Kelly sizing

RUN: python cbb_complete_system.py

================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from datetime import datetime
from typing import Dict, List, Optional
import requests
import pickle
import os

# Import the trained model components
from cbb_advanced_trained import (
    MetaEnsembleModel, FeatureEngineer, ModelConfig,
    HistoricalDataGenerator
)

API_KEY = "272be842201ff50bdfee622541e2d3ee925afac17b3126e93b81b4d58e0e6b62"
MIN_EDGE = 0.025
KELLY_FRAC = 0.25
MAX_KELLY = 0.05

# =============================================================================
# KENPOM DATABASE
# =============================================================================
class KenPomDB:
    def __init__(self):
        self.teams = {}
        self._build()
    
    def _build(self):
        np.random.seed(42)
        teams = [
            ("Auburn", 1.25), ("Tennessee", 1.15), ("Duke", 1.10), ("Iowa St.", 1.05),
            ("Alabama", 1.00), ("Houston", 1.00), ("Kansas", 1.00), ("Florida", 0.95),
            ("Marquette", 0.95), ("Kentucky", 0.90), ("Purdue", 0.92), ("Michigan St.", 0.88),
            ("Gonzaga", 0.90), ("Connecticut", 0.85), ("Texas A&M", 0.80), ("Oregon", 0.78),
            ("Illinois", 0.76), ("UCLA", 0.74), ("Wisconsin", 0.72), ("Mississippi St.", 0.70),
            ("Oklahoma", 0.68), ("Creighton", 0.68), ("Arizona", 0.70), ("Saint Mary's", 0.66),
            ("Baylor", 0.65), ("Texas Tech", 0.64), ("San Diego St.", 0.62), ("BYU", 0.60),
            ("Memphis", 0.55), ("Indiana", 0.54), ("Michigan", 0.52), ("Ohio St.", 0.50),
            ("Arkansas", 0.48), ("Xavier", 0.46), ("Clemson", 0.44), ("North Carolina", 0.50),
            ("Missouri", 0.42), ("Maryland", 0.40), ("Ole Miss", 0.42), ("Cincinnati", 0.40),
            ("Drake", 0.45), ("Dayton", 0.42), ("VCU", 0.38), ("New Mexico", 0.40),
            ("Nevada", 0.36), ("NC State", 0.35), ("Louisville", 0.30),
        ]
        
        for name, power in teams:
            s = power + np.random.normal(0, 0.08)
            adj_o = 105 + s * 12 + np.random.normal(0, 1.5)
            adj_d = 100 - s * 10 + np.random.normal(0, 1.5)
            
            self.teams[name] = {
                'name': name, 'adj_em': adj_o - adj_d,
                'adj_o': adj_o, 'adj_d': adj_d,
                'adj_t': 68 + np.random.normal(0, 3.5),
                'rank': 0,
                'efg_o': np.clip(0.52 + s * 0.02, 0.44, 0.58),
                'efg_d': np.clip(0.50 - s * 0.015, 0.44, 0.56),
                'tov_o': np.clip(0.17 - s * 0.01, 0.12, 0.24),
                'tov_d': np.clip(0.17 + s * 0.008, 0.12, 0.24),
                'orb_o': np.clip(0.29 + s * 0.012, 0.22, 0.36),
                'drb_d': np.clip(0.73 + s * 0.01, 0.66, 0.80),
                'ftr_o': np.clip(0.32 + s * 0.01, 0.22, 0.42),
                'ftr_d': np.clip(0.30 - s * 0.01, 0.22, 0.40),
            }
        
        for i, t in enumerate(sorted(self.teams.keys(), 
                             key=lambda x: self.teams[x]['adj_em'], reverse=True), 1):
            self.teams[t]['rank'] = i
    
    def get(self, name: str) -> Optional[Dict]:
        if name in self.teams:
            return self.teams[name]
        for t in self.teams:
            if name.lower() in t.lower() or t.lower() in name.lower():
                return self.teams[t]
        return None


# =============================================================================
# ODDS CLIENT
# =============================================================================
def fetch_odds():
    url = f"https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
    params = {'apiKey': API_KEY, 'regions': 'us', 
              'markets': 'h2h,spreads,totals', 'oddsFormat': 'american'}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"API Error: {e}")
        return []

def parse_game(g):
    result = {'home': g.get('home_team',''), 'away': g.get('away_team',''),
              'ml_h': None, 'ml_a': None, 'spread': None, 'total': None,
              'sp_j_h': -110, 'sp_j_a': -110, 'ov_j': -110, 'un_j': -110}
    
    mls_h, mls_a, spreads, totals = [], [], [], []
    for bk in g.get('bookmakers', []):
        for mkt in bk.get('markets', []):
            for o in mkt.get('outcomes', []):
                if mkt['key'] == 'h2h':
                    if o['name'] == result['home']: mls_h.append(o['price'])
                    else: mls_a.append(o['price'])
                elif mkt['key'] == 'spreads':
                    if o['name'] == result['home']:
                        spreads.append(o.get('point', 0))
                        result['sp_j_h'] = o['price']
                    else: result['sp_j_a'] = o['price']
                elif mkt['key'] == 'totals':
                    if o['name'] == 'Over':
                        totals.append(o.get('point', 0))
                        result['ov_j'] = o['price']
                    else: result['un_j'] = o['price']
    
    if mls_h: result['ml_h'] = int(np.median(mls_h))
    if mls_a: result['ml_a'] = int(np.median(mls_a))
    if spreads: result['spread'] = np.median(spreads)
    if totals: result['total'] = np.median(totals)
    return result


# =============================================================================
# PREDICTION WITH TRAINED MODEL
# =============================================================================
class TrainedPredictor:
    def __init__(self, model_path: str = 'cbb_trained_model.pkl'):
        self.kenpom = KenPomDB()
        self.feature_eng = FeatureEngineer()
        
        # Load or train model
        if os.path.exists(model_path):
            self.model = MetaEnsembleModel.load(model_path)
        else:
            print("No trained model found - training now...")
            self._train_model(model_path)
    
    def _train_model(self, save_path: str):
        data_gen = HistoricalDataGenerator(n_games=3000)
        X, y_win, y_margin, y_total = data_gen.generate()
        
        self.model = MetaEnsembleModel()
        self.model.train(X, y_win, y_margin, y_total)
        self.model.save(save_path)
    
    def predict_game(self, home: str, away: str, market: Dict = None) -> Optional[Dict]:
        h_stats = self.kenpom.get(home)
        a_stats = self.kenpom.get(away)
        
        if not h_stats or not a_stats:
            return None
        
        # Create features
        features = self.feature_eng.create_features(h_stats, a_stats, market=market)
        feature_vec = [features.get(f, 0) for f in self.feature_eng.get_feature_names()]
        X = np.array([feature_vec])
        
        # Get predictions
        preds = self.model.predict_full(X)
        
        return {
            'home': home, 'away': away,
            'home_rank': h_stats['rank'], 'away_rank': a_stats['rank'],
            'win_prob': float(preds['win_prob'][0]),
            'margin': float(preds['margin'][0]),
            'margin_std': float(preds['margin_std'][0]),
            'total': float(preds['total'][0]),
            'total_std': float(preds['total_std'][0]),
        }
    
    def spread_prob(self, pred: Dict, spread: float) -> float:
        z = (pred['margin'] - (-spread)) / pred['margin_std']
        return float(stats.norm.cdf(z))
    
    def total_prob(self, pred: Dict, line: float, over: bool = True) -> float:
        z = (line - pred['total']) / pred['total_std']
        return 1 - stats.norm.cdf(z) if over else float(stats.norm.cdf(z))


# =============================================================================
# BET ANALYSIS
# =============================================================================
def am_to_prob(odds): 
    return 100/(odds+100) if odds > 0 else abs(odds)/(abs(odds)+100)

def am_to_dec(odds):
    return odds/100+1 if odds > 0 else 100/abs(odds)+1

def analyze_bets(predictor: TrainedPredictor, pred: Dict, odds: Dict) -> List[Dict]:
    recs = []
    
    # Moneyline
    if odds.get('ml_h') and odds.get('ml_a'):
        h_imp = am_to_prob(odds['ml_h'])
        a_imp = am_to_prob(odds['ml_a'])
        fair_h = h_imp / (h_imp + a_imp)
        fair_a = a_imp / (h_imp + a_imp)
        
        # Home ML
        edge_h = pred['win_prob'] - fair_h
        dec_h = am_to_dec(odds['ml_h'])
        ev_h = pred['win_prob'] * (dec_h - 1) - (1 - pred['win_prob'])
        kelly_h = max(0, ev_h / (dec_h - 1) * KELLY_FRAC) if ev_h > 0 else 0
        
        recs.append({
            'type': 'ML_HOME', 'desc': f"{pred['home']} ML ({odds['ml_h']:+d})",
            'prob': pred['win_prob'], 'fair': fair_h, 'edge': edge_h,
            'ev': ev_h * 100, 'kelly': min(kelly_h, MAX_KELLY),
            'conf': _conf(edge_h, ev_h)
        })
        
        # Away ML
        away_p = 1 - pred['win_prob']
        edge_a = away_p - fair_a
        dec_a = am_to_dec(odds['ml_a'])
        ev_a = away_p * (dec_a - 1) - (1 - away_p)
        kelly_a = max(0, ev_a / (dec_a - 1) * KELLY_FRAC) if ev_a > 0 else 0
        
        recs.append({
            'type': 'ML_AWAY', 'desc': f"{pred['away']} ML ({odds['ml_a']:+d})",
            'prob': away_p, 'fair': fair_a, 'edge': edge_a,
            'ev': ev_a * 100, 'kelly': min(kelly_a, MAX_KELLY),
            'conf': _conf(edge_a, ev_a)
        })
    
    # Spread
    if odds.get('spread') is not None:
        h_cov = predictor.spread_prob(pred, odds['spread'])
        h_imp = am_to_prob(odds['sp_j_h'])
        a_imp = am_to_prob(odds['sp_j_a'])
        fair_h = h_imp / (h_imp + a_imp)
        fair_a = a_imp / (h_imp + a_imp)
        
        edge_h = h_cov - fair_h
        dec_h = am_to_dec(odds['sp_j_h'])
        ev_h = h_cov * (dec_h - 1) - (1 - h_cov)
        kelly_h = max(0, ev_h / (dec_h - 1) * KELLY_FRAC) if ev_h > 0 else 0
        
        recs.append({
            'type': 'SPREAD_HOME', 
            'desc': f"{pred['home']} {odds['spread']:+.1f} ({odds['sp_j_h']:+d})",
            'prob': h_cov, 'fair': fair_h, 'edge': edge_h,
            'ev': ev_h * 100, 'kelly': min(kelly_h, MAX_KELLY),
            'conf': _conf(edge_h, ev_h),
            'line_value': pred['margin'] - (-odds['spread'])
        })
        
        edge_a = (1 - h_cov) - fair_a
        dec_a = am_to_dec(odds['sp_j_a'])
        ev_a = (1 - h_cov) * (dec_a - 1) - h_cov
        kelly_a = max(0, ev_a / (dec_a - 1) * KELLY_FRAC) if ev_a > 0 else 0
        
        recs.append({
            'type': 'SPREAD_AWAY',
            'desc': f"{pred['away']} {-odds['spread']:+.1f} ({odds['sp_j_a']:+d})",
            'prob': 1 - h_cov, 'fair': fair_a, 'edge': edge_a,
            'ev': ev_a * 100, 'kelly': min(kelly_a, MAX_KELLY),
            'conf': _conf(edge_a, ev_a)
        })
    
    # Total
    if odds.get('total') is not None:
        ov_p = predictor.total_prob(pred, odds['total'], over=True)
        o_imp = am_to_prob(odds['ov_j'])
        u_imp = am_to_prob(odds['un_j'])
        fair_o = o_imp / (o_imp + u_imp)
        fair_u = u_imp / (o_imp + u_imp)
        
        edge_o = ov_p - fair_o
        dec_o = am_to_dec(odds['ov_j'])
        ev_o = ov_p * (dec_o - 1) - (1 - ov_p)
        kelly_o = max(0, ev_o / (dec_o - 1) * KELLY_FRAC) if ev_o > 0 else 0
        
        recs.append({
            'type': 'OVER', 'desc': f"Over {odds['total']} ({odds['ov_j']:+d})",
            'prob': ov_p, 'fair': fair_o, 'edge': edge_o,
            'ev': ev_o * 100, 'kelly': min(kelly_o, MAX_KELLY),
            'conf': _conf(edge_o, ev_o),
            'total_value': pred['total'] - odds['total']
        })
        
        edge_u = (1 - ov_p) - fair_u
        dec_u = am_to_dec(odds['un_j'])
        ev_u = (1 - ov_p) * (dec_u - 1) - ov_p
        kelly_u = max(0, ev_u / (dec_u - 1) * KELLY_FRAC) if ev_u > 0 else 0
        
        recs.append({
            'type': 'UNDER', 'desc': f"Under {odds['total']} ({odds['un_j']:+d})",
            'prob': 1 - ov_p, 'fair': fair_u, 'edge': edge_u,
            'ev': ev_u * 100, 'kelly': min(kelly_u, MAX_KELLY),
            'conf': _conf(edge_u, ev_u)
        })
    
    return recs

def _conf(edge, ev):
    if edge < MIN_EDGE or ev <= 0: return 'NO_BET'
    if edge > 0.06: return 'HIGH'
    if edge > 0.035: return 'MEDIUM'
    return 'LOW'


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("🏀 ADVANCED SYNDICATE SYSTEM - TRAINED META-ENSEMBLE")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now()}")
    print(f"API Key: {API_KEY[:16]}...{API_KEY[-8:]}")
    
    # Initialize predictor (loads trained model)
    predictor = TrainedPredictor()
    print(f"✓ Model loaded (Brier: {predictor.model.training_history[-1]['brier_score']:.4f})")
    print(f"✓ {len(predictor.kenpom.teams)} teams in database")
    
    # Fetch odds
    print("\nFetching odds...")
    raw = fetch_odds()
    
    if not raw:
        print("Using demo games...")
        games = [
            {'home_team': 'Auburn', 'away_team': 'Ohio St.', 'bookmakers': [{'markets': [
                {'key': 'h2h', 'outcomes': [{'name': 'Auburn', 'price': -450}, {'name': 'Ohio St.', 'price': 350}]},
                {'key': 'spreads', 'outcomes': [{'name': 'Auburn', 'price': -110, 'point': -10.5}, {'name': 'Ohio St.', 'price': -110, 'point': 10.5}]},
                {'key': 'totals', 'outcomes': [{'name': 'Over', 'price': -110, 'point': 148.5}, {'name': 'Under', 'price': -110, 'point': 148.5}]}
            ]}]},
            {'home_team': 'Duke', 'away_team': 'NC State', 'bookmakers': [{'markets': [
                {'key': 'h2h', 'outcomes': [{'name': 'Duke', 'price': -550}, {'name': 'NC State', 'price': 400}]},
                {'key': 'spreads', 'outcomes': [{'name': 'Duke', 'price': -108, 'point': -12.0}, {'name': 'NC State', 'price': -112, 'point': 12.0}]},
                {'key': 'totals', 'outcomes': [{'name': 'Over', 'price': -105, 'point': 155.5}, {'name': 'Under', 'price': -115, 'point': 155.5}]}
            ]}]},
            {'home_team': 'Kansas', 'away_team': 'Missouri', 'bookmakers': [{'markets': [
                {'key': 'h2h', 'outcomes': [{'name': 'Kansas', 'price': -800}, {'name': 'Missouri', 'price': 550}]},
                {'key': 'spreads', 'outcomes': [{'name': 'Kansas', 'price': -105, 'point': -15.5}, {'name': 'Missouri', 'price': -115, 'point': 15.5}]},
                {'key': 'totals', 'outcomes': [{'name': 'Over', 'price': -110, 'point': 149.5}, {'name': 'Under', 'price': -110, 'point': 149.5}]}
            ]}]},
            {'home_team': 'Florida', 'away_team': 'North Carolina', 'bookmakers': [{'markets': [
                {'key': 'h2h', 'outcomes': [{'name': 'Florida', 'price': -200}, {'name': 'North Carolina', 'price': 170}]},
                {'key': 'spreads', 'outcomes': [{'name': 'Florida', 'price': -110, 'point': -4.5}, {'name': 'North Carolina', 'price': -110, 'point': 4.5}]},
                {'key': 'totals', 'outcomes': [{'name': 'Over', 'price': -110, 'point': 152.5}, {'name': 'Under', 'price': -110, 'point': 152.5}]}
            ]}]},
            {'home_team': 'Oregon', 'away_team': 'UCLA', 'bookmakers': [{'markets': [
                {'key': 'h2h', 'outcomes': [{'name': 'Oregon', 'price': 135}, {'name': 'UCLA', 'price': -155}]},
                {'key': 'spreads', 'outcomes': [{'name': 'Oregon', 'price': -110, 'point': 3.0}, {'name': 'UCLA', 'price': -110, 'point': -3.0}]},
                {'key': 'totals', 'outcomes': [{'name': 'Over', 'price': -108, 'point': 145.5}, {'name': 'Under', 'price': -112, 'point': 145.5}]}
            ]}]},
            {'home_team': 'Drake', 'away_team': 'Dayton', 'bookmakers': [{'markets': [
                {'key': 'h2h', 'outcomes': [{'name': 'Drake', 'price': 125}, {'name': 'Dayton', 'price': -145}]},
                {'key': 'spreads', 'outcomes': [{'name': 'Drake', 'price': -110, 'point': 2.5}, {'name': 'Dayton', 'price': -110, 'point': -2.5}]},
                {'key': 'totals', 'outcomes': [{'name': 'Over', 'price': -110, 'point': 140.5}, {'name': 'Under', 'price': -110, 'point': 140.5}]}
            ]}]},
        ]
    else:
        games = raw
    
    print(f"\nAnalyzing {len(games)} games...\n")
    
    all_recs = []
    
    for g in games:
        odds = parse_game(g)
        pred = predictor.predict_game(odds['home'], odds['away'], 
                                      market={'spread': odds.get('spread', 0), 
                                             'total': odds.get('total', 145)})
        if not pred:
            continue
        
        print("-" * 70)
        print(f"🏀 {odds['away']} (#{pred['away_rank']}) @ {odds['home']} (#{pred['home_rank']})")
        print(f"   MODEL: Win% {pred['win_prob']*100:.1f}% | Margin {pred['margin']:.1f}")
        print(f"          Total {pred['total']:.1f} (σ={pred['total_std']:.1f})")
        if odds['spread']:
            print(f"   MARKET: Spread {odds['spread']:+.1f} | Total {odds['total']}")
        
        recs = analyze_bets(predictor, pred, odds)
        for r in recs:
            r['home'], r['away'] = pred['home'], pred['away']
        all_recs.extend(recs)
        
        value = [r for r in recs if r['conf'] != 'NO_BET']
        if value:
            best = max(value, key=lambda x: x['edge'])
            emoji = {'HIGH': '🔥', 'MEDIUM': '✅', 'LOW': '⚠️'}[best['conf']]
            print(f"\n   {emoji} BEST: {best['desc']}")
            print(f"      Model {best['prob']*100:.1f}% vs Fair {best['fair']*100:.1f}%")
            print(f"      Edge: {best['edge']*100:+.2f}% | EV: {best['ev']:+.2f}%")
        else:
            print("\n   ❌ No value detected")
        print()
    
    # Summary
    value_bets = [r for r in all_recs if r['conf'] != 'NO_BET']
    
    print("=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Value Bets: {len(value_bets)}")
    print(f"   🔥 HIGH: {len([r for r in value_bets if r['conf'] == 'HIGH'])}")
    print(f"   ✅ MEDIUM: {len([r for r in value_bets if r['conf'] == 'MEDIUM'])}")
    print(f"   ⚠️  LOW: {len([r for r in value_bets if r['conf'] == 'LOW'])}")
    
    if value_bets:
        print("\n" + "=" * 70)
        print("🎯 TOP BETS BY EDGE")
        print("=" * 70)
        
        for i, b in enumerate(sorted(value_bets, key=lambda x: x['edge'], reverse=True)[:10], 1):
            emoji = {'HIGH': '🔥', 'MEDIUM': '✅', 'LOW': '⚠️'}[b['conf']]
            print(f"\n{i}. {emoji} {b['desc']}")
            print(f"   {b['away']} @ {b['home']}")
            print(f"   Model: {b['prob']*100:.1f}% | Fair: {b['fair']*100:.1f}%")
            print(f"   Edge: {b['edge']*100:+.2f}% | EV: {b['ev']:+.2f}%")
            print(f"   Kelly: {b['kelly']*100:.2f}%")
        
        # Export
        df = pd.DataFrame(value_bets)
        df.to_csv('trained_recommendations.csv', index=False)
        print(f"\n✓ Exported to trained_recommendations.csv")


if __name__ == "__main__":
    main()
