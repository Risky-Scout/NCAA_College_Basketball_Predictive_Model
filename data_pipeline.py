#!/usr/bin/env python3
"""
================================================================================
NCAA College Basketball Predictive Model - Data Pipeline
================================================================================

Data fetching, team database, and syndicate data integration.

================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import requests
import sqlite3
import json

from config import ODDS_API_KEY, DATA_DIR, syndicate_config


# =============================================================================
# TEAM DATABASE
# =============================================================================
class TeamDatabase:
    """
    In-memory team ratings database with KenPom-style metrics.

    In production, this would pull from a real database or API.
    """

    def __init__(self):
        self.teams = self._build_team_database()

    def _build_team_database(self) -> Dict[str, Dict]:
        """Build team database with realistic ratings"""
        teams = {}

        # Top 25 + Other major teams with realistic stats
        team_data = [
            # (name, adj_em, adj_o, adj_d, tempo, rank)
            ('Auburn', 28.5, 118.2, 89.7, 67.5, 1),
            ('Duke', 26.8, 120.1, 93.3, 70.2, 2),
            ('Tennessee', 25.9, 114.8, 88.9, 65.1, 3),
            ('Alabama', 25.2, 117.5, 92.3, 72.8, 4),
            ('Iowa State', 24.1, 113.2, 89.1, 66.4, 5),
            ('Houston', 23.8, 111.9, 88.1, 64.2, 6),
            ('Florida', 23.2, 116.4, 93.2, 69.8, 7),
            ('Kansas', 22.9, 117.8, 94.9, 68.3, 8),
            ('Marquette', 22.1, 118.5, 96.4, 70.1, 9),
            ('Oregon', 21.5, 114.2, 92.7, 67.9, 10),
            ('Kentucky', 21.0, 116.9, 95.9, 71.4, 11),
            ('Michigan State', 20.4, 113.5, 93.1, 66.8, 12),
            ('Purdue', 19.8, 119.2, 99.4, 67.5, 13),
            ('UConn', 19.2, 114.8, 95.6, 66.2, 14),
            ('Gonzaga', 18.9, 117.3, 98.4, 71.8, 15),
            ('Texas A&M', 18.5, 110.8, 92.3, 64.5, 16),
            ('Texas Tech', 18.1, 109.5, 91.4, 63.8, 17),
            ('UCLA', 17.8, 113.2, 95.4, 68.5, 18),
            ('Wisconsin', 17.4, 111.8, 94.4, 62.1, 19),
            ('Memphis', 17.0, 112.5, 95.5, 69.4, 20),
            ('North Carolina', 16.5, 115.8, 99.3, 72.5, 21),
            ('Illinois', 16.2, 114.2, 98.0, 68.9, 22),
            ('Baylor', 15.8, 110.9, 95.1, 66.2, 23),
            ('St. John\'s', 15.4, 112.8, 97.4, 67.8, 24),
            ('Ole Miss', 15.0, 109.5, 94.5, 65.4, 25),
            # Additional teams
            ('Ohio State', 12.5, 108.2, 95.7, 66.5, 35),
            ('NC State', 8.5, 105.8, 97.3, 67.2, 55),
            ('Missouri', 6.2, 103.5, 97.3, 68.8, 72),
            ('Indiana', 11.5, 109.2, 97.7, 69.1, 42),
            ('Michigan', 10.8, 107.5, 96.7, 65.8, 48),
            ('Syracuse', 5.5, 104.2, 98.7, 70.5, 85),
            ('Louisville', 3.2, 102.8, 99.6, 69.2, 105),
            ('Arizona', 14.5, 111.8, 97.3, 68.4, 28),
            ('Texas', 13.2, 110.5, 97.3, 66.9, 32),
            ('Creighton', 16.8, 113.5, 96.7, 67.5, 22),
            ('Arkansas', 11.2, 108.8, 97.6, 71.2, 45),
            ('Villanova', 12.8, 109.5, 96.7, 65.4, 38),
            ('San Diego State', 14.8, 106.2, 91.4, 62.8, 27),
            ('BYU', 13.5, 110.2, 96.7, 68.1, 31),
            ('Xavier', 10.2, 107.8, 97.6, 67.8, 52),
        ]

        for name, adj_em, adj_o, adj_d, tempo, rank in team_data:
            teams[name] = self._create_team_stats(name, adj_em, adj_o, adj_d, tempo, rank)

        return teams

    def _create_team_stats(self, name: str, adj_em: float, adj_o: float,
                           adj_d: float, tempo: float, rank: int) -> Dict:
        """Create full team stats dictionary with four factors"""
        # Derive four factors from efficiency ratings
        quality = adj_em / 30  # Normalized quality (-1 to 1)

        return {
            'name': name,
            'adj_em': adj_em,
            'adj_o': adj_o,
            'adj_d': adj_d,
            'adj_t': tempo,
            'rank': rank,
            # Four factors (derived with some variance)
            'efg_o': np.clip(0.51 + quality * 0.03 + np.random.normal(0, 0.01), 0.44, 0.58),
            'efg_d': np.clip(0.50 - quality * 0.02 + np.random.normal(0, 0.01), 0.44, 0.56),
            'tov_o': np.clip(0.17 - quality * 0.015 + np.random.normal(0, 0.01), 0.12, 0.24),
            'tov_d': np.clip(0.17 + quality * 0.012 + np.random.normal(0, 0.01), 0.12, 0.24),
            'orb_o': np.clip(0.29 + quality * 0.015 + np.random.normal(0, 0.015), 0.22, 0.36),
            'drb_d': np.clip(0.73 + quality * 0.012 + np.random.normal(0, 0.015), 0.66, 0.80),
            'ftr_o': np.clip(0.32 + quality * 0.015 + np.random.normal(0, 0.02), 0.22, 0.42),
            'ftr_d': np.clip(0.30 - quality * 0.012 + np.random.normal(0, 0.02), 0.22, 0.40),
            # Advanced
            'sos': quality * 3 + np.random.normal(0, 1.5),
            'luck': np.random.normal(0, 0.025),
        }

    def get(self, team: str) -> Optional[Dict]:
        """Get team stats by name (fuzzy matching)"""
        if team in self.teams:
            return self.teams[team]

        # Try fuzzy match
        team_lower = team.lower()
        for name, stats in self.teams.items():
            if name.lower() in team_lower or team_lower in name.lower():
                return stats

        return None


# =============================================================================
# ODDS API CLIENT
# =============================================================================
class OddsAPIClient:
    """
    Client for The Odds API with multi-book support.

    Integrates with syndicate data module for enhanced features.
    """

    def __init__(self, api_key: str = ODDS_API_KEY):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
        self.opening_lines: Dict[str, Dict] = {}

    def fetch_games(self) -> List[Dict]:
        """Fetch games from The Odds API"""
        params = {
            'apiKey': self.api_key,
            'regions': 'us,us2',
            'markets': 'h2h,spreads,totals',
            'oddsFormat': 'american',
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Odds API error: {e}")
            return []

    def parse_game(self, game_data: Dict) -> Dict:
        """Parse game data with full odds breakdown"""
        game_id = game_data.get('id', '')
        home = game_data.get('home_team', '')
        away = game_data.get('away_team', '')

        result = {
            'game_id': game_id,
            'home': home,
            'away': away,
            'commence_time': game_data.get('commence_time', ''),
        }

        # Aggregate odds across bookmakers
        spreads = []
        totals = []
        home_mls = []
        away_mls = []
        spread_juices = {'home': [], 'away': []}
        total_juices = {'over': [], 'under': []}

        sharp_books = syndicate_config.sharp_books

        for bookie in game_data.get('bookmakers', []):
            book_key = bookie.get('key', '')
            is_sharp = book_key in sharp_books

            for market in bookie.get('markets', []):
                mkt_key = market.get('key')

                for outcome in market.get('outcomes', []):
                    if mkt_key == 'spreads':
                        if outcome.get('name') == home:
                            spreads.append({
                                'point': outcome.get('point'),
                                'price': outcome.get('price'),
                                'book': book_key,
                                'is_sharp': is_sharp,
                            })
                            spread_juices['home'].append(outcome.get('price'))
                        else:
                            spread_juices['away'].append(outcome.get('price'))

                    elif mkt_key == 'totals':
                        if outcome.get('name') == 'Over':
                            totals.append({
                                'point': outcome.get('point'),
                                'price': outcome.get('price'),
                                'book': book_key,
                            })
                            total_juices['over'].append(outcome.get('price'))
                        else:
                            total_juices['under'].append(outcome.get('price'))

                    elif mkt_key == 'h2h':
                        if outcome.get('name') == home:
                            home_mls.append({'price': outcome.get('price'), 'book': book_key})
                        else:
                            away_mls.append({'price': outcome.get('price'), 'book': book_key})

        # Calculate consensus values
        if spreads:
            # Sharp book consensus
            sharp_spreads = [s['point'] for s in spreads if s['is_sharp'] and s['point']]
            result['sharp_spread'] = np.median(sharp_spreads) if sharp_spreads else None

            # Market consensus
            all_spreads = [s['point'] for s in spreads if s['point']]
            result['spread'] = np.median(all_spreads) if all_spreads else None

            # Best available
            result['best_home_spread'] = max([s['point'] for s in spreads if s['point']], default=None)

        if totals:
            all_totals = [t['point'] for t in totals if t['point']]
            result['total'] = np.median(all_totals) if all_totals else None

        if home_mls:
            result['ml_home'] = int(np.median([m['price'] for m in home_mls]))
        if away_mls:
            result['ml_away'] = int(np.median([m['price'] for m in away_mls]))

        # Juice
        if spread_juices['home']:
            result['spread_juice_home'] = int(np.median(spread_juices['home']))
        if spread_juices['away']:
            result['spread_juice_away'] = int(np.median(spread_juices['away']))
        if total_juices['over']:
            result['over_juice'] = int(np.median(total_juices['over']))
        if total_juices['under']:
            result['under_juice'] = int(np.median(total_juices['under']))

        # Track opening lines
        if game_id not in self.opening_lines:
            self.opening_lines[game_id] = {
                'spread': result.get('spread'),
                'total': result.get('total'),
                'timestamp': datetime.now().isoformat(),
            }
            result['line_movement'] = 0
            result['opening_spread'] = result.get('spread')
        else:
            opening = self.opening_lines[game_id]
            if result.get('spread') and opening.get('spread'):
                result['line_movement'] = result['spread'] - opening['spread']
                result['opening_spread'] = opening['spread']

        return result


# =============================================================================
# HISTORICAL DATA MANAGER
# =============================================================================
class HistoricalDataManager:
    """
    Manages historical game data for training and CLV tracking.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DATA_DIR / 'cbb_historical.db')
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                date TEXT,
                season INTEGER,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER,
                home_spread_open REAL,
                home_spread_close REAL,
                total_open REAL,
                total_close REAL,
                created_at TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                game_id TEXT,
                timestamp TEXT,
                bet_type TEXT,
                side TEXT,
                line REAL,
                model_prob REAL,
                market_prob REAL,
                edge REAL,
                closing_prob REAL,
                clv REAL,
                result INTEGER,
                profit REAL
            )
        ''')

        conn.commit()
        conn.close()

    def record_prediction(self, prediction: Dict):
        """Record a prediction for CLV tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        pred_id = f"{prediction['game_id']}_{prediction['bet_type']}_{prediction['side']}"

        cursor.execute('''
            INSERT OR REPLACE INTO predictions
            (prediction_id, game_id, timestamp, bet_type, side, line,
             model_prob, market_prob, edge)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pred_id,
            prediction['game_id'],
            datetime.now().isoformat(),
            prediction['bet_type'],
            prediction['side'],
            prediction.get('line'),
            prediction['model_prob'],
            prediction['market_prob'],
            prediction['edge'],
        ))

        conn.commit()
        conn.close()

    def get_clv_summary(self, days: int = 30) -> Dict:
        """Get CLV performance summary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute('''
            SELECT
                COUNT(*) as n_bets,
                AVG(clv) as avg_clv,
                AVG(edge) as avg_edge,
                SUM(profit) as total_profit,
                SUM(CASE WHEN result = 1 THEN 1 ELSE 0 END) as wins
            FROM predictions
            WHERE timestamp >= ? AND result IS NOT NULL
        ''', (cutoff,))

        row = cursor.fetchone()
        conn.close()

        if row and row[0] > 0:
            return {
                'n_bets': row[0],
                'avg_clv': row[1] or 0,
                'avg_edge': row[2] or 0,
                'total_profit': row[3] or 0,
                'wins': row[4] or 0,
                'win_rate': row[4] / row[0] if row[0] else 0,
            }

        return {'n_bets': 0}
