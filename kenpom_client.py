#!/usr/bin/env python3
"""
================================================================================
KenPom Data Client - NCAA College Basketball
================================================================================

Fetches team ratings from KenPom.com (requires subscription).
Falls back to Barttorvik.com (free, same metrics) if KenPom unavailable.

SYNDICATE-GRADE: Uses real KenPom data for maximum accuracy.

Usage:
    client = KenPomClient(email="your@email.com", password="yourpass")
    ratings = client.get_current_ratings()

Environment Variables:
    KENPOM_EMAIL - Your KenPom subscription email
    KENPOM_PASSWORD - Your KenPom subscription password

================================================================================
"""

import os
import re
import requests
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import json
from pathlib import Path

# Cache settings
CACHE_DIR = Path(__file__).parent / "data"
CACHE_DIR.mkdir(exist_ok=True)
RATINGS_CACHE_FILE = CACHE_DIR / "team_ratings_cache.json"
CACHE_DURATION_HOURS = 4  # Refresh every 4 hours for freshest data


class KenPomClient:
    """
    Client for fetching KenPom team ratings.

    SYNDICATE-GRADE DATA SOURCE

    Primary: KenPom.com (requires $25/year subscription)
    Fallback: Barttorvik.com (free, same data format)

    Data includes:
    - Adjusted Efficiency Margin (AdjEM)
    - Adjusted Offensive Efficiency (AdjO)
    - Adjusted Defensive Efficiency (AdjD)
    - Adjusted Tempo (AdjT)
    - Strength of Schedule (SoS)
    - Luck rating
    - All 362 D1 teams
    """

    KENPOM_BASE = "https://kenpom.com"
    KENPOM_LOGIN_URL = "https://kenpom.com/index.php"
    BARTTORVIK_URL = "https://barttorvik.com/trank.php"

    def __init__(self, email: str = None, password: str = None):
        """
        Initialize KenPom client.

        Args:
            email: KenPom subscription email (or set KENPOM_EMAIL env var)
            password: KenPom subscription password (or set KENPOM_PASSWORD env var)
        """
        self.email = email or os.environ.get('KENPOM_EMAIL')
        self.password = password or os.environ.get('KENPOM_PASSWORD')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })
        self._logged_in = False
        self._ratings_cache: Dict[str, Dict] = {}
        self._cache_time: Optional[datetime] = None

    def _login_kenpom(self) -> bool:
        """Login to KenPom with credentials."""
        if not self.email or not self.password:
            print("KenPom: No credentials provided")
            return False

        try:
            print(f"KenPom: Attempting login for {self.email}...")

            # First, get the main page to establish session
            main_page = self.session.get(self.KENPOM_BASE, timeout=30)

            # KenPom uses a simple form POST for login
            # The login form is on the main page when not logged in
            login_data = {
                'email': self.email,
                'password': self.password,
                'submit': 'Login!'
            }

            # Post to the login handler
            response = self.session.post(
                self.KENPOM_LOGIN_URL,
                data=login_data,
                timeout=30,
                allow_redirects=True
            )

            # Check if login successful by looking for subscriber-only content
            # or the logout link
            if 'logout' in response.text.lower() or 'subscriber' in response.text.lower():
                self._logged_in = True
                print("KenPom: Login successful!")
                return True

            # Check for error messages
            if 'invalid' in response.text.lower() or 'incorrect' in response.text.lower():
                print("KenPom: Invalid credentials")
                return False

            # Check if we can see the full ratings (subscribers see more columns)
            if 'AdjEM' in response.text and 'SoS' in response.text:
                self._logged_in = True
                print("KenPom: Login successful (verified by content)")
                return True

            print("KenPom: Login status unclear, trying to fetch data anyway...")
            self._logged_in = True  # Try anyway
            return True

        except Exception as e:
            print(f"KenPom: Login error - {e}")
            return False

    def _fetch_kenpom_ratings(self) -> Optional[pd.DataFrame]:
        """Fetch ratings from KenPom.com."""
        if not self._logged_in:
            if not self._login_kenpom():
                return None

        try:
            print("KenPom: Fetching ratings...")
            response = self.session.get(self.KENPOM_BASE, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the main ratings table
            # KenPom uses id="ratings-table" for the main table
            table = soup.find('table', {'id': 'ratings-table'})

            if not table:
                # Try alternative selectors
                table = soup.find('table', class_='ratings-table')

            if not table:
                # Look for any table with team data
                tables = soup.find_all('table')
                for t in tables:
                    if t.find('td', string=re.compile(r'Duke|Kentucky|Kansas', re.I)):
                        table = t
                        break

            if not table:
                print("KenPom: Could not find ratings table")
                # Save page for debugging
                debug_path = CACHE_DIR / "kenpom_debug.html"
                with open(debug_path, 'w') as f:
                    f.write(response.text)
                print(f"KenPom: Saved debug page to {debug_path}")
                return None

            # Parse table
            rows = []
            tbody = table.find('tbody') or table

            for tr in tbody.find_all('tr'):
                cols = tr.find_all(['td', 'th'])
                if len(cols) < 5:
                    continue

                try:
                    # Extract team name (usually in a link)
                    team_cell = cols[1] if len(cols) > 1 else cols[0]
                    team_link = team_cell.find('a')
                    team_name = team_link.text.strip() if team_link else team_cell.text.strip()

                    # Clean team name (remove seed numbers, etc.)
                    team_name = re.sub(r'^\d+\s*', '', team_name).strip()

                    if not team_name or team_name.lower() in ['team', 'rk', 'rank']:
                        continue

                    # Parse numeric values
                    def parse_float(cell, default=0.0):
                        try:
                            text = cell.text.strip()
                            # Remove any non-numeric chars except . and -
                            text = re.sub(r'[^\d.\-+]', '', text)
                            return float(text) if text else default
                        except:
                            return default

                    def parse_int(cell, default=0):
                        try:
                            text = cell.text.strip()
                            text = re.sub(r'[^\d]', '', text)
                            return int(text) if text else default
                        except:
                            return default

                    row = {
                        'rank': parse_int(cols[0], 999),
                        'team': team_name,
                        'conf': cols[2].text.strip() if len(cols) > 2 else '',
                        'record': cols[3].text.strip() if len(cols) > 3 else '',
                        'adj_em': parse_float(cols[4]) if len(cols) > 4 else 0,
                        'adj_o': parse_float(cols[5]) if len(cols) > 5 else 100,
                        'adj_o_rank': parse_int(cols[6]) if len(cols) > 6 else 0,
                        'adj_d': parse_float(cols[7]) if len(cols) > 7 else 100,
                        'adj_d_rank': parse_int(cols[8]) if len(cols) > 8 else 0,
                        'adj_t': parse_float(cols[9]) if len(cols) > 9 else 68,
                        'adj_t_rank': parse_int(cols[10]) if len(cols) > 10 else 0,
                        'luck': parse_float(cols[11]) if len(cols) > 11 else 0,
                        'sos_adj_em': parse_float(cols[12]) if len(cols) > 12 else 0,
                        'sos_opp_o': parse_float(cols[13]) if len(cols) > 13 else 0,
                        'sos_opp_d': parse_float(cols[14]) if len(cols) > 14 else 0,
                        'ncsos_adj_em': parse_float(cols[15]) if len(cols) > 15 else 0,
                    }

                    # Only add valid teams
                    if row['team'] and row['rank'] < 400:
                        rows.append(row)

                except Exception as e:
                    continue

            if rows:
                df = pd.DataFrame(rows)
                print(f"KenPom: Fetched {len(df)} teams")
                return df

            print("KenPom: No valid team data found in table")
            return None

        except Exception as e:
            print(f"KenPom: Fetch error - {e}")
            return None

    def _fetch_barttorvik_ratings(self, season: int = 2025) -> Optional[pd.DataFrame]:
        """Fetch ratings from Barttorvik (free KenPom alternative)."""
        url = f"{self.BARTTORVIK_URL}?year={season}&csv=1"

        try:
            print(f"Barttorvik: Fetching {season} ratings...")
            response = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()

            # Parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))

            # Standardize column names to match KenPom format
            column_mapping = {
                'Team': 'team',
                'Conf': 'conf',
                'Rec': 'record',
                'AdjOE': 'adj_o',
                'AdjDE': 'adj_d',
                'AdjTempo': 'adj_t',
                'Barthag': 'barthag',
                'WAB': 'wab',
                'Rank': 'rank',
            }

            # Rename columns that exist
            for old, new in column_mapping.items():
                if old in df.columns:
                    df = df.rename(columns={old: new})

            # Calculate adj_em if not present
            if 'adj_em' not in df.columns and 'adj_o' in df.columns and 'adj_d' in df.columns:
                df['adj_em'] = df['adj_o'] - df['adj_d']

            # Add rank if not present
            if 'rank' not in df.columns:
                df = df.sort_values('adj_em', ascending=False).reset_index(drop=True)
                df['rank'] = df.index + 1

            # Add placeholder columns
            for col in ['luck', 'sos_adj_em', 'adj_o_rank', 'adj_d_rank', 'adj_t_rank']:
                if col not in df.columns:
                    df[col] = 0

            print(f"Barttorvik: Fetched {len(df)} teams")
            return df

        except Exception as e:
            print(f"Barttorvik: Fetch error - {e}")
            return None

    def _load_cache(self) -> Optional[Dict[str, Dict]]:
        """Load ratings from cache file."""
        if not RATINGS_CACHE_FILE.exists():
            return None

        try:
            with open(RATINGS_CACHE_FILE) as f:
                cache = json.load(f)

            cache_time = datetime.fromisoformat(cache.get('timestamp', '2000-01-01'))
            age_hours = (datetime.now() - cache_time).total_seconds() / 3600

            if age_hours > CACHE_DURATION_HOURS:
                print(f"Cache: Expired ({age_hours:.1f} hours old)")
                return None

            teams = cache.get('teams', {})
            source = cache.get('source', 'unknown')
            print(f"Cache: Loaded {len(teams)} teams from {source} ({age_hours:.1f} hours old)")
            return teams

        except Exception as e:
            print(f"Cache: Load error - {e}")
            return None

    def _save_cache(self, ratings: Dict[str, Dict], source: str):
        """Save ratings to cache file."""
        try:
            cache = {
                'timestamp': datetime.now().isoformat(),
                'source': source,
                'teams': ratings
            }
            with open(RATINGS_CACHE_FILE, 'w') as f:
                json.dump(cache, f, indent=2)
            print(f"Cache: Saved {len(ratings)} teams from {source}")
        except Exception as e:
            print(f"Cache: Save error - {e}")

    def get_current_ratings(self, force_refresh: bool = False) -> Dict[str, Dict]:
        """
        Get current team ratings.

        Priority:
        1. Memory cache (if recent)
        2. File cache (if recent)
        3. KenPom.com (if credentials available)
        4. Barttorvik.com (free fallback)

        Returns:
            Dict mapping team name to stats dict
        """
        # Check memory cache
        if not force_refresh and self._ratings_cache and self._cache_time:
            age_hours = (datetime.now() - self._cache_time).total_seconds() / 3600
            if age_hours < CACHE_DURATION_HOURS:
                return self._ratings_cache

        # Check file cache
        if not force_refresh:
            cached = self._load_cache()
            if cached:
                self._ratings_cache = cached
                self._cache_time = datetime.now()
                return cached

        # Try KenPom first
        df = None
        source = None

        if self.email and self.password:
            df = self._fetch_kenpom_ratings()
            if df is not None:
                source = 'KenPom'

        # Fallback to Barttorvik
        if df is None:
            print("Falling back to Barttorvik (free KenPom alternative)...")
            df = self._fetch_barttorvik_ratings()
            if df is not None:
                source = 'Barttorvik'

        if df is None:
            print("WARNING: Could not fetch live ratings")
            if self._ratings_cache:
                print("Using stale memory cache")
                return self._ratings_cache
            # Try loading stale file cache as last resort
            if RATINGS_CACHE_FILE.exists():
                try:
                    with open(RATINGS_CACHE_FILE) as f:
                        cache = json.load(f)
                    print("Using stale file cache as last resort")
                    return cache.get('teams', {})
                except:
                    pass
            return {}

        # Convert DataFrame to dict
        ratings = {}
        for _, row in df.iterrows():
            team_name = str(row.get('team', '')).strip()
            if not team_name:
                continue

            stats = {
                'name': team_name,
                'rank': int(row.get('rank', 999)),
                'adj_em': float(row.get('adj_em', 0)),
                'adj_o': float(row.get('adj_o', 100)),
                'adj_d': float(row.get('adj_d', 100)),
                'adj_t': float(row.get('adj_t', 68)),
                'luck': float(row.get('luck', 0)),
                'sos': float(row.get('sos_adj_em', 0)),
                'conf': str(row.get('conf', '')),
                'record': str(row.get('record', '')),
            }

            # Add derived four factors
            self._add_four_factors(stats)
            ratings[team_name] = stats

        # Cache results
        self._ratings_cache = ratings
        self._cache_time = datetime.now()
        self._save_cache(ratings, source)

        print(f"Loaded {len(ratings)} teams from {source}")
        return ratings

    def _add_four_factors(self, stats: Dict):
        """Add estimated four factors based on efficiency ratings."""
        quality = stats.get('adj_em', 0) / 30  # Normalize to ~[-1, 1]

        stats['efg_o'] = round(np.clip(0.51 + quality * 0.03, 0.44, 0.58), 4)
        stats['efg_d'] = round(np.clip(0.50 - quality * 0.02, 0.44, 0.56), 4)
        stats['tov_o'] = round(np.clip(0.17 - quality * 0.015, 0.12, 0.24), 4)
        stats['tov_d'] = round(np.clip(0.17 + quality * 0.012, 0.12, 0.24), 4)
        stats['orb_o'] = round(np.clip(0.29 + quality * 0.015, 0.22, 0.36), 4)
        stats['drb_d'] = round(np.clip(0.73 + quality * 0.012, 0.66, 0.80), 4)
        stats['ftr_o'] = round(np.clip(0.32 + quality * 0.015, 0.22, 0.42), 4)
        stats['ftr_d'] = round(np.clip(0.30 - quality * 0.012, 0.22, 0.40), 4)

    def get_team(self, team_name: str) -> Optional[Dict]:
        """Get stats for a specific team with fuzzy matching."""
        ratings = self.get_current_ratings()

        # Exact match
        if team_name in ratings:
            return ratings[team_name]

        # Case-insensitive exact match
        team_lower = team_name.lower()
        for name, stats in ratings.items():
            if name.lower() == team_lower:
                return stats

        # Fuzzy match (contains)
        for name, stats in ratings.items():
            if team_lower in name.lower() or name.lower() in team_lower:
                return stats

        # Try common name variations
        variations = {
            'UConn': 'Connecticut',
            'Connecticut': 'UConn',
            'NC State': 'N.C. State',
            'N.C. State': 'NC State',
            'UNC': 'North Carolina',
            'North Carolina': 'UNC',
            'USC': 'Southern California',
            'LSU': 'Louisiana St.',
            'UCF': 'Central Florida',
            'UNLV': 'Nevada-Las Vegas',
            'SMU': 'Southern Methodist',
            'TCU': 'Texas Christian',
            'VCU': 'Virginia Commonwealth',
            'BYU': 'Brigham Young',
        }

        if team_name in variations:
            alt_name = variations[team_name]
            for name, stats in ratings.items():
                if alt_name.lower() in name.lower():
                    return stats

        return None

    def get_top_teams(self, n: int = 25) -> List[Dict]:
        """Get top N teams by AdjEM."""
        ratings = self.get_current_ratings()
        sorted_teams = sorted(ratings.values(), key=lambda x: x.get('rank', 999))
        return sorted_teams[:n]


class LiveTeamDatabase:
    """
    Team database that fetches live data from KenPom/Barttorvik.

    Drop-in replacement for the hardcoded TeamDatabase.
    """

    def __init__(self, kenpom_email: str = None, kenpom_password: str = None):
        self.client = KenPomClient(email=kenpom_email, password=kenpom_password)
        self._teams: Optional[Dict[str, Dict]] = None

    @property
    def teams(self) -> Dict[str, Dict]:
        if self._teams is None:
            self._teams = self.client.get_current_ratings()
        return self._teams

    def get(self, team: str) -> Optional[Dict]:
        """Get team stats by name."""
        return self.client.get_team(team)

    def refresh(self):
        """Force refresh of team data."""
        self._teams = self.client.get_current_ratings(force_refresh=True)


# =============================================================================
# MAIN - Test the client
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("KENPOM CLIENT TEST")
    print("=" * 70)

    # Check for credentials
    email = os.environ.get('KENPOM_EMAIL')
    password = os.environ.get('KENPOM_PASSWORD')

    if email:
        print(f"KenPom credentials found for: {email}")
    else:
        print("No KenPom credentials - will use Barttorvik fallback")

    # Test fetch
    client = KenPomClient(email=email, password=password)
    ratings = client.get_current_ratings()

    print(f"\n{'='*70}")
    print(f"RESULTS: Loaded {len(ratings)} teams")
    print(f"{'='*70}")

    # Show top 25
    if ratings:
        print("\nTop 25 Teams:")
        print("-" * 60)
        sorted_teams = sorted(ratings.items(), key=lambda x: x[1].get('rank', 999))
        for name, stats in sorted_teams[:25]:
            print(f"  {stats['rank']:3d}. {name:25s} AdjEM: {stats['adj_em']:+6.2f}  "
                  f"O: {stats['adj_o']:.1f}  D: {stats['adj_d']:.1f}")

        print(f"\n{'='*70}")
        print("CLIENT TEST COMPLETE")
        print(f"{'='*70}")
