#!/usr/bin/env python3
"""
================================================================================
KenPom Data Client - NCAA College Basketball
================================================================================

Fetches team ratings from KenPom.com (requires subscription).
Falls back to Barttorvik.com (free, same metrics) if KenPom unavailable.

Usage:
    client = KenPomClient(email="your@email.com", password="yourpass")
    ratings = client.get_current_ratings()

Environment Variables:
    KENPOM_EMAIL - Your KenPom subscription email
    KENPOM_PASSWORD - Your KenPom subscription password

================================================================================
"""

import os
import requests
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import json
import sqlite3
from pathlib import Path

# Cache settings
CACHE_DIR = Path(__file__).parent / "data"
CACHE_DIR.mkdir(exist_ok=True)
RATINGS_CACHE_FILE = CACHE_DIR / "team_ratings_cache.json"
CACHE_DURATION_HOURS = 6  # Refresh every 6 hours


class KenPomClient:
    """
    Client for fetching KenPom team ratings.

    Primary: KenPom.com (requires subscription)
    Fallback: Barttorvik.com (free, same data format)
    """

    KENPOM_LOGIN_URL = "https://kenpom.com/login.php"
    KENPOM_RATINGS_URL = "https://kenpom.com/index.php"
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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self._logged_in = False
        self._ratings_cache: Dict[str, Dict] = {}
        self._cache_time: Optional[datetime] = None

    def _login_kenpom(self) -> bool:
        """Login to KenPom with credentials."""
        if not self.email or not self.password:
            print("KenPom credentials not provided")
            return False

        try:
            # Get login page for CSRF token
            login_page = self.session.get(self.KENPOM_LOGIN_URL, timeout=30)

            # Submit login form
            login_data = {
                'email': self.email,
                'password': self.password,
                'submit': 'Login'
            }

            response = self.session.post(
                self.KENPOM_LOGIN_URL,
                data=login_data,
                timeout=30
            )

            # Check if login successful (redirects to main page)
            if 'logout' in response.text.lower() or response.url == self.KENPOM_RATINGS_URL:
                self._logged_in = True
                print("KenPom login successful")
                return True
            else:
                print("KenPom login failed - check credentials")
                return False

        except Exception as e:
            print(f"KenPom login error: {e}")
            return False

    def _fetch_kenpom_ratings(self) -> Optional[pd.DataFrame]:
        """Fetch ratings from KenPom.com."""
        if not self._logged_in:
            if not self._login_kenpom():
                return None

        try:
            response = self.session.get(self.KENPOM_RATINGS_URL, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', {'id': 'ratings-table'})

            if not table:
                print("Could not find ratings table on KenPom")
                return None

            # Parse table rows
            rows = []
            for tr in table.find_all('tr')[1:]:  # Skip header
                cols = tr.find_all('td')
                if len(cols) >= 12:
                    try:
                        row = {
                            'rank': int(cols[0].text.strip()),
                            'team': cols[1].text.strip(),
                            'conf': cols[2].text.strip(),
                            'record': cols[3].text.strip(),
                            'adj_em': float(cols[4].text.strip()),
                            'adj_o': float(cols[5].text.strip()),
                            'adj_o_rank': int(cols[6].text.strip()),
                            'adj_d': float(cols[7].text.strip()),
                            'adj_d_rank': int(cols[8].text.strip()),
                            'adj_t': float(cols[9].text.strip()),
                            'adj_t_rank': int(cols[10].text.strip()),
                            'luck': float(cols[11].text.strip()) if cols[11].text.strip() else 0,
                            'sos': float(cols[12].text.strip()) if len(cols) > 12 else 0,
                        }
                        rows.append(row)
                    except (ValueError, IndexError):
                        continue

            if rows:
                df = pd.DataFrame(rows)
                print(f"Fetched {len(df)} teams from KenPom")
                return df

            return None

        except Exception as e:
            print(f"KenPom fetch error: {e}")
            return None

    def _fetch_barttorvik_ratings(self, season: int = 2025) -> Optional[pd.DataFrame]:
        """Fetch ratings from Barttorvik (free KenPom alternative)."""
        url = f"{self.BARTTORVIK_URL}?year={season}&csv=1"

        try:
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
            }

            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

            # Calculate adj_em if not present
            if 'adj_em' not in df.columns and 'adj_o' in df.columns and 'adj_d' in df.columns:
                df['adj_em'] = df['adj_o'] - df['adj_d']

            # Add rank
            df = df.sort_values('adj_em', ascending=False).reset_index(drop=True)
            df['rank'] = df.index + 1

            # Add placeholder columns
            for col in ['luck', 'sos', 'adj_o_rank', 'adj_d_rank', 'adj_t_rank']:
                if col not in df.columns:
                    df[col] = 0

            print(f"Fetched {len(df)} teams from Barttorvik")
            return df

        except Exception as e:
            print(f"Barttorvik fetch error: {e}")
            return None

    def _load_cache(self) -> Optional[Dict[str, Dict]]:
        """Load ratings from cache file."""
        if not RATINGS_CACHE_FILE.exists():
            return None

        try:
            with open(RATINGS_CACHE_FILE) as f:
                cache = json.load(f)

            cache_time = datetime.fromisoformat(cache.get('timestamp', '2000-01-01'))
            if datetime.now() - cache_time > timedelta(hours=CACHE_DURATION_HOURS):
                print("Cache expired")
                return None

            print(f"Loaded {len(cache.get('teams', {}))} teams from cache")
            return cache.get('teams', {})

        except Exception as e:
            print(f"Cache load error: {e}")
            return None

    def _save_cache(self, ratings: Dict[str, Dict]):
        """Save ratings to cache file."""
        try:
            cache = {
                'timestamp': datetime.now().isoformat(),
                'teams': ratings
            }
            with open(RATINGS_CACHE_FILE, 'w') as f:
                json.dump(cache, f, indent=2)
            print(f"Cached {len(ratings)} teams")
        except Exception as e:
            print(f"Cache save error: {e}")

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
            if datetime.now() - self._cache_time < timedelta(hours=CACHE_DURATION_HOURS):
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
        if self.email and self.password:
            df = self._fetch_kenpom_ratings()

        # Fallback to Barttorvik
        if df is None:
            print("Falling back to Barttorvik...")
            df = self._fetch_barttorvik_ratings()

        if df is None:
            print("WARNING: Could not fetch live ratings, using cache or defaults")
            if self._ratings_cache:
                return self._ratings_cache
            return self._get_default_ratings()

        # Convert DataFrame to dict
        ratings = {}
        for _, row in df.iterrows():
            team_name = row['team']
            ratings[team_name] = {
                'name': team_name,
                'rank': int(row.get('rank', 0)),
                'adj_em': float(row.get('adj_em', 0)),
                'adj_o': float(row.get('adj_o', 100)),
                'adj_d': float(row.get('adj_d', 100)),
                'adj_t': float(row.get('adj_t', 68)),
                'luck': float(row.get('luck', 0)),
                'sos': float(row.get('sos', 0)),
                'conf': row.get('conf', ''),
                'record': row.get('record', ''),
            }

            # Add derived four factors if available
            self._add_four_factors(ratings[team_name])

        # Cache results
        self._ratings_cache = ratings
        self._cache_time = datetime.now()
        self._save_cache(ratings)

        return ratings

    def _add_four_factors(self, stats: Dict):
        """Add estimated four factors based on efficiency ratings."""
        quality = stats.get('adj_em', 0) / 30  # Normalize to ~[-1, 1]

        stats['efg_o'] = np.clip(0.51 + quality * 0.03, 0.44, 0.58)
        stats['efg_d'] = np.clip(0.50 - quality * 0.02, 0.44, 0.56)
        stats['tov_o'] = np.clip(0.17 - quality * 0.015, 0.12, 0.24)
        stats['tov_d'] = np.clip(0.17 + quality * 0.012, 0.12, 0.24)
        stats['orb_o'] = np.clip(0.29 + quality * 0.015, 0.22, 0.36)
        stats['drb_d'] = np.clip(0.73 + quality * 0.012, 0.66, 0.80)
        stats['ftr_o'] = np.clip(0.32 + quality * 0.015, 0.22, 0.42)
        stats['ftr_d'] = np.clip(0.30 - quality * 0.012, 0.22, 0.40)

    def _get_default_ratings(self) -> Dict[str, Dict]:
        """Return minimal default ratings if all else fails."""
        print("WARNING: Using hardcoded default ratings")
        defaults = {}
        # This should rarely be used - only if network completely fails
        # and no cache exists
        return defaults

    def get_team(self, team_name: str) -> Optional[Dict]:
        """Get stats for a specific team with fuzzy matching."""
        ratings = self.get_current_ratings()

        # Exact match
        if team_name in ratings:
            return ratings[team_name]

        # Fuzzy match
        team_lower = team_name.lower()
        for name, stats in ratings.items():
            if name.lower() in team_lower or team_lower in name.lower():
                return stats

        return None


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
    print("=" * 60)
    print("KENPOM CLIENT TEST")
    print("=" * 60)

    # Check for credentials
    email = os.environ.get('KENPOM_EMAIL')
    password = os.environ.get('KENPOM_PASSWORD')

    if email and password:
        print(f"KenPom credentials found for: {email}")
    else:
        print("No KenPom credentials - will use Barttorvik")

    # Test fetch
    client = KenPomClient(email=email, password=password)
    ratings = client.get_current_ratings()

    print(f"\nLoaded {len(ratings)} teams")

    # Show top 10
    if ratings:
        sorted_teams = sorted(ratings.items(), key=lambda x: x[1].get('rank', 999))
        print("\nTop 10 Teams:")
        for name, stats in sorted_teams[:10]:
            print(f"  {stats['rank']:3d}. {name:20s} AdjEM: {stats['adj_em']:+6.1f}")
