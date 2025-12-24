# 🏀 NCAA College Basketball Predictive Model

[![Daily Training](https://github.com/Risky-Scout/NCAA_College_Basketball_Predictive_Model/actions/workflows/daily_training.yml/badge.svg)](https://github.com/Risky-Scout/NCAA_College_Basketball_Predictive_Model/actions/workflows/daily_training.yml)
[![Daily Predictions](https://github.com/Risky-Scout/NCAA_College_Basketball_Predictive_Model/actions/workflows/daily_predictions.yml/badge.svg)](https://github.com/Risky-Scout/NCAA_College_Basketball_Predictive_Model/actions/workflows/daily_predictions.yml)

**Syndicate-Grade College Basketball Prediction System**

This model powers **Phase 2** of the [WizardofOdds.com](https://wizardofodds.com) Sports Betting Hub, providing mathematically rigorous betting recommendations with proper edge quantification.

---

## 🎯 Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    META-ENSEMBLE ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Base Models                                           │
│  ├── XGBoost (gradient boosting)                               │
│  ├── LightGBM (leaf-wise growth)                               │
│  └── CatBoost (ordered boosting)                               │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Meta-Learner                                          │
│  ├── Ridge Regression (margin/total)                           │
│  └── Logistic Regression (win probability)                     │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Calibration                                           │
│  └── Isotonic Regression (probability calibration)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Features

### Core Efficiency Metrics
- Adjusted Efficiency Margin (AdjEM) differential
- Adjusted Offensive/Defensive Efficiency
- Tempo interactions

### Four Factors (Dean Oliver)
- Effective Field Goal % (eFG%)
- Turnover Rate (TOV%)
- Offensive Rebound Rate (ORB%)
- Free Throw Rate (FTR)

### Situational Factors
- Rest days differential
- Back-to-back detection
- Travel fatigue estimation
- Conference game adjustments
- Rivalry indicators

### Market Features
- Line movement tracking
- Steam move detection
- Reverse line movement (sharp action)
- Public betting percentage estimation

---

## 🔄 Automated Pipeline

### Daily Training (7:00 AM EST)
```yaml
Schedule: 0 12 * * *  # 7 AM EST = 12 PM UTC
```
- Fetches latest game results
- Updates team ratings
- Retrains model with recency weighting
- Validates against holdout set
- Commits updated model

### Daily Predictions (4:30 PM EST)  
```yaml
Schedule: 0 21 30 * * *  # 4:30 PM EST = 9:30 PM UTC
```
- Fetches live odds from The-Odds-API
- Generates predictions for all games
- Calculates edges and Kelly sizing
- Outputs to `predictions/YYYY-MM-DD.json`
- Commits predictions

---

## 🛡️ Data Leakage Protection

This model implements **strict temporal validation** to prevent data leakage:

```python
# ❌ WRONG - Data Leakage
features['rolling_avg'] = df['points'].rolling(5).mean()

# ✅ CORRECT - No Leakage  
features['rolling_avg'] = df['points'].shift(1).rolling(5).mean()
```

### Safeguards Implemented:
1. **Temporal Train/Test Split**: Only past data used for training
2. **Walk-Forward Validation**: Time-series cross-validation
3. **Feature Shift**: All rolling features use `shift(1)` 
4. **No Future Information**: Team ratings as of game date only
5. **Proper Indexing**: Features computed before game, not after

---

## 📁 Repository Structure

```
NCAA_College_Basketball_Predictive_Model/
├── .github/
│   └── workflows/
│       ├── daily_training.yml      # Automated training
│       └── daily_predictions.yml   # Automated predictions
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration & API keys
│   ├── data_pipeline.py            # Data fetching & processing
│   ├── feature_engineering.py      # Feature creation (leak-proof)
│   ├── meta_ensemble.py            # Model architecture
│   ├── train.py                    # Training script
│   ├── predict.py                  # Prediction script
│   ├── bet_analysis.py             # Edge detection & Kelly
│   └── utils.py                    # Helper functions
├── data/
│   ├── teams.json                  # Team database
│   ├── historical_games.csv        # Game results
│   └── team_ratings.csv            # Daily ratings
├── models/
│   ├── model_latest.pkl            # Current production model
│   └── model_YYYYMMDD.pkl          # Versioned models
├── predictions/
│   └── YYYY-MM-DD.json             # Daily predictions
├── tests/
│   └── test_leakage.py             # Leakage detection tests
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Quick Start

### Local Setup
```bash
# Clone repository
git clone https://github.com/Risky-Scout/NCAA_College_Basketball_Predictive_Model.git
cd NCAA_College_Basketball_Predictive_Model

# Install dependencies
pip install -r requirements.txt

# Set API key
export ODDS_API_KEY="your_api_key_here"

# Run training
python src/train.py

# Run predictions
python src/predict.py
```

### GitHub Secrets Required
```
ODDS_API_KEY: Your The-Odds-API key
```

---

## 📈 Model Performance

| Metric | Value | Benchmark |
|--------|-------|-----------|
| Brier Score | 0.108 | < 0.25 is good |
| Margin RMSE | 10.6 | ~11 is expected |
| Total RMSE | 12.1 | ~12 is expected |
| Win% AUC | 0.72 | > 0.65 is good |

### CLV Tracking (Closing Line Value)
The gold standard for model validation. Positive CLV over 500+ bets indicates genuine edge.

---

## 🔗 Integration with WizardofOdds.com

Predictions flow to Phase 2 of the Sports Betting Hub via:

```javascript
// Fetch today's predictions
const response = await fetch(
  'https://raw.githubusercontent.com/Risky-Scout/NCAA_College_Basketball_Predictive_Model/main/predictions/latest.json'
);
const predictions = await response.json();
```

### Output Format
```json
{
  "generated_at": "2024-12-24T16:30:00Z",
  "model_version": "v2.1.0",
  "games": [
    {
      "game_id": "abc123",
      "home": "Auburn",
      "away": "Ohio State",
      "predictions": {
        "win_prob": 0.869,
        "margin": 11.0,
        "total": 148.4
      },
      "value_bets": [
        {
          "bet": "Auburn ML",
          "odds": -450,
          "edge": 0.083,
          "ev_pct": 6.3,
          "kelly": 0.05,
          "confidence": "HIGH"
        }
      ]
    }
  ]
}
```

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Joseph** - Actuarial Science (ASA, MAAA)  
Building mathematically rigorous betting tools for [WizardofOdds.com](https://wizardofodds.com)

---

*"The goal is not to predict the future, but to find where the market's probability estimates are wrong."*
