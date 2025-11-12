# Childline Kenya Call Volume Prediction

Simple but effective ML pipeline for predicting hourly call volumes.

## 🎯 Results

- **Leaderboard Score**: ~28 RMSE
- **Approach**: XGBoost with 5 calendar features

### ⚠️ Important Caveat

The strong performance may be due to:
- **Overfitting to test set outliers**
- Test period having similar patterns to training
- Lucky feature alignment with test characteristics

**Not guaranteed to generalize to different time periods.**

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## 🔧 Features

Only 5 simple calendar features:
- `hour` - Hour of day (0-23)
- `dayofweek` - Day of week (0-6)
- `dayofyear` - Day of year (1-365)
- `dayofmonth` - Day of month (1-31)
- `weekofyear` - Week number (1-52)

## 🤖 Model

```python
XGBRegressor(
    n_estimators=42,
    learning_rate=0.11,
    max_depth=7,
    subsample=0.74
)
```

## 💡 Why It Works

✅ No recursive forecasting = no error accumulation  
✅ Calendar features always known for future  
✅ Simple patterns generalize well  

## ⚠️ Limitations

- May not generalize to different time periods
- No external factors (holidays, events)
- High score may be due to test set peculiarities
- Temporal stability assumptions

## 📊 Performance

| Metric | Training | Leaderboard |
|--------|----------|-------------|
| RMSE | ~12-15 | ~28 |
| MAE | ~8-10 | - |

## 📁 Structure

```
childline-kenya-prediction/
├── main.py
├── README.md
├── requirements.txt
├── data/
│   ├── train.csv
│   └── Sample_Submission.csv
└── outputs/
    ├── submission.csv
    └── feature_importance.csv
```

## 🔬 Lessons Learned

1. Simple can beat complex (sometimes)
2. Recursive forecasting often hurts more than helps
3. Calendar patterns are powerful
4. Single test score can be misleading
5. Overfitting to test is a real risk

## 📧 Contact

Your Name - nashaa182@gmail.com

---

**Disclaimer**: Performance may not replicate on different data. High score may partly reflect test set characteristics rather than true model quality. Always validate on multiple time periods.
