---
name: backtest-review
description: "Review backtest-related changes for event backtester, walk-forward, Monte Carlo"
whenToUse: "When reviewing changes to event_backtester.py, walkforward_backtester.py, or backtest-related modules"
---

# Backtest Review Skill

## Overview
Review changes to backtesting modules for correctness, especially:
- `event_backtester.py` — event-driven backtest (Order→Execution→Position→Risk→Report)
- `walkforward_backtester.py` — Walk-Forward + Bootstrap Monte Carlo
- Any module that affects backtest results

## Checklist

### Event Backtester
- [ ] Order lifecycle: Order → Execution → Position → Risk → Report
- [ ] Transaction costs modeled: commission 0.03% + stamp tax 0.1% + slippage
- [ ] Position tracking is accurate (long/short, P&L)
- [ ] Risk limits enforced
- [ ] No lookahead bias

### Walk-Forward
- [ ] Train/test window boundaries correct
- [ ] No data leakage between folds
- [ ] Bootstrap sampling preserves temporal order
- [ ] Monte Carlo confidence intervals computed correctly

### Metrics
- [ ] Sharpe ratio uses correct risk-free rate
- [ ] Max drawdown computed from equity curve
- [ ] Win rate, profit factor calculated correctly
- [ ] Annualization factor correct

## Common Pitfalls
- Using future data in backtest (lookahead bias)
- Not accounting for trading costs
- Ignoring position sizing limits
- Overfitting walk-forward parameters
