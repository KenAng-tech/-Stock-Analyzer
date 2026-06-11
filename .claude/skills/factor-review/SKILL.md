---
name: factor-review
description: "Review 21-factor model changes for momentum, value, volatility, volume, liquidity, quality, technical, sentiment"
whenToUse: "When reviewing changes to multi_factor_model_v2.py or any factor calculation module"
---

# Factor Model Review Skill

## Overview
Review changes to the 21-factor model:
- `multi_factor_model_v2.py` — main factor model

## 21 Factors by Category

### Momentum (5)
- Price momentum (1/3/6/12 month)
- Relative strength index
- Rate of change

### Value (3)
- P/E ratio
- P/B ratio
- EV/EBITDA

### Volatility (2)
- Historical volatility
- Downside deviation

### Volume (3)
- Volume trend
- Volume-price correlation
- OBV (On-Balance Volume)

### Liquidity (1)
- Turnover rate

### Quality (2)
- ROE
- Debt-to-equity

### Technical (4)
- MACD signal
- Bollinger Band position
- ATR-based signal
- Moving average crossover

### Sentiment (1)
- Chinese sentiment score

## Checklist
- [ ] Factor normalization correct (z-score/min-max)
- [ ] Missing data handled (NaN → 0 or median)
- [ ] No lookahead in factor calculation
- [ ] Factor orthogonality maintained
- [ ] Weight aggregation correct
- [ ] Final score range consistent

## Common Pitfalls
- Survivorship bias in factor calculation
- Not adjusting for stock splits/dividends
- Incorrect lookback window
- Overweighting correlated factors
