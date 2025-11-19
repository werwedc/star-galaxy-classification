## Project Overview

Binary classification of night-sky objects into stars and galaxies. The models consume magnitudes (`u`, `g`, `r`, `i`, `z`) and statistical errors of these magnitudes

Originally built as an internal university competition entry

## Two Approaches

### 1. `3ModelSolution.py` – Tree Trio Solution

- Ensemble of **XGBoost**, **LightGBM**, and **CatBoost**.
- Trees natively handle NaN values
- This consistently outperformed a single XGBoost model on the public leaderboard while keeping runtime low.

### 2. `TopologyAware3Model+ANN_Solution.py` – ANN Specialists

- Starts with the same tree trio.
- Adds four specialist ANNs for the most frequent null signatures `['11111', '10111', '11110', '01111']`.

### 3. Solo LGBM
- Turned out to perform the best

F1 score on test dataset
| Approach     | F1 Score   |
| -------------| ---------- |
| Solo LGBM    | 0.8506     |
| Solo XGBoost | 0.8473     |
| Solo CatBoost| 0.8280 |
| Trio Solution | 0.8486|
| Trio Solution + 4 ANNs| 0.8483|
