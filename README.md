## Project Overview

Binary classification of night-sky objects into stars and galaxies with an emphasis on coping with occasional null cells in photometric bands and their error estimates. The models consume magnitudes (`u`, `g`, `r`, `i`, `z`) and statistical errors

Originally built as an internal university competition entry

## Two Approaches

### 1. `3ModelSolution.py` – Tree Trio Solution

- Ensemble of **XGBoost**, **LightGBM**, and **CatBoost**.
- Trees natively handle NaN values
- This consistently outperformed a single XGBoost model on the public leaderboard while keeping runtime low.

### 2. `TopologyAware3Model+ANN_Solution.py` – ANN Specialists

- Starts with the same tree trio.
- Adds four specialist ANNs for the most frequent null signatures `['11111', '10111', '11110', '01111']`.
- In practice, the solution with ANNs didn`t surpass the tree-only ensemble, but matched it