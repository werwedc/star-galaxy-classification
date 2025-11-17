import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

DATA_PATH = './'
OPTIMAL_THRESHOLD = 0.43
N_FOLDS = 5

print("Loading Data...")
try:
    train_df = pd.read_csv(f"{DATA_PATH}data.csv", index_col=0)
    test_df = pd.read_csv(f"{DATA_PATH}test.csv", index_col=0)
except:
    DATA_PATH = '/content/drive/MyDrive/Data/'
    train_df = pd.read_csv(f"{DATA_PATH}data.csv", index_col=0)
    test_df = pd.read_csv(f"{DATA_PATH}test.csv", index_col=0)


def add_features(df):
    df = df.copy()
    mag_cols = ['u', 'g', 'r', 'i', 'z']
    df['pattern_id'] = df[mag_cols].notna().astype(int).astype(str).agg(''.join, axis=1)
    return df


train_df = add_features(train_df)
test_df = add_features(test_df)

le = LabelEncoder()
all_patterns = pd.concat([train_df['pattern_id'], test_df['pattern_id']]).unique()
le.fit(all_patterns)
train_df['pattern_id'] = le.transform(train_df['pattern_id'])
test_df['pattern_id'] = le.transform(test_df['pattern_id'])

features = ['u', 'g', 'r', 'i', 'z', 'Err_u', 'Err_g', 'Err_r', 'Err_i', 'Err_z', 'pattern_id']

X = train_df[features]
y = train_df['TARGET']
X_test = test_df[features]

print(f"Data Ready. Training on {len(X)} rows. Predicting for {len(X_test)} rows.")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
test_probs_sum = np.zeros(len(X_test))

print(f"\nStarting {N_FOLDS}-Fold Ensemble Training...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    clf_xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=-1
    )

    clf_lgb = LGBMClassifier(
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        random_state=42, n_jobs=-1, verbose=-1
    )

    clf_cat = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        loss_function='Logloss', verbose=0, random_seed=42, allow_writing_files=False
    )

    ensemble = VotingClassifier(
        estimators=[('xgb', clf_xgb), ('lgb', clf_lgb), ('cat', clf_cat)],
        voting='soft'
    )

    ensemble.fit(X_tr, y_tr)
    test_probs_sum += ensemble.predict_proba(X_test)[:, 1]

    print(f"   Fold {fold + 1}/{N_FOLDS} Complete.")

avg_test_probs = test_probs_sum / N_FOLDS
final_preds = (avg_test_probs >= OPTIMAL_THRESHOLD).astype(int)

submission = pd.DataFrame({
    'ID': test_df.index,
    'TARGET': final_preds
})

submission['ID'] = submission['ID'].astype(int)

filename = "final_submission.csv"
save_path = f"{DATA_PATH}{filename}"
submission.to_csv(save_path, index=False)
