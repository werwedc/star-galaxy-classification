import os

# CPU is enough for this task
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

DATA_PATH = './'
OPTIMAL_THRESHOLD = 0.43
N_FOLDS = 5
# We use ANN specialists for the most frequent null signatures
SPECIALIST_PATTERNS = ['11111', '10111', '11110', '01111']

print("Loading Data...")
try:
    train_df = pd.read_csv(f"{DATA_PATH}data.csv", index_col=0)
    test_df = pd.read_csv(f"{DATA_PATH}test.csv", index_col=0)
except:
    # Google Drive data
    DATA_PATH = '/content/drive/MyDrive/Data/'
    train_df = pd.read_csv(f"{DATA_PATH}data.csv", index_col=0)
    test_df = pd.read_csv(f"{DATA_PATH}test.csv", index_col=0)

mag_cols = ['u', 'g', 'r', 'i', 'z']
err_cols = ['Err_u', 'Err_g', 'Err_r', 'Err_i', 'Err_z']
all_physics = mag_cols + err_cols


def get_signature(df):
    return df[mag_cols].notna().astype(int).astype(str).agg(''.join, axis=1)


train_df['signature'] = get_signature(train_df)
test_df['signature'] = get_signature(test_df)


def get_valid_features(pattern_str):
    valid_indices = [i for i, char in enumerate(pattern_str) if char == '1']
    valid_mags = [mag_cols[i] for i in valid_indices]
    valid_errs = [err_cols[i] for i in valid_indices]
    return valid_mags + valid_errs


le = LabelEncoder()
all_sigs = pd.concat([train_df['signature'], test_df['signature']]).unique()
le.fit(all_sigs)
train_df['pattern_id'] = le.transform(train_df['signature'])
test_df['pattern_id'] = le.transform(test_df['signature'])

tree_features = all_physics + ['pattern_id']
X_tree = train_df[tree_features].values
y_all = train_df['TARGET'].values
X_tree_test = test_df[tree_features].values


def build_ann(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss="binary_crossentropy")
    return model


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
test_probs_final = np.zeros(len(test_df))

print(f"\nStarting {N_FOLDS}-Fold Ensemble Training...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tree, y_all)):
    X_tr_tree, y_tr = X_tree[train_idx], y_all[train_idx]
    X_val_tree, y_val = X_tree[val_idx], y_all[val_idx]

    clf_xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                            eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=-1)
    clf_xgb.fit(X_tr_tree, y_tr)

    clf_lgb = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                             random_state=42, n_jobs=-1, verbose=-1)
    clf_lgb.fit(X_tr_tree, y_tr)

    clf_cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05,
                                 loss_function='Logloss', verbose=0, random_seed=42, allow_writing_files=False)
    clf_cat.fit(X_tr_tree, y_tr)

    val_preds_base = (clf_xgb.predict_proba(X_val_tree)[:, 1] +
                      clf_lgb.predict_proba(X_val_tree)[:, 1] +
                      clf_cat.predict_proba(X_val_tree)[:, 1]) / 3.0

    test_preds_base = (clf_xgb.predict_proba(X_tree_test)[:, 1] +
                       clf_lgb.predict_proba(X_tree_test)[:, 1] +
                       clf_cat.predict_proba(X_tree_test)[:, 1]) / 3.0

    val_preds_final = val_preds_base.copy()
    test_preds_final = test_preds_base.copy()

    for pattern in SPECIALIST_PATTERNS:
        active_feats = get_valid_features(pattern)
        tr_indices = train_df.index[train_idx]
        val_indices = train_df.index[val_idx]

        mask_tr = train_df.loc[tr_indices, 'signature'] == pattern
        mask_val = train_df.loc[val_indices, 'signature'] == pattern

        if mask_tr.sum() > 100:
            X_spec_tr = train_df.loc[tr_indices[mask_tr], active_feats].values
            y_spec_tr = train_df.loc[tr_indices[mask_tr], 'TARGET'].values

            scaler_spec = StandardScaler()
            X_spec_tr_scaled = scaler_spec.fit_transform(X_spec_tr)

            ann = build_ann(input_dim=len(active_feats))
            es = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            ann.fit(X_spec_tr_scaled, y_spec_tr, epochs=50, batch_size=32, callbacks=[es], verbose=0)

            if mask_val.sum() > 0:
                X_spec_val = train_df.loc[val_indices[mask_val], active_feats].values
                X_spec_val_scaled = scaler_spec.transform(X_spec_val)

                p_ann = ann.predict(X_spec_val_scaled).flatten()
                mask_val_array = (train_df.iloc[val_idx]['signature'] == pattern).values
                p_trees = val_preds_base[mask_val_array]

                p_combined = (p_trees * 3.0 + p_ann) / 4.0
                val_preds_final[mask_val_array] = p_combined

            mask_test = test_df['signature'] == pattern
            if mask_test.sum() > 0:
                X_spec_test = test_df.loc[mask_test, active_feats].values
                X_spec_test_scaled = scaler_spec.transform(X_spec_test)

                t_ann = ann.predict(X_spec_test_scaled).flatten()
                t_trees = test_preds_base[mask_test]

                t_combined = (t_trees * 3.0 + t_ann) / 4.0
                test_preds_final[mask_test] = t_combined

    fold_pred_classes = (val_preds_final >= OPTIMAL_THRESHOLD).astype(int)
    f1 = f1_score(y_val, fold_pred_classes)
    print(f"   Fold {fold + 1} F1: {f1:.4f}")

    test_probs_final += test_preds_final

avg_test_probs = test_probs_final / N_FOLDS
final_preds = (avg_test_probs >= OPTIMAL_THRESHOLD).astype(int)

# Save the submission to the competition format
submission = pd.DataFrame({
    'ID': test_df.index,
    'TARGET': final_preds
})
submission.to_csv("final_submission_specialists.csv", index=False)
print(f"\nSaved.")