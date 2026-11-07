import numpy as np, pandas as pd, tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ---------------- 1. Load ----------------
COLS = ['unit','cycle','setting1','setting2','setting3'] + [f's{i}' for i in range(1, 22)]
load = lambda p: pd.read_csv(p, sep=r'\s+', header=None, names=COLS)
train, test = load('Dataset/train_FD001.txt'), load('Dataset/test_FD001.txt')
y_test = np.loadtxt('Dataset/RUL_FD001.txt')

W   = 30      # window length (cycles)
CAP = 10**6     # piecewise-linear RUL cap  -> set to 10**6 once for the "uncapped" ablation

# ---------------- 2. Target ----------------
train['RUL'] = (train.groupby('unit')['cycle'].transform('max') - train['cycle']).clip(upper=CAP)

# FD001 = single operating condition -> settings carry no info; these sensors are constant
DROP  = ['setting1','setting2','setting3','s1','s5','s6','s10','s16','s18','s19']
FEATS = [c for c in COLS if c not in DROP + ['unit','cycle']]   # 14 sensors

# ---------------- 3. Engine-wise split (no window leakage) ----------------
rng = np.random.default_rng(42)
val_units = rng.choice(train['unit'].unique(), 20, replace=False)
tr = train[~train['unit'].isin(val_units)].copy()
va = train[ train['unit'].isin(val_units)].copy()

scaler = MinMaxScaler().fit(tr[FEATS])          # fit on train engines only
for df in (tr, va, test):
    df[FEATS] = scaler.transform(df[FEATS])

# ---------------- 4. Windows ----------------
def make_windows(df):
    X, y = [], []
    for _, g in df.groupby('unit'):
        a, r = g[FEATS].values, g['RUL'].values
        for i in range(len(g) - W + 1):
            X.append(a[i:i+W]); y.append(r[i+W-1])
    return np.array(X, np.float32), np.array(y, np.float32)

def last_windows(df):                            # one window per test engine
    out = []
    for _, g in df.groupby('unit'):
        a = g[FEATS].values
        if len(a) < W:
            a = np.vstack([np.repeat(a[:1], W - len(a), 0), a])
        out.append(a[-W:])
    return np.array(out, np.float32)

Xtr, ytr = make_windows(tr)
Xva, yva = make_windows(va)
Xte      = last_windows(test)
print(Xtr.shape, Xva.shape, Xte.shape)

# ---------------- 5. Metrics ----------------
rmse = lambda y, p: float(np.sqrt(np.mean((p - y) ** 2)))
def nasa_score(y, p):                            # late predictions penalised harder
    d = p - y
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))

# ---------------- 6. Models ----------------
def build_lstm():
    m = tf.keras.Sequential([
        tf.keras.layers.Input((W, len(FEATS))),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0), loss='mse')   # CHANGED: gradient clipping
    return m

flat = lambda x: x.reshape(len(x), -1)
Ftr, Fva, Fte = flat(Xtr), flat(Xva), flat(Xte)

YS = float(ytr.max())                            # CHANGED: target scale (125 when capped)

results = {'LSTM': [], 'XGBoost': [], 'RandomForest': []}
preds_lstm = None

for seed in [0, 1, 2]:
    # ---- LSTM (CHANGED: scaled target, longer patience, LR decay, per-seed print) ----
    tf.keras.utils.set_random_seed(seed)
    m = build_lstm()
    cbs = [tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)]
    m.fit(Xtr, ytr / YS, validation_data=(Xva, yva / YS), epochs=200, batch_size=256,
          callbacks=cbs, verbose=0)
    p = np.clip(m.predict(Xte, verbose=0).ravel() * YS, 0, None)
    r = (rmse(y_test, p), nasa_score(y_test, p))
    print(f"seed {seed} LSTM:    RMSE {r[0]:.2f} | score {r[1]:.0f}")
    results['LSTM'].append(r)
    if seed == 0: preds_lstm = p

    # ---- XGBoost (unchanged) ----
    xgb = XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=4, subsample=0.8,
                       colsample_bytree=0.5, early_stopping_rounds=50, random_state=seed)
    xgb.fit(Ftr, ytr, eval_set=[(Fva, yva)], verbose=False)
    p = np.clip(xgb.predict(Fte), 0, None)
    results['XGBoost'].append((rmse(y_test, p), nasa_score(y_test, p)))

    # ---- Random Forest (unchanged) ----
    rf = RandomForestRegressor(n_estimators=300, max_features=0.3, min_samples_leaf=5,
                               n_jobs=-1, random_state=seed).fit(Ftr, ytr)
    p = np.clip(rf.predict(Fte), 0, None)
    results['RandomForest'].append((rmse(y_test, p), nasa_score(y_test, p)))

# ---------------- 7. Report ----------------
print(f"\nFD001 official test set (100 engines), CAP={CAP}, 3 seeds")
for k, v in results.items():
    v = np.array(v)
    print(f"{k:13s} RMSE {v[:,0].mean():5.2f} ± {v[:,0].std():.2f} | "
          f"NASA score {v[:,1].mean():6.0f} ± {v[:,1].std():.0f}")

# ---------------- 8. Plot (seed 0 LSTM) ----------------
import matplotlib.pyplot as plt
o = np.argsort(y_test)
plt.figure(figsize=(10, 4))
plt.plot(y_test[o], label='True RUL'); plt.plot(preds_lstm[o], label='LSTM prediction', alpha=.8)
plt.xlabel('Test engines (sorted by true RUL)'); plt.ylabel('RUL (cycles)'); plt.legend()
plt.tight_layout(); plt.savefig(f'rul_predictions_cap{CAP}.png', dpi=150)