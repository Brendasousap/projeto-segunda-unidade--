"""
Problema 1 — Adaline para Ionosphere
Instale: pip install numpy pandas scikit-learn matplotlib ucimlrepo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neural_network import MLPRegressor

# Carrega o dataset UCI Ionosphere
ionosphere_ae = fetch_ucirepo(id=52)
X_ae = ionosphere_ae.data.features.copy()
y_raw_ae = ionosphere_ae.data.targets.iloc[:, 0].astype(str).str.lower()

# UCI usa g/b: good -> +1; bad -> -1
y_ae = y_raw_ae.map({"g": 1, "good": 1, "b": -1, "bad": -1})
if y_ae.isna().any():
    raise ValueError(f"Rótulos inesperados: {y_raw_ae[y_ae.isna()].unique()}")

# Verificações e remoção de colunas constantes
print("Valores ausentes por coluna:\n", X_ae.isna().sum()[lambda s: s > 0])
X_ae = X_ae.select_dtypes(include=[np.number])
constantes_ae = X_ae.columns[X_ae.nunique(dropna=False) <= 1].tolist()
print("Atributos constantes removidos:", constantes_ae)
X_ae = X_ae.drop(columns=constantes_ae)
X_ae = X_ae.fillna(X_ae.median(numeric_only=True))

X_train_ae, X_test_ae, y_train_ae, y_test_ae = train_test_split(
    X_ae, y_ae, test_size=0.20, random_state=42, stratify=y_ae
)
scaler_ae = StandardScaler()
X_train_ae = scaler_ae.fit_transform(X_train_ae)
X_test_ae = scaler_ae.transform(X_test_ae)

# Adaline por descida do gradiente batch:
# saída linear; minimiza MSE. Predição final usa sinal (limiar 0).
etas_ae = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
epocas_ae = 200
perdas_ae = {}
modelos_ae = {}

Xb_train_ae = np.c_[np.ones(len(X_train_ae)), X_train_ae]
Xb_test_ae = np.c_[np.ones(len(X_test_ae)), X_test_ae]
yt_ae = np.asarray(y_train_ae, dtype=float)

for eta_ae in etas_ae:
    rng_ae = np.random.default_rng(42)
    w_ae = rng_ae.normal(0, 0.01, Xb_train_ae.shape[1])
    hist_ae = []
    for epoca_ae in range(epocas_ae):
        saida_ae = Xb_train_ae @ w_ae
        erro_ae = yt_ae - saida_ae
        loss_ae = np.mean(erro_ae ** 2) / 2
        hist_ae.append(loss_ae)
        grad_ae = -(Xb_train_ae.T @ erro_ae) / len(yt_ae)
        w_ae -= eta_ae * grad_ae
        if not np.isfinite(w_ae).all():
            hist_ae = [np.inf] * epocas_ae
            break
    perdas_ae[eta_ae] = hist_ae
    modelos_ae[eta_ae] = w_ae

plt.figure(figsize=(9, 5))
for eta_ae, hist_ae in perdas_ae.items():
    plt.plot(range(1, len(hist_ae)+1), hist_ae, label=f"eta={eta_ae:g}")
plt.xlabel("Época")
plt.ylabel("Loss (MSE/2)")
plt.title("Adaline — convergência por taxa de aprendizagem")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("p1_curvas_adaline.png", dpi=160)
plt.show()

eta_melhor_ae = min(
    [e for e in etas_ae if np.isfinite(perdas_ae[e][-1])],
    key=lambda e: perdas_ae[e][-1]
)
print("Eta selecionado pela menor loss final:", eta_melhor_ae)
w_melhor_ae = modelos_ae[eta_melhor_ae]
pred_cont_ae = Xb_test_ae @ w_melhor_ae
y_pred_ae = np.where(pred_cont_ae >= 0, 1, -1)

print("Matriz de confusão (ordem [-1, +1]):\n",
      confusion_matrix(y_test_ae, y_pred_ae, labels=[-1, 1]))
print("Acurácia:", accuracy_score(y_test_ae, y_pred_ae))
print(classification_report(y_test_ae, y_pred_ae, labels=[-1, 1],
                            target_names=["bad (-1)", "good (+1)"], zero_division=0))
