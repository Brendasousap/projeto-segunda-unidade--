"""
Problema 2 — MLP para NSL-KDD
Instale: pip install pandas numpy scikit-learn torch matplotlib
Baixe KDDTrain+.txt e KDDTest+.txt do dataset NSL-KDD e coloque-os
na mesma pasta deste script.
"""
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

col_names_ae = [
"duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
"urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
"num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
"is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
"rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
"dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
"dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
"dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
"attack_type","difficulty_level"
]
train_ae = pd.read_csv("KDDTrain+.txt", names=col_names_ae)
test_ae = pd.read_csv("KDDTest+.txt", names=col_names_ae)
y_train_ae = (train_ae["attack_type"].astype(str).str.lower() != "normal").astype(np.float32).to_numpy()
y_test_ae = (test_ae["attack_type"].astype(str).str.lower() != "normal").astype(np.float32).to_numpy()

X_train_df_ae = train_ae.drop(columns=["attack_type","difficulty_level"])
X_test_df_ae = test_ae.drop(columns=["attack_type","difficulty_level"])
categoricas_ae = ["protocol_type","service","flag"]
X_train_df_ae = pd.get_dummies(X_train_df_ae, columns=categoricas_ae)
X_test_df_ae = pd.get_dummies(X_test_df_ae, columns=categoricas_ae)
X_train_df_ae, X_test_df_ae = X_train_df_ae.align(X_test_df_ae, join="left", axis=1, fill_value=0)
X_test_df_ae = X_test_df_ae.reindex(columns=X_train_df_ae.columns, fill_value=0)

# Reserva os últimos 20% do treino como validação, sem embaralhar a divisão.
n_ae = len(X_train_df_ae)
n_val_ae = int(0.20 * n_ae)
X_tr_df_ae, X_val_df_ae = X_train_df_ae.iloc[:-n_val_ae], X_train_df_ae.iloc[-n_val_ae:]
y_tr_ae, y_val_ae = y_train_ae[:-n_val_ae], y_train_ae[-n_val_ae:]
scaler_ae = StandardScaler()
X_tr_ae = scaler_ae.fit_transform(X_tr_df_ae).astype(np.float32)
X_val_ae = scaler_ae.transform(X_val_df_ae).astype(np.float32)
X_te_ae = scaler_ae.transform(X_test_df_ae).astype(np.float32)

torch.manual_seed(42)
device_ae = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MLP_ae(nn.Module):
    def __init__(self, n_features_ae):
        super().__init__()
        self.net_ae = nn.Sequential(
            nn.Linear(n_features_ae, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x_ae):
        return self.net_ae(x_ae)

train_ds_ae = TensorDataset(torch.tensor(X_tr_ae), torch.tensor(y_tr_ae).reshape(-1,1))
train_dl_ae = DataLoader(train_ds_ae, batch_size=256, shuffle=True)
Xv_ae, yv_ae = torch.tensor(X_val_ae).to(device_ae), torch.tensor(y_val_ae).reshape(-1,1).to(device_ae)
Xt_ae, yt_ae = torch.tensor(X_te_ae).to(device_ae), torch.tensor(y_test_ae).reshape(-1,1).to(device_ae)

model_ae = MLP_ae(X_tr_ae.shape[1]).to(device_ae)
loss_fn_ae = nn.BCEWithLogitsLoss()
optimizer_ae = torch.optim.Adam(model_ae.parameters(), lr=1e-3)
train_losses_ae, val_losses_ae = [], []
best_val_ae, best_state_ae, patience_ae, wait_ae = float("inf"), None, 6, 0

for epoch_ae in range(50):
    model_ae.train()
    total_ae, count_ae = 0.0, 0
    for xb_ae, yb_ae in train_dl_ae:
        xb_ae, yb_ae = xb_ae.to(device_ae), yb_ae.to(device_ae)
        optimizer_ae.zero_grad()
        logits_ae = model_ae(xb_ae)
        loss_ae = loss_fn_ae(logits_ae, yb_ae)
        loss_ae.backward()
        optimizer_ae.step()
        total_ae += loss_ae.item() * len(xb_ae)
        count_ae += len(xb_ae)
    train_loss_ae = total_ae / count_ae
    model_ae.eval()
    with torch.no_grad():
        val_loss_ae = loss_fn_ae(model_ae(Xv_ae), yv_ae).item()
    train_losses_ae.append(train_loss_ae); val_losses_ae.append(val_loss_ae)
    print(f"Época {epoch_ae+1:02d}: treino={train_loss_ae:.4f}, validação={val_loss_ae:.4f}")
    if val_loss_ae < best_val_ae:
        best_val_ae = val_loss_ae
        best_state_ae = copy.deepcopy(model_ae.state_dict())
        wait_ae = 0
    else:
        wait_ae += 1
        if wait_ae >= patience_ae:
            print("Parada antecipada.")
            break

model_ae.load_state_dict(best_state_ae)
plt.figure(figsize=(8,5))
plt.plot(train_losses_ae, label="Treino")
plt.plot(val_losses_ae, label="Validação")
plt.xlabel("Época"); plt.ylabel("BCE loss"); plt.title("MLP — loss")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("p2_curvas_mlp.png", dpi=160); plt.show()

model_ae.eval()
with torch.no_grad():
    probs_ae = torch.sigmoid(model_ae(Xt_ae)).cpu().numpy().ravel()
pred_ae = (probs_ae >= 0.5).astype(int)
cm_ae = confusion_matrix(y_test_ae, pred_ae, labels=[0,1])
precision_ae, recall_ae, f1_ae, _ = precision_recall_fscore_support(
    y_test_ae, pred_ae, average="binary", zero_division=0)
print("Matriz de confusão [normal, ataque]:\n", cm_ae)
print("Acurácia:", accuracy_score(y_test_ae, pred_ae))
print("Precisão ataque:", precision_ae, "Recall ataque:", recall_ae, "F1 ataque:", f1_ae)
