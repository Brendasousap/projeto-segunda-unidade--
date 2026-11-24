"""
Problema 3 — CNN EuroSAT RGB
Instale: pip install torch torchvision matplotlib scikit-learn numpy
O dataset é baixado automaticamente pela torchvision.
"""
import copy, random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import EuroSAT
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

SEED_ae = 42
random.seed(SEED_ae); np.random.seed(SEED_ae); torch.manual_seed(SEED_ae)
device_ae = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_ae = "./data"

# Carrega imagens sem normalização para calcular estatísticas só do treino
base_ae = EuroSAT(root=root_ae, download=True, transform=transforms.ToTensor())
targets_ae = np.array(base_ae.targets)
indices_ae = np.arange(len(base_ae))
train_idx_ae, temp_idx_ae = train_test_split(indices_ae, test_size=0.30, stratify=targets_ae, random_state=SEED_ae)
val_idx_ae, test_idx_ae = train_test_split(temp_idx_ae, test_size=0.50, stratify=targets_ae[temp_idx_ae], random_state=SEED_ae)

# Média e desvio calculados sobre pixels das imagens do conjunto de treino
sums_ae = torch.zeros(3); sqs_ae = torch.zeros(3); n_pixels_ae = 0
for i_ae in train_idx_ae:
    img_ae, _ = base_ae[i_ae]
    sums_ae += img_ae.sum(dim=(1,2))
    sqs_ae += (img_ae**2).sum(dim=(1,2))
    n_pixels_ae += img_ae.shape[1] * img_ae.shape[2]
mean_ae = sums_ae / n_pixels_ae
std_ae = torch.sqrt(sqs_ae / n_pixels_ae - mean_ae**2).clamp_min(1e-6)
print("Classes:", base_ae.classes)
print("Imagens por classe:", {c: int((targets_ae == i).sum()) for i,c in enumerate(base_ae.classes)})
norm_ae = transforms.Normalize(mean_ae.tolist(), std_ae.tolist())

# Grid de cinco imagens por classe (antes da normalização)
fig_ae, axes_ae = plt.subplots(10, 5, figsize=(10, 18))
for class_i_ae, class_name_ae in enumerate(base_ae.classes):
    class_ids_ae = np.where(targets_ae == class_i_ae)[0][:5]
    for j_ae, idx_ae in enumerate(class_ids_ae):
        img_ae, _ = base_ae[int(idx_ae)]
        axes_ae[class_i_ae,j_ae].imshow(img_ae.permute(1,2,0))
        axes_ae[class_i_ae,j_ae].axis("off")
        if j_ae == 0: axes_ae[class_i_ae,j_ae].set_ylabel(class_name_ae, fontsize=8)
plt.tight_layout(); plt.savefig("p3_grade_eurosat.png", dpi=150); plt.show()

class TransformSubset_ae(torch.utils.data.Dataset):
    def __init__(self, base, indices, augment=False):
        self.base, self.indices, self.augment = base, list(indices), augment
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30)
        ])
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        # base retorna tensor em [0,1]; augmentation apenas treino
        if self.augment:
            img = self.aug(img)
        return norm_ae(img), label

def run_model_ae(augment_ae=False):
    tr_ds_ae = TransformSubset_ae(base_ae, train_idx_ae, augment_ae)
    va_ds_ae = TransformSubset_ae(base_ae, val_idx_ae, False)
    te_ds_ae = TransformSubset_ae(base_ae, test_idx_ae, False)
    tr_dl_ae = DataLoader(tr_ds_ae, batch_size=64, shuffle=True, num_workers=0)
    va_dl_ae = DataLoader(va_ds_ae, batch_size=64, shuffle=False, num_workers=0)
    te_dl_ae = DataLoader(te_ds_ae, batch_size=64, shuffle=False, num_workers=0)

    class CNN_ae(nn.Module):
        def __init__(self):
            super().__init__()
            self.features_ae = nn.Sequential(
                nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2)
            )
            self.classifier_ae = nn.Sequential(
                nn.Flatten(), nn.Linear(128*8*8,128), nn.ReLU(), nn.Linear(128,10)
            )
        def forward(self, x): return self.classifier_ae(self.features_ae(x))

    model_ae = CNN_ae().to(device_ae)
    opt_ae = torch.optim.Adam(model_ae.parameters(), lr=1e-3)
    criterion_ae = nn.CrossEntropyLoss()
    hist_ae = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_loss_ae, best_state_ae, wait_ae, patience_ae = float("inf"), None, 0, 5

    for epoch_ae in range(30):
        model_ae.train(); sum_loss_ae = 0; correct_ae = 0; total_ae = 0
        for x_ae, y_ae in tr_dl_ae:
            x_ae, y_ae = x_ae.to(device_ae), y_ae.to(device_ae)
            opt_ae.zero_grad(); logits_ae = model_ae(x_ae)
            loss_ae = criterion_ae(logits_ae, y_ae); loss_ae.backward(); opt_ae.step()
            sum_loss_ae += loss_ae.item()*len(y_ae)
            correct_ae += (logits_ae.argmax(1)==y_ae).sum().item(); total_ae += len(y_ae)
        tr_loss_ae, tr_acc_ae = sum_loss_ae/total_ae, correct_ae/total_ae
        model_ae.eval(); val_sum_ae = 0; val_correct_ae = 0; val_total_ae = 0
        with torch.no_grad():
            for x_ae,y_ae in va_dl_ae:
                x_ae,y_ae=x_ae.to(device_ae),y_ae.to(device_ae)
                logits_ae=model_ae(x_ae); loss_ae=criterion_ae(logits_ae,y_ae)
                val_sum_ae += loss_ae.item()*len(y_ae)
                val_correct_ae += (logits_ae.argmax(1)==y_ae).sum().item(); val_total_ae += len(y_ae)
        vl_ae, va_ae = val_sum_ae/val_total_ae, val_correct_ae/val_total_ae
        for key_ae,val_ae in [("train_loss",tr_loss_ae),("val_loss",vl_ae),("train_acc",tr_acc_ae),("val_acc",va_ae)]:
            hist_ae[key_ae].append(val_ae)
        print(f"aug={augment_ae} época={epoch_ae+1:02d} train_loss={tr_loss_ae:.4f} val_loss={vl_ae:.4f} train_acc={tr_acc_ae:.4f} val_acc={va_ae:.4f}")
        if vl_ae < best_loss_ae:
            best_loss_ae=vl_ae; best_state_ae=copy.deepcopy(model_ae.state_dict()); wait_ae=0
        else:
            wait_ae += 1
            if wait_ae >= patience_ae: break

    model_ae.load_state_dict(best_state_ae)
    fig_ae, ax_ae = plt.subplots(1,2,figsize=(11,4))
    ax_ae[0].plot(hist_ae["train_loss"],label="Treino"); ax_ae[0].plot(hist_ae["val_loss"],label="Validação")
    ax_ae[0].set_title("Loss"); ax_ae[0].legend(); ax_ae[0].grid(True)
    ax_ae[1].plot(hist_ae["train_acc"],label="Treino"); ax_ae[1].plot(hist_ae["val_acc"],label="Validação")
    ax_ae[1].set_title("Acurácia"); ax_ae[1].legend(); ax_ae[1].grid(True)
    plt.tight_layout(); plt.savefig(f"p3_curvas_augment_{augment_ae}.png",dpi=150); plt.show()

    model_ae.eval(); y_true_ae=[]; y_pred_ae=[]
    with torch.no_grad():
        for x_ae,y_ae in te_dl_ae:
            logits_ae=model_ae(x_ae.to(device_ae))
            y_pred_ae.extend(logits_ae.argmax(1).cpu().numpy()); y_true_ae.extend(y_ae.numpy())
    print("Augmentation:",augment_ae)
    print("Acurácia teste:",accuracy_score(y_true_ae,y_pred_ae))
    print("Matriz de confusão:\n",confusion_matrix(y_true_ae,y_pred_ae))
    print(classification_report(y_true_ae,y_pred_ae,target_names=base_ae.classes,zero_division=0))
    return hist_ae

hist_sem_ae = run_model_ae(False)
hist_com_ae = run_model_ae(True)
