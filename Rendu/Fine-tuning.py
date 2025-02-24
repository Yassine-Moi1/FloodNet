import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

"""Features: 
	1. Dataset is distributed same as FloodNet challenge.
	2. Masks are same size as original images.
	3. Total class: 10 ('Background':0, 'Building-flooded':1, 'Building-non-flooded':2, 'Road-flooded':3, 'Road-non-flooded':4, 'Water':5, 'Tree':6, 'Vehicle':7, 'Pool':8, 'Grass':9).
	4. Total image: 2343 (Train: 1445, Val: 450, Test: 448)"""

# Définition des chemins
DATA_DIR = r"C:\Users\yasss\Desktop\Cours\stage audentiel\FloodNet-Supervised_v1.0"  # Remplace par le chemin réel
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "train-org-img")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "train-label-img")
VAL_IMG_DIR = os.path.join(DATA_DIR, "val", "val-org-img")
VAL_MASK_DIR = os.path.join(DATA_DIR, "val", "val-label-img")

# Hyperparamètres
IMG_SIZE = 256  # Taille d'entrée du modèle
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4  # Taux d'apprentissage
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset personnalisé pour FloodNet
class FloodNetDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = os.listdir(img_dir)
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx].replace(".jpg", "_lab.png"))
        
        image = cv2.imread(img_path,1)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        

        mask = ((mask == 4)|(mask == 0)).astype(np.float32)  # Binarisation, choix arbitraire, les classes 0 et 4 sont considérées comme 'praticables' (cf en haut)

        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(mask).unsqueeze(0)  # Ajout d'une dimension
        
        return image, mask

# Transformations des images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# Chargement des données
train_dataset = FloodNetDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=transform)

val_dataset = FloodNetDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Modèle U-Net avec EfficientNet-B3
model = smp.Unet(encoder_name="efficientnet-b3", encoder_weights="imagenet", in_channels=3, classes=1)
model = model.to(DEVICE)

# Fonction de perte et optimiseur
criterion = smp.losses.DiceLoss(mode="binary")
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Entraînement du modèle (fine-tuning)
print("Entraînement en cours...")
train_losses = []
val_losses = []
best_val_loss = float("inf")

# Ajout d'un scheduler pour ajuster le taux d'apprentissage
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', threshold=1e-2, factor=0.1, patience=2, verbose=True)  #Finalement pas utilisé car résultats suffisants, mais on peut ameliorer le modèle avec

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader):
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            val_loss = criterion(outputs, masks)
            running_val_loss += val_loss.item()
    val_loss = running_val_loss / len(val_loader)
    val_losses.append(val_loss)
    
    print(f" Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Sauvegarde des poids pour chaque epoch
    scheduler.step(val_loss)
    torch.save(model.state_dict(), f"unet_floodnet_epoch_{epoch+1}.pth")
    
    # Sauvegarde du meilleur modèle
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_unet_floodnet.pth")
        print(" Nouveau modèle sauvegardé avec meilleure loss de validation !")

# Affichage du graphique de la loss
plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Courbe de Loss - Train vs Validation")
plt.show()

print("Entraînement terminé et modèles sauvegardés.")

