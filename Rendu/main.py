# Importation des bibliothèques nécessaires
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from ultralytics import YOLO
import matplotlib.pyplot as plt
import networkx as nx
from helper import preprocess_image, segment_image, detect_obstacles, plan_path_with_checkpoints, visualize_results, find_first_praticable_point, find_farthest_praticable_point,get_random_checkpoints, update_path

# Définition des chemins vers les fichiers nécessaires
image_path = r"C:\Users\yasss\Desktop\Cours\stage audentiel\FloodNet-Supervised_v1.0\test\test-org-img\9001.jpg"
model_yolo_path = 'yolov8n.pt'  # Modèle YOLO pour détection d'obstacles

# Étape 1 : Détection des contours avec l'algorithme de Canny
edges = preprocess_image(image_path)  # Détection de contours avec Canny

# Étape 2 : Segmentation sémantique avec U-Net
model = smp.Unet(encoder_name="efficientnet-b3", encoder_weights=None, in_channels=3, classes=1)  # Pas de poids pré-entraînés car on charge les nôtres
model.load_state_dict(torch.load("unet_floodnet_epoch_10.pth", map_location=torch.device("cpu")))  
model.eval()
segmentation_map = segment_image(image_path, model)

# Étape 3 : Détection des obstacles avec YOLO
obstacles_image, yolo_results = detect_obstacles(image_path, model_yolo_path)

# Étape 4 : Fusion des résultats (Contours + Segmentation)
binary_map = (segmentation_map > 0.5).astype(np.uint8) # Binarisation de la carte de segmentation, seuil de 0.5 pour classifier praticable/impraticable

# Intégration des contours dans la carte binaire
edges_resized = cv2.resize(edges, (binary_map.shape[1], binary_map.shape[0]))   # Redimensionne edges pour qu'il corresponde à binary_map
binary_map[edges_resized > 50] = 0  # Les pixels détectés comme contours deviennent impraticables

# Étape 5 : Détection des points de départ et d'arrivée
start = find_first_praticable_point(binary_map)
goal = find_farthest_praticable_point(binary_map, start)

# Étape 6 : Planification de chemin avec A*
num_checkpoints = 2
checkpoints = get_random_checkpoints(binary_map, num_checkpoints)
path = plan_path_with_checkpoints(binary_map, start, goal,checkpoints)

# Étape 7 : Visualisation finale des résultats
visualize_results(image_path, segmentation_map, obstacles_image, path,binary_map,checkpoints)


