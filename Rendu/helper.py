import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from ultralytics import YOLO
import matplotlib.pyplot as plt
import networkx as nx
import random
import itertools

def preprocess_image(image_path):
    """Charge l'image et applique une détection de contours."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def segment_image(image_path,model):
    """Utilise un modèle pré-entraîné de segmentation pour classifier les pixels."""

    image = cv2.imread(image_path,1)
    image = cv2.resize(image, (256, 256))

    image_tensor = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        logits= model(image_tensor).squeeze().cpu().numpy()
        segmentation_map = 1 / (1 + np.exp(-logits))  # Application de la sigmoïde pour normaliser entre 0 et 1

    return segmentation_map

def detect_obstacles(image_path, model_yolo):
    """Utilise YOLO pour détecter les obstacles et les annoter."""
    model = YOLO(model_yolo)
    results = model(image_path)
    annotated_image = results[0].plot()  # Image avec annotations
    return annotated_image, results


def find_first_praticable_point(binary_map):
    """Trouve le premier pixel praticable (1) dans la carte"""
    for x in range(binary_map.shape[0]):
        for y in range(binary_map.shape[1]):
            if binary_map[x, y] == 1:
                return (x, y)
    return None  # Si aucune zone praticable n'est trouvée


def find_farthest_praticable_point(binary_map, start):
    """Trouve le point praticable (1) le plus éloigné du point de départ"""
    max_dist = 0
    farthest_point = start
    
    for x in range(binary_map.shape[0]):
        for y in range(binary_map.shape[1]):
            if binary_map[x, y] == 1:
                dist = np.linalg.norm(np.array(start) - np.array((x, y)))
                if dist > max_dist:
                    max_dist = dist
                    farthest_point = (x, y)
    
    return farthest_point


def get_random_checkpoints(binary_map, num_checkpoints):
    """Sélectionne k checkpoints aléatoires parmi les points praticables (valeur 1)."""
    if num_checkpoints == 0:
        return []  # Pas de checkpoints
    
    # Trouver tous les points praticables
    practicable_points = [(x, y) for x in range(binary_map.shape[0]) 
                          for y in range(binary_map.shape[1]) if binary_map[x, y] == 1]
    
    # Vérifier qu'il y a assez de points praticables
    if len(practicable_points) < num_checkpoints:
        print(f" Seulement {len(practicable_points)} points praticables trouvés, impossible d'en choisir {num_checkpoints} !")
        return practicable_points  # Retourne tous les points si k est trop grand
    
    # Sélectionner "num_checkpoints" points aléatoires
    checkpoints = random.sample(practicable_points, num_checkpoints)
    
    return checkpoints


def plan_path_with_checkpoints(binary_map, start, goal, checkpoints):
    """Planifie un chemin en passant par les checkpoints de façon optimisée, tout en atteignant goal en dernier."""

    # Créer le graphe
    G = nx.grid_2d_graph(binary_map.shape[0], binary_map.shape[1])
    nodes_to_remove = [(x, y) for x, y in G.nodes if binary_map[x, y] == 0]
    G.remove_nodes_from(nodes_to_remove)

    # Vérification de la validité des points
    all_points = [start] + checkpoints + [goal]
    for point in all_points:
        if point not in G:
            print(f"Le point {point} est impraticable !")
            return []

    # Trouver l’ordre optimal en testant aussi goal dans la permutation
    shortest_path = None
    min_distance = float('inf')

    for perm in itertools.permutations(checkpoints + [goal]):  # Inclut goal dans l'optimisation
        if perm[-1] != goal:
            continue  # On s'assure que goal est le dernier point

        current_position = start
        temp_path = []
        total_distance = 0
        valid = True

        for checkpoint in perm:  # goal est aussi optimisé mais forcé en dernier
            try:
                sub_path = nx.astar_path(G, current_position, checkpoint)
                temp_path.extend(sub_path[:-1])  # Ajouter le sous-chemin sans doublons
                total_distance += len(sub_path)
                current_position = checkpoint
            except nx.NetworkXNoPath:
                valid = False
                break  # On arrête si un chemin est impossible

        if valid and total_distance < min_distance:
            min_distance = total_distance
            shortest_path = temp_path + [goal]  # Assurer que goal est bien le dernier point

    return shortest_path if shortest_path else []



def visualize_results(image_path, segmentation_map, obstacles_image, path, binary_map,checkpoints):
    """Affiche les résultats: segmentation, obstacles et chemin optimal avec correction d'échelle."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Charger l'image originale
    original = cv2.imread(image_path,1)
    orig_h, orig_w = original.shape[:2]  # Dimensions de l'image originale
    bin_h, bin_w = binary_map.shape[:2]  # Dimensions de la carte binaire

    # Calcul des facteurs d'échelle
    scale_x = orig_w / bin_w
    scale_y = orig_h / bin_h

    # Affichage des images
    axes[0].imshow(original)  # Image originale en couleurs
    axes[0].set_title('Image originale')

    axes[1].imshow(segmentation_map, cmap='viridis')
    axes[1].set_title('Segmentation')

    axes[2].imshow(obstacles_image)
    axes[2].set_title('Planification du chemin')

    # Vérification du chemin
    print(f"Nombre de points dans le chemin: {len(path)}")
    print(f"Taille image originale: {orig_w}x{orig_h}, Taille binary_map: {bin_w}x{bin_h}")

    # Tracé du chemin avec correction d'échelle
    for (x, y) in path:
        axes[2].scatter(y * scale_x, x * scale_y, color='red', s=2)  # Adapter à l'échelle

    # Tracé les checkpoints (croix bleues "X")
    for (x, y) in checkpoints:
        axes[2].scatter(y * scale_x, x * scale_y, color='blue', marker='x', s=50, linewidths=2)

    plt.show()

def update_path(G, binary_map, current_position, goal, checkpoints):
    """Met à jour le graphe et recalcule un nouveau chemin en tenant compte des checkpoints."""

    # Suppression des nouveaux obstacles
    nodes_to_remove = [(x, y) for x, y in G.nodes if binary_map[x, y] == 0]
    G.remove_nodes_from(nodes_to_remove)

    # Vérifier si le goal est toujours praticable
    if goal not in G:
        print(" Le point d'arrivée est maintenant impraticable !")
        return []

    # Vérifier la praticabilité des checkpoints
    valid_checkpoints = [cp for cp in checkpoints if cp in G]
    if len(valid_checkpoints) < len(checkpoints):
        print(" Certains checkpoints sont devenus impraticables et seront ignorés.")  # On pourrait faire mieux ici, par exemple prendre un nouveau checkpoint proche, mais il faudrait faire une distinction de cas si le plus proche est trop loin

    # Recalculer le chemin avec la nouvelle version optimisée qui inclut les checkpoints
    try:
        new_path = plan_path_with_checkpoints(binary_map, current_position, goal, valid_checkpoints)
        print(" Nouveau chemin calculé en raison d'un obstacle.")
    except nx.NetworkXNoPath:
        print(" Aucun chemin possible après modification des obstacles !")
        new_path = []

    return new_path
