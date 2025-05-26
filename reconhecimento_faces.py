import os
import cv2
import numpy as np

# Caminho da base de dados ORL
DATASET_PATH = 'att_faces/orl_faces'
IMG_HEIGHT = 112
IMG_WIDTH = 92

def carregar_imagens():
    imagens = []
    rotulos = []
    nomes_classes = []

    for idx, pasta in enumerate(sorted(os.listdir(DATASET_PATH))):
        pasta_path = os.path.join(DATASET_PATH, pasta)
        if os.path.isdir(pasta_path):
            nomes_classes.append(pasta)
            for nome_arquivo in os.listdir(pasta_path):
                caminho_img = os.path.join(pasta_path, nome_arquivo)
                img = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_redimensionada = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    imagens.append(img_redimensionada.flatten())
                    rotulos.append(idx)

    return np.array(imagens), np.array(rotulos), nomes_classes

if __name__ == "__main__":
    X, y, nomes_classes = carregar_imagens()
    print(f"Imagens carregadas: {X.shape[0]}")
    print(f"Formato de cada imagem vetorizada: {X.shape[1]}")
    print(f"Número de classes: {len(nomes_classes)}")
