import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import cv2
import os
import seaborn as sns
from reconhecimento_faces import carregar_imagens, IMG_HEIGHT, IMG_WIDTH


# =========================
# Funções de PCA e SVM
# =========================
def aplicar_pca(X, n_componentes=50):
    pca = PCA(n_components=n_componentes, whiten=True, random_state=42)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

def treinar_svm(X_pca, y):
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    return svm.fit(X_pca, y)

def validar_modelo(svm, X_pca, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(svm, X_pca, y, cv=skf)

    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', xticks_rotation=90)
    plt.title("Matriz de Confusão (5-Fold CV)")
    plt.tight_layout()
    plt.show()

    return y_pred


# =========================
# Funções para Visualização
# =========================
def plot_imagens_classe(X, y, classe, nomes_classes):
    imgs = X[y == classe][:9]
    fig, axs = plt.subplots(3, 3, figsize=(8, 6))
    fig.suptitle(f"9 imagens da classe prevista: {nomes_classes[classe]}", fontsize=14)
    for i, ax in enumerate(axs.flat):
        if i < len(imgs):
            ax.imshow(imgs[i].reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def top_5_classes(modelo, vetor_pca, nomes_classes, X, y):
    probs = modelo.predict_proba([vetor_pca])[0]
    top5_posicoes = np.argsort(probs)[-5:][::-1]
    top5_classes = modelo.classes_[top5_posicoes]

    fig, axs = plt.subplots(1, 5, figsize=(14, 4))
    fig.suptitle("Top 5 classes mais prováveis", fontsize=14)

    for i, classe_idx in enumerate(top5_classes):
        imagens_classe = X[y == classe_idx]
        if imagens_classe.shape[0] == 0:
            imagem = np.zeros((IMG_HEIGHT, IMG_WIDTH))
        else:
            imagem = imagens_classe[0].reshape(IMG_HEIGHT, IMG_WIDTH)

        axs[i].imshow(imagem, cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f"{nomes_classes[classe_idx]}\n{probs[top5_posicoes[i]]*100:.2f}%")

    plt.tight_layout()
    plt.show()


def mostrar_tsne(X_pca, y, nomes_classes):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)

    plt.figure(figsize=(14, 10))
    scatter = sns.scatterplot(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        hue=y,
        palette='tab20',
        legend='full'
    )

    handles, labels = scatter.get_legend_handles_labels()
    new_labels = []
    new_handles = []

    for handle, label in zip(handles, labels):
        if label.isdigit():
            idx = int(label)
            if idx < len(nomes_classes):
                new_labels.append(nomes_classes[idx])
                new_handles.append(handle)

    plt.legend(
        new_handles,
        new_labels,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.,
        frameon=True,
        title='Classes',
        ncol=1
    )

    plt.title("t-SNE dos vetores PCA (colorido por classe)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.show()


# =========================
# Classificação Dinâmica
# =========================
def classificar_imagem(caminho_img, pca, modelo_svm, X_orig, y, nomes_classes):
    img = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro ao carregar a imagem.")
        return
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    vetor = img.flatten()
    vetor_pca = pca.transform([vetor])[0]

    classe_predita = modelo_svm.predict([vetor_pca])[0]
    print(f"\nClasse prevista: {nomes_classes[classe_predita]}")

    # 1. Mostrar imagem de teste com classe prevista
    plt.imshow(img, cmap='gray')
    plt.title(f"Imagem de teste - Classe prevista: {nomes_classes[classe_predita]}")
    plt.axis('off')
    plt.show()

    # 2. Mostrar 9 imagens da classe prevista
    plot_imagens_classe(X_orig, y, classe_predita, nomes_classes)

    # 3. Mostrar Top-5 classes mais prováveis
    top_5_classes(modelo_svm, vetor_pca, nomes_classes, X_orig, y)

    return vetor_pca


# =========================
# Execução Principal
# =========================
if __name__ == "__main__":
    # Carregar dados
    X, y, nomes_classes = carregar_imagens()

    # PCA
    pca, X_pca = aplicar_pca(X)

    # SVM
    svm = treinar_svm(X_pca, y)

    # Escolha do caminho da imagem de teste
    caminho_teste = input("Digite o caminho da imagem de teste (ex: att_faces/orl_faces/s1/1.pgm): ")

    # Classificação da imagem
    vetor_pca_img = classificar_imagem(caminho_teste, pca, svm, X, y, nomes_classes)

    # 4. Matriz de Confusão
    validar_modelo(svm, X_pca, y)

    # 5. t-SNE
    mostrar_tsne(X_pca, y, nomes_classes)
