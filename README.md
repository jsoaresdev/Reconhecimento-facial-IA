
# 🔍 Reconhecimento Facial com IA (PCA + SVM)

Este projeto implementa um sistema de **Reconhecimento Facial** utilizando técnicas de **Aprendizado de Máquina**, combinando **Análise de Componentes Principais (PCA)** para redução de dimensionalidade e **Máquinas de Vetores de Suporte (SVM)** para classificação. A base de dados utilizada é a clássica **ORL Faces Database**, composta por imagens de rostos.

## 🚀 Funcionalidades
- Leitura e pré-processamento das imagens da base ORL.
- Redimensionamento das imagens para padrão 112x92 pixels.
- Aplicação de **PCA** para redução da dimensionalidade.
- Treinamento de um classificador **SVM** com kernel RBF.
- Avaliação do modelo com:
  - Acurácia no conjunto de teste.
  - Validação cruzada (5-fold cross-validation).
  - Matriz de confusão.
- Visualização dos dados com **t-SNE**.


## ⚙️ Tecnologias Utilizadas
- Python
- Scikit-Learn
- NumPy
- Matplotlib
- ImageIO
- Scikit-Image (resize)
- t-SNE para visualização

## 🧠 Como Executar
1. Clone o repositório:
```bash
git clone https://github.com/jsoaresdev/Reconhecimento-facial-IA.git
```
2. Navegue até a pasta do projeto:
```bash
cd Reconhecimento-facial-IA
```
3. Instale as dependências:
```bash
pip install -r requirements.txt
```
4. Execute o script:
```bash
python reconhecimento_facial_pca_svm.py
```

## 📊 Resultados Esperados
- Acurácia média com validação cruzada entre **92% a 97%**.
- Matriz de confusão representando os acertos e erros.
- Visualização dos clusters via **t-SNE**.

## 📚 Base de Dados
- [ORL Faces Database (AT&T)](https://cam-orl.co.uk/facedatabase.html)
- Contém 40 indivíduos com 10 imagens cada, totalizando 400 imagens em escala de cinza.

## 🤝 Contribuição
Sinta-se à vontade para abrir issues, propor melhorias ou enviar pull requests.

## 🧑‍💻 Autor
- João Soares - [@jsoaresdev](https://github.com/jsoaresdev)

## 📝 Licença
Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
