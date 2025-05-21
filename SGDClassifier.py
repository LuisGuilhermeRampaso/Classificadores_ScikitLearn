import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Carrega o dataset
cifar10 = fetch_openml('cifar_10', version=1, cache=True)
X = cifar10.data[:10000]  # Apenas 10 mil imagens
y = cifar10.target[:10000].astype(np.uint8)

# Divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X / 255.0, y, test_size=0.2, random_state=42)

# Treina o modelo
model = SGDClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusão - SGDClassifier")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.show()

# Relatório
print("Relatório de Classificação - SGDClassifier:")
print(classification_report(y_test, y_pred))
