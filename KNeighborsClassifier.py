from sklearn.neighbors import KNeighborsClassifier

# Treina o modelo
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Matriz de Confusão - KNeighborsClassifier")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.show()

# Relatório
print("Relatório de Classificação - KNeighborsClassifier:")
print(classification_report(y_test, y_pred))
