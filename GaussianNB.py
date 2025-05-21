from sklearn.naive_bayes import GaussianNB

# Treina o modelo
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title("Matriz de Confusão - GaussianNB")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.show()

# Relatório
print("Relatório de Classificação - GaussianNB:")
print(classification_report(y_test, y_pred))
