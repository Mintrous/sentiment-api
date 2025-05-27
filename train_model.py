import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# ler o dataset
data = pd.read_csv('imdb_dataset.csv')

# separar em x e y (review e sentimento)
X = data['review']
# binarizar
y = data['sentiment'].map({'positive': 1, 'negative': 0})

# dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# vetorizar
vectorizer = CountVectorizer(stop_words='english')
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# treinar o modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vect, y_train)

print('\nmodelo treinado!\n')

# Fazer previsões com o conjunto de teste
y_pred = model.predict(X_test_vect)

# Calcular acuracia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia no conjunto de teste: {accuracy:.2f}")

# Matriz de confusão
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# salvar modelo e vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

