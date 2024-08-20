chemin="C:/Users/hrywa/OneDrive/Bureau/Méthodes_d_apprentissage/Data/projet/Data.csv"

import pandas as pd

# Charger les données depuis Data.csv
data = pd.read_csv(chemin)

# Afficher les premières lignes pour comprendre la structure des données
print(data.head())

# Obtenir des statistiques descriptives
print(data.describe())

# Vérifier le déséquilibre de classes
class_counts = data['Label'].value_counts()
print(class_counts)

# Visualiser la répartition des classes
import matplotlib.pyplot as plt
class_counts.plot(kind='bar')
plt.title('Répartition des classes')
plt.xlabel('Label')
plt.ylabel('Nombre d\'individus')
plt.show()


#############
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Séparer les données en features (X) et labels (y)
X = data.drop('Label', axis=1)
y = data['Label']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle de régression logistique
model = LogisticRegression()
model.fit(X_train, y_train)

# Prédire les classes sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer le modèle
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Imprimer les résultats
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
#######

