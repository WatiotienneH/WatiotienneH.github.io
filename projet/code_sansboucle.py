from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ttest_ind

#### Fonction pour évaluer les modèles
def evaluate_model(model, X_test, y_test, model_name):
    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Affichage du rapport de classification
    print(f"\nRapport de classification pour {model_name}:\n", classification_report(y_test, y_pred))

    # Calcul de la matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Affichage de la matrice de confusion avec seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('')
    plt.ylabel('')
    plt.title(f'Matrice de Confusion - {model_name}')
    plt.show()

    # Calcul du taux d'erreur
    error_rate = (y_pred != y_test).mean()
    print(f"Taux d'erreur pour {model_name}: {error_rate:.2%}")
    return error_rate

meilleur_modele=["nom",100]

##### Chargement et traitement des données
data = pd.read_csv("C:/Users/hrywa/OneDrive/Bureau/Méthodes_d_apprentissage/projet/Data.csv", sep=";")

print(data.head())
print(data.info)
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

# Étude des données
plt.matshow(np.corrcoef(np.transpose(X)))
plt.colorbar()
plt.title('Matrice de Corrélation')
plt.show()


####### Sélection des variables
selection =True
if selection :
    selected_variables = []
    class_0 = X[Y == "goodware"]
    class_1 = X[Y == "malware"]
    # Appliquer le test t de Student pour chaque variable
    for column in X.columns:
        t_stat, p_value = ttest_ind(class_0[column], class_1[column], equal_var=False)

        # Choisissez un seuil de significativité, par exemple, 0.05
        if p_value < 0.05:
            selected_variables.append(column)

    # Afficher les variables sélectionnées
    print("Variables sélectionnées par différence de moyennes significative :", selected_variables)

X=X[selected_variables]

###### Définition de X_train et X_test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

# Calcul de la proportion dans y_train
prop_goodware_train = (y_train == 'goodware').mean()
prop_malware_train = (y_train == 'malware').mean()

# Calcul de la proportion dans y_test
prop_goodware_test = (y_test == 'goodware').mean()
prop_malware_test = (y_test == 'malware').mean()

# Affichage des résultats
print(f"Proportion de goodware dans y_train : {prop_goodware_train:.2%}")
print(f"Proportion de malware dans y_train : {prop_malware_train:.2%}")
print(f"Proportion de goodware dans y_test : {prop_goodware_test:.2%}")
print(f"Proportion de malware dans y_test : {prop_malware_test:.2%}")
###### SMOTE
def apply_smote(X_train, y_train):
    # Appliquer SMOTE sur l'ensemble d'apprentissage
    smote = SMOTE(sampling_strategy='auto')
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print((y_train_resampled != "goodware").count())
    print((y_train_resampled != "malware").count())
    return X_train_resampled, y_train_resampled

# Utilisation optionnelle de SMOTE
use_smote = True  # Changez à False si vous ne voulez pas utiliser SMOTE

if use_smote:
    X_train, y_train = apply_smote(X_train, y_train)

#####  SVM avec bagging et sans
svm_base_classifier = SVC()

# Paramètres à tester pour Bagging SVM
bagging_svm = BaggingClassifier(estimator=SVC(), n_estimators=50)
bagging_svm.fit(X_train, y_train)
erreur=evaluate_model(bagging_svm, X_test, y_test, 'Bagging SVM')

if erreur < meilleur_modele[1]:
    meilleur_modele[0]= "SVM avec bagging"
    meilleur_modele[1]= erreur

# Paramètres à tester pour SVM sans Bagging/Boosting
param_grid_svm = {'C': [80, 90, 70, 85, 75],
                  'gamma': [0.0001, 0.0005, 0.001, 0.00009],
                  'kernel': ['linear', 'rbf']}

grid_search_svm = GridSearchCV(svm_base_classifier, param_grid_svm, cv=5)
grid_search_svm.fit(X_train, y_train)
best_svm = grid_search_svm.best_estimator_

erreur=evaluate_model(best_svm, X_test, y_test, 'Best SVM sans Bagging/Boosting')
if erreur < meilleur_modele[1]:
    meilleur_modele[0]= "SVM classique"
    meilleur_modele[1]= erreur
###### Cart ave bagging et sans
cart_base_classifier = DecisionTreeClassifier()

# Paramètres à tester pour Bagging Decision Tree (CART)
bagging_cart = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50)
bagging_cart.fit(X_train, y_train)
erreur=evaluate_model(bagging_cart, X_test, y_test, 'Bagging Decision Tree (CART)')
if erreur < meilleur_modele[1]:
    meilleur_modele[0]= "CART avec bagging"
    meilleur_modele[1]= erreur

# Paramètres à tester pour Decision Tree (CART) sans Bagging/Boosting
param_grid_cart = {'max_depth': list(range(1, 50))}

grid_search_cart = GridSearchCV(cart_base_classifier, param_grid_cart, cv=5)
grid_search_cart.fit(X_train, y_train)
best_cart = grid_search_cart.best_estimator_


erreur=evaluate_model(best_cart, X_test, y_test, 'Best Decision Tree (CART) sans Bagging/Boosting')
if erreur < meilleur_modele[1]:
    meilleur_modele[0]= "CART classique"
    meilleur_modele[1]= erreur
###### logic avec boosting et sans
# Paramètres à tester pour Boosting Logistic Regression
# Boosting Logistic Regression
boosting_logreg = AdaBoostClassifier(
    estimator=LogisticRegression(max_iter=1000),
    n_estimators=50,
    algorithm='SAMME'
)
boosting_logreg.fit(X_train, y_train)
erreur = evaluate_model(boosting_logreg, X_test, y_test, 'Boosting Logistic Regression')
if erreur < meilleur_modele[1]:
    meilleur_modele[0] = "régression logistique avec boosting"
    meilleur_modele[1] = erreur

# Paramètres à tester pour Logistic Regression sans Bagging/Boosting
param_grid_logreg = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
logreg_base_classifier = LogisticRegression(max_iter=1000)
grid_search_logreg = GridSearchCV(logreg_base_classifier, param_grid_logreg, cv=5)
grid_search_logreg.fit(X_train, y_train)
best_logreg = grid_search_logreg.best_estimator_

erreur = evaluate_model(best_logreg, X_test, y_test, 'Best Logistic Regression sans Bagging/Boosting')
if erreur < meilleur_modele[1]:
    meilleur_modele[0] = "régression logistique classique"
    meilleur_modele[1] = erreur
####### Random forest avec bagging et sans
rf_base_classifier = RandomForestClassifier()

# Paramètres à tester pour Bagging Random Forest
bagging_rf = BaggingClassifier(estimator=RandomForestClassifier(), n_estimators=50)
bagging_rf.fit(X_train, y_train)
erreur=evaluate_model(bagging_rf, X_test, y_test, 'Bagging Random Forest')
if erreur < meilleur_modele[1]:
    meilleur_modele[0]= "Randdom forest avec bagging"
    meilleur_modele[1]= erreur

# Paramètres à tester pour Random Forest sans Bagging/Boosting
param_grid_rf = {'n_estimators': [10, 20, 30, 40, 50], 'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20, 30, 40, 50]}
grid_search_rf = GridSearchCV(rf_base_classifier, param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_

# Évaluer les performances des modèles  Random Forest classique
erreur=evaluate_model(best_rf, X_test, y_test, 'Best Random Forest without Bagging/Boosting')
if erreur < meilleur_modele[1]:
    meilleur_modele[0]= "randdom forest classique"
    meilleur_modele[1]= erreur

##### Plotting ROC curves for all models
# Plotting ROC curves for models that exist
fig, ax = plt.subplots(figsize=(8, 8))


# Courbe ROC pour SVM
fpr_svm, tpr_svm, _ = roc_curve(y_test, best_svm.decision_function(X_test), pos_label='malware')
auc_svm = auc(fpr_svm, tpr_svm)
ax.plot(fpr_svm, tpr_svm, lw=2, label=f'SVM (AUC = {auc_svm:.2f})')

# Courbe ROC pour Bagging SVM
fpr_bagging_svm, tpr_bagging_svm, _ = roc_curve(y_test, bagging_svm.decision_function(X_test), pos_label='malware')
auc_bagging_svm = auc(fpr_bagging_svm, tpr_bagging_svm)
ax.plot(fpr_bagging_svm, tpr_bagging_svm, lw=2, label=f'Bagging SVM (AUC = {auc_bagging_svm:.2f})')

# Courbe ROC pour Decision Tree (CART)
fpr_cart, tpr_cart, _ = roc_curve(y_test, best_cart.predict_proba(X_test)[:, 1], pos_label='malware')
auc_cart = auc(fpr_cart, tpr_cart)
ax.plot(fpr_cart, tpr_cart, lw=2, label=f'Decision Tree (AUC = {auc_cart:.2f})')

# Courbe ROC pour Bagging Decision Tree (CART)
fpr_bagging_cart, tpr_bagging_cart, _ = roc_curve(y_test, bagging_cart.predict_proba(X_test)[:, 1], pos_label='malware')
auc_bagging_cart = auc(fpr_bagging_cart, tpr_bagging_cart)
ax.plot(fpr_bagging_cart, tpr_bagging_cart, lw=2, label=f'Bagging Decision Tree (AUC = {auc_bagging_cart:.2f})')


# Ajout des labels et du titre
ax.set_xlabel('Taux de faux positifs')
ax.set_ylabel('Taux de vrais positifs')
ax.set_title('Courbes ROC pour différents modèles')
ax.legend(loc='lower right')

# Afficher le plot
plt.show()

fig, ax = plt.subplots(figsize=(8, 8))

# Courbe ROC pour Logistic Regression
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, best_logreg.predict_proba(X_test)[:, 1], pos_label='malware')
auc_logreg = auc(fpr_logreg, tpr_logreg)
ax.plot(fpr_logreg, tpr_logreg, lw=2, label=f'Logistic Regression (AUC = {auc_logreg:.2f})')

# Courbe ROC pour Boosting Logistic Regression
fpr_boosting_logreg, tpr_boosting_logreg, _ = roc_curve(y_test, boosting_logreg.predict_proba(X_test)[:, 1], pos_label='malware')
auc_boosting_logreg = auc(fpr_boosting_logreg, tpr_boosting_logreg)
ax.plot(fpr_boosting_logreg, tpr_boosting_logreg, lw=2, label=f'Boosting Logistic Regression (AUC = {auc_boosting_logreg:.2f})')

# Courbe ROC pour Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, best_rf.predict_proba(X_test)[:, 1], pos_label='malware')
auc_rf = auc(fpr_rf, tpr_rf)
ax.plot(fpr_rf, tpr_rf, lw=2, label=f'Random Forest (AUC = {auc_rf:.2f})')

# Courbe ROC pour Bagging Random Forest
fpr_bagging_rf, tpr_bagging_rf, _ = roc_curve(y_test, bagging_rf.predict_proba(X_test)[:, 1], pos_label='malware')
auc_bagging_rf = auc(fpr_bagging_rf, tpr_bagging_rf)
ax.plot(fpr_bagging_rf, tpr_bagging_rf, lw=2, label=f'Bagging Random Forest (AUC = {auc_bagging_rf:.2f})')

# Ajout des labels et du titre
ax.set_xlabel('Taux de faux positifs')
ax.set_ylabel('Taux de vrais positifs')
ax.set_title('Courbes ROC pour différents modèles')
ax.legend(loc='lower right')

# Afficher le plot
plt.show()