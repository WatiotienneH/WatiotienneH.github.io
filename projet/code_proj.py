##### Attention : le code prend du temps à s'exécuter car il effectue 15 itérations de chaque algorithme



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
from imblearn.under_sampling import NearMiss
from scipy.stats import fisher_exact
from sklearn.neighbors import KNeighborsClassifier
import warnings
from scipy.stats import t
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')
#### Fonction pour évaluer les modèles et fonction pour appliquer le réenchantillonage
matrix=False
def evaluate_model(model, X_test, y_test, model_name,show_confusion_matrix =False,show_rapport=False):
    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)
    if show_rapport :
        # Affichage du rapport de classification
        print(f"\nRapport de classification pour {model_name}:\n", classification_report(y_test, y_pred))

    # Calcul de la matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)
    if show_confusion_matrix:
        # Affichage de la matrice de confusion avec seaborn
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('')
        plt.ylabel('')
        plt.title(f'Matrice de Confusion - {model_name}')
        plt.show()

    # Calcul du taux d'erreur
    error_rate = (y_pred != y_test).mean()
    print(f"Taux d'erreur pour {model_name}: {error_rate:.4%}")
    return error_rate



def apply_sampling(X_train, y_train, method='smote'):
    if method == 'smote':
        smote = SMOTE(sampling_strategy='auto')
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    elif method == 'nearmiss':
        nearmiss = NearMiss(sampling_strategy='auto')
        X_resampled, y_resampled = nearmiss.fit_resample(X_train, y_train)
    else:
        return X_train, y_train

    return X_resampled, y_resampled

##### Chargement et traitement des données
data = pd.read_csv("C:/Users/hrywa/OneDrive/Bureau/Méthodes_d_apprentissage/projet/Data.csv", sep=";")
#dans le fichier la derniere ligne est vide je l'ai retiré à la main

print(data.head())
print(data.info)
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]



####### Sélection des variables
selection = True
if selection:
    selected_variables = []
    class_0 = X[Y == "goodware"]
    class_1 = X[Y == "malware"]

    for column in X.columns:
        # Construire le tableau de contingence
        contingency_table = pd.crosstab(X[column], Y)

        # Vérifier si le tableau de contingence a la taille 2x2 (sinon la varibale a une seul modalité
        if contingency_table.shape == (2, 2):
            _, p_value = fisher_exact(contingency_table)
            # Si la valeur p est inférieure au seuil de significativité, ajouter la variable
            if p_value < 0.05:
                selected_variables.append(column)

    print("Nombre de variables sélectionnées :", len(selected_variables))

X = X[selected_variables]


# Étude des données
plt.matshow(np.corrcoef(np.transpose(X)))
plt.colorbar()
plt.title('Matrice de Corrélation')
plt.show()



#### On définie le dictionnaire pour stocker les erreurs

# Initialisation du dictionnaire pour stocker les erreurs
errors_dict = {model_name: [] for model_name in ["SVM avec bagging", "SVM classique",
                                                  "CART avec bagging", "CART classique", "régression logistique avec boosting", "régression logistique classique",
                                                "randdom forest classique","KNN avec bagging","KNN classique"]}

### EVALUATION DES MODELES


nbr_iter=15
show_confusion_matrix = False

for i in range (nbr_iter):
    #--------------- Définition de X_train et X_test --------------------
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
#--------------------- SMOTE / NEARMISS------------------------

    sampling_method = 'smote' # SMOTE/NEARMISS/NONE


    X_train_resampled, y_train_resampled = apply_sampling(X_train, y_train, method=sampling_method)

    print("Proportion apres resampled",(y_train_resampled=='goodware').mean())
# ------------------ SVM avec bagging et sans -----------------------
    # Initialisation du modèle SVM
    svm_base_classifier = SVC()

    # Entraînez un modèle Bagging SVM avec des hyperparamètres prédéfinis
    bagging_svm = BaggingClassifier(base_estimator=svm_base_classifier, n_estimators=50)
    bagging_svm.fit(X_train_resampled, y_train_resampled)
    erreur = evaluate_model(bagging_svm, X_test, y_test, 'Bagging SVM', show_confusion_matrix=False)
    errors_dict["SVM avec bagging"].append(erreur)

    # Paramètres à tester pour SVM sans Bagging/Boosting
    param_grid_svm = {'C': [10, 100],
                    'gamma': [0.0001, 0.001],
                    'kernel': ['rbf']}
    grid_search_svm = GridSearchCV(svm_base_classifier, param_grid_svm, cv=5)
    grid_search_svm.fit(X_train_resampled, y_train_resampled)
    best_svm = grid_search_svm.best_estimator_

    erreur = evaluate_model(best_svm, X_test, y_test, 'Best SVM sans Bagging/Boosting', show_confusion_matrix=False)
    errors_dict["SVM classique"].append(erreur)

    # ----------------- CART avec bagging et sans -----------------------
    # Initialisation du modèle CART
    cart_base_classifier = DecisionTreeClassifier()

    # Paramètres à tester pour Bagging CART avec GridSearchCV
    param_grid_cart_bagging = {'base_estimator__max_depth': [10, 30, 50]}
    bagging_cart = GridSearchCV(BaggingClassifier(DecisionTreeClassifier(), n_estimators=50), param_grid_cart_bagging, cv=5)
    bagging_cart.fit(X_train_resampled, y_train_resampled)

    best_bagging_cart = bagging_cart.best_estimator_
    erreur = evaluate_model(best_bagging_cart, X_test, y_test, 'Bagging Decision Tree (CART)', show_confusion_matrix=False)
    errors_dict["CART avec bagging"].append(erreur)

    # Paramètres à tester pour Decision Tree (CART) sans Bagging/Boosting
    param_grid_cart = {'max_depth': [10, 30, 50]}
    grid_search_cart = GridSearchCV(cart_base_classifier, param_grid_cart, cv=5)
    grid_search_cart.fit(X_train_resampled, y_train_resampled)
    best_cart = grid_search_cart.best_estimator_

    erreur = evaluate_model(best_cart, X_test, y_test, 'Best Decision Tree (CART) sans Bagging/Boosting', show_confusion_matrix=False)
    errors_dict["CART classique"].append(erreur)

    # -------------- Régression logistique avec boosting et sans ---------
    # Initialisation du modèle de régression logistique
    logreg_base_classifier = LogisticRegression(max_iter=1000)

    # Paramètres à tester pour Boosting Logistic Regression
    param_grid_logreg_boosting = {'base_estimator__C': [0.001, 0.01, 0.1, 1, 10]}
    boosting_logreg = GridSearchCV(AdaBoostClassifier(LogisticRegression(max_iter=1000), n_estimators=50, algorithm='SAMME'), param_grid_logreg_boosting, cv=5)
    boosting_logreg.fit(X_train_resampled, y_train_resampled)

    best_boosting_logreg = boosting_logreg.best_estimator_
    erreur = evaluate_model(best_boosting_logreg, X_test, y_test, 'Boosting Logistic Regression', show_confusion_matrix=False)
    errors_dict["régression logistique avec boosting"].append(erreur)

    # Paramètres à tester pour Logistic Regression sans Bagging/Boosting
    param_grid_logreg = {'C': [0.001, 0.01, 0.1, 1, 10]}
    grid_search_logreg = GridSearchCV(logreg_base_classifier, param_grid_logreg, cv=5)
    grid_search_logreg.fit(X_train_resampled, y_train_resampled)
    best_logreg = grid_search_logreg.best_estimator_

    erreur = evaluate_model(best_logreg, X_test, y_test, 'Best Logistic Regression sans Bagging/Boosting', show_confusion_matrix=False)
    errors_dict["régression logistique classique"].append(erreur)

    # ------------- Random forest  ---------------
    # Initialisation du modèle Random Forest
    rf_base_classifier = RandomForestClassifier()
    # Paramètres à tester pour Random Forest
    param_grid_rf = {'n_estimators': [200], 'max_depth': [10, 30, 50]}
    grid_search_rf = GridSearchCV(rf_base_classifier, param_grid_rf, cv=5)
    grid_search_rf.fit(X_train_resampled, y_train_resampled)
    best_rf = grid_search_rf.best_estimator_

    erreur = evaluate_model(best_rf, X_test, y_test, 'Best Random Forest without Bagging/Boosting', show_confusion_matrix=False)
    errors_dict["randdom forest classique"].append(erreur)

    # --------------------------- KNN Avec bagging ---------------------------
    # Paramètres à tester pour Bagging KNN avec GridSearchCV
    param_grid_knn_bagging = {'base_estimator__n_neighbors': [5, 10, 15]}
    bagging_knn = GridSearchCV(BaggingClassifier(KNeighborsClassifier(), n_estimators=50), param_grid_knn_bagging, cv=5)
    bagging_knn.fit(X_train_resampled, y_train_resampled)
    best_bagging_knn = bagging_knn.best_estimator_

    # Utilisation de best_estimator_ pour Bagging KNN
    best_bagging_knn = bagging_knn.best_estimator_
    erreur = evaluate_model(best_bagging_knn, X_test, y_test, 'Bagging KNN', show_confusion_matrix=False)
    errors_dict["KNN avec bagging"].append(erreur)

    # ------------------------ KNN Classique ----------------------------
    # Paramètres pour le GridSearch avec KNN
    knn_params = {'n_neighbors': [5, 10, 15]}
    grid_knn = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
    grid_knn.fit(X_train_resampled, y_train_resampled)
    best_knn = grid_knn.best_estimator_

    best_knn = grid_knn.best_estimator_
    erreur = evaluate_model(best_knn, X_test, y_test, 'KNN classique', show_confusion_matrix=False)
    errors_dict["KNN classique"].append(erreur)

    print(" ---------------------------- ITERATION TERMINER -------------------------------")


##### Etude des erreurs


# Calcul de la moyenne des erreurs
mean_errors_dict = {model_name: np.mean(errors) for model_name, errors in errors_dict.items()}
std_errors_dict = {model_name: np.std(errors) for model_name, errors in errors_dict.items()}

t_score = t.ppf(0.975, df=nbr_iter - 1)


margin_of_error_dict = {model_name: t_score * (std_errors_dict[model_name] / np.sqrt(nbr_iter)) for model_name in mean_errors_dict}

confidence_intervals_dict = {model_name: (mean_errors_dict[model_name] - margin_of_error_dict[model_name],
                                          mean_errors_dict[model_name] + margin_of_error_dict[model_name])
                             for model_name in mean_errors_dict}


for model_name, interval in confidence_intervals_dict.items():
    print(f"Confidence interval for {model_name}: {interval[0]:.4%} to {interval[1]:.4%}")

best_model = min(confidence_intervals_dict, key=lambda k: confidence_intervals_dict[k][0])
print(f"\nMeilleur modèle: {best_model} avec une erreur moyenne de {mean_errors_dict[best_model]:.4%}")



# Convert errors_dict to a DataFrame for easier plotting
errors_df = pd.DataFrame(errors_dict)

# Plotting boxplots for all models on the same graph
plt.figure(figsize=(14, 10))
sns.boxplot(data=errors_df, width=0.6)
plt.title('Boxplots des erreurs pour chaque modèle')
plt.ylabel('Taux d\'erreur')
plt.xticks(rotation=30, ha='center')  # Rotate x-axis labels for better readability
plt.subplots_adjust(bottom=0.2)
plt.show()



##### Plotting ROC curves for all models
# Plotting ROC curves for models that exist
fig, ax = plt.subplots(figsize=(8, 8))


# Courbe ROC pour SVM
fpr_svm, tpr_svm, _ = roc_curve(y_test, best_svm.decision_function(X_test), pos_label='malware')
auc_svm = auc(fpr_svm, tpr_svm)
ax.plot(fpr_svm, tpr_svm, lw=2, label=f'SVM (AUC = {auc_svm:.2f})')


# Courbe ROC pour Decision Tree (CART)
fpr_cart, tpr_cart, _ = roc_curve(y_test, best_cart.predict_proba(X_test)[:, 1], pos_label='malware')
auc_cart = auc(fpr_cart, tpr_cart)
ax.plot(fpr_cart, tpr_cart, lw=2, label=f'Decision Tree (AUC = {auc_cart:.2f})')


# Courbe ROC pour Logistic Regression
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, best_logreg.predict_proba(X_test)[:, 1], pos_label='malware')
auc_logreg = auc(fpr_logreg, tpr_logreg)
ax.plot(fpr_logreg, tpr_logreg, lw=2, label=f'Logistic Regression (AUC = {auc_logreg:.2f})')

# Courbe ROC pour Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, best_rf.predict_proba(X_test)[:, 1], pos_label='malware')
auc_rf = auc(fpr_rf, tpr_rf)
ax.plot(fpr_rf, tpr_rf, lw=2, label=f'Random Forest (AUC = {auc_rf:.2f})')


fpr_knn, tpr_knn, _ = roc_curve(y_test, best_knn.predict_proba(X_test)[:, 1], pos_label='malware')
auc_knn = auc(fpr_knn, tpr_knn)
ax.plot(fpr_knn, tpr_knn, lw=2, label=f'KNN avec bagging (AUC = {auc_knn:.2f})')


# Ajout des labels et du titre
ax.set_xlabel('Taux de faux positifs')
ax.set_ylabel('Taux de vrais positifs')
ax.set_title('Courbes ROC pour différents modèles')
ax.legend(loc='lower right')


# Afficher le plot
plt.show()


#### Identification des varibales d'importance

# Initialisation et entraînement des modèles
rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=50)
cart_classifier = DecisionTreeClassifier(max_depth=50)
logreg_classifier = LogisticRegression(max_iter=1000)

rf_classifier.fit(X, Y)
cart_classifier.fit(X, Y)
logreg_classifier.fit(X, Y)

# Extraction des importances et coefficients
importances_rf = rf_classifier.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1][:15]  # 15 premières valeurs

importances_cart = cart_classifier.feature_importances_
indices_cart = np.argsort(importances_cart)[::-1][:15]

coefficients = logreg_classifier.coef_[0]
indices_logreg = np.argsort(np.abs(coefficients))[::-1][:15]

# Fonction pour créer un graphique
def plot_feature_importances(model_name, importances, indices, title):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    bars = plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), [X.columns[i] for i in indices], rotation=80)
    plt.subplots_adjust(bottom=0.4)
    plt.show()

# Tracé des importances pour chaque modèle
plot_feature_importances("Random Forest", importances_rf, indices_rf, "Importances des caractéristiques - Random Forest")

plot_feature_importances("Decision Tree", importances_cart, indices_cart, "Importances des caractéristiques - Decision Tree")
plot_feature_importances("Régression Logistique", coefficients, indices_logreg, "Coefficients - Régression Logistique")

##### Graphique pour les besoins du rapport
if True :
    # Données fournies
    data = {
        "Sans Rééchantillonnage": {
            'SVM avec bagging': (0.012172502690732036, 0.014709217739375492),
            'SVM classique': (0.015487527759906906, 0.018204228512494525),
            'CART avec bagging': (0.010618684949352213, 0.014470920785414813),
            'CART classique': (0.012101466037733214, 0.01549710027051051),
            'reg log avec boosting': (0.017991502488776512, 0.02215186668685073),
            'reg log classique': (0.015428390235078129, 0.0182633660373233),
            'random forest classique': (0.007425932246268157, 0.011570483524341165),
            'KNN avec bagging': (0.012786493349983195, 0.016962610592669135),
            'KNN classique': (0.012167877881840518, 0.016028059991516706)
        },
        "SMOTE": {
            'SVM avec bagging': (0.010376421199082393, 0.01447423590963923),
            'SVM classique': (0.013297567022786316, 0.017526805737070314),
            'CART avec bagging': (0.0070662353950733964, 0.01037701430624082),
            'CART classique': (0.008550917625579642, 0.011401292649211279),
            'reg log avec boosting': (0.016148051008506636, 0.021486357593643902),
            'reg log classique': (0.011828247059141251, 0.01660663944026138),
            'random forest classique': (0.00454936535677636, 0.006920168693402853),
            'KNN avec bagging': (0.01115869786205063, 0.013691959246670994),
            'KNN classique': (0.011354228323083396, 0.015288543480978729)
        },
        "NEARMISS": {
            'SVM avec bagging': (0.11849288965865609, 0.1561785559805315),
            'SVM classique': (0.15054679010318536, 0.1634317045204706),
            'CART avec bagging': (0.13782708894674772, 0.150344954064005),
            'CART classique': (0.14221911928855327, 0.16172353304119588),
            'reg log avec boosting': (0.16023313473216685, 0.16999386646257622),
            'reg log classique': (0.14376146236016654, 0.155999589013788),
            'random forest classique': (0.18546647972917463, 0.19697079625648845),
            'KNN avec bagging': (0.04410207804421319, 0.04932683473953829),
            'KNN classique': (0.04117730809933161, 0.06634957362109849)
        }
    }
    paires = [
        ('SVM classique', 'SVM avec bagging'),
        ('CART classique', 'CART avec bagging'),
        ('reg log classique', 'reg log avec boosting'),
        ('KNN classique', 'KNN avec bagging'),
        ('random forest classique',)
    ]

    colors = {
        "Sans Rééchantillonnage": 'blue',
        "SMOTE": 'green',
        "NEARMISS": 'red'
    }
    for pair in paires:
        plt.figure(figsize=(10, 6))

        for modèle in pair:
            for method, scores in data.items():
                if modèle in scores:
                    ci_lower, ci_upper = scores[modèle]
                    plt.plot([method + '\n' + modèle, method + '\n' + modèle], [ci_lower, ci_upper], color=colors[method])

        plt.title(f'Comparaison de {pair[0]} avec variantes')
        plt.ylabel('Performance')
        plt.xlabel('Modèle')

        # Création de la légende pour les couleurs
        legend_elements = [plt.Line2D([0], [0], color=color, lw=1, label=method) for method, color in colors.items()]
        plt.legend(handles=legend_elements, loc='upper left')

        plt.show()