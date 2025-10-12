import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import svm, metrics
from itertools import product
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter
import numpy as np

##################################################################
########### Cross validation

def cross_validation(X, Y, hyperparams_grid, model_method, test_size=0.3, k_fold=5):
    """
    Effectue une cross-validation manuelle sur plusieurs hyperparamètres.
    
    Args:
        X, Y: données et labels
        hyperparams_grid: dict des hyperparamètres à tester, ex:
            {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
        model_method: implémentation du modèle
        test_size: proportion du test split
        k_fold: nombre de folds pour la cross-validation
    
    Returns:
        best_params: dict des meilleurs hyperparamètres
        test_accuracy: précision finale sur le jeu de test
    """

    # Split train/test global
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, stratify=Y, random_state=42)
    
    # Préparer la grille de combinaisons (cartésien)
    keys = list(hyperparams_grid.keys())
    values = list(hyperparams_grid.values())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    skf = StratifiedKFold(n_splits=k_fold)
    results = np.zeros((len(param_combinations), k_fold + 1))  # moyenne en dernière colonne

    # Boucle sur les combinaisons d’hyperparamètres
    for i, params in enumerate(param_combinations):
        for j, (train_index, val_index) in enumerate(skf.split(X_train, Y_train)):
            
            # Construire le modèle avec les hyperparamètres courants
            model = model_method(params)
            
            # Entraînement
            model.fit(X_train[train_index], Y_train[train_index])
            
            # Validation
            predictions = model.predict(X_train[val_index])
            results[i, j] = metrics.accuracy_score(Y_train[val_index], predictions)
        
        # Moyenne des scores
        results[i, -1] = np.mean(results[i, :-1])
    
    # Sélection des meilleurs hyperparamètres
    best_index = np.argmax(results[:, -1])
    best_params = param_combinations[best_index]

    #Moyenne et écartype
    cv_scores = results[best_index, :-1]
    mean_acc = np.mean(cv_scores)
    std_acc = np.std(cv_scores)
    
    # Réentraîner sur tout le jeu d’entraînement
    best_model = model_method(best_params)
    best_model.fit(X_train, Y_train)
    
    # Évaluer sur le jeu de test
    test_predictions = best_model.predict(X_test)
    test_accuracy = metrics.accuracy_score(Y_test, test_predictions)
    
    return test_accuracy*100, mean_acc*100, std_acc*100

##################################################################
########### DecisionTree Bagging Pipeline

def make_decision_tree_bagging(n):

    """
    Retourne une pipeline d'une méthode de bagging avec comme learner,
    une DecisionTree simple.
    """

    return make_pipeline(
        StandardScaler(),
        BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            n_estimators=n,
            random_state=42,
            n_jobs=-1
        )
)

##################################################################
########### DecisionTree Bagging Fit

def decion_tree_bagging_fit(X, Y, n, test_size=0.3, k_fold=5):

    """
    Splitte les données et apprend un modèle de bagging basé sur un learner
    Decision Tree.
    """

    # Split train/test global
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, stratify=Y, random_state=42)
    
    # Construire le modèle avec les hyperparamètres courants
    model = make_decision_tree_bagging(n)
    
    # Entraînement
    model.fit(X_train, Y_train)
    
    # Prédiction
    predictions = model.predict(X_test)
    accuracy = metrics.accuracy_score(Y_test, predictions)
    
    return accuracy*100

##################################################################
########### Sampling methods

def apply_sampling_methods(X, y, method="random"):

    """
    Apprend la méthode de sampling choisie.
    """
    if method == "random":
        sampler = RandomOverSampler(random_state=42)
    elif method == "smote":
        sampler = SMOTE(random_state=42)
    else:
        raise ValueError("Méthode non reconnue. Utilise 'random' ou 'smote'.")

    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res

##################################################################
########### Cross validation

def imblearn_cross_validation(X, Y, hyperparams_grid, model_method, sampling_method, test_size=0.3, k_fold=5):

    """
    Apprend une cross-validation sur les datasets déséquilibrés.s
    """

    # Split train/test global
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, stratify=Y, random_state=42)

    #Sampling methods
    X_res, y_res = apply_sampling_methods(X_train, Y_train, method=sampling_method)
    
    # Préparer la grille de combinaisons (cartésien)
    keys = list(hyperparams_grid.keys())
    values = list(hyperparams_grid.values())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    skf = StratifiedKFold(n_splits=k_fold)
    results = np.zeros((len(param_combinations), k_fold + 1))  # moyenne en dernière colonne

    # Boucle sur les combinaisons d’hyperparamètres
    for i, params in enumerate(param_combinations):
        for j, (train_index, val_index) in enumerate(skf.split(X_res, y_res)):
            
            # Construire le modèle avec les hyperparamètres courants
            model = model_method(params)
            
            # Entraînement
            model.fit(X_res[train_index], y_res[train_index])
            
            # Validation
            predictions = model.predict(X_res[val_index])
            results[i, j] = metrics.recall_score(y_res[val_index], predictions)
        
        # Moyenne des scores
        results[i, -1] = np.mean(results[i, :-1])
    
    # Sélection des meilleurs hyperparamètres
    best_index = np.argmax(results[:, -1])
    best_params = param_combinations[best_index]

    #Moyenne et écartype
    cv_scores = results[best_index, :-1]
    mean_recall = np.mean(cv_scores)
    std_recall = np.std(cv_scores)
    
    # Réentraîner sur tout le jeu d’entraînement
    best_model = model_method(best_params)
    best_model.fit(X_res, y_res)
    
    # Évaluer sur le jeu de test
    test_predictions = best_model.predict(X_test)
    test_recall = metrics.recall_score(Y_test, test_predictions)
    
    return test_recall*100, mean_recall*100, std_recall*100