from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC

##################################################################
########### Penalized Logistic Regression

def make_penalized_logreg(params):
    """
    Crée un modèle de régression logistique pénalisée avec scaling.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(**params, max_iter=1000, random_state=42)
    )

##################################################################
########### Bagging ensemble method

def make_bagging(params):
    """
    Crée un BaggingClassifier avec un estimateur de base.
    """
    base_estimator = params.pop("estimator", DecisionTreeClassifier())
    return BaggingClassifier(
        estimator=base_estimator,
        random_state=42,
        **params
    )

##################################################################
########### Random Forest

def make_random_forest(params):
    """
    Crée un RandomForestClassifier.
    """
    return RandomForestClassifier(
        random_state=42,
        **params
    )

##################################################################
########### Boosting ensemble method

def make_adaboost(params):
    """
    Crée un AdaBoostClassifier avec arbre faible par défaut.
    """
    base_estimator = params.pop("estimator", DecisionTreeClassifier(max_depth=1))
    return AdaBoostClassifier(
        estimator=base_estimator,
        random_state=42,
        **params
    )

##################################################################
########### Stacking ensemble method

def make_stacking(params):
    """
    Crée un StackingClassifier avec quelques modèles de base.
    """
    estimators = params.pop("estimators", [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('svc', SVC(probability=True))
    ])
    
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=params.get("cv", 5),
        passthrough=params.get("passthrough", False)
    )

##################################################################
########### Gradient boosting

def make_gradient_boosting(params):
    """
    Crée un GradientBoostingClassifier.
    """
    return GradientBoostingClassifier(
        random_state=42,
        **params
    )

    