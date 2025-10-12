import numpy as np

##################################################################
########### Datasets properties

def summarize_datasets(X, y):

    """
    Calcule les différents ratio sur chaque dataset.
    """

    n_samples, n_features = X.shape
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    majority, minority = counts.max(), counts.min()
    imbalance_ratio = round(minority / majority, 3) if n_classes > 1 else np.nan

    summaries = [n_samples, n_features, n_classes, majority, minority, imbalance_ratio]

    return summaries

##################################################################
########### Datasets splitting

def split_datasets_by_balance(df, sample_threshold=300, imbalance_threshold=0.1):

    """
    Splitte les datasets selon l'imbalance_ratio.
    """
    
    # Filtrer les petits datasets
    filtered_df = df[df["nsamples"] >= sample_threshold].copy()

    # Datasets très déséquilibrés : ratio < 0.2
    highly_imbalanced = filtered_df[
        (filtered_df["imbalanceratio"] < imbalance_threshold)
    ].index.tolist()

    # Les autres
    balanced_or_moderate = filtered_df[
        ~filtered_df.index.isin(highly_imbalanced)
    ].index.tolist()

    return balanced_or_moderate, highly_imbalanced