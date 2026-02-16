"""
Consistent Multiclass Algorithms for Complex Performance Measures Algorithm 2 p6
"""

import sys
import logging as log

import numpy as np
from sklearn.metrics import confusion_matrix

import scripts.classifier as classifier

def run_algo(data, nb_class, hparam, argv):
    """ Tune classifier with bisection algorithm """

    outputs = {"confusions": {"train": np.zeros((1, nb_class, nb_class), dtype=int),
                              "valid": np.zeros((1, nb_class, nb_class), dtype=int),
                              "test": np.zeros((1, nb_class, nb_class), dtype=int)},
               "t_values": [-1]}

    a_mat = np.zeros((nb_class, nb_class))
    np.fill_diagonal(a_mat, 1+argv.beta**2) #remplir la diagonale avec 1+beta^2
    a_mat[0, 0] = 0 #mettre a_00 = 0

    b_mat = np.full((nb_class, nb_class), 1+argv.beta**2) #remplir toute la matrice avec 1+beta^2
    b_mat[0, 0] = 0 #mettre b_00 = 0
    b_mat[0, 1:] = 1 #mettre la premiere ligne (sauf b_00) a 1
    b_mat[1:, 0] = argv.beta**2 #mettre la premiere colonne (sauf b_00) a beta^2

    classif = classifier.get_classifier(argv, hparam, {class_i:1.0 for class_i in range(nb_class)})

    try:
        classif.predict_proba
    except AttributeError:
        log.error("Bisection algorithm requires classifier with 'predict_proba' method")
        sys.exit(0)

    classif.fit(data["train"]["exemples"], data["train"]["labels"])

    inf_bound = 0
    sup_bound = 1
    tuned_classif = [classif, np.ones((nb_class, nb_class)), -1]#on initialise tous les coûts à 1

    # TOO LONG, each time converge after ~10 iterations !
    # nb_iter = int(kappa*(label_train.shape[0]+label_valid.shape[0]))
    nb_iter = int(argv.kappa)

    for iter_i in range(nb_iter):
        log.debug("tuning iteration n°%d/%d...", iter_i, nb_iter)

        gamma = (inf_bound+sup_bound)/2 #bisection step

        loss_mat = -(a_mat-gamma*b_mat) #À appronfondir : matrice des couts
        loss_mat = (loss_mat-loss_mat.min())/(loss_mat.max()-loss_mat.min()) #normalisation entre 0 et 1

        conf = confusion_matrix(data["valid"]["labels"],
                                predict(data["valid"]["exemples"], [classif, loss_mat])) #correspond à notre profil d'erreur
        
        phi = (a_mat*conf).sum()/(b_mat*conf).sum() #À appronfondir : calcul de la performance

        if phi >= gamma:
            inf_bound = gamma
            tuned_classif = [classif, loss_mat, gamma]
        else:
            sup_bound = gamma

        log.debug("\t gamma = %f; phi = %f", gamma, phi)

    for subset in ["train", "valid", "test"]:
        preds = predict(data[subset]["exemples"], tuned_classif)

        outputs["confusions"][subset][0] = confusion_matrix(preds, data[subset]["labels"])
    
    return outputs #return outputs

def predict(dataset, tuned_classif):
    """ Get tuned prediction """

    pred = tuned_classif[0].predict_proba(dataset)#on recupere les probabilites predites par le classifieur

    loss_pred = np.matmul(np.transpose(tuned_classif[1]), np.transpose(pred))

    return np.argmin(loss_pred, axis=0) #on retourne la classe avec le cout minimum
