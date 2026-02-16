"""
    Manage baseline algorithms
"""

import numpy as np

import scripts.classifier as classifier

def run_algo(data, nb_class, hparam, argv):
    """ Use baseline algorithm on given options """

    outputs = {"confusions": {"train": np.zeros((1, nb_class, nb_class), dtype=int),
                              "valid": np.zeros((1, nb_class, nb_class), dtype=int),
                              "test": np.zeros((1, nb_class, nb_class), dtype=int)},
               "predictions": {"train": np.zeros((1, data["train"]["labels"].shape[0], nb_class), dtype=np.float32),
                               "valid": np.zeros((1, data["valid"]["labels"].shape[0], nb_class), dtype=np.float32),
                               "test": np.zeros((1, data["test"]["labels"].shape[0], nb_class), dtype=np.float32)},
               "t_values": [-1]}

    conf_mats = outputs["confusions"]
    preds = outputs["predictions"]

    if argv.classif.lower() == "ir":
        # class weight = 1 - <class proportion>
        class_w = {class_i:1-(data["train"]["labels"] == class_i).sum()/data["train"]["labels"].shape[0]#on calcule le poids de chaque classe en cas d'"imbalance ratio"
                   for class_i in range(nb_class)}
    else:
        class_w = {class_i:1.0 for class_i in range(nb_class)} #poids uniforme pour chaque classe

    classif = classifier.get_classifier(argv, hparam, class_w) #on recupere le classifieur choisi avec les hyperparametres et les poids de classes

    classif.fit(data["train"]["exemples"], data["train"]["labels"]) #on entraine le classifieur

    for subset in ["train", "valid", "test"]:
        out_iter = classifier.get_confusion(data, nb_class, subset, classif)

        if nb_class == 2:
            conf_mats[subset][0], preds[subset][0, :, 0] = out_iter
        else:
            conf_mats[subset][0], preds[subset][0] = out_iter

    return outputs #on retourne les matrices de confusion et les predictions pour chaque subset
