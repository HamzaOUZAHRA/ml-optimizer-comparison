#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_momentum_heatmap.py

Génère un heat-map AUC pour l'optimiseur Momentum (via LogisticRegression) en fonction de (lr, beta).
Placez ce fichier dans src/optim/, puis exécutez-le depuis la racine du projet :
    python src/optim/plot_momentum_heatmap.py
"""
import sys, os

# 1) Ajouter src/ au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

# 2) Modules utilitaires
from preprocess import load_data, drop_highly_correlated_features, split_scale
from logistic import LogisticRegression


def main():
    # Chargement et nettoyage
    X_df, y_series = load_data(os.path.join('data', 'raw', 'breast-cancer.csv'))
    X_df, _ = drop_highly_correlated_features(X_df, threshold=0.9)
    X = X_df.values
    y = y_series.values

    # Split + scale
    X_train, X_test, y_train, y_test = split_scale(X, y, test_size=0.2, random_state=42)

    # SMOTE sur le train
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Grille de lr et beta
    etas  = np.logspace(-3, -1, 5)    # de 0.001 à 0.1
    betas = np.linspace(0.7, 0.99, 5)  # de 0.7 à 0.99
    auc_matrix = np.zeros((len(etas), len(betas)))

    # Boucle pour mesurer l'AUC
    for i, eta in enumerate(etas):
        for j, beta in enumerate(betas):
            # Utilise la classe LogisticRegression avec momentum interne
            model = LogisticRegression(
                optimizer='momentum', lr=eta, beta=beta,
                max_iter=200, tol=1e-6
            )
            model.fit(X_train_res, y_train_res)
            y_proba = model.predict_proba(X_test)
            auc_matrix[i, j] = roc_auc_score(y_test, y_proba)

        # Trace du heat-map
    fig, ax = plt.subplots()
    im = ax.imshow(
        auc_matrix,
        origin='lower',
        extent=[betas[0], betas[-1], etas[0], etas[-1]],
        aspect='auto'
    )
    ax.set_xlabel(r'\beta (momentum)')
    ax.set_ylabel(r'\eta (learning rate)')
    ax.set_title('Heat-map AUC pour Momentum')
    fig.colorbar(im, ax=ax, label='AUC')
    plt.tight_layout()

    # Affichage interactif (sans sauvegarde)
    plt.show()

if __name__ == '__main__':
    main()
