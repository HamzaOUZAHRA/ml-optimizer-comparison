#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_coef_importance.py

Affiche un bar chart des 5 descripteurs radiomiques les plus influents,
d’après la valeur absolue des coefficients appris par LogisticRegression(momentum).
Placez ce fichier dans src/optim/, puis exécutez depuis la racine :
    python src/optim/plot_coef_importance.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from preprocess import load_data, drop_highly_correlated_features, split_scale
from logistic import LogisticRegression
from imblearn.over_sampling import SMOTE

def main():
    # 1) Chargement + nettoyage
    csv_path = os.path.join('data', 'raw', 'breast-cancer.csv')
    X_df, y = load_data(csv_path)
    X_df, _ = drop_highly_correlated_features(X_df, threshold=0.9)

    # 2) Split + scale
    X_train, X_test, y_train, y_test = split_scale(X_df.values, y.values,
                                                  test_size=0.2, random_state=42)

    # 3) SMOTE
    sm = SMOTE(random_state=42)
    X_tr, y_tr = sm.fit_resample(X_train, y_train)

    # 4) Entraînement du modèle
    model = LogisticRegression(
        optimizer='momentum',
        lr=0.01,
        beta=0.9,
        max_iter=500,
        tol=1e-6
    )
    model.fit(X_tr, y_tr)

    # 5) Extraction et tri des coefficients
    coefs = model.w  # vecteur de taille n_features
    feat_names = np.array(X_df.columns)
    # On prend top 5 par magnitude
    top_idx = np.argsort(np.abs(coefs))[::-1][:5]

    # 6) Plot
    plt.figure(figsize=(6,4))
    plt.barh(feat_names[top_idx][::-1], coefs[top_idx][::-1])
    plt.xlabel('Coefficient (poids)')
    plt.title('Top 5 descripteurs par |coefficient|')
    plt.tight_layout()

    # 7) Sauvegarde & affichage
    out_dir = os.path.join('report','figs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'coef_importance.png')
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Sauvegardé → {out_path}")

if __name__ == '__main__':
    main()
