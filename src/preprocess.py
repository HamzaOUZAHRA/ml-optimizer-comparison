#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess.py
Fournit les fonctions de chargement et prétraitement des données pour le projet ML+OPT.
"""
import os
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_data(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Charge un CSV, supprime les colonnes ID non numériques, renvoie (X_df, y_series)."""
    df = pd.read_csv(csv_path)
    # Suppression colonnes ID ou non numériques
    drop_cols = [c for c in df.columns if c.lower() in {"id", "unnamed: 0"}]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Identification de la colonne cible
    if any(c.lower() == "diagnosis" for c in df.columns):
        diag_col = next(c for c in df.columns if c.lower() == "diagnosis")
        y = df[diag_col].map({"B": 0, "M": 1})
        X_df = df.drop(columns=[diag_col])
    else:
        y = df.iloc[:, -1]
        X_df = df.iloc[:, :-1]

    # Conversion en numérique
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.dropna(axis=1)

    return X_df, y.astype(float)


def drop_highly_correlated_features(
    X_df: pd.DataFrame, threshold: float = 0.9
) -> Tuple[pd.DataFrame, List[str]]:
    """Supprime les features dont la corrélation absolue > threshold."""
    corr_mat = X_df.corr(numeric_only=True).abs()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    X_reduced = X_df.drop(columns=to_drop)
    return X_reduced, to_drop


def split_scale(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split train/test puis standardisation (stats du train)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    eps = 1e-8
    X_train_scaled = (X_train - mean) / (std + eps)
    X_test_scaled = (X_test - mean) / (std + eps)
    return X_train_scaled, X_test_scaled, y_train, y_test  

# Pas de "main" exécuté ici ; ce module fournit uniquement des utilitaires.
