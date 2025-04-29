# Optimisation des algorithmes d’apprentissage  
*Classification du cancer du sein – Breast-Cancer Wisconsin*

Ce dépôt compare quatre optimiseurs : **Gradient Descent (GD), Momentum, Adam, L-BFGS**  
sur deux modèles supervisés : **Régression logistique** et **MLP (1 couche cachée ReLU)**.  
Le jeu de données utilisé est **Breast-Cancer Wisconsin (Diagnostic)** (569 échantillons).

---

## 1. Arborescence principale
project/
├── data/
│   ├── raw/
│   │   └── breast-cancer.csv    # Jeu de données brut
│   └── processed/               # Données pré-traitées générées
├── src/
│   ├── preprocess.py            # Script de pré-traitement et sauvegarde des .npy
│   ├── logistic.py              # Classe LogisticRegression (GD, Momentum, Adam, L-BFGS)
│   ├── mlp.py                   # Classe MLP (GD, Momentum, Adam, L-BFGS)
│   ├── optim/                   # Implémentations des mises à jour
│   │   ├── sgd.py
│   │   ├── momentum.py
│   │   ├── adam.py
│   │   └── lbfgs.py
│   │   └── plot_coef_importance.py
│   │   └── plot_momentum_heatmap.py
│   ├── compare__LR_optimizers.py    # Convergence & performance pour LR
│   └── compare_mlp_optimizers.py# Convergence & performance pour MLP
└── report_main.tex              # Rapport LaTeX complet
```



## 2. Installation rapide

```bash
git clone https://github.com/HamzaOUZAHRA/ml-optimizer-comparison
cd ML-optimization
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

3. Pipeline d’exécution

Étape	Script & commande	Détails
(1) Pré-traitement	python src/preprocess.py	• supprime features corrélées (ρ>0.9)
• standardise
• équilibre via SMOTE
→ crée data/processed/{X_train,X_test,y_train,y_test}.npy
(2) LR : comparaison	python src/compare_optimizers.py	Affiche :
• loss vs itérations
• loss vs temps
• métriques (Acc, AUC, etc.)
(3) MLP : comparaison	python src/compare_mlp_optimizers.py	Idem pour le réseau de neurones
(4) Heat-map Momentum	python src/optim/plot_momentum_heatmap.py	Génère report/figs/heatmap_momentum.png
Scripts unitaires (pour tester un seul modèle) :

bash
Copier
Modifier
python src/logistic.py     # régression logistique seule
python src/mlp.py          # MLP seul
Toutes les figures se retrouvent dans report/figs/.

4. Reproductibilité & réglages
Graine fixée : np.random.seed(42) + random_state=42 dans train_test_split et SMOTE.

Hyper-paramètres (lr, β, max_iter…) disponibles en table dans le rapport et directement modifiables en tête de chaque script.

Environnement : voir requirements.txt et la section Protocole expérimental du PDF.

5. Rapport
ML_optimization.pdf synthétise :

Problème & données – justification médicale, pré-traitement.

Méthodes – formules clés & complexité.

Protocole – split 80/20, hyper-paramètres, mesure du temps.

Résultats & analyses – courbes de convergence, heat-map (η × β), métriques complètes.

Interprétabilité – top-5 features via coefficients (proxy SHAP).

Reproductibilité – lien GitHub, seed, requirements.

Conclusion & perspectives – recommandations opérationnelles.



```

