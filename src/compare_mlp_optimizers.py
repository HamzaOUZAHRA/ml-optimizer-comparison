import time
import numpy as np
import matplotlib.pyplot as plt

from mlp import MLP
from utilis import (
    load_processed_data,
    evaluate_model,
    plot_loss_histories,
    plot_loss_vs_time
)

def main():
    X_train, y_train, X_test, y_test = load_processed_data()
    layer_sizes = [X_train.shape[1], 16, 1]

    optimizers = [
        ('SGD',      {'optimizer': 'gd',       'lr': 0.01, 'max_iter': 1000}),
        ('Momentum', {'optimizer': 'momentum', 'lr': 0.01, 'beta':0.9, 'max_iter': 1000}),
        ('Adam',     {'optimizer': 'adam',     'lr': 0.01, 'beta':0.9, 'max_iter': 1000}),
        ('L-BFGS',   {'optimizer': 'lbfgs',    'max_iter': 500}),
    ]

    results = {}

    for name, params in optimizers:
        print(f"\n>>> Training MLP with {name} <<<")
        mlp = MLP(
            layer_sizes,
            optimizer=params['optimizer'],
            lr=params.get('lr', 0.01),
            max_iter=params.get('max_iter', 1000),
            tol=1e-6,
            beta=params.get('beta', 0.9)
        )

        start = time.time()
        mlp.fit(X_train, y_train)
        total_time = time.time() - start

        # Evaluate
        y_pred = mlp.predict(X_test)
        y_prob = mlp.predict_proba(X_test)
        _ = evaluate_model(name, y_test, y_pred, y_prob)

        # Build time history
        iters = len(mlp.loss_history)
        time_hist = np.linspace(0, total_time, iters)

        results[name] = {
            'loss':      mlp.loss_history,
            'time_hist': time_hist
        }

    # Plot Loss vs Iterations
    plot_loss_histories(
        [res['loss'] for res in results.values()],
        list(results.keys()),
        title='MLP: Loss vs Iterations'
    )
    plt.show()

    # Plot Loss vs Time
    plot_loss_vs_time(
        [res['time_hist'] for res in results.values()],
        [res['loss']      for res in results.values()],
        list(results.keys()),
        title='MLP: Loss vs Time'
    )
    plt.show()

if __name__ == '__main__':
    main()
