import time
import numpy as np
import matplotlib.pyplot as plt

from logistic import LogisticRegression
from utilis import (
    load_processed_data,
    evaluate_model,
    plot_loss_histories,
    plot_loss_vs_time
)

def main():
    X_train, y_train, X_test, y_test = load_processed_data()

    optimizers = [
        ('GD',       {'lr': 0.1,   'max_iter': 500}),
        ('Momentum', {'lr': 0.1,   'max_iter': 500}),
        ('Adam',     {'lr': 0.01,  'max_iter': 500}),
        ('L-BFGS',   {'max_iter': 200}),
    ]

    results = {}

    for name, params in optimizers:
        key = name.lower().replace('-', '')  # matches LogisticRegression API
        print(f"\n>>> Training LogisticRegression with {name} <<<")
        model = LogisticRegression(optimizer=key, **params)

        start = time.time()
        model.fit(X_train, y_train)
        total_time = time.time() - start

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        _ = evaluate_model(name, y_test, y_pred, y_prob)

        # Build time history
        iters = len(model.loss_history)
        time_hist = np.linspace(0, total_time, iters)

        results[name] = {
            'loss':      model.loss_history,
            'time_hist': time_hist
        }

    # Plot Loss vs Iterations
    plot_loss_histories(
        [res['loss'] for res in results.values()],
        list(results.keys()),
        title='Logistic Regression: Loss vs Iterations'
    )
    plt.show()

    # Plot Loss vs Time
    plot_loss_vs_time(
        [res['time_hist'] for res in results.values()],
        [res['loss']      for res in results.values()],
        list(results.keys()),
        title='Logistic Regression: Loss vs Time'
    )
    plt.show()

if __name__ == '__main__':
    main()
