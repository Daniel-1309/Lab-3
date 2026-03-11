import pandas as pd
import mlflow
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Use the same tracking backend as the UI
mlflow.set_tracking_uri("sqlite:///mlflow.db")


def tune_model(X_train_path, y_train_path, X_test_path, y_test_path):

    # Load data
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    # Hyperparameters to test
    C_values = [0.1, 1, 10]
    gamma_values = [0.01, 0.1]

    for c in C_values:
        for g in gamma_values:

            # Start a new run for each experiment
            with mlflow.start_run(run_name=f"C={c}_gamma={g}"):

                print(f"Training with C={c}, gamma={g}")

                model = make_pipeline(
                    RBFSampler(gamma=g, n_components=500, random_state=42),
                    LinearSVR(C=c)
                )

                model.fit(X_train, y_train.values.ravel())

                preds = model.predict(X_test)

                mse = mean_squared_error(y_test, preds)

                # Log parameters
                mlflow.log_param("model", "RBFSampler + LinearSVR")
                mlflow.log_param("C", c)
                mlflow.log_param("gamma", g)

                # Log metric
                mlflow.log_metric("mse", mse)


if __name__ == "__main__":
    tune_model(
        "data/X_train.csv",
        "data/y_train.csv",
        "data/X_test.csv",
        "data/y_test.csv"
    )