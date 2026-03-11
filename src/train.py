from sklearn.svm import SVR
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("sqlite:///mlflow.db")

def train_model(X_train_path, y_train_path):
    # Load the preprocessed training data
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    # Initialize the SVR model
    model = SVR()

    with mlflow.start_run():

        model.fit(X_train, y_train.values.ravel())

        mlflow.log_param("model", "SVR")

        mlflow.sklearn.log_model(model, "model")

        preds = model.predict(X_train)
        mse = mean_squared_error(y_train, preds)

        mlflow.log_metric("mse", mse)

    # Save model as pkl file
    joblib.dump(model, 'models/model.pkl')

    return model

if __name__ == "__main__":
    train_model("data/X_train.csv", "data/y_train.csv")