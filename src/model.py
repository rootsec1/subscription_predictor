import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Create a class for building and using a machine learning model
class Model():
    training_df = None

    def __init__(self, training_df: pd.DataFrame):
        # Initialize the class with a training dataframe
        self.training_df = training_df

    def __split_data__(self, y_feature: str, X_feature_list: list = [], X_feature_list_to_drop: list = []):
        # Split data into training and validation sets
        X = pd.DataFrame()
        if len(X_feature_list_to_drop) > 0:
            X = self.training_df.drop(columns=X_feature_list_to_drop)
        else:
            X = self.training_df[X_feature_list]
        y = self.training_df[y_feature]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=0)
        return X_train, X_val, y_train, y_val

    def train_model(self):
        # Train a RandomForestClassifier model
        model = RandomForestClassifier()
        X_train, X_test, y_train, y_test = self.__split_data__(
            X_feature_list_to_drop=["account.id", "label", "billing.zip.code", "billing.city", "package"],
            y_feature="label"
        )
        model.fit(X_train, y_train)
        self.__save_model__(model)
        return model, X_test, y_test

    def __save_model__(self, model):
        # Save the trained model to a file using pickle
        with open("../models/classifier.pkl", "wb") as f:
            pickle.dump(model, f)

    def __load_model__(self):
        # Load a saved model from a file
        try:
            with open("../models/classifier.pkl", "rb") as f:
                model = pickle.load(f)
            return model
        except:
            print("Model not found. Please train the model first.")

    def predict_and_return_accuracy_and_roc(self):
        # Train the model, make predictions, and calculate accuracy and ROC AUC
        model, X_test, y_test = self.train_model()
        y_pred = model.predict(X_test)
        acc_score_val = accuracy_score(y_test, y_pred)
        roc_auc_score_val = roc_auc_score(y_test, y_pred)
        return acc_score_val, roc_auc_score_val

    def run_inference(self, input_data: list):
        # Load the trained model and make predictions on new input data
        model = self.__load_model__()
        y_pred = model.predict(input_data)
        return y_pred[0]
