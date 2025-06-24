import os
import sys
import pandas as pd
import joblib
from dataclasses import dataclass
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'svm.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_embeddings_path, train_target_path, test_embeddings_path, test_target_path):
        logging.info("Entering model training process.")
        try:
            logging.info(f"Loading train embeddings from {train_embeddings_path} and targets from {train_target_path}")
            X_train = pd.read_csv(train_embeddings_path)
            y_train = pd.read_csv(train_target_path)['target']
            logging.info(f"Loading test embeddings from {test_embeddings_path} and targets from {test_target_path}")
            X_test = pd.read_csv(test_embeddings_path)
            y_test = pd.read_csv(test_target_path)['target']
            logging.info("Data loading successful")

            if X_train.shape[0] != y_train.shape[0]:
                raise ValueError("Mismatch between train embeddings and target sizes")
            if X_test.shape[0] != y_test.shape[0]:
                raise ValueError("Mismatch between test embeddings and target sizes")
            if X_train.shape[1] != X_test.shape[1]:
                raise ValueError("Mismatch between train and test feature dimensions")

            logging.info("Initializing Linear SVM model with GridSearchCV")
            svm_linear = SVC(kernel='linear', probability=True)
            param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
            grid_search = GridSearchCV(
                estimator=svm_linear,
                param_grid=param_grid,
                scoring='f1',
                cv=5,
                verbose=2,
                n_jobs=-1
            )

            logging.info("Training model with GridSearchCV")
            grid_search.fit(X_train, y_train)
            logging.info(f"Best parameters: {grid_search.best_params_}")
            logging.info(f"Best F1 score from CV: {grid_search.best_score_:.4f}")

            logging.info("Evaluating model on test set")
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred, average='weighted')
            test_precision = precision_score(y_test, y_pred, zero_division=0)
            test_recall = recall_score(y_test, y_pred, zero_division=0)
            test_roc_auc = roc_auc_score(y_test, y_pred)
            logging.info("Test set metrics:")
            logging.info(f"- Accuracy: {test_accuracy:.4f}")
            logging.info(f"- F1 Score: {test_f1:.4f}")
            logging.info(f"- Precision: {test_precision:.4f}")
            logging.info(f"- Recall: {test_recall:.4f}")
            logging.info(f"- ROC AUC Score: {test_roc_auc:.4f}")
            logging.info("\nClassification Report:")
            logging.info(classification_report(y_test, y_pred, output_dict=False))

            logging.info(f"Saving trained model to {self.model_trainer_config.trained_model_file_path}")
            joblib.dump(best_model, self.model_trainer_config.trained_model_file_path)
            logging.info("Model saved successfully")
            logging.info("Model training completed")
            return {
                'test_accuracy': test_accuracy,
                'test_f1': test_f1,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_roc_auc': test_roc_auc,
                'best_params': grid_search.best_params_
            }
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = ModelTrainer()
    train_embeddings_path = os.path.join('artifacts', 'train_embeddings.csv')
    train_target_path = os.path.join('artifacts', 'train_target.csv')
    test_embeddings_path = os.path.join('artifacts', 'test_embeddings.csv')
    test_target_path = os.path.join('artifacts', 'test_target.csv')
    results = obj.initiate_model_training(
        train_embeddings_path,
        train_target_path,
        test_embeddings_path,
        test_target_path
    )
    print("Training Results:", results)