import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# For more complex models, you might add:
# from lightgbm import LGBMClassifier
# from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import logging
import os # For setting environment variables

# Import the data processing pipeline if you decide to integrate it here
from data_processing import get_data_processing_pipeline # Assuming get_data_processing_pipeline returns the ColumnTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set MLflow tracking URI (local 'mlruns' folder by default if not set)
# You can set it explicitly like:
# mlflow.set_tracking_uri("file:///path/to/your/mlruns")
# For local tracking, it automatically creates 'mlruns' in the current directory where train.py is executed.
# It's good practice to set an experiment name
mlflow.set_experiment("Credit_Risk_Model_Training")
logging.info("MLflow experiment set to 'Credit_Risk_Model_Training'")

def train_model(data_path, test_size=0.2, random_state=42):
    """
    Loads processed data, trains a Logistic Regression model, evaluates it,
    and logs the experiment details to MLflow.

    Args:
        data_path (str): Path to the processed customer-level CSV file.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random number generation for reproducibility.
    """
    logging.info(f"Starting model training for data at: {data_path}")

    # --- 1. Load Data ---
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Loaded data with shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Error: Processed data file not found at '{data_path}'.")
        return

    # --- 2. Separate Features (X) and Target (y) ---
    # Remember: 'is_high_risk' is your target variable.
    # 'CustomerId' and 'Cluster' are identifiers/intermediate results, not features for the model.
    # The other RFM features (log_transformed), and other aggregates are your X.

    # Identify numerical and categorical features for the final preprocessing step
    # These lists should match those used in get_data_processing_pipeline() in data_processing.py
    # You need to ensure these features are present in your customer_features_with_risk.csv
    # Based on Task 4's `prepare_customer_data` output:
    feature_columns = [
        'Recency_log', 'Frequency_log', 'Monetary_log', # Log-transformed RFM
        'avg_transaction_amount', 'std_transaction_amount', # Other numerical aggregates
        'unique_product_categories', 'unique_channels' # Other numerical aggregates
        # Add any other categorical aggregates if you introduced them.
        # Example: 'most_frequent_product_category_encoded' (if you plan to encode it)
    ]

    # Filter out features that might not be available if not all aggregations were fully implemented
    # (e.g., if you skipped 'unique_product_categories' etc.)
    X = df[feature_columns]
    y = df['is_high_risk']
    logging.info(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")
    logging.info(f"Target distribution (is_high_risk):\n{y.value_counts(normalize=True)}")


    # --- 3. Apply Final Preprocessing Pipeline (from data_processing.py) ---
    # This pipeline will scale numerical features and encode categorical features
    # for the model. It needs to be fitted on training data and transform train/test data.

    # Get the final preprocessing ColumnTransformer from data_processing.py
    # This preprocessor expects the exact feature columns defined within it.
    # Ensure that `feature_columns` list above includes all features that this `preprocessor` expects.
    data_preprocessor = get_data_processing_pipeline()
    logging.info("Initialized data preprocessing pipeline from data_processing.py.")


    # --- 4. Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y # Stratify for imbalanced target
    )
    logging.info(f"Data split into Train (X:{X_train.shape}, y:{y_train.shape}) and Test (X:{X_test.shape}, y:{y_test.shape}).")

    # --- 5. Fit and Transform Data using Preprocessor ---
    # Fit the preprocessor ONLY on the training data to prevent data leakage.
    X_train_processed = data_preprocessor.fit_transform(X_train)
    # Transform both training and testing data using the fitted preprocessor.
    X_test_processed = data_preprocessor.transform(X_test)
    logging.info(f"Features processed using pipeline. X_train_processed shape: {X_train_processed.shape}")


    # --- 6. MLflow Run ---
    # Start an MLflow run to log all aspects of this experiment.
    # The 'with' statement ensures the run is properly ended.
    with mlflow.start_run():
        # --- Log Parameters ---
        # Model parameters
        solver = 'liblinear' # 'liblinear' is good for small datasets and L1/L2 regularization
        penalty = 'l1' # L1 regularization for feature selection
        C = 0.1 # Inverse of regularization strength; smaller values mean stronger regularization
        # Due to class imbalance, use 'balanced' class weight
        class_weight = 'balanced'

        mlflow.log_param("model_type", "Logistic Regression")
        mlflow.log_param("solver", solver)
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("C", C)
        mlflow.log_param("class_weight", class_weight)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        logging.info(f"MLflow logging parameters: solver={solver}, penalty={penalty}, C={C}, class_weight={class_weight}")

        # --- 7. Initialize and Train Model ---
        # Logistic Regression is a good interpretable baseline
        model = LogisticRegression(
            solver=solver,
            penalty=penalty,
            C=C,
            class_weight=class_weight, # Handles class imbalance by weighting
            random_state=random_state
        )
        logging.info("Training Logistic Regression model...")
        model.fit(X_train_processed, y_train)
        logging.info("Model training complete.")

        # --- 8. Evaluate Model ---
        y_pred = model.predict(X_test_processed)
        y_proba = model.predict_proba(X_test_processed)[:, 1] # Probability of the positive class (high risk)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0) # zero_division=0 to handle cases where no positive predictions are made
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)

        logging.info(f"Model Evaluation on Test Set:")
        logging.info(f"  Accuracy: {accuracy:.4f}")
        logging.info(f"  Precision: {precision:.4f}")
        logging.info(f"  Recall: {recall:.4f}")
        logging.info(f"  F1-Score: {f1:.4f}")
        logging.info(f"  ROC-AUC: {roc_auc:.4f}")

        # --- 9. Log Metrics to MLflow ---
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        logging.info("MLflow metrics logged.")

        # --- 10. Log Model to MLflow ---
        # It's best practice to log the *entire pipeline* including preprocessing
        # so that when you load the model, it comes with its transformers.
        # However, for this structure, our preprocessing is a separate pipeline (ColumnTransformer).
        # We can save the model itself, and then combine the preprocessor and model into a final
        # inference pipeline later (e.g., in `src/predict.py`).
        # For now, let's log the trained Logistic Regression model.

        # To log the full preprocessor + model as a single sklearn pipeline:
        # combined_pipeline = Pipeline(steps=[
        #     ('preprocessor', data_preprocessor),
        #     ('classifier', model)
        # ])
        # mlflow.sklearn.log_model(
        #     sk_model=combined_pipeline,
        #     artifact_path="credit_risk_model_pipeline",
        #     registered_model_name="LogisticRegressionCreditRiskModel" # Optional: Register for versioning
        # )
        # logging.info("MLflow logged combined preprocessor and model.")

        # Or, just log the trained model:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="credit_risk_model",
            registered_model_name="LogisticRegressionCreditRiskModel" # Optional: Register for versioning
        )
        logging.info("MLflow logged Logistic Regression model.")

        logging.info("MLflow run completed successfully.")

# You can also explore other models here (e.g., LGBMClassifier)
# def train_lgbm_model(data_path, test_size=0.2, random_state=42):
#     # ... similar structure as train_model but with LGBMClassifier ...
#     pass # Implement this if you want to experiment with GBM

if __name__ == '__main__':
    logging.info("--- Starting train.py script ---")

    # Path to the processed data created in Task 4
    PROCESSED_DATA_PATH = 'data/processed/customer_features_with_risk.csv'

    # Call the training function
    train_model(PROCESSED_DATA_PATH)

    logging.info("--- train.py script finished ---")