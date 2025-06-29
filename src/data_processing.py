import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder # MinMaxScaler is an option too

# Optional: For logging inside the script
import logging
# Configure logging to display messages with a specific format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to convert a specified date column to datetime objects
    and extract time-based features (hour, day of week, month, year).
    """
    def __init__(self, date_column='TransactionStartTime'):
        # Initialize with the name of the column containing date/time information
        self.date_column = date_column

    def fit(self, X, y=None):
        # This transformer does not learn any parameters from the data during fitting,
        # so it simply returns itself.
        return self

    def transform(self, X):
        """
        Transforms the input DataFrame by extracting date-time features.

        Args:
            X (pd.DataFrame): The input DataFrame containing the date column.

        Returns:
            pd.DataFrame: The DataFrame with new date-time features added.
        """
        # Create a copy to avoid modifying the original DataFrame in place
        X_copy = X.copy()

        # Convert the specified date column to datetime objects.
        # 'errors='coerce'' will turn unparseable dates into NaT (Not a Time),
        # preventing errors and allowing imputation later if needed.
        X_copy[self.date_column] = pd.to_datetime(X_copy[self.date_column], errors='coerce')

        # Extract various time-based features from the datetime column
        # .dt accessor is used to access datetime properties of a Series
        X_copy['transaction_hour'] = X_copy[self.date_column].dt.hour
        X_copy['transaction_day_of_week'] = X_copy[self.date_column].dt.dayofweek # Monday=0, Sunday=6
        X_copy['transaction_month'] = X_copy[self.date_column].dt.month
        X_copy['transaction_year'] = X_copy[self.date_column].dt.year

        # Log the action for tracking purposes
        logging.info(f"Extracted date features from '{self.date_column}'.")

        # Return the DataFrame with the newly created features
        return X_copy

def get_data_processing_pipeline():
    """
    Returns a sklearn.pipeline.Pipeline object for preprocessing the raw data.
    This pipeline includes a custom date feature extractor and a ColumnTransformer
    for numerical and categorical feature preprocessing.
    """

    # Define lists of features based on their type and how they will be processed.
    # These lists are based on the expected columns *after* the DateFeatureExtractor runs.
    # 'Amount' and 'Value' are original numerical features.
    # 'transaction_hour', 'transaction_day_of_week', 'transaction_month', 'transaction_year'
    # are new numerical features extracted by DateFeatureExtractor.
    numerical_features = ['Amount', 'Value',
                          'transaction_hour', 'transaction_day_of_week',
                          'transaction_month', 'transaction_year']

    # Categorical features that will be One-Hot Encoded.
    # 'ProductId' and 'ProviderId' are included here, but be mindful of high cardinality
    # which might lead to many new columns. For now, OneHotEncoder handles them.
    categorical_features = ['CurrencyCode', 'CountryCode', 'ProductCategory',
                            'ChannelId', 'PricingStrategy', 'ProviderId', 'ProductId']

    # --- Define preprocessing steps for numerical features ---
    # This pipeline handles missing values (imputation) and scales numerical data.
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Fills missing numerical values with the median
        ('scaler', StandardScaler())                   # Standardizes features (mean=0, variance=1)
        # Alternative scaler: ('scaler', MinMaxScaler()) # Scales features to a [0, 1] range
    ])
    logging.info("Defined numerical feature transformer pipeline.")

    # --- Define preprocessing steps for categorical features ---
    # This pipeline handles missing values and encodes categorical data.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Fills missing categorical values with the most frequent category
        ('onehot', OneHotEncoder(handle_unknown='ignore'))     # Converts categorical data into numerical format
                                                               # 'handle_unknown='ignore'' prevents errors if new, unseen categories appear during inference
    ])
    logging.info("Defined categorical feature transformer pipeline.")

    # --- Combine transformers using ColumnTransformer ---
    # ColumnTransformer applies different transformers to different columns of the DataFrame.
    # It takes a list of (name, transformer, columns) tuples.
    # 'remainder='drop'' means any columns not specified in `transformers` will be dropped.
    # This ensures that identifier columns (like TransactionId, AccountId) and the original
    # 'TransactionStartTime' (if not dropped by DateFeatureExtractor) are not passed to the model.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Drops any columns not explicitly listed in `numerical_features` or `categorical_features`
    )
    logging.info("Defined ColumnTransformer to apply specific transformations to numerical and categorical features.")

    # --- Assemble the full preprocessing pipeline ---
    # The order of steps in the main Pipeline is crucial:
    # 1. 'date_feature_extractor': Extracts new features from 'TransactionStartTime'.
    #    The output of this step is a DataFrame with the new date features.
    # 2. 'preprocessor': Takes the output from 'date_feature_extractor' and applies
    #    numerical and categorical transformations using the ColumnTransformer.
    full_pipeline = Pipeline(steps=[
        ('date_feature_extractor', DateFeatureExtractor(date_column='TransactionStartTime')),
        ('preprocessor', preprocessor)
    ])
    logging.info("Assembled full data processing pipeline.")

    return full_pipeline

if __name__ == '__main__':
    # This block allows you to run the script directly to test the pipeline.
    logging.info("--- Starting test of data_processing.py script ---")

    # Define the path to your raw data.
    # This path assumes you are running the script from the 'credit-risk-model' root directory.
    RAW_DATA_PATH = 'data/raw/data.csv'

    try:
        # Load the raw dataset
        df_raw = pd.read_csv(RAW_DATA_PATH)
        logging.info(f"Successfully loaded raw data from '{RAW_DATA_PATH}' with shape: {df_raw.shape}")
    except FileNotFoundError:
        logging.error(f"Error: Raw data file not found at '{RAW_DATA_PATH}'. Please ensure 'data.csv' is in the 'data/raw/' directory.")
        exit() # Exit the script if the data file is not found

    # Separate features (X) from the target variable ('FraudResult').
    # Note: 'FraudResult' is the direct fraud label. Your credit risk proxy ('is_high_risk')
    # will be engineered in Task 4. For now, we just exclude 'FraudResult' from the features.
    # Also, explicitly drop identifier columns that are not features.
    # The `remainder='drop'` in ColumnTransformer will also handle this, but explicit drop
    # here ensures X_raw only contains columns relevant for the pipeline's first step.
    columns_to_exclude_from_X = ['FraudResult', 'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
    X_raw = df_raw.drop(columns=columns_to_exclude_from_X, errors='ignore') # 'errors='ignore'' prevents error if column not found
    y_fraud = df_raw['FraudResult'] # Keep the original FraudResult for reference, though not used by this pipeline directly.

    logging.info(f"Prepared raw features (X_raw) with shape: {X_raw.shape}")

    # Get the complete data processing pipeline
    pipeline = get_data_processing_pipeline()

    # Fit and Transform the raw data using the pipeline.
    # This step applies all the defined transformations in sequence.
    logging.info("Fitting and transforming data using the pipeline...")
    X_transformed = pipeline.fit_transform(X_raw)
    logging.info(f"Data transformation completed. Shape of transformed data: {X_transformed.shape}")

    # Display a small part of the transformed data to verify.
    # The output of ColumnTransformer with OneHotEncoder is typically a NumPy array (dense or sparse).
    print("\n--- Sample of Transformed Data (First 5 rows, first 10 columns) ---")
    # If the output is a sparse matrix, convert it to a dense array for printing
    if hasattr(X_transformed, 'toarray'):
        print(X_transformed[:5, :10].toarray())
    else:
        print(X_transformed[:5, :10]) # Print first 5 rows and first 10 columns

    logging.info("--- Test of data_processing.py script completed successfully ---")

