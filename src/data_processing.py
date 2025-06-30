import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
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

class RFMCalculator(BaseEstimator, TransformerMixin):
    """
    Custom transformer to calculate Recency, Frequency, and Monetary (RFM) values
    for each customer from transaction data.
    """
    def __init__(self, customer_id_col='CustomerId', transaction_time_col='TransactionStartTime',
                 amount_col='Value', snapshot_date=None):
        self.customer_id_col = customer_id_col
        self.transaction_time_col = transaction_time_col
        self.amount_col = amount_col
        self.snapshot_date = snapshot_date # Will be determined dynamically if None

    def fit(self, X, y=None):
        # If snapshot_date is not provided, calculate it from the training data
        if self.snapshot_date is None:
            # Ensure transaction_time_col is datetime for max()
            X[self.transaction_time_col] = pd.to_datetime(X[self.transaction_time_col], errors='coerce')
            self.snapshot_date_ = X[self.transaction_time_col].max() + pd.Timedelta(days=1)
        else:
            self.snapshot_date_ = pd.to_datetime(self.snapshot_date)
        logging.info(f"RFM snapshot date set to: {self.snapshot_date_}")
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Ensure transaction_time_col is datetime
        X_copy[self.transaction_time_col] = pd.to_datetime(X_copy[self.transaction_time_col], errors='coerce')

        # Calculate RFM metrics
        rfm_df = X_copy.groupby(self.customer_id_col).agg(
            Recency=(self.transaction_time_col, lambda date: (self.snapshot_date_ - date.max()).days),
            Frequency=(self.transaction_time_col, 'count'), # Count of transactions
            Monetary=(self.amount_col, 'sum') # Sum of transaction values
        ).reset_index()

        # Handle cases where Monetary might be negative if 'Value' can be negative and you want absolute spending
        # For RFM, Monetary usually implies positive contribution.
        # If 'Value' can be negative (e.g., refunds), you might want to use abs() or filter.
        # For this context, assuming 'Value' is the absolute value of the transaction.
        # If 'Value' is not always positive, consider:
        # rfm_df['Monetary'] = rfm_df['Monetary'].abs() # Or sum of positive values only

        logging.info(f"Calculated RFM metrics for {len(rfm_df)} customers.")

        # RFM features are often highly skewed, apply log transformation (log1p handles zeros)
        # Add a small constant to Recency if it can be 0, to avoid log(0) issues.
        # Recency of 0 means transaction on snapshot date, so 1 day difference.
        # If (snapshot_date - date.max()).days can be 0, add 1.
        rfm_df['Recency'] = rfm_df['Recency'].apply(lambda x: x + 1 if x == 0 else x) # Add 1 if Recency is 0
        rfm_df['Recency_log'] = np.log1p(rfm_df['Recency'])
        rfm_df['Frequency_log'] = np.log1p(rfm_df['Frequency'])
        rfm_df['Monetary_log'] = np.log1p(rfm_df['Monetary']) # Use Monetary_log for clustering

        # Drop original RFM columns if only log-transformed versions are needed for clustering
        # rfm_df = rfm_df.drop(columns=['Recency', 'Frequency', 'Monetary'])

        # Rename for clarity if needed
        rfm_df = rfm_df.rename(columns={
            'Recency': 'recency_days', # Keep original for potential interpretation
            'Frequency': 'transaction_frequency',
            'Monetary': 'total_monetary_value'
        })


        # The output of this transformer is a customer-level DataFrame with RFM features.
        # This DataFrame will then be merged with other customer-level aggregated features.
        return rfm_df
    # Modify the existing get_data_processing_pipeline function
def get_data_processing_pipeline():
    """
    Returns a sklearn.pipeline.Pipeline object for end-to-end preprocessing,
    including date feature extraction, RFM calculation, customer clustering,
    and final feature scaling/encoding for customer-level data.
    """

    # --- Step 1: Extract Date Features (Transaction-level) ---
    date_feature_extractor = DateFeatureExtractor(date_column='TransactionStartTime')
    logging.info("Pipeline step: DateFeatureExtractor defined.")

    # --- Step 2: Calculate RFM Metrics (Aggregates to Customer-level) ---
    # This step will transform transaction-level data into customer-level RFM data.
    # It needs the original TransactionStartTime and Value/Amount columns.
    rfm_calculator = RFMCalculator(
        customer_id_col='CustomerId',
        transaction_time_col='TransactionStartTime',
        amount_col='Value' # Assuming 'Value' is the monetary amount
    )
    logging.info("Pipeline step: RFMCalculator defined.")

    # --- Step 3: Customer Clustering and High-Risk Labeling ---
    # This step takes RFM features (log-transformed) and assigns the 'is_high_risk' label.
    # It operates on the customer-level data output by RFMCalculator.
    # The RFMCalculator output will have 'Recency_log', 'Frequency_log', 'Monetary_log'
    # as well as 'recency_days', 'transaction_frequency', 'total_monetary_value'
    customer_cluster_labeler = CustomerClusterAndLabeler(
        n_clusters=3,
        random_state=42,
        rfm_features=['Recency_log', 'Frequency_log', 'Monetary_log'],
        customer_id_col='CustomerId'
    )
    logging.info("Pipeline step: CustomerClusterAndLabeler defined.")

    # --- Step 4: Aggregate Other Customer-Level Features ---
    # You need to define how to aggregate other transaction-level features
    # (like average Amount, count of categories, etc.) to the customer level.
    # This is not a standard sklearn transformer, so we might need a custom one
    # or handle it outside the main pipeline for now, then merge.

    # For a truly end-to-end pipeline, you'd need a custom transformer that:
    # 1. Takes the raw transaction data.
    # 2. Extracts date features.
    # 3. Calculates RFM.
    # 4. Aggregates other numerical/categorical transaction features to customer level.
    # 5. Merges all customer-level features.
    # 6. Performs clustering and labeling.
    # 7. Applies final scaling/encoding to the *combined* customer-level features.

    # This is getting complex for a single pipeline.
    # Let's simplify the pipeline structure for now:
    # The pipeline will output the RFM and is_high_risk.
    # Other customer-level features (e.g., average transaction amount) will be calculated
    # separately and merged *after* this pipeline, before final training.

    # REVISED PLAN:
    # The `get_data_processing_pipeline` will focus on generating the RFM and `is_high_risk`
    # for each customer. It will also handle the initial transaction-level date feature extraction.
    # A separate function will then aggregate other features to customer level and merge.

    # Let's make `get_data_processing_pipeline` return a pipeline that takes raw transactions
    # and outputs a customer-level DataFrame with RFM and `is_high_risk`.
    # The final preprocessing for *all* features (including RFM and other aggregates)
    # will happen *after* this pipeline, or this pipeline will only output RFM+target.

    # Let's make the pipeline output the RFM features + cluster + is_high_risk.
    # The other numerical/categorical features (like Product Category, ChannelId)
    # will need to be aggregated to the customer level separately and then joined.

    # This indicates that the full data processing might be a multi-stage process:
    # Stage 1: Transaction-level feature extraction (DateFeatureExtractor).
    # Stage 2: Customer-level aggregation (RFM, other aggregates).
    # Stage 3: Clustering and target labeling.
    # Stage 4: Final preprocessing (scaling, encoding) on the combined customer-level features.

    # To keep `data_processing.py` as the central script for "model-ready format",
    # let's define a function that orchestrates these stages.
    # The `get_data_processing_pipeline` will return the pipeline for the *final* scaling/encoding
    # of the customer-level features, *after* RFM and target are created.

    # Let's redefine `get_data_processing_pipeline` to return a pipeline that expects
    # a customer-level DataFrame (with RFM, other aggregates, and the target).
    # And provide a separate function `prepare_customer_data` that takes raw data,
    # performs RFM, clustering, and other aggregations, then returns the DataFrame
    # that `get_data_processing_pipeline` expects.

    # This is a more modular approach.

    # --- Final Preprocessing for Customer-Level Features (after RFM, Clustering, and other aggregations) ---
    # This part will be used *after* we have generated all customer-level features.
    # For now, we define it, but its application will be in the `if __name__ == '__main__':` block,
    # or in `src/train.py`.

    # Define numerical and categorical features that will be present in the *final customer-level* DataFrame
    # These will include RFM features (log-transformed) and other aggregated numerical/categorical features.
    final_numerical_features = [
        'recency_days', 'transaction_frequency', 'total_monetary_value', # Original RFM values
        'Recency_log', 'Frequency_log', 'Monetary_log', # Log-transformed RFM values
        # Add placeholders for other aggregated numerical features you might create (e.g., avg_amount_per_customer)
        # 'avg_transaction_amount_per_customer',
        # 'total_transactions_per_customer',
        # 'std_transaction_amount_per_customer',
    ]

    # Add placeholders for aggregated categorical features (e.g., most frequent channel, count of unique products)
    final_categorical_features = [
        # Placeholder for aggregated categorical features
        # 'most_frequent_channel',
        # 'most_frequent_product_category'
    ]

    # Numerical transformer for the final customer-level features
    final_numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical transformer for the final customer-level features
    final_categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # ColumnTransformer for the final customer-level features
    final_preprocessor = ColumnTransformer(
        transformers=[
            ('num', final_numerical_transformer, final_numerical_features),
            ('cat', final_categorical_transformer, final_categorical_features)
        ],
        remainder='drop' # Drop any other columns (like CustomerId, Cluster, is_high_risk - target)
    )
    logging.info("Defined final preprocessor for customer-level features.")

    # This pipeline will be used *after* all customer-level features and target are generated.
    # For now, it's just the final scaling/encoding.
    return final_preprocessor # This pipeline expects customer-level data as input


# --- New function to orchestrate the full data preparation from raw to customer-level with target ---
def prepare_customer_data(df_raw, snapshot_date=None):
    """
    Orchestrates the full data preparation process from raw transaction data
    to a customer-level DataFrame with engineered features and the 'is_high_risk' target.

    Args:
        df_raw (pd.DataFrame): The raw transaction DataFrame.
        snapshot_date (str or datetime, optional): The date to calculate Recency against.
                                                   If None, it's derived from max transaction date.

    Returns:
        pd.DataFrame: A customer-level DataFrame with features and 'is_high_risk' target.
    """
    logging.info("Starting full customer data preparation process.")

    # Step 1: Extract Date Features from raw transactions
    date_extractor = DateFeatureExtractor(date_column='TransactionStartTime')
    df_with_date_features = date_extractor.fit_transform(df_raw.copy())
    logging.info(f"Shape after date feature extraction: {df_with_date_features.shape}")

    # Step 2: Calculate RFM Metrics (aggregates to customer level)
    rfm_calculator = RFMCalculator(
        customer_id_col='CustomerId',
        transaction_time_col='TransactionStartTime',
        amount_col='Value',
        snapshot_date=snapshot_date
    )
    # Fit and transform on the data with date features (though RFM only needs original time/value)
    # RFMCalculator will handle its own datetime conversion internally.
    df_rfm = rfm_calculator.fit_transform(df_with_date_features.copy())
    logging.info(f"Shape after RFM calculation (customer-level): {df_rfm.shape}")

    # Step 3: Customer Clustering and High-Risk Labeling
    # This operates on the df_rfm which contains RFM_log features
    customer_labeler = CustomerClusterAndLabeler(
        rfm_features=['Recency_log', 'Frequency_log', 'Monetary_log'],
        customer_id_col='CustomerId'
    )
    # Fit and transform to identify clusters and assign labels
    df_customer_features = customer_labeler.fit_transform(df_rfm.copy())
    logging.info(f"Shape after clustering and labeling: {df_customer_features.shape}")

    # Step 4: Aggregate other transaction-level features to customer level
    # This is where you would add other customer-level aggregations
    # beyond RFM. For example:
    # - Average transaction amount per customer
    # - Count of unique product categories per customer
    # - Most frequent channel used by customer
    # For simplicity, let's just use the RFM features and the target for now.
    # If you want to add more, you'd define aggregation logic here and merge.

    # Example of other aggregations (you can expand this based on your EDA):
    # Group original raw data by CustomerId to get other aggregates
    other_customer_aggregates = df_raw.groupby('CustomerId').agg(
        avg_transaction_amount=('Amount', 'mean'),
        std_transaction_amount=('Amount', 'std'),
        unique_product_categories=('ProductCategory', lambda x: x.nunique()),
        unique_channels=('ChannelId', lambda x: x.nunique())
        # You can add more complex aggregations here, e.g., mode for categorical
    ).reset_index()

    # Merge RFM/Cluster data with other aggregates
    final_customer_df = pd.merge(df_customer_features, other_customer_aggregates,
                                 on='CustomerId', how='left')
    logging.info(f"Shape after merging other customer aggregates: {final_customer_df.shape}")

    # Handle potential NaNs from std() on single-transaction customers
    final_customer_df['std_transaction_amount'] = final_customer_df['std_transaction_amount'].fillna(0)


    # The final_customer_df now contains:
    # CustomerId, recency_days, transaction_frequency, total_monetary_value,
    # Recency_log, Frequency_log, Monetary_log, Cluster, is_high_risk,
    # avg_transaction_amount, std_transaction_amount, unique_product_categories, unique_channels

    logging.info("Full customer data preparation completed.")
    return final_customer_df
if __name__ == '__main__':
    logging.info("--- Starting full test of data_processing.py script ---")

    RAW_DATA_PATH = 'data/raw/data.csv'

    try:
        df_raw = pd.read_csv(RAW_DATA_PATH)
        logging.info(f"Successfully loaded raw data from '{RAW_DATA_PATH}' with shape: {df_raw.shape}")
    except FileNotFoundError:
        logging.error(f"Error: Raw data file not found at '{RAW_DATA_PATH}'. Please ensure 'data.csv' is in the 'data/raw/' directory.")
        exit()

    # Prepare the customer-level DataFrame with RFM and target
    # Pass the raw DataFrame to the new preparation function
    customer_df = prepare_customer_data(df_raw.copy()) # Pass a copy to avoid modifying original df_raw

    logging.info(f"Customer-level DataFrame prepared with shape: {customer_df.shape}")
    logging.info("Sample of Customer-level DataFrame head:")
    print(customer_df.head())

    logging.info("\nDistribution of 'is_high_risk' target variable:")
    print(customer_df['is_high_risk'].value_counts(normalize=True))

    # Now, apply the final preprocessing pipeline to the customer-level features
    # Exclude 'CustomerId', 'Cluster' and 'is_high_risk' from features for the final pipeline
    features_for_final_pipeline = customer_df.drop(columns=['CustomerId', 'Cluster', 'is_high_risk'], errors='ignore')
    target_variable = customer_df['is_high_risk']

    # Get the final preprocessing pipeline
    final_preprocessor_pipeline = get_data_processing_pipeline()

    # Fit and transform the customer-level features
    logging.info("Fitting and transforming customer-level features with final preprocessor...")
    X_final_transformed = final_preprocessor_pipeline.fit_transform(features_for_final_pipeline)
    logging.info(f"Final transformed feature matrix shape: {X_final_transformed.shape}")

    # Save the final processed customer-level data for Task 5
    # This DataFrame should contain all features (before final scaling/encoding) and the target.
    # Or, you can save the X_final_transformed and target_variable separately.
    # For simplicity, let's save the customer_df with all features and the target.
    # In Task 5, you'll load this and then apply the `final_preprocessor_pipeline`
    # to X_train and X_test.

    # Let's save the `customer_df` as it contains all the features and the target for next step.
    OUTPUT_PROCESSED_DATA_PATH = 'data/processed/customer_features_with_risk.csv'
    customer_df.to_csv(OUTPUT_PROCESSED_DATA_PATH, index=False)
    logging.info(f"Processed customer data saved to '{OUTPUT_PROCESSED_DATA_PATH}'")

    logging.info("--- Full test of data_processing.py script completed successfully ---")