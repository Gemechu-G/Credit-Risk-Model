    from pydantic import BaseModel
    from typing import Optional, List

    # Define the input schema for a new customer prediction.
    # These fields must exactly match the engineered features that your
    # CreditRiskPredictorPipeline expects after all preprocessing steps
    # (RFM, other aggregations, and before final scaling/encoding).
    # This means the client calling the API is responsible for providing these
    # already engineered features.

    class CustomerFeatures(BaseModel):
        # RFM features (log-transformed are used by the model)
        Recency_log: float
        Frequency_log: float
        Monetary_log: float

        # Other numerical aggregates
        avg_transaction_amount: float
        std_transaction_amount: float
        unique_product_categories: int
        unique_channels: int

        # Add other features if you added more aggregations in data_processing.py
        # For example, if you aggregated categorical features and encoded them.
        # If you included ProductId/ProviderId in OHE in data_processing.py,
        # then the API client would need to send all those one-hot encoded columns.
        # This is why passing raw data and having the model handle it is better,
        # but for this guide, we simplified the pyfunc model input.
        # If your data_preprocessor also expects one-hot encoded columns for
        # CurrencyCode, CountryCode, ProductCategory, ChannelId, PricingStrategy, ProviderId, ProductId,
        # then you would need to list all of them here, e.g.:
        # CurrencyCode_UGX: float
        # ProductCategory_Airtime: float
        # ... and so on for all OHE columns.
        # For simplicity, we'll assume the model only uses the numerical features for now,
        # or that the client sends the OHE columns if they were part of the training.
        # For a production system, you'd need a robust way to handle all OHE columns.

        # Let's assume the final preprocessor in data_processing.py only takes numerical features
        # and the categorical features are handled by the OHE part of the preprocessor.
        # If the model was trained on OHE features, these need to be part of the input.
        # The `feature_columns` in `train.py` should guide this.
        # The `get_data_processing_pipeline` in `data_processing.py` returns a ColumnTransformer
        # that handles both numerical and categorical. So, the client needs to send the *raw*
        # numerical and categorical features that go into that ColumnTransformer.

        # Let's revert to the more robust API input: The API takes the original raw features
        # that the `data_processing.py` pipeline (specifically, the `ColumnTransformer` part) expects.
        # This means the `CreditRiskPredictor.predict` method needs to handle the transformation
        # from these raw features to the final model input.

        # RE-REFINEMENT: The `CreditRiskPredictor` should take the *raw transaction details*
        # and internally call `prepare_customer_data` to get the engineered features,
        # then pass those through the `data_preprocessor` and `model`.
        # This is the most user-friendly API.

        # So, the input should be the original transaction features that are used to derive RFM and other aggregates.
        # This means the API needs to be able to look up historical data for a customer to compute RFM.
        # This is beyond a simple FastAPI example without a database.

        # FOR THIS PROJECT'S SCOPE AND SIMPLICITY:
        # The API will expect the *already engineered customer-level features* as input.
        # This means the client must compute RFM, other aggregates, and send them.
        # The `CustomerFeatures` model below reflects this.

        # Original transaction IDs might be useful for tracking but not for prediction
        customer_id: str


    # Define the output schema for the prediction
    class PredictionResponse(BaseModel):
        customer_id: str
        risk_probability: float
        risk_category: str # "High Risk" or "Low Risk"
        # Optional: Add credit_score if you implement it later
        # credit_score: Optional[int] = None
    