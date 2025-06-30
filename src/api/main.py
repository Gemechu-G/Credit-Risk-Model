    from fastapi import FastAPI, HTTPException
    from src.api.pydantic_models import CustomerFeatures, PredictionResponse
    import mlflow.pyfunc
    import pandas as pd
    import logging
    import os

    # Configure logging for the API
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    app = FastAPI(
        title="Credit Risk Probability API",
        description="Predicts credit risk probability for new customers based on engineered features."
    )

    # --- MLflow Model Loading Configuration ---
    # These should match the registered model name and version from your train.py
    MODEL_NAME = "CreditRiskPredictorPipeline"
    MODEL_VERSION = 1 # Or "Production" if you've promoted it in MLflow UI

    # Global variables to hold the loaded model
    # It's loaded once on startup for efficiency
    credit_risk_predictor = None

    @app.on_event("startup")
    async def load_model():
        """
        Loads the MLflow pyfunc model when the FastAPI application starts up.
        """
        global credit_risk_predictor
        try:
            # Construct the MLflow model URI
            model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
            logging.info(f"Attempting to load MLflow model from URI: {model_uri}")

            # Load the custom pyfunc model (which includes preprocessor and predictor)
            credit_risk_predictor = mlflow.pyfunc.load_model(model_uri)
            logging.info(f"MLflow model '{MODEL_NAME}' version '{MODEL_VERSION}' loaded successfully.")

        except Exception as e:
            logging.error(f"Error loading MLflow model: {e}", exc_info=True)
            # Raise an exception to prevent the application from starting if the model can't be loaded
            raise HTTPException(status_code=500, detail=f"Failed to load ML model: {e}")

    @app.get("/")
    async def root():
        """
        Root endpoint for a simple health check.
        """
        return {"message": "Credit Risk Probability API is running!"}


    @app.post("/predict", response_model=PredictionResponse)
    async def predict_credit_risk(features: CustomerFeatures):
        """
        Predicts the credit risk probability for a new customer based on their engineered features.

        Args:
            features (CustomerFeatures): Pydantic model containing the engineered features
                                        for a single customer.

        Returns:
            PredictionResponse: Pydantic model containing the customer ID,
                                predicted risk probability, and risk category.
        """
        if credit_risk_predictor is None:
            logging.error("Model not loaded. Cannot make predictions.")
            raise HTTPException(status_code=500, detail="Model not loaded. Please try again later.")

        try:
            # Convert the incoming Pydantic model to a Pandas DataFrame.
            # The `predict` method of our `CreditRiskPredictor` expects a DataFrame.
            # Ensure the order and names of columns in the DataFrame match what the
            # preprocessor within the `CreditRiskPredictor` expects.
            # model_dump() is for Pydantic v2. For v1, use .dict()
            input_df = pd.DataFrame([features.model_dump()])
            logging.info(f"Received input features: {input_df.columns.tolist()}")

            # Make prediction using the loaded MLflow pyfunc model.
            # The `predict` method of CreditRiskPredictor handles both preprocessing and prediction.
            prediction_output_df = credit_risk_predictor.predict(input_df)

            # Extract results (assuming prediction_output_df has 'risk_probability' and 'risk_category')
            risk_probability = prediction_output_df['risk_probability'].iloc[0]
            risk_category = prediction_output_df['risk_category'].iloc[0]

            logging.info(f"Prediction for customer {features.customer_id}: Probability={risk_probability:.4f}, Category={risk_category}")

            return PredictionResponse(
                customer_id=features.customer_id,
                risk_probability=risk_probability,
                risk_category=risk_category
            )

        except Exception as e:
            logging.error(f"Prediction failed for customer {features.customer_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    