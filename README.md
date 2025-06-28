# Credit-Risk-Model
## Credit Scoring Business Understanding

### Basel II and Model Interpretability

The Basel II Capital Accord emphasizes the importance of effective risk management in financial institutions, especially for credit risk. It requires banks to use internal rating systems that are transparent, auditable, and explainable. This has direct implications for machine learning: models used for credit scoring must be interpretable, well-documented, and backed by solid reasoning. For Bati Bank, this means using models that are not only accurate but also explain how predictions are made. Models like Logistic Regression, combined with tools like Weight of Evidence (WoE), align well with these expectations due to their simplicity and traceability.

---

### Need for a Proxy Variable

In our dataset, there is no explicit label that indicates whether a customer has defaulted. This is a common scenario in alternative data settings. Therefore, we need to engineer a proxy target variable that identifies potentially high-risk customers. One way to do this is by using RFM (Recency, Frequency, Monetary) analysis to find disengaged customers who are less active or spend less frequently — a possible indicator of credit risk. However, using a proxy introduces uncertainty: it may not fully reflect actual defaults and may misclassify reliable customers as risky. This could lead to unfair decisions, reduced customer trust, and regulatory scrutiny if not handled carefully.

---

### Trade-offs: Simple vs Complex Models

There is a trade-off between model simplicity and performance. Simple models like Logistic Regression with Weight of Evidence (WoE) are highly interpretable, making them suitable for regulated industries like finance. They allow for straightforward documentation, explainability, and easier compliance with auditing requirements. On the other hand, complex models like Gradient Boosting Machines (e.g., XGBoost) typically achieve higher predictive performance but are harder to interpret. In regulated environments, using complex models requires additional explainability tools (e.g., SHAP or LIME) and may still face resistance unless their decisions can be clearly justified.

