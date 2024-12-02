# Customer Churn Prediction: A Comparative Analysis Using Bayesian and Frequentist Approaches

## Overview
This project analyzes customer churn using both **Bayesian** and **Frequentist** approaches. By leveraging a real-world dataset, it provides insights into customer behavior, aiding stakeholders in retention strategies. The project explores uncertainty quantification, interpretable modeling, and predictive performance to compare these two methodologies.

---

## Features
- **Dual Modeling Approach**:
  - **Bayesian Logistic Regression**: Incorporates prior beliefs and uncertainty quantification.
  - **Frequentist Logistic Regression**: Provides baseline performance for comparison.
- **Exploratory Data Analysis (EDA)**:
  - Visualizations and statistical tests to identify key predictors.
- **Robust Preprocessing**:
  - Data cleaning, transformation, and feature engineering.
- **Model Reliability**:
  - Evaluation with accuracy, precision, recall, F1 score, and ROC-AUC metrics.
- **Future Directions**:
  - Suggestions for hierarchical priors, time-series analysis, and non-linear extensions.

---

## Dataset
The project uses the **Telco Customer Churn** dataset from Kaggle, containing 7,032 customer records. Key attributes:
- **Demographics**: Gender, age, dependent status.
- **Account Information**: Tenure, payment methods, charges.
- **Services**: Subscriptions like phone, internet, online security, and streaming.

Dataset preprocessing steps:
1. Converted `TotalCharges` from string to numeric.
2. Removed rows with missing values.
3. One-hot encoded categorical variables.
4. Normalized numerical features using `StandardScaler`.

---

## Methodology

### Bayesian Approach
#### Bayesian Logistic Regression
The Bayesian model predicts churn probability, \( P(Y_{\text{churn}} = 1 | X) \), where:
- **Outcome**: Binary (0 = No Churn, 1 = Churn).
- **Features (X)**: Demographics, service usage, and charges.

**Components**:
1. **Prior**: Distributions reflecting initial beliefs about parameters (e.g., Normal, Uniform).
2. **Likelihood**: The probability of observed data given the parameters.
3. **Posterior**: Combines prior and likelihood to update beliefs.

#### Sampling
- Implemented using PyMC with the No-U-Turn Sampler (NUTS).
- **Iterations**: 1,000 warm-up, 1,000 posterior draws.
- Priors include:
  - **Numerical Features**: Normal or Gamma.
  - **Binary Features**: Uniform.
  - **Intercept**: Normal.

### Frequentist Approach
- Standard logistic regression without prior beliefs.
- Optimized using Maximum Likelihood Estimation (MLE).

---

## Exploratory Data Analysis
Key findings:
- Strong correlations between tenure and total charges.
- Customers with shorter tenures and higher monthly charges are more likely to churn.
- Statistically significant relationships for features like contract type and online security.

---

## Results

### Bayesian Logistic Regression
- **Training Set**:
  - Accuracy: 80.6%, ROC-AUC: 0.843.
- **Test Set**:
  - Accuracy: 78.8%, ROC-AUC: 0.824.
- Offers credible intervals and uncertainty quantification.

### Frequentist Logistic Regression
- **Training Set**:
  - Accuracy: 80.6%, ROC-AUC: 0.851.
- **Test Set**:
  - Accuracy: 78.7%, ROC-AUC: 0.832.
- Slightly better recall and F1 scores.

---

## Posterior Analysis
- **Minimal Effects**: Gender, streaming services.
- **Moderate Effects**: Online security, total charges.
- **Strong Effects**: Tenure, fiber optic service.

**Odds Ratios**:
- Fiber optic service: \( OR = 2.72 \).
- Tenure: \( OR = 0.19 \), the strongest protective factor.

---

## Visualizations
- **Correlation Matrix**: Highlights relationships between numerical features.
- **Posterior Predictive Checks**: Validates model reliability.
- **ROC-AUC Curves**: Compares model performance across datasets.

---

## Conclusion
- **Bayesian Model**: Superior in uncertainty estimation and interpretability.
- **Frequentist Model**: Slightly better recall but lacks uncertainty quantification.

---

## Future Work
1. Use hierarchical or informative priors for Bayesian modeling.
2. Address data imbalance with Bayesian oversampling techniques.
3. Explore non-linear models like Bayesian Neural Networks or Random Forests.
4. Incorporate time-series analysis for temporal patterns.
5. Develop online learning models for real-time churn prediction.

---

## Getting Started
### Prerequisites
- Python 3.8+
- Libraries: `pymc`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/customer-churn-bayes
   cd customer-churn-bayes
