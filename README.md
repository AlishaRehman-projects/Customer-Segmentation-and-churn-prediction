# Customer-Segmentation-and-churn-prediction
ML pipeline for telecom customer churn prediction and segmentation using Random Forest classification and K-Means clustering on the Telco Customer Churn dataset.
This project analyzes telecom customer data to predict churn and segment customers based on behavior and risk profiles.
Key steps:

EDA — explored relationships between churn and tenure, monthly charges, contract type, internet service, payment method, and tech support
Data Cleaning — handled missing/invalid values, dropped irrelevant columns, one-hot encoded categorical features
Churn Prediction — built and tuned a Random Forest classifier (with class balancing for imbalanced churn data), evaluated using accuracy, precision, recall, confusion matrix, and ROC-AUC (achieved ~0.857 AUC)
Feature Selection — used feature importance analysis to drop low-impact features and improve model efficiency
Customer Segmentation — applied K-Means clustering (with elbow method for optimal k) on tenure, charges, and churn probability to segment customers into groups like Budget Loyal Customers, High Risk New Customers, and Loyal Premium Customers

Tech stack: Python, pandas, scikit-learn, imbalanced-learn (SMOTE), matplotlib, seaborn
Dataset: Telco Customer Churn dataset
