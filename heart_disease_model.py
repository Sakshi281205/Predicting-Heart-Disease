# Heart Disease Prediction using Bayesian Network

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load the dataset
df = pd.read_csv("heart_disease.csv")  # Use your local file path if needed

# Remove duplicates and missing values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Normalize numeric columns
numeric_cols = df.select_dtypes(include='number').columns
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Discretize normalized numeric columns into 5 bins, only if enough unique values
for col in numeric_cols:
    if df[col].nunique() >= 5:
        df[col] = pd.qcut(df[col], q=5, labels=False, duplicates='drop')

# Save cleaned dataset
df.to_csv("heart_disease_cleaned.csv", index=False)

# Define the Bayesian Network structure
model = DiscreteBayesianNetwork([
    ("age", "fbs"),
    ("fbs", "target"),
    ("target", "chol"),
    ("target", "thalach")
])

# Fit the model using Maximum Likelihood Estimation
model.fit(df, estimator=MaximumLikelihoodEstimator)

# Visualize the Bayesian Network
plt.figure(figsize=(6, 4))
G = nx.DiGraph()
G.add_edges_from(model.edges())
nx.draw(G, with_labels=True, node_size=2000, node_color='lightblue', font_size=10)
plt.title("Bayesian Network for Heart Disease")
plt.savefig("bn_visualization.png")
plt.show()

# Perform inference
infer = VariableElimination(model)

# Inference 1: Probability of heart disease for age bin 2
print("\nP(target | age = 2):")
print(infer.query(["target"], evidence={"age": 2}))

# Inference 2: Cholesterol distribution given heart disease
print("\nP(chol | target = 1):")
print(infer.query(["chol"], evidence={"target": 1}))
