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
df = pd.read_csv("https://bit.ly/3T1A7Rs")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Normalize numeric columns
numeric_cols = df.select_dtypes(include='number').columns
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save cleaned dataset
df.to_csv("heart_disease_cleaned.csv", index=False)

# Define the structure
model = BayesianNetwork([
    ("age", "fbs"),
    ("fbs", "target"),
    ("target", "chol"),
    ("target", "thalach")
])

# Train the model
model.fit(df, estimator=MaximumLikelihoodEstimator)

# Visualize Bayesian Network
plt.figure(figsize=(6, 4))
nx.draw(model, with_labels=True, node_size=2000, node_color='lightblue', font_size=10)
plt.title("Bayesian Network for Heart Disease")
plt.savefig("bn_visualization.png")
plt.show()

# Inference
infer = VariableElimination(model)

# Example Inference 1: P(target | age=0.6)
print("P(target | age=0.6)")
print(infer.query(["target"], evidence={"age": 0.6}))

# Example Inference 2: P(chol | target=1)
print("P(chol | target=1)")
print(infer.query(["chol"], evidence={"target": 1}))


