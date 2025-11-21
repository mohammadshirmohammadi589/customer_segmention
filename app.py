import gradio as gr
import pandas as pd
import joblib
import numpy as np

# Load model & scaler
model = joblib.load("model.pkl")            # KMeans / GMM / etc.
scaler = joblib.load("scaler.pkl")          # StandardScaler
pca_model = joblib.load("pca.pkl")          # PCA model for 2D projection

# Feature names expected:
FEATURES = [
    "Quantity", "UnitPrice", "CustomerID", "CountryEncoded",
    "Recency", "Frequency", "Monetary"
]

def segment_customer(quantity, unit_price, customer_id, country, recency, frequency, monetary):
    # Create dataframe from inputs
    df = pd.DataFrame([[
        quantity, unit_price, customer_id, country,
        recency, frequency, monetary
    ]], columns=FEATURES)

    # Scale
    X_scaled = scaler.transform(df)

    # Predict cluster
    cluster = model.predict(X_scaled)[0]

    # PCA projection
    pca_point = pca_model.transform(X_scaled)[0]

    return {
        "Cluster Assigned": f"Cluster {cluster}",
        "PCA x": float(pca_point[0]),
        "PCA y": float(pca_point[1])
    }

inputs = [
    gr.Number(label="Quantity"),
    gr.Number(label="Unit Price"),
    gr.Number(label="Customer ID"),
    gr.Number(label="Country (Encoded)"),
    gr.Number(label="Recency"),
    gr.Number(label="Frequency"),
    gr.Number(label="Monetary")
]

outputs = [
    gr.JSON(label="Segmentation Result")
]

iface = gr.Interface(
    fn=segment_customer,
    inputs=inputs,
    outputs=outputs,
    title="Customer Segmentation Model",
    description="Provide customer attributes to predict their cluster group."
)

if __name__ == "__main__":
    iface.launch()
