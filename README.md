# ✅ Customer Segmentation Using Machine Learning


## 📌 Project Overview

Customer segmentation is a fundamental step in understanding customer behavior and designing personalized marketing strategies.
The goal of this project is to divide customers into meaningful groups based on behavioral and demographic features using machine learning techniques.

This segmentation helps businesses:

Identify high-value customer groups

Understand purchasing behavior

Optimize marketing campaigns

Reduce churn by targeting risky customer segments

Allocate resources more efficiently

This project uses clustering algorithms and dimensionality reduction techniques to uncover hidden patterns in the customer data.


## 📊 Dataset Overview

The dataset contains standard customer attributes commonly used in segmentation:

customerID ,gender , SeniorCitizen ,Partner ,dependents , tenure , PhoneService , MultipleLines , InternetService , OnlineSecurity , OnlineBackup , DeviceProtection , TechSupport , StreamingTV , StreamingMovies , Contract , PaperlessBilling , PaymentMethod , MonthlyCharges , TotalCharges , Churn



## 🧼 Preprocessing & Feature Engineering

The following steps were performed:

✔ Data Cleaning

Handling missing values

Fixing inconsistent data types

Converting categorical features to numeric using one-hot encoding

✔ Feature Scaling

StandardScaler was applied to numerical features to normalize the data for clustering.

✔ Dimensionality Reduction

PCA (Principal Component Analysis) was used to reduce high-dimensional data into 2–3 dimensions for clear visualization.


## 📈 Exploratory Data Analysis (EDA)

Several visualizations were used to understand the data distribution and discover patterns:

🔹 Distribution Plots

To analyze MonthlyCharges, Tenure, TotalCharges, etc.

🔹 Correlation Heatmap

To identify important relationships between customer features.

🔹 PCA Scatter Plot

To visualize clusters clearly in 2D and reduce noise/overlap.

🔹 Boxplots & Countplots

To study behavior differences across genders, contracts, payment methods.


### 🤖 Modeling

Multiple clustering models were evaluated to find the most meaningful segmentation:

Models Used:

KMeans Clustering

Hierarchical Agglomerative Clustering

DBSCAN

Gaussian Mixture Models (GMM)

MiniBatch KMeans

Model Selection Process

To choose the optimal model, several metrics and techniques were used:

✔ Elbow Method

To determine optimal number of clusters.

✔ Silhouette Score

To evaluate cluster separation and cohesion.

✔ Davies–Bouldin Index

To validate cluster quality.

KMeans combined with PCA performed the best and was selected as the final model.


### 🎯 Results & Cluster Insights

The algorithm identified meaningful customer segments, such as:

Cluster 1 — High-Value Loyal Customers

. Long tenure

. High monthly charges

. Low churn rate

Cluster 2 — Price-Sensitive Customers

. Low monthly charges

. Short tenure

. High churn risk

Cluster 3 — Multi-Service Customers

. Subscribed to many add-ons

. Medium churn risk

Cluster 4 — Minimal Service Users

. Only basic services

. Low engagement

These insights can be used for personalized marketing and churn reduction.


### 💰 Business Impact

Customer segmentation enables:

. Targeted retention campaigns

. Personalized offers

. Optimized promotions

. Increased customer lifetime value (CLV)

. Reduced marketing waste


### 🌐 Deployment on Hugging Face Spaces

This project is deployed using Hugging Face Spaces with a Gradio interface.

How to deploy on Hugging Face:

Go to Hugging Face Spaces:
https://huggingface.co/spaces

Create a New Space

Select the Gradio template

Upload the following files into your Space:

app.py

model.pkl (if needed)

requirements.txt 

```python
import gradio as gr

def predict_cluster(input_data):
    # load model & scaler
    return model.predict([input_data])[0]

iface = gr.Interface(
    fn=predict_cluster,
    inputs=[gr.Number(), gr.Number(), gr.Number(), ...],
    outputs="text",
    title="Customer Segmentation",
)

iface.launch()
```


### 🖥 How to Run Locally

#### 1️⃣ Install dependencies:
```python
1. Install dependencies:
```

#### 2️⃣ Run the HuggingFace (Gradio) App Locally:
```python
python app.py
```


### 📁 Project Structure
````markdown
├── data/
├── models/
├── notebooks/
│   ├── EDA.ipynb
│   ├── Modeling.ipynb
├── app.py
├── requirements.txt
├── README.md

````


## 📌 Conclusion

This customer segmentation project provides businesses with a powerful analytical tool to deeply understand customer behavior and optimize decision-making processes. By combining machine learning, PCA visualization,  the project delivers both analytical power and user-friendly accessibility.


## 🔮 Future Improvements

Testing the model on larger and more diverse datasets

Improving model accuracy with advanced feature engineering

Creating a more interactive and analytical dashboard in Hugging Face

Adding an API layer for integration with CRM systems
