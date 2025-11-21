# ✅ README — Customer Segmentation Using Machine Learning

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

Feature	Description
customerID	Unique identifier
gender	Male / Female
SeniorCitizen	Whether the customer is a senior citizen
Partner	Has partner or not
Dependents	Has dependents or not
tenure	Customer relationship duration
PhoneService	Phone service status
MultipleLines	Multiple phone lines
InternetService	DSL / Fiber / None
OnlineSecurity, OnlineBackup, TechSupport, etc.	Service add-ons
Contract	Month-to-month / One-year / Two-year
PaymentMethod	Payment type
MonthlyCharges, TotalCharges	Financial metrics
Churn	Whether customer left the company

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

##3 💰 Business Impact

Customer segmentation enables:

. Targeted retention campaigns

. Personalized offers

. Optimized promotions

. Increased customer lifetime value (CLV)

. Reduced marketing waste

### 🖥 Streamlit Application

The project includes a Streamlit web application allowing users to:

. Upload customer data

. Run segmentation in real time

. Visualize clusters on PCA plots

. See cluster statistics and summaries

. Download segmentation results

Run the app:
```python
streamlit run app.py
```
### 📁 Project Structure
````markdown
├── data/
│   ├── raw_data.csv
│   └── processed_data.csv
│
├── notebooks/
│   └── segmentation_EDA_and_Modeling.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── visualization.py
│   └── utils.py
│
├── app.py
├── requirements.txt
└── README.md
````

### ⚙️ How to Run the Project
#### 1️⃣ Install dependencies:

```python
pip install -r requirements.txt
```
#### 2️⃣ Run the modeling notebook:
Open Jupyter Notebook:

```python
jupyter notebook notebooks/segmentation_EDA_and_Modeling.ipynb
```

#### 3️⃣ Run the Streamlit app:

```python
streamlit run app.py
```
## 📌 Conclusion

This customer segmentation project provides businesses with a powerful analytical tool to deeply understand customer behavior and optimize decision-making processes. By combining machine learning, PCA visualization, and a Streamlit interface, the project delivers both analytical power and user-friendly accessibility.

## 🚀 Future Improvements

Add segmentation tracking over time

Deploy the model as an API

Add deep learning–based clustering (Autoencoders + KMeans)

Integrate churn prediction together with segmentation
