import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Configure Streamlit page
st.set_page_config(page_title="Clustering App", layout="wide")

st.title("Customer Personality Analysis - Clustering")
st.write("A simple Streamlit GUI for clustering analysis.")

# Helper to find the correct dataset file
@st.cache_data
def load_dataset():
    files_to_try = [
        "Dataset.csv", 
        "Customer_Personality_Analysis_Dataset.csv", 
        "Customer Personality Analysis Dataset.csv"
    ]
    for file in files_to_try:
        if os.path.exists(file):
            return pd.read_csv(file)
    st.error("Dataset not found! Please ensure the CSV file is in the same directory.")
    return None

df_raw = load_dataset()

if df_raw is not None:
    st.subheader("Dataset Preview")
    st.dataframe(df_raw.head())
    
    if st.button("Run Clustering"):
        with st.spinner("Preprocessing Data and Running Clustering..."):
            # 1. Preprocessing
            df = df_raw.copy()
            df = df.dropna()
            
            num_cols = [
                "Year_Birth", "MntWines", "MntFruits", "MntMeatProducts",
                "MntFishProducts", "MntSweetProducts", "MntGoldProds",
                "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
                "Kidhome", "Teenhome"
            ]
            for col in num_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if "Year_Birth" in df.columns:
                df["Age"] = 2026 - df["Year_Birth"]
            
            cols_to_sum = ["MntWines","MntFruits","MntMeatProducts", "MntFishProducts","MntSweetProducts","MntGoldProds"]
            available_cols = [c for c in cols_to_sum if c in df.columns]
            if available_cols:
                df["Spent"] = df[available_cols].sum(axis=1)

            if "Marital_Status" in df.columns:
                df["Living_With"] = df["Marital_Status"].replace({
                    "Married":"Partner","Together":"Partner",
                    "Absurd":"Alone","Widow":"Alone",
                    "YOLO":"Alone","Divorced":"Alone","Single":"Alone"
                })

            purchases_cols = ["NumWebPurchases","NumCatalogPurchases","NumStorePurchases"]
            available_purchases = [c for c in purchases_cols if c in df.columns]
            if available_purchases:
                df["Total_Purchases"] = df[available_purchases].sum(axis=1)
            
            if "Kidhome" in df.columns and "Teenhome" in df.columns:
                df["Children"] = df["Kidhome"] + df["Teenhome"]
                if "Living_With" in df.columns:
                    df["Family_Size"] = df["Living_With"].replace({"Alone":1,"Partner":2}) + df["Children"]
                df["Is_Parent"] = np.where(df["Children"] > 0, 1, 0)

            if "Education" in df.columns:
                df["Education"] = df["Education"].replace({
                    "Basic":"Undergraduate","2n Cycle":"Undergraduate",
                    "Graduation":"Graduate","Master":"Postgraduate","PhD":"Postgraduate"
                })

            rename_dict = {
                "MntWines":"Wines","MntFruits":"Fruits","MntMeatProducts":"Meat",
                "MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"
            }
            df = df.rename(columns=rename_dict)

            drop_cols = ["Marital_Status","Dt_Customer","Year_Birth","ID", "Unnamed: 0"]
            df = df.drop(columns=[c for c in drop_cols if c in df.columns])
            
            cat_cols = [c for c in ["Education","Living_With"] if c in df.columns]
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
            
            # Feature Selection
            var_selector = VarianceThreshold(threshold=0.01)
            var_selector.fit(df)
            selected_features = df.columns[var_selector.get_support()]
            df = df[selected_features]

            corr_matrix = df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_features = [col for col in upper.columns if any(upper[col] > 0.9)]
            df = df.drop(columns=high_corr_features)

            # Scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df)

            # PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # Use X_pca for clustering
            X = X_pca
            
            # 2. KMeans Clustering
            # The notebook's best params: we'll use a fixed k for simplicity, e.g., k=4 or allow user to select.
            # Let's just use k=4 as a good default based on typical Customer Personality Analysis, 
            # or just simple KMeans with 4 clusters. Let's use 4.
            k = 4
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=500, random_state=42)
            labels = kmeans.fit_predict(X)
            
            # Scores
            sil_score = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            
            # Add labels to raw dataframe (only for rows that were not dropped)
            # Find the indices of kept rows
            valid_indices = df_raw.dropna().index
            df_final = df_raw.iloc[valid_indices].copy()
            df_final["Cluster"] = labels
            
            # Display Metrics
            st.subheader("Clustering Results")
            col1, col2 = st.columns(2)
            col1.metric("Silhouette Score", round(sil_score, 4))
            col2.metric("Davies-Bouldin Score", round(db_score, 4))
            
            # PCA Plot
            st.subheader("PCA 2D Scatter Plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, alpha=0.8)
            ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                       c='red', marker='X', s=200, label='Centroids')
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            ax.set_title(f"KMeans Clustering (k={k})")
            ax.legend()
            st.pyplot(fig)
            
            # Cluster Counts
            st.subheader("Cluster Counts")
            counts = df_final["Cluster"].value_counts().sort_index()
            st.bar_chart(counts)
            
            # Download Button
            st.subheader("Download Labeled Data")
            csv = df_final.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Dataset with Cluster Labels",
                data=csv,
                file_name="clustered_dataset.csv",
                mime="text/csv",
            )
