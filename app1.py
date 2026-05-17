import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

st.set_page_config(
    page_title="Clustering Fine-Tuning",
    layout="wide"
)

st.title("Customer Segmentation - Clustering Fine-Tuning")
st.write("Exhaustive Hyperparameter Fine-Tuning for KMeans, DBSCAN, and Agglomerative Clustering")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.drop_duplicates()
    
    st.subheader("Original Dataset Preview")
    st.dataframe(df.head())
    
    # Preprocessing identical to clustering_finetuning.ipynb
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
            
    df = df.dropna()
    
    if "Year_Birth" in df.columns:
        df["Age"] = 2026 - df["Year_Birth"]
        
    spend_cols = ["MntWines","MntFruits","MntMeatProducts", "MntFishProducts","MntSweetProducts","MntGoldProds"]
    existing_spend = [c for c in spend_cols if c in df.columns]
    if existing_spend:
        df["Spent"] = df[existing_spend].sum(axis=1)
        
    if "Marital_Status" in df.columns:
        df["Living_With"] = df["Marital_Status"].replace({
            "Married":"Partner","Together":"Partner",
            "Absurd":"Alone","Widow":"Alone",
            "YOLO":"Alone","Divorced":"Alone","Single":"Alone"
        })
        
    purchases_cols = ["NumWebPurchases","NumCatalogPurchases","NumStorePurchases"]
    ex_purch = [c for c in purchases_cols if c in df.columns]
    if ex_purch:
        df["Total_Purchases"] = df[ex_purch].sum(axis=1)
        
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
        
    df = df.rename(columns={
        "MntWines":"Wines","MntFruits":"Fruits","MntMeatProducts":"Meat",
        "MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"
    })
    
    drop_cols = ["Marital_Status","Dt_Customer","Year_Birth","ID","Unnamed: 0"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    cat_cols = ["Education", "Living_With"]
    ex_cat = [c for c in cat_cols if c in df.columns]
    if ex_cat:
        df = pd.get_dummies(df, columns=ex_cat, drop_first=True)
        
    # Ensure all columns are numeric for VarianceThreshold
    df = df.select_dtypes(include=[np.number])
    df = df.dropna()
    
    var_selector = VarianceThreshold(threshold=0.01)
    var_selector.fit(df)
    selected_features = df.columns[var_selector.get_support()]
    df = df[selected_features]
    
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [col for col in upper.columns if any(upper[col] > 0.9)]
    df = df.drop(columns=high_corr_features)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    pca2 = PCA(n_components=3)
    X_pca2 = pca2.fit_transform(X_scaled)
    X = X_pca2[:, :2] # Using first 2 components for model and visualization
    
    st.sidebar.title("Model Fine-Tuning Settings")
    model_name = st.sidebar.selectbox("Choose Model", ["KMeans", "DBSCAN", "Agglomerative"])
    
    if model_name == "KMeans":
        k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 4)
        init = st.sidebar.selectbox("Initialization Method", ["k-means++", "random"])
        n_init = st.sidebar.selectbox("Number of Init (n_init)", [10, 20, 30])
        
        model = KMeans(n_clusters=k, init=init, n_init=n_init, max_iter=500, random_state=42)
        labels = model.fit_predict(X)
        
        try:
            sil = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
        except ValueError:
            sil, db_score = None, None
            
        st.subheader("KMeans Results")
        if sil is not None:
            st.write(f"**Silhouette Score:** {sil:.4f} (higher is better)")
            st.write(f"**Davies-Bouldin Score:** {db_score:.4f} (lower is better)")
        st.write(f"**Inertia:** {model.inertia_:.2f}")
        
    elif model_name == "DBSCAN":
        eps = st.sidebar.slider("EPS", 0.15, 1.0, 0.25, step=0.05)
        min_samples = st.sidebar.slider("Min Samples", 3, 20, 5)
        metric = st.sidebar.selectbox("Metric", ["euclidean", "manhattan"])
        
        model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = model.fit_predict(X)
        
        n_cl = len(set(labels)) - (1 if -1 in labels else 0)
        mask = labels != -1
        
        st.subheader("DBSCAN Results")
        st.write(f"**Number of Clusters:** {n_cl}")
        noise_pct = 100 * (~mask).sum() / len(labels)
        st.write(f"**Noise:** {noise_pct:.1f}%")
        
        if n_cl >= 2 and mask.sum() > 10:
            sil = silhouette_score(X[mask], labels[mask])
            db_score = davies_bouldin_score(X[mask], labels[mask])
            st.write(f"**Silhouette Score (excl. noise):** {sil:.4f}")
            st.write(f"**Davies-Bouldin Score (excl. noise):** {db_score:.4f}")
        else:
            st.write("Not enough clusters/points to calculate Silhouette & Davies-Bouldin scores.")
            
    else: # Agglomerative
        k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 4)
        linkage = st.sidebar.selectbox("Linkage", ["ward", "complete", "average", "single"])
        
        if linkage == "ward":
            metric = st.sidebar.selectbox("Metric", ["euclidean"])
        else:
            metric = st.sidebar.selectbox("Metric", ["euclidean", "manhattan", "cosine"])
            
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metric)
        labels = model.fit_predict(X)
        
        try:
            sil = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
        except ValueError:
            sil, db_score = None, None
            
        st.subheader("Agglomerative Results")
        if sil is not None:
            st.write(f"**Silhouette Score:** {sil:.4f} (higher is better)")
            st.write(f"**Davies-Bouldin Score:** {db_score:.4f} (lower is better)")
            
    # Visualization
    st.subheader("Cluster Visualization (PCA)")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if model_name == "KMeans":
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=25)
        centers = model.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
        plt.colorbar(scatter, ax=ax)
        ax.legend()
    elif model_name == "DBSCAN":
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma', s=25)
        plt.colorbar(scatter, ax=ax)
    else:
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=25)
        plt.colorbar(scatter, ax=ax)
        
    ax.set_title(f"{model_name} Clustering")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
        
    st.pyplot(fig)
    
    # Results DataFrame
    result_df = pd.DataFrame()
    result_df["Cluster_Label"] = labels
    st.subheader("Cluster Labels")
    st.dataframe(result_df.head(20))
