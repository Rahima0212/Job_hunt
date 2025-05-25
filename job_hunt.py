import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Paths
DATA_DIR = 'data'
MODEL_PATH = os.path.join(DATA_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(DATA_DIR, 'vectorizer.pkl')
CLUSTERED_JOBS_PATH = os.path.join(DATA_DIR, 'jobs_clustered.csv')
ELBOW_PLOT_PATH = os.path.join(DATA_DIR, 'elbow_plot.png')

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


def preprocess_skills(skills_series: pd.Series) -> pd.Series:
    """Convert skills text to lowercase and handle missing values."""
    return skills_series.fillna('').str.lower()


def find_optimal_clusters(X, max_clusters: int = 15) -> int:
    """
    Use the elbow and silhouette methods to determine the optimal number of clusters.
    Also saves a plot of distortion and silhouette scores.
    """
    distortions = []
    silhouette_scores = []
    K = list(range(2, max_clusters + 1))

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

        if k > 1:
            score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(score)

    # Plot elbow and silhouette analysis
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')

    plt.subplot(1, 2, 2)
    plt.plot(K[1:], silhouette_scores, 'rx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')

    plt.tight_layout()
    plt.savefig(ELBOW_PLOT_PATH)
    plt.close()

    # Use second derivative to estimate elbow
    second_derivative = np.gradient(np.gradient(distortions))
    optimal_k = K[np.argmax(second_derivative)]
    return optimal_k


def cluster_jobs(jobs_csv_path: str, n_clusters: int = None) -> pd.DataFrame:
    """
    Cluster job postings based on their skills using TF-IDF and KMeans.
    Saves the model, vectorizer, clustered jobs CSV, and returns the DataFrame.
    """
    df = pd.read_csv(jobs_csv_path)
    skills = preprocess_skills(df['Skills'])

    vectorizer = TfidfVectorizer(token_pattern=r'[^,;]+')
    X = vectorizer.fit_transform(skills)

    if n_clusters is None:
        n_clusters = find_optimal_clusters(X)
        print(f"Optimal number of clusters determined: {n_clusters}")

    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans_model.fit_predict(X)

    # Save model and vectorizer
    with open(MODEL_PATH, 'wb') as f_model:
        pickle.dump(kmeans_model, f_model)
    with open(VECTORIZER_PATH, 'wb') as f_vec:
        pickle.dump(vectorizer, f_vec)

    df.to_csv(CLUSTERED_JOBS_PATH, index=False)
    return df


def load_model():
    """Load the saved KMeans model and TF-IDF vectorizer."""
    with open(MODEL_PATH, 'rb') as f_model:
        kmeans_model = pickle.load(f_model)
    with open(VECTORIZER_PATH, 'rb') as f_vec:
        vectorizer = pickle.load(f_vec)
    return kmeans_model, vectorizer


def assign_cluster_to_new_jobs(jobs_df: pd.DataFrame) -> pd.DataFrame:
    """Assign cluster labels to new job entries based on trained model and vectorizer."""
    kmeans_model, vectorizer = load_model()
    skills = preprocess_skills(jobs_df['Skills'])
    X = vectorizer.transform(skills)
    jobs_df['Cluster'] = kmeans_model.predict(X)
    return jobs_df
