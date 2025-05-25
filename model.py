
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# File paths
DATA_DIR = 'data'
MODEL_PATH = os.path.join(DATA_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(DATA_DIR, 'vectorizer.pkl')
CLUSTERED_JOBS_PATH = os.path.join(DATA_DIR, 'jobs_clustered.csv')
ELBOW_PLOT_PATH = os.path.join(DATA_DIR, 'elbow_plot.png')

os.makedirs(DATA_DIR, exist_ok=True)

def preprocess_skills(skills_series):
    return skills_series.fillna('').str.lower()

def find_optimal_clusters(X, max_clusters=10):
    distortions = []
    silhouette_scores = []
    K = range(2, max_clusters + 1)

    for k in K:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        distortions.append(model.inertia_)
        silhouette_scores.append(silhouette_score(X, model.labels_))

    second_derivative = np.gradient(np.gradient(distortions))
    optimal_k = K[np.argmax(second_derivative)]
    return optimal_k

def cluster_jobs(jobs_csv_path, n_clusters=None):
    df = pd.read_csv(jobs_csv_path)
    skills = preprocess_skills(df['Skills'])
    vectorizer = TfidfVectorizer(token_pattern=r'[^,;]+')
    X = vectorizer.fit_transform(skills)

    if n_clusters is None:
        n_clusters = find_optimal_clusters(X)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = model.fit_predict(X)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)

    df.to_csv(CLUSTERED_JOBS_PATH, index=False)
    return df

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def assign_cluster_to_new_jobs(jobs_df):
    model, vectorizer = load_model()
    skills = preprocess_skills(jobs_df['Skills'])
    X = vectorizer.transform(skills)
    jobs_df['Cluster'] = model.predict(X)
    return jobs_df
