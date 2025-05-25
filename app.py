import os
import pandas as pd
import streamlit as st

from model import cluster_jobs, assign_cluster_to_new_jobs, CLUSTERED_JOBS_PATH
from scrape import scrape_karkidi_jobs, save_jobs_to_csv

# --- App Config ---
st.set_page_config(page_title="JobHunt: Smart Job Discovery", layout="wide")
st.title("🔎 JobHunt: Smart Job Discovery & Alerts")

# --- Load or Cluster Jobs ---
@st.cache_data(show_spinner=False)
def get_clustered_jobs():
    if not os.path.exists(CLUSTERED_JOBS_PATH):
        st.info("Clustering jobs for the first time...")
        return cluster_jobs('data/jobs.csv')
    return pd.read_csv(CLUSTERED_JOBS_PATH)

df = get_clustered_jobs()

# --- Skill Filter UI ---
st.header("Find Jobs by Your Skills")

all_skills = sorted({
    skill.strip().lower()
    for skills in df['Skills'].dropna()
    for skill in skills.split(',') if skill.strip()
})

user_skills = st.multiselect(
    "Select your skills of interest:",
    all_skills,
    help="You'll see jobs that require at least one of your selected skills."
)

# --- Filtered Job Results ---
if user_skills:
    filtered_df = df[df['Skills'].str.lower().apply(
        lambda s: any(skill in s for skill in user_skills) if isinstance(s, str) else False)]
    
    st.success(f"Found {len(filtered_df)} jobs matching your skills!")
    st.dataframe(filtered_df[['Title', 'Company', 'Location', 'Experience', 'Skills', 'Summary']])
else:
    st.info("Select skills to see matching jobs.")

# --- Job Refresh Button ---
if st.button("🔄 Refresh Job Listings Now"):
    with st.spinner("Fetching latest jobs and updating clusters..."):
        latest_jobs = scrape_karkidi_jobs(keyword="data science", pages=2)
        save_jobs_to_csv(latest_jobs)
        cluster_jobs('data/jobs.csv')
        df = get_clustered_jobs()
    st.success("Job listings updated!")
