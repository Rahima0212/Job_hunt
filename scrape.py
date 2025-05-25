import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

DATA_DIR = 'data'
JOBS_CSV_PATH = os.path.join(DATA_DIR, 'jobs.csv')
os.makedirs(DATA_DIR, exist_ok=True)

def scrape_karkidi_jobs(keyword="data science", pages=2):
    all_jobs = []

    for page in range(1, pages + 1):
        url = f"https://www.karkidi.com/search/page:{page}?q={keyword.replace(' ', '+')}"
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            listings = soup.select('.jobContainer')

            for job in listings:
                title = job.select_one('.jobTitle').text.strip() if job.select_one('.jobTitle') else ''
                company = job.select_one('.companyName').text.strip() if job.select_one('.companyName') else ''
                location = job.select_one('.location').text.strip() if job.select_one('.location') else ''
                experience = job.select_one('.experience').text.strip() if job.select_one('.experience') else ''
                skills = job.select_one('.skills').text.strip() if job.select_one('.skills') else ''
                summary = job.select_one('.description').text.strip() if job.select_one('.description') else ''

                all_jobs.append({
                    'Title': title,
                    'Company': company,
                    'Location': location,
                    'Experience': experience,
                    'Skills': skills,
                    'Summary': summary
                })
        except Exception as e:
            print(f"Failed to scrape page {page}: {e}")

    return pd.DataFrame(all_jobs)

def save_jobs_to_csv(df, path=JOBS_CSV_PATH):
    df.to_csv(path, index=False)
