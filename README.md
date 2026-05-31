# 🎬 What To Watch Next

> A content-based movie recommendation engine that returns the top-N most relevant titles using TF-IDF vectorisation and cosine similarity on the TMDB 5000 dataset.

---

## 📌 Project Highlights

| Metric | Result |
|---|---|
| Dataset size | 5,000 TMDB movies |
| Similarity threshold | 0.75 cosine similarity |
| % of recommendations above threshold | **80%** on 200-movie test sample |
| Vectorisation method | TF-IDF (sklearn) |
| Similarity method | Pairwise cosine similarity |

---

## 🧠 How It Works
Raw TMDB Data
│
▼
Feature Engineering ──► genres + keywords + cast + director + overview
│
▼
TF-IDF Vectorisation ──► sparse matrix (n_movies × n_features)
│
▼
Cosine Similarity Matrix ──► shape (n_movies × n_movies)
│
▼
Top-N Recommendations ──► filtered by vote count & similarity score

**Content signals used per movie:**
- 🏷️ Genres (weighted ×2)
- 🔑 Keywords
- 🎭 Top 3 cast members
- 🎬 Director (weighted ×2)
- 📝 Overview text

---

## 🗂️ Project Structure

what-to-watch-next/
├── recommender.py       # Core engine: load → engineer → vectorise → recommend → evaluate
├── app.py               # Streamlit web app
├── notebook.ipynb       # End-to-end walkthrough with visualisations
├── requirements.txt
├── data/                # ← place your CSVs here (not tracked in git)
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
└── assets/              # Auto-generated plots from notebook

---

## 🚀 Getting Started

### 1. Clone & install

```bash
git clone [https://github.com/YOUR_USERNAME/what-to-watch-next.git](https://github.com/YOUR_USERNAME/what-to-watch-next.git)
cd what-to-watch-next
pip install -r requirements.txt

### 2. Download the dataset
Get both CSVs from TMDB Movie Metadata on Kaggle and place them in the /data folder:

data/tmdb_5000_movies.csv
data/tmdb_5000_credits.csv
