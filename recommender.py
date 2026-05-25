"""
What To Watch Next – Content-Based Movie Recommendation Engine
=============================================================
Uses TF-IDF vectorisation + cosine similarity on TMDB metadata
to return the top-N most relevant movies for any given title.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os


# ──────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────

def load_data(movies_path: str, credits_path: str) -> pd.DataFrame:
    """
    Load and merge TMDB movies + credits CSVs.

    Parameters
    ----------
    movies_path  : path to tmdb_5000_movies.csv
    credits_path : path to tmdb_5000_credits.csv

    Returns
    -------
    Merged DataFrame with relevant columns.
    """
    movies  = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # tmdb_5000_credits uses 'movie_id'; align to 'id'
    if "movie_id" in credits.columns:
        credits.rename(columns={"movie_id": "id"}, inplace=True)

    df = movies.merge(credits, on="id")

    # Keep only what we need
    cols = ["id", "title_x", "overview", "genres",
            "keywords", "cast", "crew", "vote_average", "vote_count"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df.rename(columns={"title_x": "title"}, inplace=True)

    return df


# ──────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────────

def _parse_list_field(value: str, key: str = "name", limit: int = None) -> list:
    """Safely parse a JSON-like string field into a list of strings."""
    try:
        items = ast.literal_eval(value)
        names = [item[key] for item in items if key in item]
        return names[:limit] if limit else names
    except (ValueError, TypeError):
        return []


def _get_director(crew_str: str) -> str:
    """Extract the director's name from the crew field."""
    try:
        crew = ast.literal_eval(crew_str)
        for member in crew:
            if member.get("job") == "Director":
                return member.get("name", "").replace(" ", "")
        return ""
    except (ValueError, TypeError):
        return ""


def build_feature_soup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine genres, keywords, top cast, director, and overview
    into a single 'soup' string per movie used for TF-IDF.
    """
    df = df.copy()

    df["genres"]   = df["genres"].apply(lambda x: _parse_list_field(x))
    df["keywords"] = df["keywords"].apply(lambda x: _parse_list_field(x))
    df["cast"]     = df["cast"].apply(lambda x: _parse_list_field(x, limit=3))
    df["director"] = df["crew"].apply(_get_director)
    df["overview"] = df["overview"].fillna("")

    def make_soup(row):
        # Weight genres & director by repeating them
        genres   = " ".join(row["genres"]) + " " + " ".join(row["genres"])
        keywords = " ".join(row["keywords"])
        cast     = " ".join([c.replace(" ", "") for c in row["cast"]])
        director = row["director"] + " " + row["director"]
        overview = row["overview"]
        return f"{genres} {keywords} {cast} {director} {overview}"

    df["soup"] = df.apply(make_soup, axis=1)
    return df


# ──────────────────────────────────────────────
# 3. SIMILARITY MATRIX
# ──────────────────────────────────────────────

def build_similarity_matrix(df: pd.DataFrame):
    """
    Vectorise the 'soup' column with TF-IDF and compute
    pairwise cosine similarity across all titles.

    Returns
    -------
    cosine_sim : ndarray of shape (n_movies, n_movies)
    indices    : Series mapping title → DataFrame index
    """
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["soup"])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Map lowercase title → positional index for fast lookup
    indices = pd.Series(df.index, index=df["title"].str.lower()).drop_duplicates()

    return cosine_sim, indices


# ──────────────────────────────────────────────
# 4. RECOMMENDATION
# ──────────────────────────────────────────────

def get_recommendations(
    title: str,
    df: pd.DataFrame,
    cosine_sim: np.ndarray,
    indices: pd.Series,
    top_n: int = 10,
    min_votes: int = 50,
    similarity_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Return the top-N most similar movies to *title*.

    Parameters
    ----------
    title                : Movie title (case-insensitive).
    df                   : Processed DataFrame.
    cosine_sim           : Precomputed cosine similarity matrix.
    indices              : Title-to-index mapping.
    top_n                : Number of recommendations to return.
    min_votes            : Filter out movies with fewer votes.
    similarity_threshold : Only include results above this score.

    Returns
    -------
    DataFrame with columns: title, similarity_score, vote_average, vote_count
    """
    title_lower = title.strip().lower()

    if title_lower not in indices:
        raise ValueError(
            f"'{title}' not found in the dataset. "
            "Check spelling or try a different title."
        )

    idx = indices[title_lower]
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort by similarity descending, skip the movie itself (score == 1.0 at idx)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx]

    # Apply similarity threshold
    sim_scores = [s for s in sim_scores if s[1] >= similarity_threshold]

    # Grab indices and scores
    movie_indices = [s[0] for s in sim_scores]
    scores        = [round(s[1], 4) for s in sim_scores]

    result = df.iloc[movie_indices][["title", "vote_average", "vote_count"]].copy()
    result["similarity_score"] = scores

    # Filter by minimum vote count
    result = result[result["vote_count"] >= min_votes]

    return result.head(top_n).reset_index(drop=True)


# ──────────────────────────────────────────────
# 5. EVALUATION
# ──────────────────────────────────────────────

def evaluate_recommendations(
    df: pd.DataFrame,
    cosine_sim: np.ndarray,
    indices: pd.Series,
    sample_size: int = 200,
    top_n: int = 10,
    threshold: float = 0.75,
    seed: int = 42,
) -> dict:
    """
    Evaluate the fraction of top-N recommendations that exceed
    a cosine similarity threshold across a random movie sample.

    Returns
    -------
    dict with keys: sample_size, threshold, pct_above_threshold,
                    mean_similarity, median_similarity
    """
    rng = np.random.default_rng(seed)
    sample_titles = rng.choice(df["title"].values, size=min(sample_size, len(df)), replace=False)

    all_scores = []
    skipped    = 0

    for title in sample_titles:
        try:
            recs = get_recommendations(title, df, cosine_sim, indices, top_n=top_n)
            all_scores.extend(recs["similarity_score"].tolist())
        except ValueError:
            skipped += 1

    if not all_scores:
        return {"error": "No scores collected – check your dataset."}

    scores_arr = np.array(all_scores)
    pct_above  = round(float(np.mean(scores_arr >= threshold)) * 100, 2)

    return {
        "sample_size"          : len(sample_titles) - skipped,
        "threshold"            : threshold,
        "pct_above_threshold"  : pct_above,
        "mean_similarity"      : round(float(scores_arr.mean()), 4),
        "median_similarity"    : round(float(np.median(scores_arr)), 4),
        "total_recs_evaluated" : len(scores_arr),
    }


# ──────────────────────────────────────────────
# 6. PIPELINE ENTRY POINT
# ──────────────────────────────────────────────

def build_pipeline(movies_path: str, credits_path: str):
    """
    Full pipeline: load → engineer features → build similarity matrix.

    Returns
    -------
    (df, cosine_sim, indices) ready for get_recommendations()
    """
    print("📂  Loading data …")
    df = load_data(movies_path, credits_path)
    print(f"    {len(df):,} movies loaded.")

    print("🔧  Engineering features …")
    df = build_feature_soup(df)

    print("📐  Building TF-IDF + cosine similarity matrix …")
    cosine_sim, indices = build_similarity_matrix(df)
    print(f"    Matrix shape: {cosine_sim.shape}")

    return df, cosine_sim, indices


# ──────────────────────────────────────────────
# 7. CLI DEMO
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
    MOVIES_PATH  = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
    CREDITS_PATH = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")

    if not os.path.exists(MOVIES_PATH):
        print("⚠️  Dataset not found.")
        print("   Download from https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
        print("   and place both CSVs inside the /data folder.")
        sys.exit(1)

    df, cosine_sim, indices = build_pipeline(MOVIES_PATH, CREDITS_PATH)

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "The Dark Knight"
    print(f"\n🎬  Top 10 recommendations for: '{query}'\n{'─'*50}")

    try:
        recs = get_recommendations(query, df, cosine_sim, indices, top_n=10)
        print(recs.to_string(index=False))
    except ValueError as e:
        print(f"Error: {e}")

    print(f"\n📊  Running evaluation on 200-movie sample …")
    metrics = evaluate_recommendations(df, cosine_sim, indices)
    for k, v in metrics.items():
        print(f"   {k}: {v}")
