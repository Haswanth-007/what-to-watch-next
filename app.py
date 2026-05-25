"""
What To Watch Next – Streamlit Web App
Run with:  streamlit run app.py
"""

import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from recommender import build_pipeline, get_recommendations, evaluate_recommendations

# ── Page config ──────────────────────────────
st.set_page_config(
    page_title="What To Watch Next",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 What To Watch Next")
st.markdown(
    "Content-based movie recommendation engine powered by **TF-IDF** "
    "vectorisation and **cosine similarity** on the TMDB 5000 dataset."
)

# ── Load data (cached) ───────────────────────
DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
MOVIES_PATH  = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
CREDITS_PATH = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")


@st.cache_resource(show_spinner="Building similarity matrix …")
def load_engine():
    return build_pipeline(MOVIES_PATH, CREDITS_PATH)


if not os.path.exists(MOVIES_PATH):
    st.error(
        "Dataset not found. "
        "Download from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) "
        "and place both CSVs in the `/data` folder."
    )
    st.stop()

df, cosine_sim, indices = load_engine()

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_n     = st.slider("Number of recommendations", 5, 20, 10)
    min_votes = st.slider("Minimum vote count", 0, 500, 50, step=25)
    show_eval = st.checkbox("Show evaluation metrics", value=False)

# ── Search ────────────────────────────────────
all_titles = sorted(df["title"].dropna().tolist())
query = st.selectbox("Search for a movie you liked:", options=[""] + all_titles)

if query:
    try:
        recs = get_recommendations(
            query, df, cosine_sim, indices,
            top_n=top_n, min_votes=min_votes
        )

        st.subheader(f"Top {top_n} picks similar to *{query}*")
        st.dataframe(
            recs.rename(columns={
                "title"            : "Title",
                "similarity_score" : "Similarity Score",
                "vote_average"     : "Avg Rating",
                "vote_count"       : "Vote Count",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Quick bar chart
        st.bar_chart(recs.set_index("title")["similarity_score"])

    except ValueError as e:
        st.error(str(e))

# ── Evaluation panel ─────────────────────────
if show_eval:
    with st.expander("📊 Model Evaluation", expanded=True):
        with st.spinner("Evaluating on 200-movie sample …"):
            metrics = evaluate_recommendations(df, cosine_sim, indices)

        col1, col2, col3 = st.columns(3)
        col1.metric("% above 0.75 threshold", f"{metrics['pct_above_threshold']}%")
        col2.metric("Mean similarity",         metrics["mean_similarity"])
        col3.metric("Median similarity",       metrics["median_similarity"])
        st.caption(
            f"Evaluated {metrics['total_recs_evaluated']} recommendations "
            f"across {metrics['sample_size']} randomly sampled movies."
        )
