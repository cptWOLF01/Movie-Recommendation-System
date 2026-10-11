"""
=============================================================
  Movie Recommendation System — model.py
  Content-Based Filtering using CountVectorizer + Cosine Similarity
=============================================================

HOW IT WORKS (in simple words):
  1. We load a CSV file with ~4800 movies.
  2. For each movie we combine its keywords, cast, genres, and director
     into one big string called "combined_features".
  3. We convert those strings into numbers using CountVectorizer
     (essentially counting how many times each word appears).
  4. We measure how "similar" two movies are by computing the
     cosine similarity of their number-vectors.
  5. When a user picks a movie, we find its similarity scores
     against every other movie and return the top 5 highest scores.
"""

import os                          # used to build file paths
import pandas as pd                # DataFrame operations (loading CSV, filtering rows)
import numpy as np                 # numerical operations on arrays

# CountVectorizer  — converts text into a word-frequency matrix
# TfidfVectorizer  — like CountVectorizer but down-weights very common words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# cosine_similarity — measures angle between two feature vectors (0 = different, 1 = same)
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Build the path to the CSV relative to wherever this script lives.
# os.path.dirname(__file__) → folder containing model.py
# os.path.join(...)         → joins folder + "data" + "movie_dataset.csv"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "movie_dataset.csv")

# These are the four columns we will use as features for similarity.
# They give the model information about what a movie is "about".
FEATURE_COLS = ["keywords", "cast", "genres", "director"]


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — load_data
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the movie dataset CSV into a pandas DataFrame.

    Parameters
    ----------
    path : str
        File path to the CSV file.  Defaults to data/movie_dataset.csv.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with all columns from the CSV.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at the given path.
    """
    # Check that the file actually exists before trying to open it
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Please place movie_dataset.csv inside the 'data/' folder."
        )

    # pd.read_csv() reads the file and returns a DataFrame (like an Excel table)
    df = pd.read_csv(path)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — preprocess
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame and build the 'combined_features' column.

    Steps performed:
      1. Keep only 'title' + the 4 feature columns.
      2. Fill any missing (NaN) values with an empty string "".
      3. Lowercase everything and strip extra whitespace.
      4. Concatenate the 4 columns into one 'combined_features' string.
      5. Reset the integer index so iloc lookups work cleanly.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame returned by load_data().

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with an extra 'combined_features' column.
    """
    # Work on a copy so we never mutate the original
    df = df.copy()

    # ── Step 1: Keep only the columns we need ────────────────────────────────
    # We check `if c in df.columns` to avoid KeyError if a column is missing
    cols_to_keep = ["title"] + FEATURE_COLS
    df = df[[c for c in cols_to_keep if c in df.columns]]

    # ── Step 2: Fill missing values ──────────────────────────────────────────
    # Some rows have NaN (blank) in keywords, cast, etc.
    # We replace NaN with "" so that string operations don't crash.
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # ── Step 3: Normalize text ───────────────────────────────────────────────
    # Lowercase → so "Action" and "action" are treated as the same word.
    # split() + join() → collapses multiple spaces into one.
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: " ".join(str(x).lower().split()))

    # ── Step 4: Combine all features into one text string ────────────────────
    # Example result for Avatar:
    # "culture clash future space war space colony ..."   (keywords)
    # + " sam worthington zoe saldana ..."                (cast)
    # + " action adventure fantasy science fiction"       (genres)
    # + " james cameron"                                  (director)
    df["combined_features"] = df[FEATURE_COLS].apply(
        lambda row: " ".join(row.values.astype(str)),
        axis=1     # axis=1 means apply the function row-by-row (not column-by-column)
    )

    # ── Step 5: Reset index ──────────────────────────────────────────────────
    # After filtering columns, the index might have gaps (0, 2, 5, …).
    # reset_index(drop=True) makes it 0, 1, 2, 3, … again.
    df = df.reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 3a — build_count_matrix  (default vectorizer)
# ─────────────────────────────────────────────────────────────────────────────
def build_count_matrix(df: pd.DataFrame):
    """
    Convert the 'combined_features' text column into a numerical matrix
    using CountVectorizer.

    CountVectorizer works by:
      - Building a vocabulary of all unique words across all movies.
      - For each movie, counting how many times each word appears.
      - Returning a sparse matrix of shape (n_movies, n_vocabulary_words).

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame (must have 'combined_features' column).

    Returns
    -------
    count_matrix : sparse matrix  (n_movies × n_words)
    cv           : fitted CountVectorizer object
    """
    # stop_words="english" — ignores very common English words like "the", "is", "and"
    #   because they appear in every movie and carry no meaning.
    # max_features=10000   — keeps only the 10,000 most frequent words to save memory.
    cv = CountVectorizer(stop_words="english", max_features=10000)

    # fit_transform() does two things:
    #   fit      — learns the vocabulary from all texts
    #   transform— converts each text into a word-count row
    count_matrix = cv.fit_transform(df["combined_features"])

    return count_matrix, cv


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 3b — build_tfidf_matrix  (optional, can be enabled in sidebar)
# ─────────────────────────────────────────────────────────────────────────────
def build_tfidf_matrix(df: pd.DataFrame):
    """
    Same as build_count_matrix but uses TF-IDF weights.

    TF-IDF (Term Frequency – Inverse Document Frequency):
      - TF  : how often a word appears in THIS movie's text.
      - IDF : penalises words that appear in MANY movies (they are less useful).
    Result: rare, distinctive words get higher weight than common ones.

    Returns
    -------
    tfidf_matrix : sparse matrix  (n_movies × n_words)
    tfidf        : fitted TfidfVectorizer object
    """
    tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
    tfidf_matrix = tfidf.fit_transform(df["combined_features"])
    return tfidf_matrix, tfidf


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 4 — compute_similarity
# ─────────────────────────────────────────────────────────────────────────────
def compute_similarity(matrix) -> np.ndarray:
    """
    Compute the Cosine Similarity between every pair of movies.

    Cosine Similarity measures the ANGLE between two vectors.
    - Score = 1.0  → vectors point in exactly the same direction (identical)
    - Score = 0.0  → vectors are perpendicular (totally different)

    The result is a square matrix of shape (n_movies × n_movies).
    similarity[i][j] = how similar movie i is to movie j.

    Parameters
    ----------
    matrix : sparse or dense matrix  (n_movies × n_features)

    Returns
    -------
    np.ndarray of shape (n_movies, n_movies)
    """
    # cosine_similarity handles sparse matrices efficiently
    return cosine_similarity(matrix)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 5 — get_recommendations   ← THE CORE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def get_recommendations(
    movie_name: str,
    df: pd.DataFrame,
    similarity_matrix: np.ndarray,
    top_n: int = 5,
) -> list:
    """
    Return the top-N most similar movies for a given movie title.

    Algorithm:
      1. Find the row index of movie_name inside df['title'].
      2. Extract that row from similarity_matrix — this gives one score
         per movie (how similar it is to movie_name).
      3. Sort all scores in descending order.
      4. Skip the first result (it's the movie itself, score = 1.0).
      5. Return the titles of the top-N remaining results.

    Parameters
    ----------
    movie_name       : str   — The movie the user selected (e.g. "Avatar")
    df               : DataFrame — Cleaned movie data with a 'title' column
    similarity_matrix: np.ndarray — (n×n) cosine similarity matrix
    top_n            : int   — How many recommendations to return (default 5)

    Returns
    -------
    list of str — top_n movie titles

    Raises
    ------
    ValueError — if the movie title is not found in the dataset
    """
    # ── Step 1: Find the movie in the dataset ────────────────────────────────
    # Convert all titles to lowercase for case-insensitive comparison
    titles_lower = df["title"].str.lower()
    query = movie_name.strip().lower()   # also lowercase the user input

    # Look for an exact match first
    matches = titles_lower[titles_lower == query]

    # If no exact match, try a partial match (movie_name is contained in title)
    if matches.empty:
        matches = titles_lower[titles_lower.str.contains(query, na=False)]

    # If still no match, raise an error so the UI can display a message
    if matches.empty:
        raise ValueError(f"Movie '{movie_name}' not found in the dataset.")

    # Take the index (row number) of the first matching movie
    movie_index = matches.index[0]

    # ── Step 2: Get similarity scores for this movie ─────────────────────────
    # similarity_matrix[movie_index] is a 1-D array of length n_movies.
    # Each value tells us how similar that movie is to our chosen movie.
    # We enumerate it so we know which row → which movie.
    sim_scores = list(enumerate(similarity_matrix[movie_index]))
    # Example: [(0, 1.0), (1, 0.23), (2, 0.41), …]
    #           index 0 is the movie itself (score 1.0 = perfect match)

    # ── Step 3: Sort by score in descending order ─────────────────────────────
    # key=lambda x: x[1]  →  sort by the score value (second element of each tuple)
    # reverse=True         →  highest score first
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # ── Step 4: Remove the movie itself ──────────────────────────────────────
    # The movie is always most similar to itself (score = 1.0).
    # We exclude it so it doesn't appear in the recommendations.
    sim_scores = [s for s in sim_scores if s[0] != movie_index]

    # ── Step 5: Take the top N results ───────────────────────────────────────
    top_indices = [s[0] for s in sim_scores[:top_n]]   # row numbers of top movies

    # Use iloc to look up the actual titles by their row numbers
    recommendations = df["title"].iloc[top_indices].tolist()
    return recommendations


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 6 — get_all_titles
# ─────────────────────────────────────────────────────────────────────────────
def get_all_titles(df: pd.DataFrame) -> list:
    """
    Return a sorted alphabetical list of all unique movie titles.
    Used to populate the dropdown menu in the Streamlit UI.

    Parameters
    ----------
    df : DataFrame with a 'title' column

    Returns
    -------
    list of str — sorted movie titles
    """
    # dropna()  — remove any rows where title is blank
    # unique()  — deduplicate identical titles
    # tolist()  — convert numpy array to a plain Python list
    # sorted()  — arrange A → Z
    return sorted(df["title"].dropna().unique().tolist())


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 7 — build_recommendation_pipeline   ← CALLED BY app.py
# ─────────────────────────────────────────────────────────────────────────────
def build_recommendation_pipeline(use_tfidf: bool = False):
    """
    Full end-to-end pipeline that wires everything together.
    Called once by app.py and cached so it doesn't re-run on every click.

    Steps:
      1. Load raw CSV → load_data()
      2. Clean & engineer features → preprocess()
      3. Vectorize text → CountVectorizer or TF-IDF
      4. Compute similarity → cosine_similarity()
      5. Extract all titles for the dropdown

    Parameters
    ----------
    use_tfidf : bool
        If True  → use TfidfVectorizer (better for long text).
        If False → use CountVectorizer (simpler, faster).

    Returns
    -------
    df               : pd.DataFrame   — cleaned movie data
    similarity       : np.ndarray     — (n×n) cosine similarity matrix
    titles           : list of str    — all movie titles for the dropdown
    """
    # Step 1 — Load the CSV file
    df_raw = load_data()

    # Step 2 — Clean, fill NaN, combine features
    df = preprocess(df_raw)

    # Step 3 — Choose vectorizer and build the feature matrix
    if use_tfidf:
        matrix, _ = build_tfidf_matrix(df)     # TF-IDF version
    else:
        matrix, _ = build_count_matrix(df)     # CountVectorizer (default)

    # Step 4 — Compute cosine similarity between every pair of movies
    similarity = compute_similarity(matrix)

    # Step 5 — Get sorted list of all titles for the UI dropdown
    titles = get_all_titles(df)

    # Return all three objects to app.py
    return df, similarity, titles


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — only runs when you execute: python model.py
# (Not executed when imported by app.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building recommendation pipeline …")

    # Run the full pipeline with CountVectorizer
    df, sim, titles = build_recommendation_pipeline(use_tfidf=False)
    print(f"Loaded {len(df)} movies  |  {len(titles)} unique titles\n")

    # Test a few movies
    for test_movie in ["Avatar", "The Dark Knight", "Inception"]:
        try:
            recs = get_recommendations(test_movie, df, sim, top_n=5)
            print(f"Top 5 for '{test_movie}':")
            for i, r in enumerate(recs, 1):
                print(f"  {i}. {r}")
            print()
        except ValueError as e:
            print(f"  Error: {e}\n")
