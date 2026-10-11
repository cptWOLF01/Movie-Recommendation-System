"""
=============================================================
  Movie Recommendation System — app.py
  Streamlit Web UI
  Run with:  python -m streamlit run app.py
=============================================================
"""

# ── Streamlit is the library that builds the web UI ──────────────────────────
import streamlit as st

# ── Import our ML functions from model.py ────────────────────────────────────
from model import build_recommendation_pipeline, get_recommendations


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# This must be the FIRST Streamlit command called in the script.
# It sets the browser tab title, icon, and layout width.
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommendation System",   # browser tab title
    page_icon="🎬",                             # tab icon
    layout="centered",                          # keep content in the middle
    initial_sidebar_state="collapsed",          # hide sidebar by default
)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Light theme: White · Yellow · Beige · Sky Blue
# We inject raw HTML/CSS so we can fully control the look of the app.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Load a modern Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    /* ── Apply the font everywhere ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main page background: sky-blue fading into beige ── */
    .stApp {
        background: linear-gradient(150deg, #d6eeff 0%, #eef7ff 30%, #fff8e7 70%, #fdf3dc 100%);
        color: #1a2e4a;   /* dark navy for all text */
    }

    /* ── Hide the default Streamlit menu, footer, and header ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Hero banner at the top of the page ── */
    .hero-banner {
        text-align: center;
        padding: 2.5rem 1rem 1rem;
    }
    /* ── Big gradient title text ── */
    .hero-banner h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1a8fd1, #FFD700, #56aee8); /* blue→yellow→skyblue */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        margin-bottom: 0.3rem;
    }
    /* ── Subtitle under the title ── */
    .hero-banner p {
        font-size: 1rem;
        color: #3a6080;
        font-weight: 400;
        margin-top: 0;
    }

    /* ── A thin coloured line used as a visual separator ── */
    .glow-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #56aee8, #FFD700, transparent);
        border: none;
        margin: 1.2rem 0 1.8rem;
    }

    /* ── The white card that wraps the dropdown and button ── */
    .search-card {
        background: rgba(255, 255, 255, 0.88);   /* semi-transparent white */
        border: 1px solid rgba(86, 174, 232, 0.5); /* sky-blue border */
        border-radius: 20px;
        padding: 2rem 2.5rem;
        backdrop-filter: blur(10px);              /* frosted-glass effect */
        box-shadow: 0 8px 32px rgba(86, 174, 232, 0.18); /* soft blue shadow */
    }

    /* ── Label text above the dropdown ── */
    .stSelectbox label {
        color: #1a5e8a !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
    }

    /* ── The dropdown box itself ── */
    .stSelectbox > div > div {
        background: #ffffff !important;
        border: 1.5px solid #56aee8 !important; /* sky-blue border */
        border-radius: 12px !important;
        color: #1a2e4a !important;
        font-size: 1rem !important;
    }

    /* ── Ensure typed text is visible (fixes white-on-white text) ── */
    .stSelectbox input {
        color: #1a2e4a !important;
    }

    /* ── Both buttons (Recommend + Clear) ── */
    .stButton > button {
        background: linear-gradient(90deg, #56aee8, #FFD700) !important; /* sky-blue → yellow */
        color: #1a2e4a !important;       /* dark text on light button */
        border: none !important;
        border-radius: 14px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        width: 100%;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(86, 174, 232, 0.3) !important;
    }
    /* ── Button lifts slightly when hovered ── */
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(255, 215, 0, 0.5) !important;
    }

    /* ── Small chips showing dataset stats ── */
    .stats-bar {
        display: flex;
        gap: 1.2rem;
        justify-content: center;
        margin: 0.8rem 0 1.6rem;
        flex-wrap: wrap;
    }
    .stat-chip {
        background: rgba(255, 255, 255, 0.8);
        border: 1.5px solid #56aee8;
        border-radius: 50px;
        padding: 0.35rem 1.1rem;
        font-size: 0.8rem;
        color: #1a5e8a;
        font-weight: 700;
    }

    /* ── Each recommended movie is shown as a card ── */
    .movie-card {
        display: flex;
        align-items: center;
        gap: 1.1rem;
        background: rgba(255, 248, 231, 0.92); /* warm beige */
        border: 1.5px solid rgba(86, 174, 232, 0.35);
        border-radius: 16px;
        padding: 1rem 1.4rem;
        margin: 0.65rem 0;
        transition: all 0.3s ease;
        animation: slideIn 0.4s ease forwards; /* fade-in from the left */
    }
    /* ── Card highlights in yellow when the mouse is over it ── */
    .movie-card:hover {
        background: rgba(255, 248, 220, 1);
        border-color: #FFD700;
        transform: translateX(6px);            /* slide right on hover */
        box-shadow: 0 4px 18px rgba(255, 215, 0, 0.3);
    }
    /* ── Rank number (#1, #2 …) with gradient colouring ── */
    .movie-rank {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1a8fd1, #f5c518);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        min-width: 2rem;
    }
    .movie-icon  { font-size: 1.7rem; }     /* trophy / star emoji */
    .movie-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1a2e4a;
    }

    /* ── Card slides in from the left when it first appears ── */
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to   { opacity: 1; transform: translateX(0); }
    }

    /* ── Red warning box when something goes wrong ── */
    .error-box {
        background: rgba(255, 100, 80, 0.1);
        border: 1px solid rgba(255, 100, 80, 0.45);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #c0392b;
        font-weight: 600;
        text-align: center;
        margin-top: 1rem;
    }

    /* ── Small uppercase label above the results list ── */
    .results-header {
        color: #1a5e8a;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 1.6rem 0 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,   # allow our raw HTML/CSS to be rendered
)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED PIPELINE LOADER
# @st.cache_resource tells Streamlit to run this function ONLY ONCE
# and reuse the result on every subsequent page interaction.
# Without caching, the model would be rebuilt on every button click (slow!).
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    Calls build_recommendation_pipeline() from model.py.
    Returns:
        df              — cleaned DataFrame of movies
        similarity_matrix — (n_movies × n_movies) cosine similarity array
        all_titles      — sorted list of every movie title
    """
    return build_recommendation_pipeline(use_tfidf=False)   # use CountVectorizer


# ─────────────────────────────────────────────────────────────────────────────
# HERO SECTION — page title and subtitle
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <h1>🎬 Movie Recommendation System</h1>
        <p>Content-Based Filtering · Cosine Similarity · DSBDA Mini Project</p>
    </div>
    <hr class="glow-divider"/>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD THE MODEL
# We show a spinner while the model is being built the very first time.
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("⏳ Loading movie data and building model…"):
    try:
        # Unpack the three objects returned by load_pipeline()
        df, similarity_matrix, all_titles = load_pipeline()
        pipeline_loaded = True   # flag so the rest of the UI knows it's ready
    except FileNotFoundError as e:
        # If the CSV is missing, show the error and stop the UI here
        st.error(str(e))
        pipeline_loaded = False


# ─────────────────────────────────────────────────────────────────────────────
# STATS BAR — shows a quick summary of what was loaded
# Only shown after the pipeline loads successfully
# ─────────────────────────────────────────────────────────────────────────────
if pipeline_loaded:
    st.markdown(
        f"""
        <div class="stats-bar">
            <div class="stat-chip">📽️ {len(df):,} Movies Loaded</div>
            <div class="stat-chip">🤖 CountVectorizer</div>
            <div class="stat-chip">📐 Cosine Similarity</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SEARCH CARD — dropdown + recommend button
# ─────────────────────────────────────────────────────────────────────────────
if pipeline_loaded:

    # Open the styled white card div
    st.markdown('<div class="search-card">', unsafe_allow_html=True)

    # ── DROPDOWN ──────────────────────────────────────────────────────────────
    # st.selectbox() renders a searchable dropdown menu.
    # The user can start typing to filter the list.
    # all_titles is a sorted list of 4800 movie names from the dataset.
    # ─────────────────────────────────────────────────────────────────────────
    selected_movie = st.selectbox(
        "🎯 Select a Movie",              # label shown above the dropdown
        options=[""] + all_titles,        # blank first option = no selection
        index=0,                          # start with blank selected
        help="Start typing to filter movies",   # tooltip on hover
    )

    # ── COLUMNS for the two buttons ───────────────────────────────────────────
    # st.columns([1, 1]) splits the row into two equal-width sections.
    # ─────────────────────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])
    with col1:
        # Primary action — trigger the recommendation
        recommend_clicked = st.button("🔍 Recommend", use_container_width=True)
    with col2:
        # Secondary action — clear results and reset
        clear_clicked = st.button("✖ Clear", use_container_width=True)

    # Close the styled card div
    st.markdown("</div>", unsafe_allow_html=True)


    # ─────────────────────────────────────────────────────────────────────────
    # CLEAR BUTTON LOGIC
    # st.rerun() forces the whole page to refresh, resetting all widgets.
    # ─────────────────────────────────────────────────────────────────────────
    if clear_clicked:
        st.rerun()


    # ─────────────────────────────────────────────────────────────────────────
    # RECOMMEND BUTTON LOGIC
    # ─────────────────────────────────────────────────────────────────────────
    if recommend_clicked:

        # Guard: user must pick a movie before clicking Recommend
        if not selected_movie:
            st.markdown(
                '<div class="error-box">⚠️ Please select a movie from the dropdown first!</div>',
                unsafe_allow_html=True,
            )

        else:
            # Show a spinner while cosine similarity is computed
            with st.spinner(f"Finding movies similar to **{selected_movie}**…"):
                try:
                    # ── CALL THE RECOMMENDATION FUNCTION ─────────────────────
                    # get_recommendations() is defined in model.py.
                    # It:
                    #   1. Finds the row index of selected_movie in df
                    #   2. Looks up that row's cosine similarity scores
                    #   3. Sorts them in descending order
                    #   4. Returns the top-5 movie titles
                    # ─────────────────────────────────────────────────────────
                    recommendations = get_recommendations(
                        selected_movie,      # the movie the user picked
                        df,                  # cleaned DataFrame
                        similarity_matrix,   # precomputed similarity matrix
                        top_n=5,             # return the 5 most similar movies
                    )

                    # ── RESULTS HEADER ────────────────────────────────────────
                    st.markdown(
                        f'<p class="results-header">'
                        f'🌟 Top 5 Movies Similar to "{selected_movie}"'
                        f'</p>',
                        unsafe_allow_html=True,
                    )

                    # ── DISPLAY EACH RECOMMENDATION AS A CARD ─────────────────
                    # We assign a rank emoji to positions 1-5
                    rank_emojis = ["🥇", "🥈", "🥉", "🎖️", "⭐"]

                    for rank, (title, emoji) in enumerate(
                        zip(recommendations, rank_emojis), start=1
                    ):
                        # Each movie gets its own animated card
                        st.markdown(
                            f"""
                            <div class="movie-card">
                                <span class="movie-rank">#{rank}</span>
                                <span class="movie-icon">{emoji}</span>
                                <span class="movie-title">{title}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # ── SUCCESS MESSAGE ───────────────────────────────────────
                    st.success(
                        f"✅ Showing top 5 recommendations for **{selected_movie}**"
                    )

                except ValueError as e:
                    # get_recommendations() raises ValueError if the movie
                    # title is not found in the dataset
                    st.markdown(
                        f'<div class="error-box">🎬 {str(e)}</div>',
                        unsafe_allow_html=True,
                    )


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <br>
    <hr class="glow-divider"/>
    <div style="text-align:center; color:#3a6080; font-size:0.82rem; padding-bottom:2rem;">
        DSBDA Mini Project &nbsp;|&nbsp; Content-Based Movie Recommendation System &nbsp;|&nbsp; 2024–25<br>
        Built with Python · scikit-learn · Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
