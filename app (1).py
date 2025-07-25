
import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load model & data
df_all = joblib.load('df_all.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Fungsi rekomendasi (sebanyak mungkin, tanpa batas default)
def recommend_film(title):
    title = title.lower()
    matches = df_all[df_all['title'].str.lower().str.contains(title, na=False)]

    if matches.empty:
        return f"Film dengan judul '{title}' tidak ditemukan."

    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [score for score in sim_scores if score[0] != idx]  # hilangkan film itu sendiri

    film_indices = [i[0] for i in sim_scores]
    result = df_all.iloc[film_indices].copy()
    return result

# 🎨 CSS Custom tampilan dark & kotak tetap
st.markdown("""
    <style>
    body {
        background-color: #0b1c2c;
    }
    .film-box {
        background-color: #1e2b3a;
        padding: 15px;
        border-radius: 15px;
        text-align: left;
        color: #ffffff;
        font-size: 15px;
        min-height: 300px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .film-title {
        font-weight: bold;
        font-size: 18px;
        margin-top: 10px;
        color: #00c6ff;
    }
    img {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# UI
st.title("🎬 Sistem Rekomendasi Film")
input_title = st.text_input("Masukkan judul film yang kamu suka:")

if st.button("Cari Rekomendasi"):
    hasil = recommend_film(input_title)

    if isinstance(hasil, str):
        st.warning(hasil)
    else:
        st.markdown("### Berikut hasil rekomendasi film untukmu:")

        for i in range(0, len(hasil), 3):
            cols = st.columns(3)
            for idx, col in enumerate(cols):
                if i + idx < len(hasil):
                    film = hasil.iloc[i + idx]
                    with col:
                        st.image(film['poster_url'], width=180)
                        st.markdown(f"""
                            <div class="film-box">
                                <div class="film-title">{film['title']}</div>
                                <div><b>Genre:</b> {film['genres']}</div>
                                <div><b>Director:</b> {film['director']}</div>
                                <div><b>Cast:</b> {film['cast']}</div>
                                <div><b>Overview:</b> {film['overview']}</div>
                            </div>
                        """, unsafe_allow_html=True)
