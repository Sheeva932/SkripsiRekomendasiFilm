
import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load model & data
df_all = joblib.load('df_all.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Fungsi rekomendasi
def recommend_film(title, num_recommendations=6):
    title = title.lower()
    matches = df_all[df_all['title'].str.lower().str.contains(title, na=False)]

    if matches.empty:
        return f"Film dengan judul '{title}' tidak ditemukan."

    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]

    film_indices = [i[0] for i in sim_scores]
    result = df_all.iloc[film_indices].copy()
    return result

# 🎨 CSS Custom agar font hitam & kotak seragam
st.markdown("""
    <style>
    .film-box {
        background-color: #f4f4f4;
        padding: 10px 15px;
        border-radius: 15px;
        text-align: left;
        color: #111111;
        font-size: 16px;
        height: 380px;
        overflow: hidden;
    }
    .film-title {
        font-weight: bold;
        font-size: 18px;
        margin-top: 10px;
        color: #007acc;
    }
    img {
        border-radius: 10px;
        height: 270px;
        object-fit: cover;
    }
    body {
        background-color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# UI
st.title("🎬 Sistem Rekomendasi Film")
st.image("poster.jpg", use_column_width=True)
input_title = st.text_input("Masukkan judul film yang kamu suka:")

if st.button("Cari Rekomendasi"):
    hasil = recommend_film(input_title)

    if isinstance(hasil, str):
        st.warning(hasil)
    else:
        st.markdown("### Berikut hasil rekomendasi film untukmu:")

        # Tampilkan dalam grid 3 kolom per baris
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
                                <div><b>Overview:</b> {film['overview'][:150]}...</div>
                            </div>
                        """, unsafe_allow_html=True)
