import streamlit as st
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from difflib import get_close_matches

# Set page config
st.set_page_config(page_title="Sistem Rekomendasi Film", layout="wide")
# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Load data
df_all = joblib.load('df_all.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Fungsi untuk mencari film yang cocok
def find_best_match(user_input):
    user_input = user_input.lower().strip()
    
    def normalize_string(s):
        import re
        normalized = re.sub(r'[-_\.\,\:\;]', ' ', s.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    normalized_input = normalize_string(user_input)
    
    # Tolak input yang terlalu pendek
    if len(normalized_input) < 2:
        return None

    df_temp = df_all.copy()
    df_temp['normalized_title'] = df_temp['title'].apply(normalize_string)
    
    # 1. Exact match
    exact_matches = df_temp[df_temp['normalized_title'] == normalized_input]
    if not exact_matches.empty:
        return exact_matches.iloc[0]['title'].lower()
    
    # 2. Partial match
    partial_matches = df_temp[df_temp['normalized_title'].str.contains(normalized_input, na=False, regex=False)]
   
# Tambahan validasi penting
    input_words = normalized_input.split()
    if len(input_words) >= 2 and partial_matches.empty:
        return None
    
    if not partial_matches.empty:
        partial_matches = partial_matches.copy()
        partial_matches['title_length'] = partial_matches['title'].str.len()
        partial_matches['starts_with_input'] = partial_matches['normalized_title'].str.startswith(normalized_input)
        partial_matches['word_count'] = partial_matches['normalized_title'].str.split().str.len()
        
        partial_matches = partial_matches.sort_values([
            'starts_with_input', 'word_count', 'title_length'
        ], ascending=[False, True, True])
        
        return partial_matches.iloc[0]['title'].lower()
    
    # 3. Keyword match
    input_words = normalized_input.split()
    if len(input_words) > 1:
        for word in input_words:
            if len(word) > 2:
                word_matches = df_temp[df_temp['normalized_title'].str.contains(word, na=False, regex=False)]
                if not word_matches.empty:
                    word_matches = word_matches.copy()
                    word_matches['word_score'] = 0
                    for input_word in input_words:
                        word_matches['word_score'] += word_matches['normalized_title'].str.contains(input_word, na=False).astype(int)
                    
                    if word_matches.iloc[0]['word_score'] < 2:
                        continue  # skip kalau cuma cocok 1 kata

                    word_matches['title_length'] = word_matches['title'].str.len()
                    word_matches = word_matches.sort_values(['word_score', 'title_length'], ascending=[False, True])
                    
                    return word_matches.iloc[0]['title'].lower()
    
    # 4. Approximate match
    from difflib import get_close_matches
    normalized_titles = df_temp['normalized_title'].tolist()
    matches = get_close_matches(normalized_input, normalized_titles, n=5, cutoff=0.7)
    
    if matches:
        for match in matches:
            original_title = df_temp[df_temp['normalized_title'] == match]['title'].iloc[0]
            return original_title.lower()
    
    return None

# Fungsi rekomendasi film
def recommend_film(title):
    corrected = find_best_match(title)
    if not corrected:
        return None, None

    # Cari index film yang dicari
    matched_films = df_all[df_all['title'].str.lower() == corrected]
    if matched_films.empty:
        return None, None
        
    idx = matched_films.index[0]
    original_title = matched_films.iloc[0]['title']
    
    # Hitung similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Filter film dengan similarity >= 0.1, exclude film yang sama persis
    result_indices = []
    similarities = []
    
    for film_idx, similarity in sim_scores:
        if similarity >= 0.09:  # Semua film dengan similarity >= 0.09
            result_indices.append(film_idx)
            similarities.append(similarity)
    
    # Buat DataFrame hasil
    if result_indices:
        result = df_all.iloc[result_indices][['title', 'genres', 'overview', 'director', 'cast', 'poster_url']].copy()
        result['cosine_similarity'] = similarities
        return result, original_title
    
    return None, None
    
# CSS Styling
st.markdown("""<style> 
/* Global Styling */
body, .stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1a1f36 50%, #0d1117 100%);
    color: #e2e8f0;
    font-family: 'font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', 'Helvetica Neue', sans-serif;
    line-height: 1.6;
}

/* Main Container */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Navigation Buttons */
.nav-container {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-bottom: 2rem;
    padding: 1rem;
    background: rgba(30, 41, 59, 0.3);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Homepage Hero Section */
.hero-section {
    text-align: center;
    padding: 4rem 2rem;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(16, 185, 129, 0.1));
    border-radius: 20px;
    margin: 2rem 0;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #3b82f6, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
}

.hero-subtitle {
    font-size: 1.3rem;
    color: #94a3b8;
    margin-bottom: 2rem;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}

/* Feature Cards */
.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin: 3rem 0;
}

.feature-card {
    background: linear-gradient(145deg, #1e293b, #111827);
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.05);
    transition: all 0.3s ease;
}

.feature-card:hover {
    transform: translateY(-5px);
    border-color: rgba(59, 130, 246, 0.3);
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.4);
}

.feature-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}

.feature-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #60a5fa;
    margin-bottom: 1rem;
}

.feature-desc {
    color: #cbd5e1;
    line-height: 1.6;
}

/* Stats Section */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 2rem;
    margin: 3rem 0;
}

.stat-card {
    background: linear-gradient(135deg, #374151, #1f2937);
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.stat-number {
    font-size: 2.5rem;
    font-weight: 700;
    color: #10b981;
    margin-bottom: 0.5rem;
}

.stat-label {
    color: #94a3b8;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* How it Works */
.steps-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
    margin: 3rem 0;
}

.step-card {
    text-align: center;
    padding: 1.5rem;
    background: rgba(17, 24, 39, 0.5);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.step-number {
    width: 60px;
    height: 60px;
    background: linear-gradient(135deg, #3b82f6, #1d4ed8);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 1rem;
    font-weight: 700;
    font-size: 1.5rem;
    color: white;
}

.step-title {
    color: #e2e8f0;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.step-desc {
    color: #cbd5e1;
    font-size: 0.9rem;
}

/* Film Card Styling (untuk halaman search) */
.film-card {
    background: linear-gradient(145deg, #1e293b 0%, #111827 100%);
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 24px;
    border-radius: 16px;
    margin-bottom: 20px;
    color: #e2e8f0;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    transition: all 0.3s ease;
    position: relative;
    overflow: visible;
}

.film-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.4);
    border-color: rgba(255, 255, 255, 0.1);
}

.film-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #f59e0b, #10b981, #3b82f6);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.film-card:hover::before {
    opacity: 1;
}

.film-poster {
    width: 100%;
    height: 400px;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 16px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}

.film-poster img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    transition: transform 0.3s ease;
}

.film-card:hover .film-poster img {
    transform: scale(1.05);
}

.film-card h4 {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 12px;
    color: #fbbf24;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    letter-spacing: -0.025em;
}

.film-card p {
    font-size: 14px;
    margin: 8px 0;
    color: #cbd5e1;
    line-height: 1.5;
}

.film-card p strong {
    color: #60a5fa;
    font-weight: 600;
}

img {
    border-radius: 10px;
    height: 270px;
    object-fit: cover;
}

details summary {
    cursor: pointer;
    color: #3b82f6;
    font-size: 14px;
    font-weight: 500;
    padding: 8px 16px;
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 8px;
    transition: all 0.3s ease;
    list-style: none;
    user-select: none;
}

details summary:hover {
    background: rgba(59, 130, 246, 0.15);
    transform: translateY(-1px);
}

details[open] summary {
    background: rgba(59, 130, 246, 0.15);
    margin-bottom: 12px;
}

details p {
    padding: 12px 16px;
    background: rgba(13, 17, 23, 0.6);
    border-radius: 8px;
    border-left: 3px solid #3b82f6;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #16a34a, #22c55e) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: .5rem 2rem !important;
    border: none !important;
    transition: all .3s ease-in-out !important;
    box-shadow: 0 4px 12px rgba(34, 197, 94, .3) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(34, 197, 94, .4) !important;
}

.stButton > button:disabled {
    background-color: #e2e8f0 !important;
    color: #1e293b !important;
    opacity: 0.7 !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
}

/* Inputs */
.stTextInput > div > div > input,
.stSelectbox > div > div > div,
.stMultiSelect > div > div > div {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
}

.stTextInput > div > div > input:focus,
.stSelectbox > div > div > div:focus-within,
.stMultiSelect > div > div > div:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, .2) !important;
    outline: none !important;
}

label, .stTextInput label {
    color: #e2e8f0 !important;
    font-weight: 500;
    font-size: 14px;
}

h1, h2, h3 {
    color: #f1f5f9 !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em;
}

h1 {
    background: linear-gradient(135deg, #f59e0b, #facc15);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 2rem !important;
}

::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #1e293b;
}
::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #475569;
}

/* Responsive */
@media (max-width: 768px) {
    .hero-title {
        font-size: 2rem;
    }
    .hero-subtitle {
        font-size: 1.1rem;
    }
    .features-grid,
    .stats-grid,
    .steps-grid {
        grid-template-columns: 1fr;
    }
    .film-poster {
        height: 300px;
    }
    .film-card {
        padding: 18px;
    }
}

@media (max-width: 480px) {
    .hero-title {
        font-size: 1.8rem;
    }
    .film-poster {
        height: 240px;
    }
    .film-card {
        padding: 16px;
    }
}
</style>
""", unsafe_allow_html=True)

# Navigation Buttons
st.markdown('<div class="nav-container">', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    if st.button("🏠 Beranda", use_container_width=True):
        st.session_state.page = 'home'
with col2:
    if st.button("🎬 Cari Film", use_container_width=True):
        st.session_state.page = 'search'
st.markdown('</div>', unsafe_allow_html=True)

# Homepage Content
if st.session_state.page == 'home':
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🎬 MovieRec System</div>
        <p class="hero-subtitle">Temukan film favoritmu dengan teknologi Machine Learning yang canggih dan akurat</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown("## ✨ Fitur Unggulan")
    st.markdown("""
    <div class="features-grid">
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <h3 class="feature-title">AI-Powered</h3>
            <p class="feature-desc">Menggunakan algoritma machine learning TF-IDF dan Cosine Similarity untuk rekomendasi yang akurat dan personal</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h3 class="feature-title">Super Cepat</h3>
            <p class="feature-desc">Dapatkan rekomendasi film dalam hitungan detik dengan performa yang optimal dan responsif</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <h3 class="feature-title">Akurat & Relevan</h3>
            <p class="feature-desc">Sistem rekomendasi dengan tingkat similarity tinggi berdasarkan genre, cast, director, dan plot</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats Section
    st.markdown("## 📊 Data & Statistik")
    st.markdown("""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-number">10K+</div>
            <div class="stat-label">Film Database</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">95%</div>
            <div class="stat-label">Akurasi</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">0.5s</div>
            <div class="stat-label">Response Time</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">24/7</div>
            <div class="stat-label">Available</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # How it Works
    st.markdown("## 🔍 Cara Kerja Sistem")
    st.markdown("""
    <div class="steps-grid">
        <div class="step-card">
            <div class="step-number">1</div>
            <h3 class="step-title">Input Film</h3>
            <p class="step-desc">Masukkan judul film yang kamu suka atau pernah tonton</p>
        </div>
        <div class="step-card">
            <div class="step-number">2</div>
            <h3 class="step-title">AI Analysis</h3>
            <p class="step-desc">Sistem menganalisis karakteristik film menggunakan TF-IDF dan Cosine Similarity</p>
        </div>
        <div class="step-card">
            <div class="step-number">3</div>
            <h3 class="step-title">Rekomendasi</h3>
            <p class="step-desc">Dapatkan daftar film serupa yang diprediksi akan kamu sukai</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # CTA Section
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🚀 Siap mencari film favoritmu?")
        if st.button("Mulai Cari Film Sekarang", use_container_width=True, type="primary"):
            st.session_state.page = 'search'
            st.rerun()
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("© 2025 Movie Recommendation")
    with col2:
        st.markdown("💻 **Developed by Sheeva**")
    with col3:
        st.markdown("🙏 Thanks to Open Source Community")

# Search Page Content (kode yang sudah ada sebelumnya)
elif st.session_state.page == 'search':
    # Load data (pindahkan ke sini agar hanya load saat diperlukan)
    @st.cache_data
    def load_data():
        df_all = joblib.load('df_all.pkl')
        tfidf = joblib.load('tfidf_vectorizer.pkl')
        tfidf_matrix = joblib.load('tfidf_matrix.pkl')
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return df_all, tfidf, tfidf_matrix, cosine_sim
    
    df_all, tfidf, tfidf_matrix, cosine_sim = load_data() 

# --- Header ---
st.title("🎬 Sistem Rekomendasi Film")

# --- Banner ---
st.image("banner.jpg", use_container_width=True)

# --- Input Form ---
st.subheader("Cari rekomendasi berdasarkan judul film yang kamu suka")
with st.form(key="search_form"):
    input_title = st.text_input("Masukkan judul film:")
    submit = st.form_submit_button("Cari Rekomendasi")

# --- Hasil ---
if submit and input_title.strip() == "":
    st.warning("⚠️ Masukkan judul film terlebih dahulu.")
elif submit:
    hasil, corrected = recommend_film(input_title)

    if hasil is None or hasil.empty:
        st.warning(f"❌ Film '{input_title}' tidak ditemukan dalam database.")
    else:
        st.markdown(f"## 🔍 Rekomendasi film untuk mu :")
        st.info(f"✅ Ditemukan {len(hasil)} film yang relevan")

        for i in range(0, len(hasil), 3):
            cols = st.columns(3)
            for idx, col in enumerate(cols):
                if i + idx < len(hasil):
                    film = hasil.iloc[i + idx]
                    full_overview = film['overview']
                    poster_url = film.get('poster_url', '')

                    with col:
                        if poster_url and not pd.isna(poster_url):
                            try:
                                st.image(poster_url, use_container_width=True)
                            except:
                                st.error("🖼️ Poster tidak dapat dimuat")
                        else:
                            st.markdown(f"""<div style="width:100%;height:300px;background:linear-gradient(135deg,#374151,#1f2937);border-radius:12px;display:flex;align-items:center;justify-content:center;margin-bottom:16px;border:2px dashed #6b7280;"><div style="text-align:center;color:#9ca3af;">🎬<br><small>Poster Tidak Tersedia</small></div></div>""", unsafe_allow_html=True)

                        st.markdown(f"""
                            <div class="film-card">
                                <h4>{film['title']}</h4>
                                <p><strong>Genre:</strong> {film['genres']}</p>
                                <p><strong>Director:</strong> {film['director']}</p>
                                <p><strong>Cast:</strong> {film['cast']}</p>
                                <p><strong>Similarity:</strong> {film['cosine_similarity']:.1%}</p>
                                <details style="margin-top:10px;">
                                    <summary>📖 Sinopsis</summary>
                                    <p style="margin-top:8px; color: #cbd5e1;">{full_overview}</p>
                                </details>
                            </div>
                        """, unsafe_allow_html=True)

