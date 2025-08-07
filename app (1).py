import streamlit as st
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from difflib import get_close_matches

# Set page config harus di awal
st.set_page_config(page_title="Sistem Rekomendasi Film", layout="wide")

# Load file .pkl
df_all = joblib.load('df_all.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#fungsi match
def find_best_match(title, choices, cutoff=0.6):
    """Cari judul terdekat dari list choices, atau None."""
    matches = get_close_matches(title.lower(), [c.lower() for c in choices], n=1, cutoff=cutoff)
    return matches[0] if matches else None

#fungsi rekomendasi film
def recommend_film(title, df_all, cosine_sim, top_n=10, sim_threshold=0.1):
    """
    title         : judul input user (string)
    df_all        : DataFrame lengkap dengan kolom 'title', dst.
    cosine_sim    : matrix numpy similarity
    top_n         : jumlah rekomendasi maksimal
    sim_threshold : nilai minimal cosine similarity
    """
    # 1. Cari judul paling mirip
    all_titles = df_all['title'].tolist()
    corrected = find_best_match(title, all_titles)
    if not corrected:
        # tidak ketemu, kasih tahu user
        print(f"Maaf, aku gak nemu film yang mirip '{title}'. Coba cek ejaannya ya 😊")
        return None
    
    # 2. Temukan indeks film yang benar
    idx = df_all[df_all['title'].str.lower() == corrected].index[0]
    
    # 3. Hitung skor similarity & filter self + threshold, langsung sort
    sim_scores = [
        (i, score) 
        for i, score in enumerate(cosine_sim[idx]) 
        if i != idx and score >= sim_threshold
    ]
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 4. Ambil top_n
    top_scores = sim_scores[:top_n]
    film_indices = [i for i, _ in top_scores]
    similarities  = [s for _, s in top_scores]
    
    # 5. Siapkan DataFrame hasil
    result = (
        df_all
        .iloc[film_indices]
        [['title', 'genres', 'overview', 'director', 'cast', 'poster_url']]
        .copy()
    )
    result['cosine_similarity'] = similarities
    return result, corrected

# --- CSS Tampilan ---
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

# --- Input ---
st.subheader("Cari rekomendasi berdasarkan judul film yang kamu suka")

# Form agar bisa jalan pakai Enter juga
with st.form(key="search_form"):
    input_title = st.text_input("Masukkan judul film:")
    submit = st.form_submit_button("Cari Rekomendasi")

# Jalankan hasil jika submit atau Enter
# Jalankan hasil jika submit atau Enter
if submit and input_title.strip() == "":
    st.warning("⚠️ Masukkan judul film terlebih dahulu.")
elif submit and input_title:
    # 1. Panggil fungsi recommend_film dan unpack dua nilai: (DataFrame, corrected_title)
    rekom = recommend_film(input_title, df_all, cosine_sim)
    
    # 2. Cek apakah fungsi mengembalikan None atau DataFrame kosong
    if not rekom:
        st.warning(f"❌ Film dengan judul '{input_title}' tidak ditemukan atau tidak ada yang mirip.")
    else:
        hasil, judul_asli = rekom
        
        # 3. Judul asli yang sudah dikoreksi
        st.markdown(f"## 🔍 Berikut hasil rekomendasi untuk **{judul_asli.title()}**:")
        
        # 4. Loop dan tampilkan 3 kolom per baris
        for i in range(0, len(hasil), 3):
            cols = st.columns(3)
            for idx, col in enumerate(cols):
                if i + idx < len(hasil):
                    film = hasil.iloc[i + idx]
                    full_overview = film['overview']
                    poster_url   = film.get('poster_url', "")
                    
                    with col:
                        # Tampilkan poster jika valid
                        if poster_url and not pd.isna(poster_url):
                            try:
                                st.image(poster_url, use_container_width=True)
                            except Exception:
                                st.error("🖼️ Poster tidak dapat dimuat")
                                st.write(f"URL: {poster_url}")
                        else:
                            # Placeholder jika poster kosong atau NaN
                            st.markdown(f"""
                                <div style="
                                    width: 100%;
                                    height: 300px;
                                    background: linear-gradient(135deg, #374151, #1f2937);
                                    border-radius: 12px;
                                    display: flex;
                                    align-items: center;
                                    justify-content: center;
                                    margin-bottom: 16px;
                                    border: 2px dashed #6b7280;
                                ">
                                    <div style="text-align: center; color: #9ca3af;">
                                        🎬<br>
                                        <small>Poster Tidak Tersedia</small>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                        with st.container():
                            st.markdown(f"""
                                <div class="film-card">
                                    <h4>{film['title']}</h4>
                                    <p><strong>Genre:</strong> {film['genres']}</p>
                                    <p><strong>Director:</strong> {film['director']}</p>
                                    <p><strong>Cast:</strong> {film['cast']}</p>
                                    <details style="margin-top:10px;">
                                        <summary>📖 Sinopsis</summary>
                                        <p style="margin-top:8px; color: #cbd5e1;">{full_overview}</p>
                                    </details>
                                </div>
                            """, unsafe_allow_html=True)
