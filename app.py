import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import requests

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="RoomieMatch AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. APPLE-INSPIRED PRO CSS (Updated for Red/Black Sidebar) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* GLOBAL RESET */
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stApp { background-color: #FAFAFA; }

    /* HIDE STREAMLIT BRANDING (But keep header visible for sidebar toggle) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header {visibility: hidden;}  <-- REMOVED THIS so you can see the sidebar arrow */

    /* --- SIDEBAR STYLING --- */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #E5E5E5;
    }

    /* Sidebar Title */
    [data-testid="stSidebar"] h1 {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111;
        margin-bottom: 2rem;
    }

    /* Input Fields (Card Style) */
    [data-testid="stSidebar"] .stTextInput div[data-baseweb="input"] {
        background-color: white;
        border: 1px solid #E0E0E0;
        border-radius: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }

    /* RED SLIDERS */
    [data-testid="stSidebar"] label {
        font-weight: 600;
        color: #333;
        font-size: 0.9rem;
    }
    div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"] {
        background-color: #FF4B4B !important;
    }
    div.stSlider > div[data-baseweb="slider"] > div > div {
        background: #FF4B4B !important;
    }

    /* BLACK BUTTON */
    div.stButton > button {
        background: #111827;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        background: #000000;
        transform: scale(1.02);
    }

    /* CARDS & CONTAINERS */
    .hero-container {
        text-align: center;
        padding: 3rem 1rem;
        animation: fadeIn 0.8s ease-in;
    }
    
    .apple-card {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.04);
        border: 1px solid #F0F0F0;
        margin-bottom: 20px;
        transition: transform 0.2s ease;
    }
    
    .apple-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
    }

    /* TYPOGRAPHY */
    h1.hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        background: -webkit-linear-gradient(120deg, #111, #555);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    p.hero-sub {
        font-size: 1.2rem;
        color: #666;
        font-weight: 400;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #111;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* BADGES */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-score { background: #E8F5E9; color: #2E7D32; }
    .badge-cluster { background: #E3F2FD; color: #1565C0; }

    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# --- 3. BACKEND LOGIC (YOUR ORIGINAL LOGIC) ---
@st.cache_data
def load_and_train():
    # Try loading CSV, if fail, generate synthetic data
    try:
        df = pd.read_csv("hostel_users_clustered.csv")
    except FileNotFoundError:
        # Fallback: Generate Synthetic Data on the fly
        names = [f"Student {i}" for i in range(101, 301)]
        majors = ['CSE', 'Business', 'Psychology', 'Engineering', 'Arts', 'Bio']
        df = pd.DataFrame({
            'Name': names,
            'Major': np.random.choice(majors, 200),
            'Contact': [f"student{i}@college.edu" for i in range(101, 301)],
            'Sleep': np.random.randint(1, 11, 200),
            'Cleanliness': np.random.randint(1, 11, 200),
            'Social': np.random.randint(1, 11, 200),
            'Noise': np.random.randint(1, 11, 200)
        })

    # Prepare ML Features
    features = df[['Sleep', 'Cleanliness', 'Social', 'Noise']]
    
    # K-Means Training
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(features)
    
    # Intelligent Labeling
    centers = df.groupby('Cluster')[['Sleep', 'Social']].mean()
    cluster_names = {}
    for cid, row in centers.iterrows():
        if row['Sleep'] > 5.5 and row['Social'] > 5.5:
            cluster_names[cid] = "🎉 Party Animal"
        elif row['Sleep'] < 5.5 and row['Social'] < 5.5:
            cluster_names[cid] = "📚 The Scholar"
        elif row['Sleep'] > 5.5 and row['Social'] < 5.5:
            cluster_names[cid] = "🦉 Night Owl"
        else:
            cluster_names[cid] = "🏅 The Athlete"
            
    df['Cluster_Name'] = df['Cluster'].map(cluster_names)
    return df, kmeans, cluster_names

# Load Logic
try:
    df, kmeans_model, cluster_map = load_and_train()
except Exception as e:
    st.error(f"Critical System Error: {e}")
    st.stop()

# --- 4. SIDEBAR UI ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712009.png", width=60)
    st.title("My Profile")
    st.markdown("Customize your habits to find your perfect tribe.")
    
    my_name = st.text_input("First Name", "Alex")
    st.write("") # Spacer
    
    st.markdown("### ⚙️ Preferences")
    s_sleep = st.slider("🌙 Sleep Schedule", 1, 10, 5, help="1=Early Riser, 10=Late Night")
    s_clean = st.slider("✨ Cleanliness", 1, 10, 5, help="1=Messy, 10=Organized")
    s_social = st.slider("🎭 Social Battery", 1, 10, 5, help="1=Introvert, 10=Extrovert")
    s_noise = st.slider("🔊 Noise Tolerance", 1, 10, 5, help="1=Quiet, 10=Loud")
    
    st.markdown("---")
    
    # ADDED: Simple session state so results don't vanish on update
    if 'search' not in st.session_state:
        st.session_state.search = False
    
    def trigger_search():
        st.session_state.search = True

    btn_search = st.button("Find Roommates", on_click=trigger_search)

# --- 5. MAIN PAGE UI ---

# Hero Section
st.markdown("""
<div style="text-align: center; margin-top: 20px; margin-bottom: 40px;">
    <div style="display: flex; align-items: center; justify-content: center; column-gap: 15px;">
        <h1 style="margin: 0; font-family: sans-serif; font-weight: 800; font-size: 3.2rem; color: #1e1e1e; letter-spacing: -1.5px;">
            RoomieMatch AI
        </h1>
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#9CA3AF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-top: 8px;">
            <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path>
            <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>
        </svg>
    </div>
    <p style="font-family: sans-serif; color: #555; font-size: 1.1rem; margin-top: 10px; font-weight: 400;">
        Intelligent hostel allocation powered by Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)

if st.session_state.search:
    # --- LOGIC ENGINE ---
    user_vector = [[s_sleep, s_clean, s_social, s_noise]]
    
    # 1. Classification
    pred_cluster = kmeans_model.predict(user_vector)[0]
    user_tribe = cluster_map.get(pred_cluster, "General")
    
    # 2. Similarity Matching
    tribe_df = df[df['Cluster'] == pred_cluster].copy()
    tribe_features = tribe_df[['Sleep', 'Cleanliness', 'Social', 'Noise']].values
    
    # Cosine Similarity
    sim_scores = cosine_similarity(user_vector, tribe_features)[0]
    tribe_df['Match_Score'] = sim_scores
    
    # Get Top 3
    results = tribe_df.sort_values(by='Match_Score', ascending=False).head(3)
    best_match = results.iloc[0]

    # --- DASHBOARD ---
    
    # Section 1: Metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="apple-card" style="text-align:center;">
            <div class="metric-label">Database</div>
            <div class="metric-value">{len(df)}</div>
            <div style="color:#888; font-size:0.8rem;">Active Students</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="apple-card" style="text-align:center;">
            <div class="metric-label">Your Tribe</div>
            <div class="metric-value">{user_tribe}</div>
            <div style="color:#888; font-size:0.8rem;">Based on AI Cluster</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="apple-card" style="text-align:center;">
            <div class="metric-label">Top Compatibility</div>
            <div class="metric-value" style="color:#2E7D32;">{int(best_match['Match_Score']*100)}%</div>
            <div style="color:#888; font-size:0.8rem;">Cosine Similarity</div>
        </div>
        """, unsafe_allow_html=True)

    # Section 2: Visual Analysis
    col_radar, col_scatter = st.columns([1, 1])
    
    with col_radar:
        st.markdown('<div class="apple-card">', unsafe_allow_html=True)
        st.markdown("**🎯 Personality Overlap**")
        
        # Radar Chart Logic
        categories = ['Sleep', 'Cleanliness', 'Social', 'Noise']
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[s_sleep, s_clean, s_social, s_noise],
            theta=categories,
            fill='toself',
            name='You',
            line_color='#007AFF'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[best_match['Sleep'], best_match['Cleanliness'], best_match['Social'], best_match['Noise']],
            theta=categories,
            fill='toself',
            name=best_match['Name'],
            line_color='#FF3B30'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=True,
            margin=dict(l=40, r=40, t=20, b=20),
            height=300
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_scatter:
        st.markdown('<div class="apple-card">', unsafe_allow_html=True)
        st.markdown(f"**🗺️ You vs The '{user_tribe}' Tribe**")
        
        # Scatter Logic
        fig_scatter = px.scatter(
            df, x='Sleep', y='Social', 
            color='Cluster_Name',
            color_discrete_sequence=['#FF3B30', '#007AFF', '#34C759', '#FF9500'],
            labels={'Sleep': 'Sleep Schedule', 'Social': 'Social Activity'}
        )
        # Add User Star
        fig_scatter.add_trace(go.Scatter(
            x=[s_sleep], y=[s_social],
            mode='markers',
            marker=dict(symbol='star', size=20, color='black'),
            name='YOU'
        ))
        fig_scatter.update_layout(
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Section 3: Recommendations
    st.markdown("### 🔥 Recommended Roommates")
    
    match_cols = st.columns(3)
    for i, (idx, row) in enumerate(results.iterrows()):
        score = int(row['Match_Score'] * 100)
        with match_cols[i]:
            st.markdown(f"""
            <div class="apple-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0; font-size:1.3rem;">{row['Name']}</h3>
                    <span class="badge badge-score">{score}%</span>
                </div>
                <div style="margin-top:10px; color:#666; font-size:0.9rem;">
                    Major: <b>{row['Major']}</b><br>
                    {row['Contact']}
                </div>
                <hr style="border:0; border-top:1px solid #eee; margin:15px 0;">
                <div style="display:flex; justify-content:space-around; text-align:center;">
                    <div>
                        <div style="font-weight:bold; font-size:1.1rem;">{row['Sleep']}</div>
                        <div style="font-size:0.7rem; color:#888;">SLEEP</div>
                    </div>
                    <div>
                        <div style="font-weight:bold; font-size:1.1rem;">{row['Social']}</div>
                        <div style="font-size:0.7rem; color:#888;">SOCIAL</div>
                    </div>
                    <div>
                        <div style="font-weight:bold; font-size:1.1rem;">{row['Cleanliness']}</div>
                        <div style="font-size:0.7rem; color:#888;">CLEAN</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    # Empty State (Before searching)
    st.info("👈 Please set your preferences in the sidebar and click **Find Roommates** to start the AI analysis.")
    
    # Show global data overview
    st.markdown("### 📊 Live System Data")
    st.dataframe(
        df[['Name', 'Major', 'Cluster_Name', 'Sleep', 'Cleanliness']].head(5),
        use_container_width=True,
        hide_index=True
    )