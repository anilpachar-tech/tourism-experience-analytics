import streamlit as st
import pandas as pd
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# ── page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="TravelIQ — Tourism Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #f0f0f0;
}

/* sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* headings */
h1, h2, h3 {
    font-family: 'Sora', sans-serif;
    color: #ffffff;
}

/* metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(10px);
}
[data-testid="stMetricLabel"]  { color: #a0a0c0 !important; font-size: 13px; }
[data-testid="stMetricValue"]  { color: #ffffff !important; font-size: 28px; font-weight: 700; }

/* buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px 28px;
    font-family: 'Sora', sans-serif;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #764ba2, #667eea);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102,126,234,0.4);
}

/* selectbox + number input */
[data-testid="stSelectbox"] > div,
[data-testid="stNumberInput"] > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: white !important;
}

/* success box */
[data-testid="stAlert"] {
    background: rgba(102, 126, 234, 0.15);
    border: 1px solid rgba(102, 126, 234, 0.4);
    border-radius: 14px;
    color: white;
}

/* dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* divider */
hr {
    border-color: rgba(255,255,255,0.1);
}

/* sidebar radio */
[data-testid="stRadio"] label {
    color: #c0c0d0 !important;
    font-size: 15px;
}

/* slider */
[data-testid="stSlider"] {
    color: white;
}

/* tabs */
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.05);
    border-radius: 8px 8px 0 0;
    color: #a0a0c0;
    font-family: 'Sora', sans-serif;
}
.stTabs [aria-selected="true"] {
    background: rgba(102,126,234,0.3) !important;
    color: white !important;
}

/* info card */
.info-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.info-card h4 {
    margin: 0 0 6px 0;
    color: #a0a0c0;
    font-size: 13px;
    font-family: 'DM Sans';
    text-transform: uppercase;
    letter-spacing: 1px;
}
.info-card p {
    margin: 0;
    color: white;
    font-size: 22px;
    font-weight: 600;
    font-family: 'Sora';
}
</style>
""", unsafe_allow_html=True)


# ── load models ───────────────────────────────────────────────
@st.cache_resource
def load():
    base     = os.path.dirname(os.path.abspath(__file__))
    reg      = joblib.load(os.path.join(base, 'models', 'reg_model.pkl'))
    cls      = joblib.load(os.path.join(base, 'models', 'cls_model.pkl'))
    le       = joblib.load(os.path.join(base, 'models', 'label_encoder.pkl'))
    ui       = joblib.load(os.path.join(base, 'models', 'user_item.pkl'))
    features = joblib.load(os.path.join(base, 'models', 'features.pkl'))
    df       = pd.read_csv(os.path.join(base, 'data', 'processed_data.csv'))
    return reg, cls, le, ui, features, df
    reg, cls, le, user_item, features, df = load()


# ── sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <div style='font-size:42px'>🌍</div>
        <div style='font-family:Sora; font-size:20px; font-weight:700; color:white;'>TravelIQ</div>
        <div style='font-size:12px; color:#a0a0c0; margin-top:4px;'>Tourism Experience Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio("", [
        "📊  Dashboard",
        "🧳  Classify Visit Mode",
        "⭐  Predict Rating",
        "🎯  Get Recommendations"
    ])

    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px; color:#606080; text-align:center; padding-top:10px;'>
        Powered by Machine Learning<br>Random Forest · XGBoost · Cosine Similarity
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "📊  Dashboard":
    st.markdown("<h1>📊 Tourism Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a0a0c0; margin-top:-10px;'>Live insights from tourism transaction data</p>", unsafe_allow_html=True)
    st.markdown("---")

    # top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions",  f"{len(df):,}")
    c2.metric("Unique Users",        f"{df['UserId'].nunique():,}")
    c3.metric("Unique Attractions",  f"{df['AttractionId'].nunique():,}")
    c4.metric("Average Rating",      f"{df['Rating'].mean():.2f} ⭐")

    st.markdown("---")

    # row 1
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🧳 Visit Mode Distribution")
        st.bar_chart(df['VisitMode'].value_counts(), color="#667eea")

    with col2:
        st.markdown("#### 🌍 Avg Rating by Continent")
        st.bar_chart(df.groupby('Continent')['Rating'].mean().sort_values(), color="#764ba2")

    # row 2
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 🗺️ Top 10 Countries by Visits")
        st.bar_chart(df['Country'].value_counts().head(10), color="#f093fb")

    with col4:
        st.markdown("#### 🏛️ Top Attraction Types")
        st.bar_chart(df['AttractionType'].value_counts().head(8), color="#4facfe")

    # row 3
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("#### 📅 Monthly Travel Trend")
        st.line_chart(df['VisitMonth'].value_counts().sort_index(), color="#43e97b")

    with col6:
        st.markdown("#### 📆 Yearly Visit Trend")
        st.bar_chart(df['VisitYear'].value_counts().sort_index(), color="#fa709a")

    # row 4 — full width
    st.markdown("#### 🏆 Top 10 Most Visited Attractions")
    st.bar_chart(df['Attraction'].value_counts().head(10), color="#fee140")


# ══════════════════════════════════════════════════════════════
# PAGE 2 — CLASSIFY VISIT MODE
# ══════════════════════════════════════════════════════════════
elif page == "🧳  Classify Visit Mode":
    st.markdown("<h1>🧳 Classify Visit Mode</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a0a0c0; margin-top:-10px;'>Predict whether a user will travel as Business, Family, Couples, Friends or Solo</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### Enter User & Attraction Details")
    col1, col2 = st.columns(2)
    vals = {}

    for i, feat in enumerate(features):
        col = col1 if i % 2 == 0 else col2
        opts = sorted(df[feat].dropna().unique().tolist())
        vals[feat] = col.selectbox(f"{feat}", opts)

    st.markdown("---")

    if st.button("🔍  Predict Visit Mode"):
        inp  = pd.DataFrame([vals])
        enc  = cls.predict(inp)[0]
        mode = le.inverse_transform([enc])[0]

        emoji_map = {
            'Business': ('💼', '#667eea', 'Corporate travel for work purposes'),
            'Family'  : ('👨‍👩‍👧', '#43e97b', 'Family vacation with kids'),
            'Couples' : ('💑', '#f093fb', 'Romantic getaway for two'),
            'Friends' : ('👫', '#fa709a', 'Fun trip with a group of friends'),
            'Solo'    : ('🧍', '#4facfe', 'Independent solo travel')
        }

        emoji, color, desc = emoji_map.get(mode, ('🧳', '#667eea', ''))

        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.07); border: 1px solid {color}55;
                    border-radius: 20px; padding: 32px; text-align:center; margin-top:16px;'>
            <div style='font-size:64px; margin-bottom:12px;'>{emoji}</div>
            <div style='font-family:Sora; font-size:32px; font-weight:700;
                        color:{color}; margin-bottom:8px;'>{mode}</div>
            <div style='color:#a0a0c0; font-size:16px;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # show all class probabilities if model supports it
        if hasattr(cls, 'predict_proba'):
            proba = cls.predict_proba(inp)[0]
            prob_df = pd.DataFrame({
                'Visit Mode' : list(le.classes_),
                'Probability': [round(p * 100, 2) for p in proba]
            }).sort_values('Probability', ascending=False)

            st.markdown("#### 📊 Probability Breakdown")
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
            st.bar_chart(prob_df.set_index('Visit Mode')['Probability'], color=color)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT RATING
# ══════════════════════════════════════════════════════════════
elif page == "⭐  Predict Rating":
    st.markdown("<h1>⭐ Predict Attraction Rating</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a0a0c0; margin-top:-10px;'>Estimate how much a user will enjoy a specific attraction</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### Enter Details")
    col1, col2 = st.columns(2)
    vals = {}

    for i, feat in enumerate(features):
        col = col1 if i % 2 == 0 else col2
        opts = sorted(df[feat].dropna().unique().tolist())
        vals[feat] = col.selectbox(f"{feat}", opts)

    st.markdown("---")

    if st.button("⭐  Predict Rating"):
        inp  = pd.DataFrame([vals])
        pred = reg.predict(inp)[0]
        pred = round(min(max(pred, 1.0), 5.0), 2)

        # color based on rating
        if pred >= 4.0:
            color = '#43e97b'
            label = 'Excellent'
            emoji = '🤩'
        elif pred >= 3.0:
            color = '#fee140'
            label = 'Good'
            emoji = '😊'
        else:
            color = '#fa709a'
            label = 'Average'
            emoji = '😐'

        filled_stars   = int(round(pred))
        unfilled_stars = 5 - filled_stars
        stars_html     = '⭐' * filled_stars + '☆' * unfilled_stars

        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.07); border: 1px solid {color}55;
                    border-radius: 20px; padding: 32px; text-align:center; margin-top:16px;'>
            <div style='font-size:48px; margin-bottom:8px;'>{emoji}</div>
            <div style='font-size:28px; margin-bottom:8px;'>{stars_html}</div>
            <div style='font-family:Sora; font-size:48px; font-weight:700;
                        color:{color}; margin-bottom:8px;'>{pred} <span style='font-size:20px; color:#a0a0c0;'>/ 5.0</span></div>
            <div style='color:{color}; font-size:18px; font-weight:600;'>{label}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
elif page == "🎯  Get Recommendations":
    st.markdown("<h1>🎯 Personalized Recommendations</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a0a0c0; margin-top:-10px;'>Attractions you are most likely to enjoy based on users similar to you</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        uid = st.number_input(
            "Enter Your User ID",
            min_value=int(df['UserId'].min()),
            max_value=int(df['UserId'].max()),
            step=1
        )
    with col2:
        n = st.slider("Number of Recommendations", 5, 20, 10)

    if uid in user_item.index:
        visited_count = user_item.loc[uid].dropna().shape[0]
        avg_given     = df[df['UserId'] == uid]['Rating'].mean()
        ic1, ic2 = st.columns(2)
        ic1.metric("Attractions Visited", visited_count)
        ic2.metric("Your Avg Rating Given", f"{avg_given:.2f} ⭐" if not pd.isna(avg_given) else "N/A")

    st.markdown("---")

    if st.button("🎯  Get My Recommendations"):
        if uid not in user_item.index:
            st.warning("⚠️ User ID not found in dataset. Please try a different one.")
        else:
            with st.spinner("Finding similar users..."):
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np

                # get this user's ratings vector
                user_vector = user_item.loc[[uid]].fillna(0).values

                # fill matrix with 0 for NaN
                filled = user_item.fillna(0).values

                
                sim_scores = cosine_similarity(user_vector, filled)[0]

                # top 10 similar users (excluding self)
                top_indices = np.argsort(sim_scores)[::-1][1:11]
                similar_users = user_item.index[top_indices].tolist()

                seen     = user_item.loc[uid].dropna().index.tolist()
                avg          = user_item.loc[similar_users].mean(axis=0, skipna=True)
                global_mean  = df['Rating'].mean()
                avg          = avg.fillna(global_mean)
                unseen       = avg.drop(index=seen, errors='ignore').sort_values(ascending=False).head(n)

                name_map = df[['AttractionId', 'Attraction']].drop_duplicates().set_index('AttractionId')['Attraction']
                type_map = df[['AttractionId', 'AttractionType']].drop_duplicates().set_index('AttractionId')['AttractionType']

                result = pd.DataFrame({
                    'Attraction'      : unseen.index.map(name_map),
                    'Type'            : unseen.index.map(type_map),
                    'Predicted Rating': unseen.values.round(2)
                }).reset_index(drop=True)
                result.index += 1

            st.markdown(f"#### 🏆 Top {n} Picks for User {uid}")
            st.dataframe(result, use_container_width=True)

            st.markdown("#### 📊 Predicted Ratings")
            st.bar_chart(result.set_index('Attraction')['Predicted Rating'], color="#667eea")

            if 'Type' in result.columns:
                st.markdown("#### 🏛️ Recommendation Type Breakdown")
                st.bar_chart(result['Type'].value_counts(), color="#f093fb")
