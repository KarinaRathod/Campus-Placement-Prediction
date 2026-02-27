import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score

# ==========================================================
# 🎨 PAGE CONFIG & STYLING
# ==========================================================
st.set_page_config(page_title="Placement AI Pro", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    .prediction-card { padding: 20px; border-radius: 15px; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 📊 DATA ENGINE
# ==========================================================
@st.cache_data
def load_and_prep_data():
    # Load your specific CSV
    df = pd.read_csv("campus_placement_data.csv")
    
    # 1. Cleaning
    df_clean = df.copy()
    if 'student_id' in df_clean.columns:
        df_clean.drop('student_id', axis=1, inplace=True)
    
    # 2. Encoding Categorical Data
    encoders = {}
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Fill NAs before encoding to avoid errors
        df_clean[col] = df_clean[col].fillna('Unknown')
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        encoders[col] = le
        
    return df, df_clean, encoders

df_raw, df_model, encoders = load_and_prep_data()

# ==========================================================
# 🧠 MODEL TRAINING ENGINE
# ==========================================================
@st.cache_resource
def train_engine(data):
    # Features: Everything except 'placed' and 'salary_lpa'
    X = data.drop(['placed', 'salary_lpa'], axis=1)
    
    # --- Placement Model (Classification) ---
    y_p = data['placed']
    clf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
    clf.fit(X, y_p)
    
    # --- Salary Model (Regression) ---
    # We only train on students who were actually placed
    placed_data = data[data['placed'] == 1]
    X_s = placed_data.drop(['placed', 'salary_lpa'], axis=1)
    y_s = placed_data['salary_lpa']
    reg = RandomForestRegressor(n_estimators=150, random_state=42)
    reg.fit(X_s, y_s)
    
    return clf, reg, X.columns

clf_model, reg_model, feature_names = train_engine(df_model)

# ==========================================================
# 🏛️ UI LAYOUT
# ==========================================================
st.title("🎓 Campus Placement Intelligence System")
st.markdown("Developed by Placement Cell Analytics | Data-Driven Career Insights")

tab1, tab2, tab3 = st.tabs(["📊 Analytics Dashboard", "🎯 Prediction Engine", "💡 Feature Insights"])

# --- TAB 1: ANALYTICS ---
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Placement Rate", f"{int(df_raw['placed'].mean()*100)}%")
    col2.metric("Avg Salary (LPA)", f"₹{df_raw['salary_lpa'].mean():.2f}")
    col3.metric("Avg Tech Score", f"{df_raw['technical_skills_score'].mean():.1f}")
    col4.metric("Dataset Samples", len(df_raw))

    st.divider()
    
    c1, c2 = st.columns([2, 1])
    with c1:
        metric = st.selectbox("Compare Outcome by:", 
                             ['degree_percentage', 'technical_skills_score', 'aptitude_score', 'work_experience_months'])
        fig = px.histogram(df_raw, x=metric, color="placed", barmode="group",
                           title=f"Distribution of {metric.replace('_',' ').title()} by Placement Status",
                           color_discrete_map={0: "#FF4B4B", 1: "#00CC96"},
                           nbins=30)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig_pie = px.pie(df_raw, names='specialization', title="Specialization Split", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

# --- TAB 2: PREDICTION ---
with tab2:
    st.subheader("📝 Input Student Profile")
    
    with st.form("prediction_form"):
        # Split inputs into three columns for better UI
        col_left, col_mid, col_right = st.columns(3)
        
        user_inputs = {}
        
        # Dynamically generate inputs based on your column list
        for i, col in enumerate(feature_names):
            # Decide which column to put the input in
            target_col = col_left if i % 3 == 0 else (col_mid if i % 3 == 1 else col_right)
            
            clean_label = col.replace('_', ' ').title()
            
            if col in encoders:
                options = list(encoders[col].classes_)
                val = target_col.selectbox(clean_label, options)
                user_inputs[col] = encoders[col].transform([val])[0]
            else:
                # Numerical handling
                min_v = float(df_raw[col].min())
                max_v = float(df_raw[col].max())
                mean_v = float(df_raw[col].mean())
                user_inputs[col] = target_col.number_input(clean_label, min_v, max_v, mean_v)

        submit = st.form_submit_button("Generate Prediction Report", type="primary")

    if submit:
        input_df = pd.DataFrame([user_inputs])
        
        # 1. Placement Prediction
        prob = clf_model.predict_proba(input_df)[0][1]
        
        st.divider()
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.markdown("### Placement Success Probability")
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                number = {'suffix': "%"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#00CC96" if prob > 0.5 else "#FF4B4B"},
                    'steps': [
                        {'range': [0, 40], 'color': "#ffebee"},
                        {'range': [40, 70], 'color': "#fff9c4"},
                        {'range': [70, 100], 'color': "#e8f5e9"}
                    ]
                }))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.markdown("### Estimated Salary Forecast")
            if prob >= 0.5:
                salary_pred = reg_model.predict(input_df)[0]
                st.success(f"## Predicted Salary: ₹ {salary_pred:.2f} LPA")
                st.info("💡 High probability detected. Focus on maintaining your technical scores to secure this bracket.")
            else:
                st.warning("## Placement Outlook: Uncertain")
                st.write("Current profile metrics are below the historical placement threshold for this dataset.")
                st.markdown("**Recommendation:** Improve Technical Skills Score or Certifications Count.")

# --- TAB 3: FEATURE INSIGHTS ---
with tab3:
    st.subheader("🧠 Model Decision Logic")
    
    # Calculate Feature Importance
    importances = clf_model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    
    fig_imp = px.bar(feat_imp, orientation='h', 
                     labels={'value': 'Importance Score', 'index': 'Factor'},
                     title="Key Drivers of Successful Placement",
                     color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.success("""
    **Expert Insight:** According to the Random Forest model, the factors at the top of this chart are the strongest predictors of whether a student gets hired. 
    Focusing on these specific areas will yield the highest 'Return on Effort' for students.
    """)

st.markdown("---")
st.caption("v2.5 Professional Edition | Built for High-Performance ML Visualization")