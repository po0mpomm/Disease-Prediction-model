import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import os
from pathlib import Path
from collections import Counter

# Set page configuration
st.set_page_config(page_title="Disease Outbreak Prediction and Map", layout="wide")

# Title
st.title("Disease Outbreak Prediction and Interactive Map")

# Check for required files located alongside this script
BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "sliced_disease_dataset.csv"
state_data_path = BASE_DIR / "new_data.csv"


if not data_path.exists():
    st.error(f"Error: {data_path} not found.")
    st.stop()
if not state_data_path.exists():
    st.error(f"Error: {state_data_path} not found.")
    st.stop()


# Load and clean symptom dataset
@st.cache_data
def load_symptom_data():
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

df = load_symptom_data()
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# Train model
@st.cache_resource
def train_model():
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    for train_idx, test_idx in split.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    rf = RandomForestClassifier(random_state=42)
    params = {
        'criterion': ['gini', 'entropy'],
        'min_samples_split': list(np.arange(2, 31)),
        'min_samples_leaf': list(np.arange(2, 51)),
        'n_estimators': [7]
    }
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    nrf = RandomizedSearchCV(rf, param_distributions=params, cv=10, n_jobs=-1, scoring='accuracy', random_state=20)
    nrf.fit(X_train, y_train_encoded)
    return nrf.best_estimator_, le, X.columns.tolist()

best_rf, le, symptom_index = train_model()

# Load state-wise disease data
@st.cache_data
def load_state_data():
    state_data = pd.read_csv(state_data_path)
    state_data.columns = state_data.columns.str.strip()
    state_data['state'] = state_data['state'].astype(str).str.strip().str.lower()
    disease_cols = state_data.columns[1:]
    state_data[disease_cols] = state_data[disease_cols].fillna(0).astype(int)
    state_disease_map = {
        row['state']: {disease: row[disease] for disease in disease_cols}
        for _, row in state_data.iterrows()
    }
    return state_disease_map, disease_cols.tolist()

state_disease_map, disease_cols = load_state_data()

# Prediction function
def predict_disease_with_state(symptom_input, user_state, model, le, symptom_index, top_n=5, alpha=1.0):
    try:
        symptoms = [s.strip().lower() for s in symptom_input.split(",")]
        input_vector = np.zeros(len(symptom_index))
        for symptom in symptoms:
            if symptom in symptom_index:
                input_vector[symptom_index.index(symptom)] = 1

        probs = model.predict_proba([input_vector])[0]
        diseases = le.inverse_transform(range(len(probs)))

        user_state = user_state.strip().lower()
        active_cases_raw = state_disease_map.get(user_state, {})
        max_cases = max(active_cases_raw.values(), default=1) or 1

        results = []
        for disease, model_prob in zip(diseases, probs):
            cases = active_cases_raw.get(disease, 0)
            case_score = np.log1p(cases) / np.log1p(max_cases) if cases > 0 else 0
            final_score = model_prob * (1 + alpha * case_score)
            results.append((disease, final_score, cases))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return []

# Tree voting function
def show_top_votes(symptom_input, model, le, symptom_index):
    symptom_list = [s.strip().lower() for s in symptom_input.split(',')]
    input_vector = np.zeros((1, len(symptom_index)))
    for symptom in symptom_list:
        if symptom in symptom_index:
            input_vector[0, symptom_index.index(symptom)] = 1
    input_df = pd.DataFrame(input_vector, columns=symptom_index)

    all_votes_raw = [est.predict(input_df)[0] for est in model.estimators_]
    all_votes_named = [le.inverse_transform([int(label)])[0] for label in all_votes_raw]
    vote_counts = Counter(all_votes_named)
    top5 = vote_counts.most_common(5)
    total_trees = len(model.estimators_)

    result = ["**Top 5 Diseases by Tree Votes:**"]
    for disease, votes in top5:
        prob = (votes / total_trees) * 100
        result.append(f"- {disease}: {votes} votes ({prob:.2f}%)")
    return "\n".join(result)

# Create tabs
tab1, tab2 = st.tabs(["Prediction Model", "Interactive Map"])

# Prediction tab
with tab1:
    st.header("Disease Prediction")
    symptom_input = st.text_input("Enter symptoms (comma-separated, e.g., cough, fever)", placeholder="e.g., cough, fever, fatigue")
    state_input = st.text_input("Enter state (e.g., delhi)", placeholder="e.g., punjab")

    if st.button("Predict Disease"):
        if not symptom_input or not state_input:
            st.warning("Please provide both symptoms and state.")
        else:
            results = predict_disease_with_state(
                symptom_input=symptom_input,
                user_state=state_input,
                model=best_rf,
                le=le,
                symptom_index=symptom_index,
                alpha=1.0
            )
            if results:
                st.subheader(f"Predicted Diseases for Symptoms: {symptom_input}")
                st.write(f"**State**: {state_input.capitalize()}")
                for i, (disease, score, cases) in enumerate(results, 1):
                    st.markdown(f"{i}. **{disease}** — Score: {score:.4f} — Active Cases: {cases}")

                st.markdown("---")
                st.markdown(show_top_votes(symptom_input, best_rf, le, symptom_index), unsafe_allow_html=True)
            else:
                st.error("No predictions available. Check input or state data.")

# Map tab
from app2 import render_disease_cluster_map

with tab2:
    render_disease_cluster_map()



# Sidebar
st.sidebar.title("About")
st.sidebar.write("This app predicts diseases based on symptoms and state-wise case data, using a RandomForest model. It also visualizes outbreak zones on an interactive map.")
st.sidebar.markdown("**Data Files**: Ensure `sliced_disease_dataset.csv`, `new_data.csv`, `all_data.csv`, and `india.geojson` sit in the project directory.")
