import hashlib

# =========================

# 🔥 PERMANENT FIX FOR REPORTLAB + PYTHON 3.12 OPENSSL ISSUE

# =========================

_original_md5 = hashlib.md5

def md5_patch(*args, **kwargs):
   kwargs.pop("usedforsecurity", None)
   return _original_md5(*args, **kwargs)

hashlib.md5 = md5_patch

# =========================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import shap
import time
from io import BytesIO
from datetime import datetime
import os
import shutil

# =========================

# 🔥 NEW IMPORTS FOR AUTO DATASET DISCOVERY

# =========================

import openml
from kaggle.api.kaggle_api_extended import KaggleApi
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
RandomForestClassifier, RandomForestRegressor,
GradientBoostingClassifier, GradientBoostingRegressor,
ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor

st.title("Intelligent AutoML & Algorithm Advisory Framework")

# =========================

# 🔥 DATASET DISCOVERY FUNCTIONS

# =========================

def extract_keywords(problem):
    problem = problem.replace(",", " ")
    problem = problem.lower()

    remove_words = [
        "predict","prediction","analysis","classification",
        "model","machine","learning","based","using",
        "build","system","detect","estimate"
    ]

    words = problem.split()

    keywords = [w for w in words if w not in remove_words]

    # fallback if empty
    if len(keywords) == 0:
        keywords = words

    return keywords

def detect_domain(problem_text):

    text = problem_text.lower()

    if any(word in text for word in ["stock","bank","finance","loan","credit","fraud","price"]):
        return "Finance"

    elif any(word in text for word in ["disease","medical","health","patient","diagnosis"]):
        return "Healthcare"

    elif any(word in text for word in ["latitude","longitude","gps","geo","location","map"]):
        return "Geography"

    elif any(word in text for word in ["sales","customer","product","retail","marketing"]):
        return "Retail"

    elif any(word in text for word in ["school","student","education","exam","grades"]):
        return "Education"

    elif any(word in text for word in ["crop","agriculture","farm","soil"]):
        return "Agriculture"

    elif any(word in text for word in ["traffic","vehicle","transport","road"]):
        return "Transportation"

    elif any(word in text for word in ["weather","climate","temperature","environment"]):
        return "Environment"

    return "General"


def search_openml(problem):

    try:
        keywords = extract_keywords(problem)

        datasets = openml.datasets.list_datasets(output_format="dataframe")

        for word in keywords:

            match = datasets[
                datasets['name'].str.contains(word, case=False, na=False)
            ]

            if len(match) > 0:

                match = match.sort_values(by="NumberOfDownloads", ascending=False)

                dataset_id = match.iloc[0]['did']

                dataset = openml.datasets.get_dataset(dataset_id)

                X, y, _, _ = dataset.get_data()

                df = pd.concat([X, y], axis=1)

                return df

    except:
        pass

    return None

def search_kaggle(problem):

    try:
        api = KaggleApi()
        api.authenticate()

        keywords = extract_keywords(problem)

        path = "datasets"

        # clear old datasets safely
        if os.path.exists(path):
            for f in os.listdir(path):
                full_path = os.path.join(path, f)

                try:
                    if os.path.isfile(full_path):
                        os.remove(full_path)
                    elif os.path.isdir(full_path):
                        shutil.rmtree(full_path)
                except Exception:
                    pass

        os.makedirs(path, exist_ok=True)

        for keyword in keywords:

            datasets = api.dataset_list(search=problem)

            for d in datasets[:3]:

                dataset_ref = d.ref

                api.dataset_download_files(dataset_ref, path=path, unzip=True)

                # search for CSV files recursively
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith(".csv"):
                            return pd.read_csv(os.path.join(root, file))

    except Exception as e:
        st.error(f"Kaggle error: {e}")

    return None

def auto_find_dataset(problem):

    # Try OpenML first
    df = search_openml(problem)

    if df is not None:
        return df, "OpenML"

    # If OpenML fails, try Kaggle
    df = search_kaggle(problem)

    if df is not None:
        return df, "Kaggle"

    # If both fail, return nothing
    return None, None

def detect_target_column(df):

    possible_targets = [
        "target", "label", "class", "output", "result",
        "price", "salary", "income", "churn",
        "disease", "diagnosis", "status", "loan_status"
    ]

    for col in df.columns:
        if col.lower() in possible_targets:
            return col

    for col in df.columns:
        if df[col].nunique() < len(df) * 0.1:
            return col

    return df.columns[-1]
# =========================

# SAFE SESSION INITIALIZATION

# =========================

if "dataset_loaded" not in st.session_state:
    st.session_state["dataset_loaded"] = False

if "dataset_problem_type" not in st.session_state:
    st.session_state["dataset_problem_type"] = None

if "auto_generated_dataset" not in st.session_state:
    st.session_state["auto_generated_dataset"] = None

if "auto_train" not in st.session_state:
    st.session_state["auto_train"] = False

if "training_in_progress" not in st.session_state:
    st.session_state["training_in_progress"] = False

if "training_completed" not in st.session_state:
    st.session_state["training_completed"] = False

if "stored_results" not in st.session_state:
    st.session_state["stored_results"] = None

if "stored_best_model" not in st.session_state:
    st.session_state["stored_best_model"] = None

if "stored_metric_name" not in st.session_state:
    st.session_state["stored_metric_name"] = None

# =========================

# AUTO TAB SWITCH SYSTEM

# =========================

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Dataset AutoML"

selected_tab = st.radio(
"Navigation",
["Dataset AutoML", "Algorithm Advisory"],
index=0 if st.session_state["active_tab"] == "Dataset AutoML" else 1,
horizontal=True
)

st.session_state["active_tab"] = selected_tab

# =========================
# TAB 1 – DATASET AUTO ML
# =========================

if st.session_state["active_tab"] == "Dataset AutoML":

    st.header("Upload Dataset for AutoML")

    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    df = None

    if st.session_state["auto_generated_dataset"] is not None:
        df = st.session_state["auto_generated_dataset"]

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

    if df is not None:

        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        # limit dataset size for faster AutoML
        if len(df) > 3000:
            df = df.sample(3000, random_state=42)
            st.info("Large dataset detected. Using 3000 samples for faster AutoML.")

        st.subheader("Basic Dataset Info")
        st.write("Shape:", df.shape)

        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

        st.success("Missing values handled automatically")

        target_column = st.selectbox("Select Target Column", df.columns)

        if target_column:

            y = df[target_column]

            if y.dtype == 'object':
                problem_type = "Classification"
            elif y.dtype in ['int64', 'float64'] and y.nunique() <= 5:
                problem_type = "Classification"
            else:
                problem_type = "Regression"

            st.success(f"Detected Problem Type: {problem_type}")
            # =========================
            # SMART HYBRID AUTO ML ADVISORY
            # =========================

            num_rows = df.shape[0]
            num_features = df.shape[1] - 1

            if problem_type == "Classification":

                if num_rows < 2000 and num_features < 10:
                    recommended_algo = "Logistic Regression"
                    reason = "Small dataset with few features."

                elif num_features > 20:
                    recommended_algo = "Extra Trees Classifier"
                    reason = "Handles high dimensional data efficiently."

                else:
                    recommended_algo = "Random Forest Classifier"
                    reason = "Strong general purpose classifier."

            else:

                if num_rows < 2000:
                    recommended_algo = "Linear Regression"
                    reason = "Works well for smaller regression datasets."

                elif num_features > 20:
                    recommended_algo = "Gradient Boosting Regressor"
                    reason = "Handles complex nonlinear relationships."

                else:
                    recommended_algo = "Random Forest Regressor"
                    reason = "Robust model for most regression problems."

            st.markdown("## ⚡ Instant Algorithm Recommendation")
            st.success(f"Recommended Algorithm: {recommended_algo}")
            st.info(reason)

            st.session_state["dataset_loaded"] = True
            st.session_state["dataset_problem_type"] = problem_type

            X = df.drop(columns=[target_column])
            y = df[target_column]

            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

            if problem_type == "Classification" and y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            results = {}
            trained_models = {}

            auto_trigger = st.session_state.get("auto_train", False)
            manual_trigger = st.button("Start Training")

            if auto_trigger or manual_trigger:
                st.session_state["auto_train"] = False
                st.session_state["training_in_progress"] = True
                st.session_state["training_completed"] = False

            if st.session_state["training_in_progress"]:

                with st.spinner("🚀 Training Multiple ML Models..."):
                    progress_bar = st.progress(0)

                    if problem_type == "Classification":

                        models = {
                            "Logistic Regression": (
                                Pipeline([
                                    ("scaler", StandardScaler()),
                                    ("model", LogisticRegression(max_iter=1000))
                                ]),
                                {}
                            ),

                            "Random Forest": (
                                RandomForestClassifier(n_estimators=100, n_jobs=-1),
                                {}
                            ),

                            "KNN": (
                                Pipeline([
                                    ("scaler", StandardScaler()),
                                    ("model", KNeighborsClassifier())
                                ]),
                                {}
                            ),

                            "Naive Bayes": (
                                GaussianNB(),
                                {}
                            )
                        }

                        metric_name = "Accuracy"

                    else:

                        models = {
                            "Linear Regression": (
                                Pipeline([
                                    ("scaler", StandardScaler()),
                                    ("model", LinearRegression())
                                ]),
                                {}
                            ),

                            "Random Forest Regressor": (
                                RandomForestRegressor(n_estimators=100, n_jobs=-1),
                                {}
                            ),

                            "KNN Regressor": (
                                Pipeline([
                                    ("scaler", StandardScaler()),
                                    ("model", KNeighborsRegressor())
                                ]),
                                {}
                            )
                        }

                        metric_name = "R² Score"


                    total_models = len(models)

                    for i, (name, (model, params)) in enumerate(models.items()):

                        try:

                            best_model = model
                            best_model.fit(X_train, y_train)

                            preds = best_model.predict(X_test)

                            if problem_type == "Classification":
                                score = accuracy_score(y_test, preds)
                            else:
                                score = r2_score(y_test, preds)

                            results[name] = score
                            trained_models[name] = best_model

                        except Exception as e:
                            st.warning(f"{name} failed: {e}")
                            continue

                        progress_bar.progress((i + 1) / total_models)

                best_model_name = max(results, key=results.get)
                best_model = trained_models[best_model_name]

                st.session_state["training_in_progress"] = False
                st.session_state["training_completed"] = True
                st.session_state["stored_results"] = results
                st.session_state["stored_best_model"] = best_model
                st.session_state["stored_best_model_name"] = best_model_name
                st.session_state["stored_metric_name"] = metric_name

            if st.session_state["training_completed"]:

                results = st.session_state["stored_results"]
                best_model = st.session_state["stored_best_model"]
                metric_name = st.session_state["stored_metric_name"]

                best_model_name = max(results, key=results.get)

                st.markdown("## 📊 Model Leaderboard")

                leaderboard_df = pd.DataFrame({
                    "Model": list(results.keys()),
                    metric_name: list(results.values())
                }).sort_values(by=metric_name, ascending=False)

                st.dataframe(leaderboard_df, use_container_width=True)

                st.markdown("## 🏆 Best Model Selected")
                st.success(f"{best_model_name} ({metric_name}: {results[best_model_name]:.4f})")

# =========================
# SHOW DOWNLOAD OPTIONS ONLY AFTER TRAINING
# =========================

if st.session_state.get("training_completed", False):

    best_model = st.session_state["stored_best_model"]
    best_model_name = st.session_state["stored_best_model_name"]
    metric_name = st.session_state["stored_metric_name"]
    results = st.session_state["stored_results"]

    score = results.get(best_model_name, 0)

    # MODEL DOWNLOAD
    model_bytes = BytesIO()
    pickle.dump(best_model, model_bytes)
    model_bytes.seek(0)

    st.download_button(
        label="Download Best Model (.pkl)",
        data=model_bytes.getvalue(),
        file_name="best_model.pkl",
        mime="application/octet-stream"
    )

    # PDF GENERATION
    pdf_buffer = BytesIO()

    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("AutoML Report", styles['Title']))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(f"Best Model: {best_model_name}", styles['Normal']))
    elements.append(Paragraph(f"Metric: {metric_name}", styles['Normal']))
    elements.append(Paragraph(f"Score: {score}", styles['Normal']))

    doc = SimpleDocTemplate(pdf_buffer)
    doc.build(elements)

    pdf_buffer.seek(0)

    st.download_button(
        label="Download AutoML Report (PDF)",
        data=pdf_buffer,
        file_name="automl_report.pdf",
        mime="application/pdf"
    )

# =========================
# TAB 2 – ALGORITHM ADVISORY
# =========================

if st.session_state["active_tab"] == "Algorithm Advisory":

    st.header("Smart Advisory Mode (Full Auto Pipeline)")

    # Problem description
    problem_description = st.text_area("Describe your Machine Learning Problem")

    # Optional keyword
    dataset_keyword = st.text_input("Dataset Keyword (Optional)")

    # Analyze button
    if st.button("Analyze Problem"):

        # Validate input
        if not problem_description and not dataset_keyword:
            st.error("Please enter a problem description or keyword.")
            st.stop()

        # Build search query
        combined_text = f"{problem_description} {dataset_keyword}"

        dataset_domain = detect_domain(combined_text)

        st.info(f"Detected Domain: {dataset_domain}")

        search_query = f"{dataset_domain} {combined_text}"

        st.info("Searching datasets from OpenML / Kaggle...")

        # Find dataset
        df_real, source = auto_find_dataset(search_query)

        if df_real is not None:

            st.success(f"Dataset discovered from {source}")
            st.dataframe(df_real.head())

            # Store dataset and trigger AutoML
            st.session_state["auto_generated_dataset"] = df_real
            st.session_state["auto_train"] = True
            st.session_state["active_tab"] = "Dataset AutoML"

            st.rerun()

        else:
            st.error("No dataset found for this problem.")