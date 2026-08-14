import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import matplotlib.pyplot as plt
from JCC_module import *
import plotly.figure_factory as ff
import plotly.graph_objects as go

#Set the title
st.set_page_config(page_title="Model Dashboard", layout="wide")

#cache used to load the models once and re-use 

#-------------------------------------------------------------------------------------------------------
#load objects of interest 
#-------------------------------------------------------------------------------------------------------


@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load(r"Original_468_files\Saved_models_Original\lr_l2_weights_model.pkl"),
        "Xg boost": joblib.load(r"Original_468_files\Saved_models_Original\best_XGB_sampler_model.pkl"),
        "Neural Network": joblib.load("best_nn_model.pkl"),
        "Scorecard logistic regression": joblib.load("scorecard.pkl")
    }

@st.cache_resource
def load_scorecard():
    return joblib.load("scorecard.pkl")  

@st.cache_data
def load_test_data():
    X_test = pd.read_csv("X_test_fe.csv")
    y_test = pd.read_csv("y_test.csv").squeeze()
    return X_test, y_test

models = load_models()
scorecard = load_scorecard()
X_test, y_test = load_test_data()

#Classes in the scorecard estimator
print(scorecard.estimator_.classes_)


#-------------------------------------------------------------------------------------------------------
#Sidebar section
#-------------------------------------------------------------------------------------------------------
st.sidebar.title("Controls")
model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))
model = models[model_name]

#Radio allows to switch between dashboard modes
mode = st.sidebar.radio("Mode", ["Evaluation", "Predict"])

st.sidebar.markdown("---")
st.sidebar.caption(f"Active model: **{model_name}**")




#-------------------------------------------------------------------------------------------------------
#Main area compnent design
#-------------------------------------------------------------------------------------------------------

st.title(f"{model_name if mode == 'Evaluation' else 'Scorecard'} — {mode}")



if mode == "Evaluation":
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    if model_name == "Scorecard logistic regression":
        # scorecard was trained on flipped labels where 1 is default and 0 is safe
        # All other models follow the class 0 is default danger class and 0 safe
        y_eval = (y_test == 0).astype(int)
        y_pred_eval = y_pred
        labels = ["Safe area (0)", "Default area (1)"]
    else:
        y_eval = y_test
        y_pred_eval = y_pred
        labels = [str(c) for c in sorted(y_test.unique())]

    report = classification_report(y_eval, y_pred_eval, output_dict=True)
    cm = confusion_matrix(y_eval, y_pred_eval)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{report['accuracy']:.3f}")
    col2.metric("Class 0 Precision", f"{report['0']['precision']:.3f}")
    col3.metric("Class 0 Recall", f"{report['0']['recall']:.3f}")
    col4.metric("Class 0 F1", f"{report['0']['f1-score']:.3f}")

    st.subheader("Confusion Matrix")
    fig = ff.create_annotated_heatmap(
        z=cm, x=labels, y=labels, colorscale="Blues", showscale=True,
    )
    fig.update_layout(
        xaxis_title="Predicted", yaxis_title="Actual",
        yaxis_autorange="reversed",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

    if y_proba is not None:
        try:
            class_0_index = list(model.classes_).index(1)
            probs = model.predict_proba(X_test)[:, class_0_index]
            fpr, tpr, _ = roc_curve(y_eval, probs if not model_name == "Scorecard logistic regression" else 1 - probs)
            roc_auc = auc(fpr, tpr)

            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"AUC (class 0) = {roc_auc:.3f}")
            ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.legend()
            st.pyplot(fig2)
        except Exception as e:
            st.warning(f"ROC curve unavailable: {e}")

else:  # Predict 
    st.caption("Predictions are generated from the credit scorecard (raw inputs, no manual encoding needed).")
    st.caption("The column chosen are the top 10 most influential variables from the logistic regression model")

    RAW_NUMERIC = [
        "total_rec_prncp", "last_fico_range_low", "out_prncp",
        "installment", "int_rate", "inq_last_6mths", "inq_last_12m",
    ]
    RAW_CATEGORICAL_OPTIONS = {
        "verification_status": ["Not Verified", "Source Verified", "Verified"],
        "home_ownership": ["RENT", "OWN", "MORTGAGE", "ANY"],
    }
    SCORECARD_COLUMNS = RAW_NUMERIC + list(RAW_CATEGORICAL_OPTIONS.keys())

    threshold = st.sidebar.slider("Bad-risk threshold", 0.0, 1.0, 0.50, 0.01)

    raw_input = {}
    cols = st.columns(3)

    for i, feat in enumerate(RAW_NUMERIC):
        #Creates the remainder index values 0,1,2 for the 3 columns
        with cols[i % 3]:
            default_val = float(X_test[feat].mean()) if feat in X_test.columns else 0.0
            raw_input[feat] = st.number_input(feat, value=default_val)

    for i, (feat, options) in enumerate(RAW_CATEGORICAL_OPTIONS.items()):
        #To pick up where the numeric column was left off
        with cols[(len(RAW_NUMERIC) + i) % 3]:
            raw_input[feat] = st.selectbox(feat, options)

    if st.button("Predict"):
        input_df = pd.DataFrame([raw_input], columns=SCORECARD_COLUMNS)

        proba_bad = scorecard.predict_proba(input_df)[:, 1][0]
        pred_bad = int(proba_bad >= threshold)

        label = "high risk" if pred_bad == 1 else "low risk"
        st.success(f"Prediction: **{label}**")
        st.metric("Probability of default (bad)", f"{proba_bad:.1%}")

        st.write("Class probabilities:")
        st.bar_chart(pd.DataFrame({
            "Probability": [proba_bad, 1 - proba_bad]
        }, index=["Bad", "Good"]))

    y_pred = model.predict(X_test)
