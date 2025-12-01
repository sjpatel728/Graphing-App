import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="Distribution Fitting Tool", layout="wide")
st.title("Distribution Fitting Tool")

# --- Data input ---
st.sidebar.header("Data Input")
input_method = st.sidebar.radio("Select data input method:", ["Manual Entry", "Upload CSV"])

data = None
if input_method == "Manual Entry":
    data_text = st.sidebar.text_area("Enter data (comma-separated):", "1,2,3,4,5,6")
    try:
        data = np.array([float(x.strip()) for x in data_text.split(",")])
    except:
        st.sidebar.error("Invalid input. Use numbers separated by commas.")
elif input_method == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] == 1:
            data = df.iloc[:,0].values
        else:
            st.sidebar.error("CSV must have only one column of numeric data.")

if data is not None:
    st.sidebar.success("Data loaded successfully!")

# --- Distribution selection ---
distributions = {
    "Normal": stats.norm,
    "Exponential": stats.expon,
    "Gamma": stats.gamma,
    "Weibull": stats.weibull_min,
    "Lognormal": stats.lognorm,
    "Beta": stats.beta,
    "Uniform": stats.uniform,
    "Chi-squared": stats.chi2,
    "Laplace": stats.laplace,
    "Cauchy": stats.cauchy
}

dist_name = st.sidebar.selectbox("Select distribution to fit:", list(distributions.keys()))
dist = distributions[dist_name]

# --- Main layout ---
if data is not None:
    x = np.linspace(min(data), max(data), 200)

    # Automatic fit
    params = dist.fit(data)
    pdf_fitted = dist.pdf(x, *params)
    mse = np.mean((np.histogram(data, bins=20, density=True)[0] - 
                   dist.pdf(np.histogram(data, bins=20, density=True)[1][:-1], *params))**2)

    tab1, tab2 = st.tabs(["Automatic Fit", "Manual Fit"])

    # --- Tab 1: Automatic Fit ---
    with tab1:
        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader("Histogram + Fitted Curve")
            fig, ax = plt.subplots(figsize=(6,4))
            ax.hist(data, bins=20, density=True, alpha=0.6, color='g')
            ax.plot(x, pdf_fitted, 'r-', lw=2)
            st.pyplot(fig)
        with col2:
            st.subheader("Fitted Parameters")
            st.write(params)
            st.write(f"Mean Squared Error: {mse:.5f}")

    # --- Tab 2: Manual Fit ---
    with tab2:
        st.subheader("Manual Adjustment")
        with st.expander("Adjust parameters"):
            sliders = []
            for i, p in enumerate(params):
                sliders.append(st.slider(f"Parameter {i}", float(p*0.5), float(p*1.5), float(p)))
        pdf_manual = dist.pdf(x, *sliders)
        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader("Histogram + Manual Curve")
            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.hist(data, bins=20, density=True, alpha=0.6, color='g')
            ax2.plot(x, pdf_manual, 'b-', lw=2)
            st.pyplot(fig2)
        with col2:
            st.subheader("Manual Parameters")
            for i, s in enumerate(sliders):
                st.write(f"Parameter {i}: {s:.4f}")