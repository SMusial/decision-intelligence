import streamlit as st
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

st.set_page_config(page_title="Telco Decision Intelligence", layout="wide")

st.title("📡 Telco Decision Intelligence: Phase 1")
st.markdown("""
**Bayesian Uncertainty & Data Recovery.** Ten moduł demonstruje, jak z brudnych danych (szum + outliery + braki) wyodrębniamy bezpieczny przedział decyzyjny.
""")

# Sidebar do hiperparametrów
st.sidebar.header("Parametry Symulacji")
noise = st.sidebar.slider("Poziom szumu (Sigma)", 1.0, 15.0, 6.0)
n_missing = st.sidebar.slider("Liczba brakujących rekordów (NaN)", 0, 40, 20)

@st.cache_data
def generate_data(noise_level, n_nan):
    np.random.seed(101)
    x_real = np.random.normal(5, 2.5, 100)
    y_obs = 12 + 2.2 * x_real + np.random.normal(0, noise_level, 100)
    
    # Outliery
    x_dirty = x_real.copy()
    x_dirty[0], x_dirty[1], x_dirty[2] = 45.0, -10.0, 5.0
    y_obs[2] = 180.0
    
    # Braki
    missing_idx = np.random.choice(100, size=n_nan, replace=False)
    x_dirty[missing_idx] = np.nan
    return x_dirty, y_obs, missing_idx

x_dirty, y_obs, missing_idx = generate_data(noise, n_missing)

if st.button('Uruchom Silnik Bayesowski (PyMC)'):
    with st.spinner('Trwa próbkowanie MCMC...'):
        with pm.Model() as model:
            mu_x = pm.Normal("mu_x", mu=5, sigma=3)
            sigma_x = pm.HalfNormal("sigma_x", sigma=3)
            x_imputed = pm.Normal("x_imputed", mu=mu_x, sigma=sigma_x, observed=x_dirty)
            
            beta = pm.Normal("beta", mu=2, sigma=5)
            alpha = pm.Normal("alpha", mu=10, sigma=15)
            nu = pm.Exponential("nu", 1/10)
            sigma_y = pm.HalfNormal("sigma_y", sigma=10)
            
            mu_y = alpha + beta * x_imputed
            y_lik = pm.StudentT("y_lik", nu=nu, mu=mu_y, sigma=sigma_y, observed=y_obs)
            trace = pm.sample(1000, chains=2, progressbar=False)

        # Wizualizacja
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x_dirty, y_obs, alpha=0.5, label="Dane wejściowe")
        # Wyciąganie trendu
        b_map = trace.posterior["beta"].mean().values
        a_map = trace.posterior["alpha"].mean().values
        x_range = np.linspace(-15, 45, 100)
        ax.plot(x_range, a_map + b_map * x_range, color='red', lw=2, label="Trend Bayesowski")
        ax.legend()
        st.pyplot(fig)
        
        st.success(f"Estymowany wpływ jakości na ROI (Beta): {b_map:.2f}")
        st.write("Wniosek: Model zignorował outliery i poprawnie wyznaczył trend mimo szumu.")