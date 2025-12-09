# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# ---- Page config ----
st.set_page_config(page_title="Resistance and Circuit Gain Analysis for 3D Circuits", layout="wide")
sns.set_style("whitegrid")

DATA_PATH = "datafiles/Insy6500_Project_Dataset_Updated.xlsx"

# ---- Helpers ----
@st.cache_data(show_spinner=True)
def load_raw(path=DATA_PATH):
    df = pd.read_excel(path)
    return df

def fix_timestamp(df):
    # try several common patterns robustly
    df = df.copy()
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format="%m/%d/%Y %H:%M:%S:%f")
    except Exception:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df

def clean_copy(df_raw):
    df = df_raw.copy()
    df = fix_timestamp(df)
    gain_cols = [
        'Gain (Vout/Vin) 40C','Gain (Vout/Vin) 60C',
        'Gain (Vout/Vin) 85C','Gain (Vout/Vin) 125C'
    ]
    res_cols = [
        'Resistor R1 40C','Resistor R1 60C','Resistor R1 85C','Resistor R1 125C',
        'Resistor R2 85C','Resistor R2 125C'
    ]
    # negative values are removed with NAN
    df[gain_cols] = df[gain_cols].where(df[gain_cols] >= 0, np.nan)
    df[res_cols] = df[res_cols].where(df[res_cols] >= 0, np.nan)
    # NAN values are changed with linear interpolation
    df[gain_cols] = df[gain_cols].interpolate(method='linear', limit_direction='both')
    df[res_cols] = df[res_cols].interpolate(method='linear', limit_direction='both')
    # Outlier data are removed after negative value removal
    for col in res_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR
        df[col] = df[col].where((df[col] >= lower) & (df[col] <= upper))
    for col in gain_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR
        df[col] = df[col].where((df[col] >= lower) & (df[col] <= upper))
    # Again NAN have been replaced with linear interpolation after outlier removal
    df[res_cols] = df[res_cols].interpolate(method='linear', limit_direction='both')
    df[gain_cols] = df[gain_cols].interpolate(method='linear', limit_direction='both')

    # derived columns
    df['hours_since_start'] = (df['timestamp'] - df['timestamp'].min()) / pd.Timedelta(hours=1)
    # Change of R1 from pristine or initial condition
    for col in ['Resistor R1 40C','Resistor R1 60C','Resistor R1 85C','Resistor R1 125C']:
        df[col + "_delta"] = (df[col] - df[col].iloc[0]).abs()
    # Change of R2 from pristine or initial condition
    for col in ['Resistor R2 85C','Resistor R2 125C']:
        df[col + "_delta"] = df[col] - df[col].iloc[0]
    # Change of Gain from pristine or initial condition
    for col in gain_cols:
        df[col + "_delta"] = df[col] - df[col].iloc[0]

    # compute drift rates for R1 deltas (slope of delta vs hours)
    drift_rates = {}
    for col in ['Resistor R1 40C_delta','Resistor R1 60C_delta','Resistor R1 85C_delta','Resistor R1 125C_delta']:
        temp_df = df[['hours_since_start', col]].dropna()
        if len(temp_df) > 1:
            slope = np.polyfit(temp_df['hours_since_start'], temp_df[col], 1)[0]
        else:
            slope = np.nan
        drift_rates[col] = slope

    return df, drift_rates

def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

# ---- Load data ----
st.title("High Temperature Operating Life Analysis of 3D printed Circuits with Resistor and Gain.")
st.markdown("This app reproduces the notebook plots with interactive tabs.")

raw = load_raw()
raw = fix_timestamp(raw)
df, drift_rates = clean_copy(raw)

# Provide download of cleaned data
def get_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# ---- Tabs UI ----
tabs = st.tabs([
    "Raw Data Overview", "Cleaning Steps", "R1 Analysis", "R2 Analysis",
    "Gain Analysis", "Distributions", "Boxplots", "Heatmap", "Drift Trends", "Raw vs Clean"
])

# 1. Raw Data Overview
with tabs[0]:
    st.header("Data Overview")
    st.subheader("Raw dataframe (first few rows)")
    st.dataframe(raw.head())
    st.write("Columns and types:")
    st.write(raw.dtypes)
    st.markdown("### Missing & Negative value summary")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Missing counts")
        st.write(raw.isna().sum())
    with col2:
        st.write("Negative counts (numeric only)")
        st.write((raw._get_numeric_data() < 0).sum())

# 2. Cleaning Steps
with tabs[1]:
    st.header("Cleaning Steps")
    st.markdown("""
    Steps applied:
    1. Parse timestamps robustly.  
    2. Replace negative values in resistor/gain columns with NaN.  
    3. Interpolate (linear) to fill NaNs.  
    4. Remove extreme outliers using IQR*3 and set them to NaN.  
    5. Interpolate again to fill NaNs created by outlier removal.  
    """)
    st.subheader("Download cleaned data")
    st.download_button("Download cleaned CSV", data=get_csv_bytes(df), file_name="cleaned_data.csv", mime="text/csv")
    st.write("Show cleaned data sample:")
    st.dataframe(df.head())

# 3. R1 Analysis
with tabs[2]:
    st.header("R1 — Time series & scatter")
    cols = ['Resistor R1 40C','Resistor R1 60C','Resistor R1 85C','Resistor R1 125C']
    show_raw_toggle = st.checkbox("Overlay raw R1 on plots", value=False)

    # Line plot over time
    fig, ax = plt.subplots(figsize=(12,5))
    for col in cols:
        ax.plot(df['timestamp'], df[col], label=col)
    if show_raw_toggle:
        for col in cols:
            ax.plot(raw['timestamp'], raw[col], linestyle='--', alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Resistance (Ohms)")
    ax.set_title("Resistor R1 Over Time")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # Scatter vs hours
    fig, ax = plt.subplots(figsize=(12,5))
    for col in cols:
        ax.scatter(df['hours_since_start'], df[col], s=10, alpha=0.6, label=col)
    ax.set_xlabel("Hours Since Start")
    ax.set_ylabel("Resistance (Ohms)")
    ax.set_title("Scatter: R1 vs Hours")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# 4. R2 Analysis
with tabs[3]:
    st.header("R2 — Time series & scatter")
    cols_r2 = ['Resistor R2 85C', 'Resistor R2 125C']
    show_raw_r2 = st.checkbox("Overlay raw R2 on plots", value=False)

    fig, ax = plt.subplots(figsize=(12,5))
    for col in cols_r2:
        ax.plot(df['timestamp'], df[col], label=col)
    if show_raw_r2:
        for col in cols_r2:
            ax.plot(raw['timestamp'], raw[col], linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Resistance (Ohms)")
    ax.set_title("Resistor R2 Over Time")
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12,5))
    for col in cols_r2:
        ax.scatter(df['hours_since_start'], df[col], s=10, alpha=0.6, label=col)
    ax.legend()
    ax.set_xlabel("Hours Since Start")
    ax.set_ylabel("Resistance (Ohms)")
    ax.set_title("Scatter: R2 vs Hours")
    st.pyplot(fig)
    plt.close(fig)

# 5. Gain Analysis
with tabs[4]:
    st.header("Gain — Time series & scatter")
    gain_cols = [
        'Gain (Vout/Vin) 40C','Gain (Vout/Vin) 60C',
        'Gain (Vout/Vin) 85C','Gain (Vout/Vin) 125C'
    ]
    fig, ax = plt.subplots(figsize=(12,5))
    for col in gain_cols[:-1]:  # first 3
        ax.plot(df['timestamp'], df[col], label=col)
    ax.set_xlabel("Time")
    ax.set_ylabel("Gain")
    ax.set_title("Gain Over Time (40–85°C)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df['timestamp'], df[gain_cols[-1]], label=gain_cols[-1])
    ax.set_xlabel("Time")
    ax.set_ylabel("Gain")
    ax.set_title("Gain Over Time (125°C)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# 6. Distributions
with tabs[5]:
    st.header("Distributions (Histograms)")
    st.write("Resistor R1 distributions:")
    for col in ['Resistor R1 40C', 'Resistor R1 60C', 'Resistor R1 85C', 'Resistor R1 125C']:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)
        plt.close(fig)

    st.write("Resistor R2 distributions:")
    for col in ['Resistor R2 85C', 'Resistor R2 125C']:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)
        plt.close(fig)

    st.write("Gain distributions for different temperatures:")
    fig, axs = plt.subplots(2, 2, figsize=(12,8))
    for ax, col in zip(axs.flatten(), gain_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# 7. Boxplots
with tabs[6]:
    st.header("Boxplots")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(data=df[['Resistor R1 40C','Resistor R1 60C','Resistor R1 85C','Resistor R1 125C']], ax=ax)
    ax.set_title("Resistor R1 Values Across Temperatures")
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,4))
    sns.boxplot(data=df[['Resistor R2 85C','Resistor R2 125C']], ax=ax)
    ax.set_title("Resistor R2 Values Across Temperatures")
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(data=df[['Gain (Vout/Vin) 40C','Gain (Vout/Vin) 60C','Gain (Vout/Vin) 85C']], ax=ax)
    ax.set_title("Gain (Vout/Vin) Distribution Across Temperatures (40–85°C)")
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,4))
    sns.boxplot(data=df[['Gain (Vout/Vin) 125C']], ax=ax)
    ax.set_title("Gain (Vout/Vin) Distribution At 125°C")
    st.pyplot(fig)
    plt.close(fig)

# 8. Heatmap
with tabs[7]:
    st.header("Correlation Heatmap (Resistors)")
    res_cols_corr = [c for c in df.columns if "Resistor R1" in c or "Resistor R2" in c]
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df[res_cols_corr].corr(), annot=True, ax=ax, cmap="coolwarm")
    ax.set_title("Resistor Correlation Heatmap")
    st.pyplot(fig)
    plt.close(fig)

# 9. Drift Trends
with tabs[8]:
    st.header("Drift Trends & Drift Rate vs Temperature")
    # plot linear trendlines of delta (R1)
    fig, ax = plt.subplots(figsize=(10,5))
    for col in ['Resistor R1 40C_delta','Resistor R1 60C_delta','Resistor R1 85C_delta','Resistor R1 125C_delta']:
        sns.regplot(x=df['hours_since_start'], y=df[col], scatter=False, ax=ax, label=col)
    ax.legend()
    ax.set_xlabel("Hours Since Start")
    ax.set_ylabel("ΔR (abs change)")
    ax.set_title("Linear Trend of Drift for R1 at All Temperatures")
    st.pyplot(fig)
    plt.close(fig)

    # R2 drift trend
    fig, ax = plt.subplots(figsize=(10,5))
    for col in ['Resistor R2 85C_delta','Resistor R2 125C_delta']:
        sns.regplot(x=df['hours_since_start'], y=df[col], scatter=False, ax=ax, label=col)
    ax.legend()
    ax.set_xlabel("Hours Since Start")
    ax.set_ylabel("ΔR")
    ax.set_title("Linear Trend of Drift for R2")
    st.pyplot(fig)
    plt.close(fig)

    # Gain drift trends
    fig, ax = plt.subplots(figsize=(10,5))
    gain_delta_cols = [c for c in df.columns if "_delta" in c and "Gain" in c]
    for col in gain_delta_cols:
        sns.regplot(x=df['hours_since_start'], y=df[col], scatter=False, ax=ax, label=col)
    ax.legend()
    ax.set_xlabel("Hours Since Start")
    ax.set_ylabel("ΔGain")
    ax.set_title("Linear Drift Trend of Gain Across Temperatures")
    st.pyplot(fig)
    plt.close(fig)

    # Drift rate vs temp scatter
    st.subheader("Drift rate (slope) for R1 deltas")
    st.write(drift_rates)
    # Convert and plot summary
    temps = [40, 60, 85, 125]
    rates = [drift_rates.get(f"Resistor R1 {t}C_delta", np.nan) for t in temps]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(temps, rates)
    ax.plot(temps, rates, marker='o')
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Drift Rate (Ohm per hour)")
    ax.set_title("Drift Rate vs Temperature")
    st.pyplot(fig)
    plt.close(fig)

# 10. Raw vs Clean Comparison
with tabs[9]:
    st.header("Raw vs Clean Comparison")
    st.markdown("Select a variable to compare raw (dashed) vs cleaned (solid).")
    all_compare_cols = [c for c in df.columns if any(k in c for k in ['Resistor', 'Gain (Vout/Vin)'] ) and '_delta' not in c]
    choice = st.selectbox("Column to compare", all_compare_cols, index=0)
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df['timestamp'], df[choice], label='Cleaned', alpha=0.9)
    ax.plot(raw['timestamp'], raw[choice], label='Raw', linestyle='--', alpha=0.5)
    ax.set_title(f"Effect of Cleaning on {choice}")
    ax.set_xlabel("Time")
    ax.set_ylabel(choice)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")
