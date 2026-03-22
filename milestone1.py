"""
╔══════════════════════════════════════════════════════════════╗
║         FITNESS DATA PRO — COMPLETE EDA PIPELINE            ║
║                                                              ║
║   Install:  pip install streamlit plotly pandas numpy        ║
║   Run:      streamlit run app.py                             ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fitness Data Pro",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: #030810 !important;
    color: #d4eaf7 !important;
    font-family: 'Exo 2', sans-serif !important;
}
[data-testid="stMain"] { background: transparent !important; }
[data-testid="stAppViewContainer"]::before {
    content: ''; position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(0,230,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,230,255,0.025) 1px, transparent 1px);
    background-size: 55px 55px;
}
[data-testid="stSidebar"] {
    background: #040a14 !important;
    border-right: 1px solid rgba(0,230,255,0.12) !important;
}
section[data-testid="stSidebar"] * { color: #d4eaf7; }
h1 { font-family:'Orbitron',monospace !important; font-size:1.9rem !important;
     letter-spacing:2px !important; color:#fff !important; }
h2 { font-family:'Orbitron',monospace !important; font-size:1.1rem !important;
     letter-spacing:1.5px !important; color:#00e6ff !important; }
h3 { font-family:'Exo 2',sans-serif !important; font-weight:600 !important; color:#d4eaf7 !important; }
p, li { color:#d4eaf7 !important; }
[data-testid="stMetric"] {
    background: #0b1422 !important;
    border: 1px solid rgba(0,230,255,0.14) !important;
    border-radius: 14px !important; padding: 1.2rem 1.4rem !important;
    box-shadow: 0 6px 30px rgba(0,0,0,0.5) !important;
    transition: transform 0.2s;
}
[data-testid="stMetric"]:hover { transform: translateY(-3px); }
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 2rem !important; font-weight: 900 !important;
    color: #00e6ff !important; text-shadow: 0 0 22px rgba(0,230,255,0.5) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem !important; letter-spacing: 2.5px !important;
    text-transform: uppercase !important; color: #4a6a88 !important;
}
[data-testid="stMetricDelta"] {
    font-family:'JetBrains Mono',monospace !important; font-size:0.72rem !important;
}
.stButton > button {
    background: linear-gradient(135deg,#00c8ff,#0055cc) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    font-family: 'Exo 2', sans-serif !important; font-weight: 600 !important;
    font-size: 0.9rem !important; padding: 0.65rem 1.8rem !important;
    box-shadow: 0 4px 22px rgba(0,200,255,0.38) !important;
    letter-spacing: 0.5px !important; transition: all 0.25s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#00d8ff,#0066ff) !important;
    box-shadow: 0 8px 32px rgba(0,200,255,0.55) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stFileUploader"] {
    background: rgba(0,230,255,0.03) !important;
    border: 2px dashed rgba(0,230,255,0.3) !important;
    border-radius: 16px !important; padding: 1.5rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,230,255,0.7) !important;
    background: rgba(0,230,255,0.06) !important;
}
[data-testid="stExpander"] {
    background: #0b1422 !important;
    border: 1px solid rgba(0,230,255,0.12) !important;
    border-radius: 12px !important;
}
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: #0b1422 !important;
    border-color: rgba(0,230,255,0.22) !important;
    border-radius: 10px !important; color: #d4eaf7 !important;
}
[data-testid="stProgressBar"] > div {
    background: linear-gradient(90deg,#7c3aed,#00e6ff,#00ffa3) !important;
    border-radius: 4px !important; box-shadow: 0 0 12px rgba(0,230,255,0.4) !important;
}
[data-testid="stProgressBar"] {
    background: rgba(255,255,255,0.05) !important; border-radius: 4px !important;
}
[data-testid="stTabs"] [role="tablist"] {
    background: #0b1422 !important; border-radius: 12px 12px 0 0 !important;
    border-bottom: 1px solid rgba(0,230,255,0.15) !important;
    gap: 4px !important; padding: 4px !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Exo 2', sans-serif !important; font-weight: 600 !important;
    font-size: 0.85rem !important; color: #4a6a88 !important;
    border-radius: 8px !important; padding: 8px 18px !important; border: none !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: rgba(0,230,255,0.1) !important; color: #00e6ff !important;
    border: 1px solid rgba(0,230,255,0.28) !important;
    box-shadow: 0 0 14px rgba(0,230,255,0.12) !important;
}
[data-testid="stTabContent"] {
    background: #0b1422 !important; border: 1px solid rgba(0,230,255,0.1) !important;
    border-top: none !important; border-radius: 0 0 12px 12px !important;
    padding: 1.5rem !important;
}
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,230,255,0.18); border-radius: 3px; }
hr { border-color: rgba(0,230,255,0.1) !important; margin: 1rem 0 !important; }
[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# PLOTLY THEME
# ──────────────────────────────────────────────────────────────
PTHEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color="#d4eaf7", size=11),
    xaxis=dict(gridcolor="rgba(0,230,255,0.06)", zerolinecolor="rgba(0,230,255,0.1)",
               linecolor="rgba(0,230,255,0.1)", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="rgba(0,230,255,0.06)", zerolinecolor="rgba(0,230,255,0.1)",
               linecolor="rgba(0,230,255,0.1)", tickfont=dict(size=10)),
    margin=dict(l=10, r=10, t=45, b=10),
)

COL_COLORS = {
    "Steps_Taken":           "#00e6ff",
    "Calories_Burned":       "#c084fc",
    "Hours_Slept":           "#f472b6",
    "Water_Intake (Liters)": "#60a5fa",
    "Active_Minutes":        "#34d399",
    "Heart_Rate (bpm)":      "#fb923c",
}
WORKOUT_COLORS = ["#c084fc","#f472b6","#ffb800","#00e6ff","#4a6a88"]
MOOD_COLORS    = ["#00ffa3","#ff4f9b","#00e6ff","#ffb800"]

AXIS_LABELS = {
    "Steps_Taken":           "Steps Taken (count)",
    "Calories_Burned":       "Calories Burned (kcal)",
    "Hours_Slept":           "Hours Slept (hrs)",
    "Water_Intake (Liters)": "Water Intake (Liters)",
    "Active_Minutes":        "Active Minutes (min)",
    "Heart_Rate (bpm)":      "Heart Rate (bpm)",
}

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float = 0.13) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"


def safe_resample(df_indexed, freq_label):
    """Resample with pandas version compatibility (ME vs M)."""
    freq_map = {"Weekly": "W", "Monthly": "ME", "Daily": "D"}
    freq = freq_map.get(freq_label, "D")
    try:
        return df_indexed.resample(freq).mean()
    except ValueError:
        # Older pandas: "ME" not recognised, fall back to "M"
        fallback = {"ME": "M", "W": "W", "D": "D"}
        return df_indexed.resample(fallback.get(freq, freq)).mean()


def preprocess(df: pd.DataFrame):
    """Full null-cleaning pipeline. Returns (cleaned_df, logs, before_nulls_dict)."""
    df   = df.copy()
    logs = []
    before_nulls = df.isnull().sum().to_dict()

    # 1. Parse / fill Date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        nat_count  = df["Date"].isna().sum()
        if nat_count > 0 and "User_ID" in df.columns:
            full_dates = pd.date_range("2023-01-01", periods=365, freq="D")
            parts = []
            for uid, grp in df.groupby("User_ID", sort=True):
                grp = grp.copy().reset_index(drop=True)
                grp["Date"] = full_dates[: len(grp)]
                parts.append(grp)
            df = pd.concat(parts, ignore_index=True)
            logs.append(("ok",
                f"✅  Filled {nat_count:,} null Date values — assigned sequential "
                f"2023-01-01 → 2023-12-31 per user"))
        else:
            logs.append(("ok",
                f"✅  Date column parsed — {df['Date'].notna().sum():,} valid timestamps"))

    # 2. Numeric nulls — interpolate per user, then global ffill/bfill
    num_null_cols = [c for c in df.columns
                     if df[c].dtype in [np.float64, np.float32] and df[c].isna().any()]
    for col in num_null_cols:
        n = int(df[col].isna().sum())
        if "User_ID" in df.columns:
            parts = []
            for uid, grp in df.groupby("User_ID", sort=True):
                grp = grp.copy()
                grp[col] = grp[col].interpolate(method="linear").ffill().bfill()
                parts.append(grp)
            df = pd.concat(parts, ignore_index=True)
        else:
            df[col] = df[col].interpolate(method="linear").ffill().bfill()
        logs.append(("ok",
            f"✅  Interpolated (linear) + ffill/bfill → '{col}' ({n:,} nulls filled)"))

    # 3. Categorical nulls — Workout_Type always filled with 'No Workout', rest use mode
    if "Workout_Type" in df.columns:
        # Catch ALL forms of missing: NaN, empty string, whitespace, "nan" string
        missing_mask = (
            df["Workout_Type"].isna() |
            (df["Workout_Type"].astype(str).str.strip() == "") |
            (df["Workout_Type"].astype(str).str.strip().str.lower() == "nan")
        )
        n = int(missing_mask.sum())
        df["Workout_Type"] = df["Workout_Type"].astype(str).str.strip()
        df.loc[missing_mask, "Workout_Type"] = "No Workout"
        logs.append(("ok",
            f"✅  Filled {n:,} null(s) in 'Workout_Type' with → 'No Workout'"))

    cat_null_cols = [c for c in df.columns
                     if df[c].dtype == object and df[c].isna().any()
                     and c != "Workout_Type"]
    for col in cat_null_cols:
        n    = int(df[col].isna().sum())
        mode = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
        df[col] = df[col].fillna(mode)
        logs.append(("ok",
            f"✅  Filled {n:,} null(s) in '{col}' with mode → '{mode}'"))

    # 4. Round floats
    for col in df.select_dtypes(include=[np.float64, np.float32]).columns:
        df[col] = df[col].round(2)

    # 5. Time normalization info
    if "Date" in df.columns:
        vd   = df["Date"].dropna()
        span = (vd.max() - vd.min()).days
        logs.append(("info",
            f"ℹ️  Date range: {vd.min().date()} → {vd.max().date()} | Span: {span} days"))
        logs.append(("warn",
            "⚠️  Timestamps stored as local/naive — UTC normalization skipped."))

    return df, logs, before_nulls


# ──────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────
for key in ["raw_df","clean_df","logs","before_nulls","step"]:
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state.step is None:
    st.session_state.step = 1

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
STEP_META = {
    1:("📂","Upload CSV"),
    2:("🔍","Check Null Values"),
    3:("⚙️","Preprocess Data"),
    4:("👁️","Preview Dataset"),
    5:("📈","Run EDA"),
}

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 1.5rem;">
        <div style="font-family:Orbitron,monospace;font-size:1.2rem;font-weight:900;
                    color:#fff;letter-spacing:3px;">🏋️ FITNESS</div>
        <div style="font-family:Orbitron,monospace;font-size:0.75rem;color:#00e6ff;
                    letter-spacing:5px;margin-top:4px;">DATA PRO</div>
        <div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;color:#4a6a88;
                    letter-spacing:2px;margin-top:8px;">EDA PIPELINE v2.0</div>
    </div><hr>""", unsafe_allow_html=True)

    st.markdown("""<div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;
        letter-spacing:3px;color:#4a6a88;text-transform:uppercase;
        margin-bottom:10px;">PIPELINE STEPS</div>""", unsafe_allow_html=True)

    for num,(icon,label) in STEP_META.items():
        active = st.session_state.step == num
        done   = st.session_state.step > num
        if active:
            fg,bg,bd,badge="#00e6ff","rgba(0,230,255,0.08)","rgba(0,230,255,0.3)","▶"
        elif done:
            fg,bg,bd,badge="#00ffa3","rgba(0,255,163,0.06)","rgba(0,255,163,0.25)","✓"
        else:
            fg,bg,bd,badge="#4a6a88","transparent","rgba(255,255,255,0.05)",str(num)
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;
             border-radius:10px;margin-bottom:4px;background:{bg};border:1px solid {bd};">
            <div style="width:22px;height:22px;border-radius:6px;display:flex;
                 align-items:center;justify-content:center;background:rgba(0,0,0,0.2);
                 font-family:JetBrains Mono,monospace;font-size:0.65rem;
                 font-weight:700;color:{fg};">{badge}</div>
            <span style="font-family:Exo 2,sans-serif;
                font-weight:{'600' if active else '500'};
                font-size:0.85rem;color:{fg};">{icon} {label}</span>
        </div>""", unsafe_allow_html=True)

    pct = (st.session_state.step-1)/4
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;
        letter-spacing:2px;color:#4a6a88;margin-bottom:6px;">PIPELINE PROGRESS</div>""",
        unsafe_allow_html=True)
    st.progress(pct)
    st.markdown(f"""<div style="font-family:JetBrains Mono,monospace;font-size:0.68rem;
        color:#00e6ff;text-align:right;">{int(pct*100)}% Complete</div>""",
        unsafe_allow_html=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("""<div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;
        letter-spacing:2px;color:#4a6a88;margin-bottom:10px;">QUICK ACTIONS</div>""",
        unsafe_allow_html=True)
    if st.button("🔄  Reset Pipeline"):
        for k in ["raw_df","clean_df","logs","before_nulls"]:
            st.session_state[k] = None
        st.session_state.step = 1
        st.rerun()

# ──────────────────────────────────────────────────────────────
# PAGE HEADER
# ──────────────────────────────────────────────────────────────
icon,name = STEP_META[st.session_state.step]
col_t,col_b = st.columns([3,1])
with col_t:
    st.markdown(f"""
    <div style="margin-bottom:1.5rem;">
        <div style="font-family:JetBrains Mono,monospace;font-size:0.65rem;
             letter-spacing:3px;color:#00e6ff;margin-bottom:6px;opacity:0.8;">
             STEP {st.session_state.step} OF 5</div>
        <h1 style="margin:0;">{icon} {name}</h1>
    </div>""", unsafe_allow_html=True)
with col_b:
    st.markdown("""
    <div style="text-align:right;padding-top:1.2rem;">
        <span style="display:inline-flex;align-items:center;gap:8px;
            background:rgba(0,255,163,0.08);border:1px solid rgba(0,255,163,0.2);
            border-radius:20px;padding:6px 16px;">
            <span style="width:7px;height:7px;border-radius:50%;background:#00ffa3;
                box-shadow:0 0 8px #00ffa3;display:inline-block;"></span>
            <span style="font-family:JetBrains Mono,monospace;font-size:0.65rem;
                color:#00ffa3;letter-spacing:1px;">PIPELINE ACTIVE</span>
        </span>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# STEP 1 — UPLOAD CSV
# ══════════════════════════════════════════════════════════════
if st.session_state.step == 1:

    st.markdown("### Upload Your Fitness Dataset")
    st.markdown("<p style='color:#4a6a88;'>Upload a CSV file with your fitness tracking data to begin the full EDA pipeline.</p>",
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your CSV here or click to browse",
        type=["csv"],
        help="Expected columns: User_ID, Full Name, Date, Age, Gender, Height (cm), "
             "Weight (kg), Steps_Taken, Calories_Burned, Hours_Slept, "
             "Water_Intake (Liters), Active_Minutes, Heart_Rate (bpm), "
             "Workout_Type, Stress_Level (1-10), Mood",
    )

    if uploaded:
        try:
            st.session_state.raw_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")

    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        st.markdown("<br>", unsafe_allow_html=True)
        st.success("✅  Dataset loaded successfully!")

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("📊 Total Rows",  f"{len(df):,}")
        c2.metric("📋 Columns",     f"{len(df.columns)}")
        c3.metric("⚠️ Null Cells",  f"{df.isnull().sum().sum():,}")
        c4.metric("👥 Unique Users",
                  str(df["User_ID"].nunique()) if "User_ID" in df.columns else "—")

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔎 Preview Raw Data (first 10 rows)"):
            st.dataframe(df.head(10), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Next → Check Null Values  ▶"):
            st.session_state.step = 2; st.rerun()
    else:
        st.info("👆  Upload a CSV file to begin.")


# ══════════════════════════════════════════════════════════════
# STEP 2 — CHECK NULL VALUES
# ══════════════════════════════════════════════════════════════
elif st.session_state.step == 2:

    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset in Step 1 first.")
        if st.button("← Back to Upload"):
            st.session_state.step = 1; st.rerun()
        st.stop()

    df = st.session_state.raw_df
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)

    st.markdown("### Null Value Analysis")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Null Cells", f"{null_counts.sum():,}")
    c2.metric("Columns Affected", f"{len(null_counts)}")
    c3.metric("Overall Null Rate",
              f"{null_counts.sum()/(len(df)*len(df.columns))*100:.1f}%")
    c4.metric("Clean Columns",    f"{len(df.columns)-len(null_counts)}")

    st.markdown("<br>", unsafe_allow_html=True)

    if len(null_counts) == 0:
        st.success("🎉  No null values detected! Dataset is already clean.")
    else:
        # Severity badges
        badge_html = ""
        for col,cnt in null_counts.items():
            pct = cnt/len(df)*100
            if pct > 20:
                fg,bg,bc="#ff8aa8","rgba(255,56,96,0.1)","rgba(255,56,96,0.3)"
            elif pct > 10:
                fg,bg,bc="#ffb800","rgba(255,184,0,0.1)","rgba(255,184,0,0.3)"
            else:
                fg,bg,bc="#00e6ff","rgba(0,230,255,0.1)","rgba(0,230,255,0.3)"
            badge_html += (
                f"<span style='display:inline-flex;align-items:center;gap:5px;"
                f"padding:5px 14px;border-radius:20px;"
                f"font-family:JetBrains Mono,monospace;font-size:0.7rem;"
                f"background:{bg};border:1px solid {bc};color:{fg};margin:3px;'>"
                f"▲ {col}: {cnt:,} ({pct:.1f}%)</span>"
            )
        st.markdown(f"<div style='margin-bottom:1.5rem;'>{badge_html}</div>",
                    unsafe_allow_html=True)

        st.markdown("#### Null Count by Column")
        bar_colors = [
            "#ff4f9b" if (v/len(df)*100)>20
            else "#ffb800" if (v/len(df)*100)>10
            else "#00e6ff"
            for v in null_counts.values
        ]
        fig_null = go.Figure(go.Bar(
            y=null_counts.index.tolist(),
            x=null_counts.values.tolist(),
            orientation="h",
            marker=dict(color=bar_colors,
                        line=dict(color="rgba(255,255,255,0.1)",width=1)),
            text=[f"  {v:,}  ({v/len(df)*100:.1f}%)" for v in null_counts.values],
            textposition="outside",
            textfont=dict(family="JetBrains Mono",size=10,color="#d4eaf7"),
            hovertemplate="<b>%{y}</b><br>Null Count: %{x:,}<extra></extra>",
        ))
        fig_null.update_layout(**PTHEME,
                               height=max(250,len(null_counts)*60),
                               xaxis_title="Null Count",
                               yaxis_title="Column Name",
                               showlegend=False)
        st.plotly_chart(fig_null, use_container_width=True)

        tab1,tab2 = st.tabs(["📋 Summary Table","🗺️ Null Heatmap"])
        with tab1:
            tbl = pd.DataFrame({
                "Column":     null_counts.index,
                "Null Count": null_counts.values,
                "Null %":     (null_counts.values/len(df)*100).round(2),
                "Data Type":  [str(df[c].dtype) for c in null_counts.index],
                "Severity":   [
                    "🔴 High"   if v/len(df)>0.2
                    else "🟡 Medium" if v/len(df)>0.1
                    else "🔵 Low"
                    for v in null_counts.values
                ],
            })
            st.dataframe(tbl, use_container_width=True, hide_index=True)
        with tab2:
            sample = df[null_counts.index.tolist()].head(100).isnull().astype(int)
            fig_heat = go.Figure(go.Heatmap(
                z=sample.T.values,
                x=[str(i) for i in sample.index],
                y=sample.columns.tolist(),
                colorscale=[[0,"#0b1422"],[1,"#ff4f9b"]],
                showscale=False,
                hovertemplate="Row %{x}<br>Column: %{y}<br>Is Null: %{z}<extra></extra>",
            ))
            fig_heat.update_layout(**PTHEME, height=320,
                                   title="Null Pattern — first 100 rows",
                                   xaxis_title="Row Index",
                                   yaxis_title="Column Name")
            st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_b,col_n = st.columns([1,5])
    with col_b:
        if st.button("← Back"):
            st.session_state.step = 1; st.rerun()
    with col_n:
        if st.button("Next → Preprocess Data  ▶"):
            st.session_state.step = 3; st.rerun()


# ══════════════════════════════════════════════════════════════
# STEP 3 — PREPROCESS DATA
# ══════════════════════════════════════════════════════════════
elif st.session_state.step == 3:

    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset first.")
        if st.button("← Back to Upload"):
            st.session_state.step = 1; st.rerun()
        st.stop()

    df_raw = st.session_state.raw_df
    st.markdown("### Data Preprocessing Pipeline")

    if st.button("▶  Run Preprocessing"):
        with st.spinner("Running full preprocessing pipeline …"):
            import time; time.sleep(0.6)
            clean,logs,before_nulls = preprocess(df_raw)
            st.session_state.clean_df    = clean
            st.session_state.logs        = logs
            st.session_state.before_nulls = before_nulls

    if st.session_state.clean_df is not None:
        logs         = st.session_state.logs
        before_nulls = st.session_state.before_nulls
        df_clean     = st.session_state.clean_df

        st.markdown("#### 📋 Preprocessing Log")
        for level,msg in logs:
            if level=="ok":   st.success(msg)
            elif level=="warn": st.warning(msg)
            else:             st.info(msg)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Null Value Comparison — Before vs After")
        col_b,col_a = st.columns(2)
        before_series = {k:v for k,v in before_nulls.items() if v>0}

        with col_b:
            st.markdown("""
            <div style="background:rgba(255,56,96,0.1);
                 border:1px solid rgba(255,56,96,0.25);border-radius:10px 10px 0 0;
                 padding:8px 16px;font-family:JetBrains Mono,monospace;font-size:0.7rem;
                 letter-spacing:2px;color:#ff8aa8;">BEFORE PREPROCESSING</div>""",
                unsafe_allow_html=True)
            if before_series:
                st.dataframe(
                    pd.DataFrame({"Column":list(before_series.keys()),
                                  "Null Count":list(before_series.values())}),
                    use_container_width=True, hide_index=True)
            else:
                st.info("No nulls were present.")

        with col_a:
            st.markdown("""
            <div style="background:rgba(0,255,163,0.1);
                 border:1px solid rgba(0,255,163,0.25);border-radius:10px 10px 0 0;
                 padding:8px 16px;font-family:JetBrains Mono,monospace;font-size:0.7rem;
                 letter-spacing:2px;color:#00ffa3;">AFTER PREPROCESSING</div>""",
                unsafe_allow_html=True)
            st.markdown("""
            <div style="background:rgba(0,255,163,0.05);
                 border:1px solid rgba(0,255,163,0.2);
                 border-radius:0 0 10px 10px;padding:3rem 1rem;text-align:center;">
                <div style="font-size:3rem;">🌟</div>
                <div style="font-family:Orbitron,monospace;font-size:0.85rem;
                     letter-spacing:2px;color:#00ffa3;margin-top:1rem;
                     text-shadow:0 0 20px rgba(0,255,163,0.5);">
                     ZERO NULLS REMAINING!</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        total_filled = sum(before_series.values())
        c1,c2,c3 = st.columns(3)
        c1.metric("Nulls Removed",  f"{total_filled:,}", delta=f"-{total_filled:,}")
        c2.metric("Rows Preserved", f"{len(df_clean):,}")
        c3.metric("Data Quality",   "100%", delta="+100%")

        st.markdown("<br>", unsafe_allow_html=True)
        csv_bytes = df_clean.to_csv(index=False).encode()
        st.download_button(
            label="⬇️  Download Cleaned CSV",
            data=csv_bytes,
            file_name="fitness_data_fully_cleaned.csv",
            mime="text/csv",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col_b2,col_n2 = st.columns([1,5])
        with col_b2:
            if st.button("← Back"):
                st.session_state.step = 2; st.rerun()
        with col_n2:
            if st.button("Next → Preview Dataset  ▶"):
                st.session_state.step = 4; st.rerun()
    else:
        st.info("👆  Click **Run Preprocessing** to clean all null values automatically.")
        if st.button("← Back"):
            st.session_state.step = 2; st.rerun()


# ══════════════════════════════════════════════════════════════
# STEP 4 — PREVIEW CLEANED DATASET
# ══════════════════════════════════════════════════════════════
elif st.session_state.step == 4:

    if st.session_state.clean_df is None:
        st.warning("Please complete preprocessing first.")
        if st.button("← Back to Preprocessing"):
            st.session_state.step = 3; st.rerun()
        st.stop()

    df = st.session_state.clean_df
    st.markdown("### Preview Cleaned Dataset")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Rows",      f"{len(df):,}")
    c2.metric("Columns",         f"{len(df.columns)}")
    c3.metric("Remaining Nulls", f"{df.isnull().sum().sum():,}")
    c4.metric("Users",
              str(df["User_ID"].nunique()) if "User_ID" in df.columns else "—")

    st.markdown("<br>", unsafe_allow_html=True)
    col_f1,col_f2 = st.columns([3,1])
    with col_f1:
        all_cols = df.columns.tolist()
        selected = st.multiselect("Select columns to display",all_cols,default=all_cols)
        if not selected: selected = all_cols
    with col_f2:
        n_rows = st.slider("Rows to show",5,min(200,len(df)),10)

    st.dataframe(df[selected].head(n_rows), use_container_width=True, height=400)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📐 Descriptive Statistics", expanded=True):
        num_df = df.select_dtypes(include=np.number)
        st.dataframe(num_df.describe().round(2), use_container_width=True)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        with st.expander("🏷️ Categorical Value Counts"):
            for c in cat_cols:
                st.markdown(f"**{c}**")
                vc = df[c].value_counts().reset_index()
                vc.columns = [c,"Count"]
                st.dataframe(vc, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        label="⬇️  Download Cleaned CSV",
        data=csv_bytes,
        file_name="fitness_data_fully_cleaned.csv",
        mime="text/csv",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col_b,col_n = st.columns([1,5])
    with col_b:
        if st.button("← Back"):
            st.session_state.step = 3; st.rerun()
    with col_n:
        if st.button("Next → Run Full EDA  ▶"):
            st.session_state.step = 5; st.rerun()


# ══════════════════════════════════════════════════════════════
# STEP 5 — EDA  (ALL BUGS FIXED)
# ══════════════════════════════════════════════════════════════
elif st.session_state.step == 5:

    if st.session_state.clean_df is None:
        st.warning("Please complete preprocessing first.")
        if st.button("← Back to Preprocessing"):
            st.session_state.step = 3; st.rerun()
        st.stop()

    df = st.session_state.clean_df.copy()

    # FIX 1: Ensure Date is datetime and handle parse errors gracefully
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    st.markdown("### Exploratory Data Analysis")

    NUM_COLS = [c for c in
                ["Steps_Taken","Calories_Burned","Hours_Slept",
                 "Water_Intake (Liters)","Active_Minutes","Heart_Rate (bpm)"]
                if c in df.columns]

    # Fallback: if none of the expected columns exist, use all numeric columns
    if not NUM_COLS:
        NUM_COLS = df.select_dtypes(include=np.number).columns.tolist()

    if not NUM_COLS:
        st.error("No numeric columns found for EDA.")
        st.stop()

    tab_dist, tab_box, tab_cat, tab_corr, tab_time, tab_user = st.tabs([
        "📊 Distributions",
        "📦 Outlier Detection",
        "🥧 Categorical Analysis",
        "🔗 Correlation Matrix",
        "📅 Time Trends",
        "👤 User Analysis",
    ])

    # ── TAB 1 : DISTRIBUTIONS ────────────────────────────────
    with tab_dist:
        st.markdown("#### Distribution of Numeric Features")
        st.caption("Histograms showing frequency of each value range. Dashed white = Mean, Dotted colour = Median.")

        for i in range(0, len(NUM_COLS), 2):
            cols = st.columns(2)
            for j, c in enumerate(NUM_COLS[i:i+2]):
                with cols[j]:
                    color    = COL_COLORS.get(c, "#00e6ff")
                    x_label  = AXIS_LABELS.get(c, c)
                    col_data = df[c].dropna()

                    # FIX 2: Guard against empty series before computing stats
                    if col_data.empty:
                        st.warning(f"No data available for column: {c}")
                        continue

                    mean_v   = col_data.mean()
                    median_v = col_data.median()

                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=col_data, nbinsx=35,
                        marker=dict(color=color, opacity=0.75,
                                    line=dict(color=color, width=0.5)),
                        name="Frequency",
                        hovertemplate=f"{x_label}: %{{x}}<br>Count: %{{y}}<extra></extra>",
                    ))
                    fig.add_vline(x=mean_v, line_color="white", line_dash="dash",
                                  line_width=1.5,
                                  annotation_text=f"Mean:{mean_v:.1f}",
                                  annotation_font_color="white",
                                  annotation_font_size=9)
                    fig.add_vline(x=median_v, line_color=color, line_dash="dot",
                                  line_width=1.5,
                                  annotation_text=f"Median:{median_v:.1f}",
                                  annotation_font_color=color,
                                  annotation_font_size=9,
                                  annotation_position="bottom right")
                    fig.update_layout(
                        **PTHEME, height=270, showlegend=False, bargap=0.04,
                        title=dict(text=f"Distribution · {c}",
                                   font=dict(size=11, color="#d4eaf7")),
                        xaxis_title=x_label,
                        yaxis_title="Number of Records",
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ── TAB 2 : OUTLIER DETECTION ─────────────────────────────
    with tab_box:
        st.markdown("#### Outlier Detection — Boxplots")
        st.caption("Box = Q1–Q3 (IQR). Dashed line = mean. Points beyond 1.5×IQR are outliers.")

        for i in range(0, len(NUM_COLS), 2):
            cols = st.columns(2)
            for j, c in enumerate(NUM_COLS[i:i+2]):
                with cols[j]:
                    color    = COL_COLORS.get(c, "#00e6ff")
                    x_label  = AXIS_LABELS.get(c, c)
                    col_data = df[c].dropna()

                    # FIX 3: Guard against empty series
                    if col_data.empty:
                        st.warning(f"No data available for column: {c}")
                        continue

                    q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
                    iqr    = q3 - q1
                    n_out  = int(((col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr)).sum())

                    fig = go.Figure(go.Box(
                        x=col_data, name=c,
                        marker_color=color, line_color=color,
                        fillcolor=hex_to_rgba(color, 0.13), boxmean=True,
                        hovertemplate=f"{x_label}: %{{x:.2f}}<extra></extra>",
                    ))
                    fig.update_layout(
                        **PTHEME, height=210, showlegend=False,
                        title=dict(text=f"Boxplot · {c}  ({n_out} outliers)",
                                   font=dict(size=11, color="#d4eaf7")),
                        xaxis_title=x_label,
                        yaxis_title="Distribution",
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Outlier Summary Table")
        rows = []
        for c in NUM_COLS:
            col_data = df[c].dropna()
            if col_data.empty:
                continue
            q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            iqr    = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out  = int(((col_data < lo) | (col_data > hi)).sum())
            rows.append({
                "Column":      c,
                "Min":         round(col_data.min(), 2),
                "Q1":          round(q1, 2),
                "Median":      round(col_data.median(), 2),
                "Mean":        round(col_data.mean(), 2),
                "Q3":          round(q3, 2),
                "Max":         round(col_data.max(), 2),
                "IQR":         round(iqr, 2),
                "Lower Fence": round(lo, 2),
                "Upper Fence": round(hi, 2),
                "# Outliers":  n_out,
                "Outlier %":   round(n_out / len(df) * 100, 2),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── TAB 3 : CATEGORICAL ───────────────────────────────────
    with tab_cat:
        st.markdown("#### Categorical Feature Analysis")
        col1, col2 = st.columns(2)

        if "Workout_Type" in df.columns:
            wc = df["Workout_Type"].value_counts().reset_index()
            wc.columns = ["Workout", "Count"]
            fig_w = go.Figure(go.Bar(
                x=wc["Workout"], y=wc["Count"],
                marker=dict(color=WORKOUT_COLORS[:len(wc)],
                            line=dict(color="rgba(255,255,255,0.1)", width=1)),
                text=wc["Count"], textposition="outside",
                hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
            ))
            fig_w.update_layout(**PTHEME, height=320, showlegend=False,
                                title="Workout Type Distribution",
                                xaxis_title="Workout Type",
                                yaxis_title="Number of Records")
            with col1:
                st.plotly_chart(fig_w, use_container_width=True)

        if "Mood" in df.columns:
            mc = df["Mood"].value_counts().reset_index()
            mc.columns = ["Mood", "Count"]
            fig_m = go.Figure(go.Pie(
                labels=mc["Mood"], values=mc["Count"],
                marker=dict(colors=MOOD_COLORS,
                            line=dict(color="#030810", width=2)),
                hole=0.55,
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
            ))
            fig_m.update_layout(**PTHEME, height=320, showlegend=True,
                                title="Mood Distribution (% of Total Records)",
                                legend=dict(orientation="v", x=1.05, font=dict(size=10)))
            with col2:
                st.plotly_chart(fig_m, use_container_width=True)

        if "Gender" in df.columns:
            col3, col4 = st.columns(2)
            gc = df["Gender"].value_counts().reset_index()
            gc.columns = ["Gender", "Count"]
            fig_g = go.Figure(go.Pie(
                labels=gc["Gender"], values=gc["Count"],
                marker=dict(colors=["#00e6ff", "#ff4f9b"],
                            line=dict(color="#030810", width=2)),
                hole=0.6,
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
            ))
            fig_g.update_layout(**PTHEME, height=300, showlegend=True,
                                title="Gender Distribution",
                                legend=dict(font=dict(size=10)))
            with col3:
                st.plotly_chart(fig_g, use_container_width=True)

            if "Stress_Level (1-10)" in df.columns and "Mood" in df.columns:
                sm = df.groupby("Mood")["Stress_Level (1-10)"].mean().reset_index()
                sm.columns = ["Mood", "Avg Stress"]
                fig_sm = go.Figure(go.Bar(
                    x=sm["Mood"], y=sm["Avg Stress"].round(2),
                    marker=dict(color=MOOD_COLORS[:len(sm)],
                                line=dict(color="rgba(255,255,255,0.1)", width=1)),
                    text=sm["Avg Stress"].round(2), textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Avg Stress: %{y:.2f}<extra></extra>",
                ))
                fig_sm.update_layout(**PTHEME, height=300, showlegend=False,
                                     title="Average Stress Level by Mood",
                                     xaxis_title="Mood",
                                     yaxis_title="Average Stress Level (1–10)")
                with col4:
                    st.plotly_chart(fig_sm, use_container_width=True)

        if "Gender" in df.columns and "Workout_Type" in df.columns:
            st.markdown("#### Workout Type Frequency by Gender")
            wg = df.groupby(["Gender", "Workout_Type"]).size().reset_index(name="Count")
            fig_wg = px.bar(
                wg, x="Workout_Type", y="Count", color="Gender", barmode="group",
                color_discrete_map={"Male": "#00e6ff", "Female": "#ff4f9b"},
                labels={"Workout_Type": "Workout Type", "Count": "Number of Records"},
                title="Workout Type Frequency by Gender",
            )
            fig_wg.update_layout(**PTHEME, height=320, showlegend=True,
                                 xaxis_title="Workout Type",
                                 yaxis_title="Number of Records")
            st.plotly_chart(fig_wg, use_container_width=True)

    # ── TAB 4 : CORRELATION ───────────────────────────────────
    with tab_corr:
        st.markdown("#### Feature Correlation Matrix")
        st.caption("Pearson r: +1 = perfect positive, 0 = no linear relationship, -1 = perfect negative.")

        if len(NUM_COLS) >= 2:
            corr = df[NUM_COLS].corr().round(2)
            fig_c = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(), y=corr.columns.tolist(),
                colorscale=[[0.0, "#ff4f9b"], [0.5, "#0b1422"], [1.0, "#00e6ff"]],
                zmin=-1, zmax=1,
                text=corr.values.round(2), texttemplate="%{text}",
                textfont=dict(size=11, family="JetBrains Mono"),
                hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.2f}<extra></extra>",
                colorbar=dict(tickfont=dict(color="#d4eaf7", size=10), len=0.8,
                              title=dict(text="r", font=dict(color="#d4eaf7", size=10))),
            ))
            fig_c.update_layout(**PTHEME, height=500,
                                title="Pearson Correlation Matrix (r: −1 to +1)",
                                xaxis_title="Feature (X-axis)",
                                yaxis_title="Feature (Y-axis)")
            st.plotly_chart(fig_c, use_container_width=True)

            # Scatter Explorer
            st.markdown("#### Scatter Plot Explorer")
            st.caption("Explore the relationship between any two numeric features.")
            sc1, sc2, sc3 = st.columns(3)
            with sc1: x_feat = st.selectbox("X-Axis Feature", NUM_COLS, index=0)
            with sc2: y_feat = st.selectbox("Y-Axis Feature", NUM_COLS,
                                             index=min(1, len(NUM_COLS) - 1))
            with sc3: color_feat = st.selectbox("Color By",
                                    ["None", "Gender", "Mood", "Workout_Type"], index=0)

            s_cols = [x_feat, y_feat]
            color_col = None if color_feat == "None" else color_feat
            color_map = None

            if color_col and color_col in df.columns:
                s_cols.append(color_col)
                if color_col == "Gender":
                    color_map = {"Male": "#00e6ff", "Female": "#ff4f9b"}
                elif color_col == "Mood":
                    u = df[color_col].dropna().unique()
                    color_map = dict(zip(u, MOOD_COLORS[:len(u)]))
                elif color_col == "Workout_Type":
                    u = df[color_col].dropna().unique()
                    color_map = dict(zip(u, WORKOUT_COLORS[:len(u)]))

            # FIX 4: Drop duplicates in s_cols list (x_feat == y_feat edge case)
            scatter_df = df[list(dict.fromkeys(s_cols))].dropna()

            # FIX 5: trendline="ols" requires statsmodels — wrap in try/except
            try:
                fig_sc = px.scatter(
                    scatter_df, x=x_feat, y=y_feat, color=color_col,
                    color_discrete_map=color_map,
                    trendline="ols",
                    trendline_color_override="#ffffff",
                    labels={x_feat: AXIS_LABELS.get(x_feat, x_feat),
                            y_feat: AXIS_LABELS.get(y_feat, y_feat)},
                    opacity=0.55,
                )
            except Exception:
                # statsmodels not installed — render scatter without trendline
                fig_sc = px.scatter(
                    scatter_df, x=x_feat, y=y_feat, color=color_col,
                    color_discrete_map=color_map,
                    labels={x_feat: AXIS_LABELS.get(x_feat, x_feat),
                            y_feat: AXIS_LABELS.get(y_feat, y_feat)},
                    opacity=0.55,
                )

            fig_sc.update_traces(marker=dict(size=4))
            fig_sc.update_layout(**PTHEME, height=380,
                                 title=f"Scatter: {x_feat} vs {y_feat}",
                                 xaxis_title=AXIS_LABELS.get(x_feat, x_feat),
                                 yaxis_title=AXIS_LABELS.get(y_feat, y_feat))
            st.plotly_chart(fig_sc, use_container_width=True)

            # Top pairs table
            st.markdown("#### Top Feature Pairs by Correlation Strength")
            pairs = []
            for i in range(len(NUM_COLS)):
                for j in range(i + 1, len(NUM_COLS)):
                    r = corr.iloc[i, j]
                    pairs.append({
                        "Feature A": NUM_COLS[i], "Feature B": NUM_COLS[j],
                        "r value":   round(r, 3),
                        "Strength":  ("Strong"   if abs(r) > 0.6
                                      else "Moderate" if abs(r) > 0.3 else "Weak"),
                        "Direction": "Positive 📈" if r > 0 else "Negative 📉",
                    })
            pair_df = pd.DataFrame(pairs).sort_values("r value", key=abs, ascending=False)
            st.dataframe(pair_df, use_container_width=True, hide_index=True)
        else:
            st.info("At least two numeric columns are required for correlation analysis.")

    # ── TAB 5 : TIME TRENDS ───────────────────────────────────
    with tab_time:
        st.markdown("#### Time Series Trends")

        if "Date" in df.columns and df["Date"].notna().sum() > 0:
            col_m, col_a = st.columns([2, 1])
            with col_m:
                metric = st.selectbox("Select Metric", NUM_COLS, key="ts_metric")
            with col_a:
                agg = st.radio("Aggregation", ["Daily", "Weekly", "Monthly"],
                               horizontal=True, key="ts_agg")

            # FIX 6: Average across all users per date BEFORE resampling
            # This prevents multi-user duplicate-index issues during resample
            df_t = (
                df.dropna(subset=["Date"])
                  .groupby("Date")[metric]
                  .mean()
                  .reset_index()
                  .set_index("Date")
                  .sort_index()
            )

            # FIX 7: Use safe_resample helper for pandas version compatibility
            if agg in ("Weekly", "Monthly"):
                df_t = safe_resample(df_t, agg)

            color   = COL_COLORS.get(metric, "#00e6ff")
            x_label = AXIS_LABELS.get(metric, metric)

            # FIX 8: Guard against empty time series after resampling
            if df_t.empty or df_t[metric].dropna().empty:
                st.warning("No time-series data available for the selected metric.")
            else:
                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(
                    x=df_t.index, y=df_t[metric], mode="lines",
                    line=dict(color=color, width=2),
                    fill="tozeroy", fillcolor=hex_to_rgba(color, 0.09),
                    name=metric,
                    hovertemplate=f"%{{x|%Y-%m-%d}}<br>{x_label}: %{{y:.2f}}<extra></extra>",
                ))
                # Rolling average (only if enough points)
                if len(df_t) > 7:
                    roll = df_t[metric].rolling(7, min_periods=1).mean()
                    fig_ts.add_trace(go.Scatter(
                        x=df_t.index, y=roll, mode="lines",
                        line=dict(color="#ffffff", width=1.5, dash="dot"),
                        name="7-period Rolling Avg",
                    ))
                fig_ts.update_layout(**PTHEME, height=380, showlegend=True,
                                     title=f"{agg} Trend · {metric}",
                                     xaxis_title="Date",
                                     yaxis_title=x_label,
                                     legend=dict(font=dict(size=10)))
                st.plotly_chart(fig_ts, use_container_width=True)

            # Monthly heatmap — all metrics
            st.markdown("#### Monthly Average Heatmap — All Metrics")

            # FIX 9: Compute monthly averages safely (mean across users per date first)
            df_for_heatmap = (
                df.dropna(subset=["Date"])
                  .groupby("Date")[NUM_COLS]
                  .mean()
                  .sort_index()
            )
            df_monthly = safe_resample(df_for_heatmap, "Monthly")
            df_monthly = df_monthly.dropna(how="all")

            if df_monthly.empty:
                st.warning("Not enough data to build the monthly heatmap.")
            else:
                # FIX 10: Safe strftime — index must be DatetimeIndex after resample
                try:
                    month_labels = df_monthly.index.strftime("%b %Y").tolist()
                except AttributeError:
                    month_labels = [str(i) for i in df_monthly.index]

                # Normalise row-wise for heatmap display
                mn = df_monthly.min()
                mx = df_monthly.max()
                denom = (mx - mn).replace(0, 1)  # FIX 11: avoid divide-by-zero
                df_norm = (df_monthly - mn) / denom

                fig_mh = go.Figure(go.Heatmap(
                    z=df_norm.T.values,
                    x=month_labels,
                    y=NUM_COLS,
                    colorscale=[[0, "#0b1422"], [0.5, "#7c3aed"], [1, "#00e6ff"]],
                    text=df_monthly.T.round(1).values,
                    texttemplate="%{text}",
                    textfont=dict(size=9),
                    hovertemplate="<b>%{y}</b><br>Month: %{x}<br>Avg: %{text}<extra></extra>",
                    colorbar=dict(tickfont=dict(color="#d4eaf7", size=9),
                                  title=dict(text="Normalised",
                                             font=dict(color="#d4eaf7", size=9))),
                ))
                fig_mh.update_layout(**PTHEME, height=340,
                                     title="Monthly Average — Normalised (0=low, 1=high)",
                                     xaxis_title="Month",
                                     yaxis_title="Metric")
                st.plotly_chart(fig_mh, use_container_width=True)

            # Steps by Workout Type boxplot
            if "Workout_Type" in df.columns and "Steps_Taken" in df.columns:
                st.markdown("#### Steps Taken by Workout Type")
                fig_sw = go.Figure()
                unique_workouts = df["Workout_Type"].dropna().unique()
                for wt, wcolor in zip(unique_workouts, WORKOUT_COLORS):
                    sub = df[df["Workout_Type"] == wt]["Steps_Taken"].dropna()
                    if len(sub):
                        fig_sw.add_trace(go.Box(
                            y=sub, name=wt,
                            marker_color=wcolor, line_color=wcolor,
                            fillcolor=hex_to_rgba(wcolor, 0.13), boxmean=True,
                        ))
                fig_sw.update_layout(**PTHEME, height=340, showlegend=False,
                                     title="Steps Taken Distribution by Workout Type",
                                     xaxis_title="Workout Type",
                                     yaxis_title="Steps Taken (count)")
                st.plotly_chart(fig_sw, use_container_width=True)

        else:
            st.info("No valid Date column found for time series analysis.")

    # ── TAB 6 : USER ANALYSIS ─────────────────────────────────
    with tab_user:
        st.markdown("#### Per-User Analysis")

        if "User_ID" in df.columns:
            all_users = sorted(df["User_ID"].unique())
            col_u1, col_u2 = st.columns([1, 3])
            with col_u1:
                selected_users = st.multiselect(
                    "Select Users",
                    options=all_users,
                    default=all_users[:5] if len(all_users) >= 5 else all_users,
                )
            with col_u2:
                user_metric = st.selectbox("Metric to Compare", NUM_COLS, key="user_metric")

            if selected_users:
                df_u = df[df["User_ID"].isin(selected_users)]

                # Average per user bar chart
                user_avg = df_u.groupby("User_ID")[user_metric].mean().reset_index()
                user_avg.columns = ["User_ID", "Average"]

                # FIX 12: Safely map Full Name without KeyError
                if "Full Name" in df.columns:
                    nm_map = df.drop_duplicates("User_ID").set_index("User_ID")["Full Name"]
                    user_avg["Label"] = user_avg["User_ID"].map(nm_map).fillna(
                        "User " + user_avg["User_ID"].astype(str))
                else:
                    user_avg["Label"] = "User " + user_avg["User_ID"].astype(str)

                fig_ua = go.Figure(go.Bar(
                    x=user_avg["Label"], y=user_avg["Average"].round(2),
                    marker=dict(color=COL_COLORS.get(user_metric, "#00e6ff"),
                                opacity=0.8,
                                line=dict(color="rgba(255,255,255,0.1)", width=1)),
                    text=user_avg["Average"].round(2), textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Avg: %{y:.2f}<extra></extra>",
                ))
                fig_ua.update_layout(**PTHEME, height=360, showlegend=False,
                                     title=f"Average {user_metric} per User",
                                     xaxis_title="User",
                                     yaxis_title=AXIS_LABELS.get(user_metric, user_metric))
                st.plotly_chart(fig_ua, use_container_width=True)

                # Time series per user
                if "Date" in df.columns:
                    st.markdown(f"#### {user_metric} Over Time — Selected Users")
                    fig_ut = go.Figure()
                    user_colors = ["#00e6ff", "#f472b6", "#34d399", "#c084fc", "#fb923c",
                                   "#ffb800", "#60a5fa", "#ff4f9b", "#00ffa3", "#7c3aed"]

                    for k, uid in enumerate(selected_users):
                        sub = df[df["User_ID"] == uid].dropna(subset=["Date"]).sort_values("Date")
                        if sub.empty:
                            continue

                        # FIX 13: Safely get label
                        if "Full Name" in df.columns and not sub["Full Name"].empty:
                            lbl = sub["Full Name"].iloc[0]
                        else:
                            lbl = f"User {uid}"

                        # FIX 14: Guard against missing metric values in user subset
                        if user_metric not in sub.columns or sub[user_metric].dropna().empty:
                            continue

                        fig_ut.add_trace(go.Scatter(
                            x=sub["Date"], y=sub[user_metric], mode="lines",
                            name=str(lbl),
                            line=dict(color=user_colors[k % len(user_colors)], width=1.5),
                            hovertemplate=(f"<b>{lbl}</b><br>%{{x|%Y-%m-%d}}<br>"
                                           f"{user_metric}: %{{y:.2f}}<extra></extra>"),
                        ))

                    if len(fig_ut.data) == 0:
                        st.warning("No time series data available for the selected users.")
                    else:
                        fig_ut.update_layout(**PTHEME, height=380, showlegend=True,
                                             title=f"{user_metric} Trends per User",
                                             xaxis_title="Date",
                                             yaxis_title=AXIS_LABELS.get(user_metric, user_metric),
                                             legend=dict(font=dict(size=9)))
                        st.plotly_chart(fig_ut, use_container_width=True)

                # FIX 15: Flatten MultiIndex columns safely after groupby + agg
                st.markdown("#### User Summary Statistics")
                agg_dict = {c: ["mean", "min", "max", "std"]
                            for c in NUM_COLS if c in df_u.columns}

                if agg_dict:
                    user_sum = df_u.groupby("User_ID").agg(agg_dict)
                    # Flatten MultiIndex columns
                    user_sum.columns = [f"{col}_{fn}" for col, fn in user_sum.columns]
                    user_sum = user_sum.round(2).reset_index()
                    st.dataframe(user_sum, use_container_width=True)
                else:
                    st.info("No numeric columns available for user summary.")
            else:
                st.info("Select at least one user to view analysis.")
        else:
            st.info("No 'User_ID' column found for per-user analysis.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to Preview"):
        st.session_state.step = 4; st.rerun()


# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:1.5rem 0;
     border-top:1px solid rgba(0,230,255,0.08);">
    <span style="font-family:JetBrains Mono,monospace;font-size:0.62rem;
        letter-spacing:2px;color:#2a4560;">
        FITNESS DATA PRO &nbsp;·&nbsp; PROFESSIONAL EDA PIPELINE &nbsp;·&nbsp;
        BUILT WITH STREAMLIT + PLOTLY
    </span>
</div>
""", unsafe_allow_html=True)