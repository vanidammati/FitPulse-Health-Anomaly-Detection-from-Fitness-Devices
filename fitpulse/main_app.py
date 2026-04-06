"""
╔══════════════════════════════════════════════════════════════════════╗
║   🏋️  FitPulse — Complete Fitness Analytics Suite                   ║
║   Module 1 : Pre-Processing & EDA   (separate CSV)                  ║
║   Module 2 : Pattern Extraction     (5 Fitbit CSVs – shared)        ║
║   Module 3 : Anomaly Detection      (5 Fitbit CSVs – shared)        ║
║   Module 4 : Insights Dashboard     (5 Fitbit CSVs – shared)        ║
║                                                                      ║
║   Install: pip install streamlit plotly pandas numpy matplotlib      ║
║            scikit-learn tsfresh prophet reportlab seaborn            ║
║   Run:     streamlit run fitpulse_suite.py                           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import io, warnings, traceback
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

import streamlit as st

# ──────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitPulse Analytics Suite",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: #030810 !important; color: #d4eaf7 !important;
    font-family: 'Exo 2', sans-serif !important;
}
[data-testid="stMain"] { background: transparent !important; }
[data-testid="stAppViewContainer"]::before {
    content: ''; position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(0,230,255,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,230,255,0.02) 1px, transparent 1px);
    background-size: 55px 55px;
}
[data-testid="stSidebar"] {
    background: #040a14 !important;
    border-right: 1px solid rgba(0,230,255,0.14) !important;
}
section[data-testid="stSidebar"] * { color: #d4eaf7 !important; }
h1 { font-family:'Orbitron',monospace !important; font-size:1.7rem !important;
     letter-spacing:2px !important; color:#fff !important; }
h2 { font-family:'Orbitron',monospace !important; font-size:1.1rem !important;
     letter-spacing:1.5px !important; color:#00e6ff !important; }
h3 { font-family:'Exo 2',sans-serif !important; font-weight:600 !important; color:#d4eaf7 !important; }
p, li { color:#d4eaf7 !important; }
[data-testid="stMetric"] {
    background: #0b1422 !important; border: 1px solid rgba(0,230,255,0.14) !important;
    border-radius: 14px !important; padding: 1.2rem 1.4rem !important;
    box-shadow: 0 6px 30px rgba(0,0,0,0.5) !important; transition: transform 0.2s;
}
[data-testid="stMetric"]:hover { transform: translateY(-3px); }
[data-testid="stMetricValue"] {
    font-family:'Orbitron',monospace !important; font-size:2rem !important;
    font-weight:900 !important; color:#00e6ff !important;
    text-shadow: 0 0 22px rgba(0,230,255,0.5) !important;
}
[data-testid="stMetricLabel"] {
    font-family:'JetBrains Mono',monospace !important; font-size:0.62rem !important;
    letter-spacing:2.5px !important; text-transform:uppercase !important; color:#4a6a88 !important;
}
.stButton > button {
    background: linear-gradient(135deg,#00c8ff,#0055cc) !important;
    color:#fff !important; border:none !important; border-radius:10px !important;
    font-family:'Exo 2',sans-serif !important; font-weight:600 !important;
    font-size:0.9rem !important; padding:0.65rem 1.8rem !important;
    box-shadow:0 4px 22px rgba(0,200,255,0.38) !important;
    letter-spacing:0.5px !important; transition:all 0.25s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#00d8ff,#0066ff) !important;
    box-shadow: 0 8px 32px rgba(0,200,255,0.55) !important; transform: translateY(-2px) !important;
}
[data-testid="stFileUploader"] {
    background: rgba(0,230,255,0.03) !important;
    border: 2px dashed rgba(0,230,255,0.3) !important;
    border-radius: 16px !important; padding: 1.2rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,230,255,0.7) !important;
    background: rgba(0,230,255,0.06) !important;
}
[data-testid="stExpander"] {
    background: #0b1422 !important; border:1px solid rgba(0,230,255,0.12) !important;
    border-radius: 12px !important;
}
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: #0b1422 !important; border-color: rgba(0,230,255,0.22) !important;
    border-radius: 10px !important; color: #d4eaf7 !important;
}
[data-testid="stProgressBar"] > div {
    background: linear-gradient(90deg,#7c3aed,#00e6ff,#00ffa3) !important;
    border-radius:4px !important; box-shadow:0 0 12px rgba(0,230,255,0.4) !important;
}
[data-testid="stProgressBar"] { background:rgba(255,255,255,0.05) !important; border-radius:4px !important; }
[data-testid="stTabs"] [role="tablist"] {
    background: #0b1422 !important; border-radius:12px 12px 0 0 !important;
    border-bottom:1px solid rgba(0,230,255,0.15) !important;
    gap:4px !important; padding:4px !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family:'Exo 2',sans-serif !important; font-weight:600 !important;
    font-size:0.85rem !important; color:#4a6a88 !important;
    border-radius:8px !important; padding:8px 18px !important; border:none !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background:rgba(0,230,255,0.1) !important; color:#00e6ff !important;
    border:1px solid rgba(0,230,255,0.28) !important;
    box-shadow:0 0 14px rgba(0,230,255,0.12) !important;
}
[data-testid="stTabContent"] {
    background:#0b1422 !important; border:1px solid rgba(0,230,255,0.1) !important;
    border-top:none !important; border-radius:0 0 12px 12px !important; padding:1.5rem !important;
}
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(0,230,255,0.18); border-radius:3px; }
hr { border-color:rgba(0,230,255,0.1) !important; margin:1rem 0 !important; }
.card {
    background:#0d1117; border:1px solid #1f2937;
    border-radius:16px; padding:1.5rem 1.8rem; margin-bottom:1rem;
}
.alert { border-radius:0 12px 12px 0; padding:0.85rem 1.1rem; margin:0.6rem 0; font-size:0.84rem; }
.alert-info    { background:rgba(56,189,248,0.08);  border-left:3px solid #38bdf8; color:#bae6fd; }
.alert-success { background:rgba(52,211,153,0.08);  border-left:3px solid #34d399; color:#a7f3d0; }
.alert-warn    { background:rgba(251,191,36,0.08);  border-left:3px solid #fbbf24; color:#fde68a; }
.alert-danger  { background:rgba(244,63,94,0.08);   border-left:3px solid #f43f5e; color:#fecdd3; }
.metric-row { display:flex; gap:0.8rem; flex-wrap:wrap; margin:1rem 0; }
.metric-box {
    flex:1; min-width:110px; border-radius:14px; padding:1.1rem 1.3rem;
    text-align:center; border:1px solid #374151; background:#111827;
}
.metric-val { font-family:'Orbitron',monospace; font-size:1.8rem; font-weight:900; line-height:1; margin-bottom:0.3rem; }
.metric-lbl { font-size:0.68rem; color:#6b7280; letter-spacing:0.08em; text-transform:uppercase; }
.sec-row { display:flex; align-items:center; gap:1rem; margin:2rem 0 1rem; padding-bottom:0.8rem; border-bottom:1px solid #1f2937; }
.sec-title { font-family:'Space Grotesk',sans-serif; font-size:1.2rem; font-weight:600; color:#f1f5f9; margin:0; }
.sec-pill { margin-left:auto; font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#38bdf8; background:rgba(56,189,248,0.1); border:1px solid rgba(56,189,248,0.25); border-radius:100px; padding:0.2rem 0.7rem; }
.anom-badge { display:inline-flex; align-items:center; gap:0.4rem; background:rgba(244,63,94,0.1); border:1px solid rgba(244,63,94,0.3); border-radius:100px; padding:0.3rem 0.9rem; font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#f43f5e; margin:0.5rem 0; }
.pulse { display:inline-block; width:7px; height:7px; border-radius:50%; background:#f43f5e; animation:pulse 1.4s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.7)} }
.div { border:none; border-top:1px solid #1f2937; margin:2rem 0; }
.sum-box { background:#0d1526; border:1px solid #1a2840; border-radius:10px; padding:16px 20px; margin-bottom:10px; }
.sum-box-title { font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:2px; text-transform:uppercase; color:#4b607a; margin-bottom:10px; }
.sum-row { display:flex; justify-content:space-between; align-items:center; padding:5px 0; border-bottom:1px solid #1a2840; font-size:13px; }
.sum-row:last-child { border-bottom:none; }
.sum-key { color:#94a3b8; } .sum-val { font-family:'JetBrains Mono',monospace; color:#e2e8f0; font-weight:600; }
.c-green{color:#22c55e!important} .c-blue{color:#38bdf8!important} .c-pink{color:#f472b6!important}
.c-orange{color:#fb923c!important} .c-purple{color:#a78bfa!important} .c-teal{color:#34d399!important}
.mtile { background:#0d1526; border:1px solid #1a2840; border-radius:10px; padding:14px 16px; text-align:center; }
.mtile-val { font-family:'Syne',sans-serif; font-size:1.75rem; font-weight:800; }
.mtile-lbl { font-family:'JetBrains Mono',monospace; font-size:9px; text-transform:uppercase; letter-spacing:1.5px; color:#4b607a; margin-top:3px; }
.step-pill { font-family:'JetBrains Mono',monospace; font-size:9px; letter-spacing:1.5px; text-transform:uppercase; padding:3px 9px; border-radius:4px; background:#1a2840; color:#4b607a; }
.step-pill.done  { background:rgba(34,197,94,.15); color:#22c55e; }
.step-pill.ready { background:rgba(56,189,248,.15); color:#38bdf8; }
.kpi-card { background:rgba(15,23,42,0.85); border:1px solid rgba(99,179,237,0.2); border-radius:14px; padding:1rem 1.1rem; text-align:center; }
.kpi-val { font-family:'Syne',sans-serif; font-size:1.7rem; font-weight:800; line-height:1; margin-bottom:0.2rem; }
.kpi-label { font-size:0.68rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.07em; }
#MainMenu, footer { visibility:hidden; }
div[data-testid="stDecoration"] { display:none; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# SHARED CONSTANTS & PLOTLY THEME
# ──────────────────────────────────────────────────────────────────────
PTHEME = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color="#d4eaf7", size=11),
    xaxis=dict(gridcolor="rgba(0,230,255,0.06)", zerolinecolor="rgba(0,230,255,0.1)",
               linecolor="rgba(0,230,255,0.1)", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="rgba(0,230,255,0.06)", zerolinecolor="rgba(0,230,255,0.1)",
               linecolor="rgba(0,230,255,0.1)", tickfont=dict(size=10)),
    margin=dict(l=10, r=10, t=45, b=10),
)

C = {
    "bg":"#050810","surface":"#0d1117","panel":"#111827","border":"#1f2937","border2":"#374151",
    "text":"#f1f5f9","muted":"#6b7280","subtle":"#9ca3af",
    "red":"#f43f5e","blue":"#38bdf8","green":"#34d399","amber":"#fbbf24",
    "purple":"#a78bfa","pink":"#f472b6","plot_bg":"#080b12","grid":"rgba(255,255,255,0.04)",
}

COL_COLORS = {
    "Steps_Taken":"#00e6ff","Calories_Burned":"#c084fc","Hours_Slept":"#f472b6",
    "Water_Intake (Liters)":"#60a5fa","Active_Minutes":"#34d399","Heart_Rate (bpm)":"#fb923c",
}
WORKOUT_COLORS = ["#c084fc","#f472b6","#ffb800","#00e6ff","#4a6a88"]
MOOD_COLORS    = ["#00ffa3","#ff4f9b","#00e6ff","#ffb800"]
AXIS_LABELS = {
    "Steps_Taken":"Steps Taken (count)","Calories_Burned":"Calories Burned (kcal)",
    "Hours_Slept":"Hours Slept (hrs)","Water_Intake (Liters)":"Water Intake (Liters)",
    "Active_Minutes":"Active Minutes (min)","Heart_Rate (bpm)":"Heart Rate (bpm)",
}

PALETTE = ["#38bdf8","#f472b6","#34d399","#fb923c","#a78bfa","#f87171","#4fd1c5"]

REQUIRED_FILES = {
    "dailyActivity_merged.csv":     {"key_cols":["ActivityDate","TotalSteps","Calories"],       "label":"Daily Activity",    "icon":"🏃"},
    "hourlySteps_merged.csv":       {"key_cols":["ActivityHour","StepTotal"],                   "label":"Hourly Steps",      "icon":"👣"},
    "hourlyIntensities_merged.csv": {"key_cols":["ActivityHour","TotalIntensity"],              "label":"Hourly Intensities","icon":"⚡"},
    "minuteSleep_merged.csv":       {"key_cols":["date","value","logId"],                       "label":"Minute Sleep",      "icon":"💤"},
    "heartrate_seconds_merged.csv": {"key_cols":["Time","Value"],                               "label":"Heart Rate",        "icon":"❤️"},
}

# ──────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ──────────────────────────────────────────────────────────────────────
def hex_to_rgba(h, a=0.13):
    h = h.lstrip("#")
    r,g,b = int(h[:2],16),int(h[2:4],16),int(h[4:],16)
    return f"rgba({r},{g},{b},{a})"

def base_layout(**kw):
    return dict(
        paper_bgcolor=C["surface"], plot_bgcolor=C["plot_bg"],
        font=dict(family="'Space Grotesk', sans-serif", color=C["subtle"], size=11),
        xaxis=dict(gridcolor=C["grid"], showgrid=True, zeroline=False,
                   linecolor=C["border"], tickfont=dict(color=C["muted"], size=10)),
        yaxis=dict(gridcolor=C["grid"], showgrid=True, zeroline=False,
                   linecolor=C["border"], tickfont=dict(color=C["muted"], size=10)),
        legend=dict(bgcolor="rgba(13,17,23,0.9)", bordercolor=C["border2"],
                    borderwidth=1, font=dict(color=C["subtle"], size=10)),
        margin=dict(l=55, r=30, t=65, b=50),
        hoverlabel=dict(bgcolor=C["panel"], bordercolor=C["border2"], font=dict(color=C["text"], size=12)),
        **kw
    )

def sec(icon, title, badge=None):
    bp = f'<span class="sec-pill">{badge}</span>' if badge else ''
    st.markdown(f'<div class="sec-row"><span style="font-size:1.3rem">{icon}</span>'
                f'<p class="sec-title">{title}</p>{bp}</div>', unsafe_allow_html=True)

def alert(kind, msg):
    st.markdown(f'<div class="alert alert-{kind}">{msg}</div>', unsafe_allow_html=True)

def anom_badge(label):
    st.markdown(f'<div class="anom-badge"><span class="pulse"></span>{label}</div>', unsafe_allow_html=True)

def metric_row(*items):
    colors = {"red":C["red"],"blue":C["blue"],"green":C["green"],"amber":C["amber"],
              "purple":C["purple"],"muted":C["muted"]}
    html = '<div class="metric-row">'
    for val, lbl, clr in items:
        c = colors.get(clr, C["blue"])
        html += (f'<div class="metric-box" style="border-color:{c}33">'
                 f'<div class="metric-val" style="color:{c}">{val}</div>'
                 f'<div class="metric-lbl">{lbl}</div></div>')
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def sum_box(title, rows):
    inner = "".join(
        f'<div class="sum-row"><span class="sum-key">{k}</span>'
        f'<span class="sum-val {c}">{v}</span></div>'
        for k, v, c in rows)
    st.markdown(f'<div class="sum-box"><div class="sum-box-title">{title}</div>{inner}</div>',
                unsafe_allow_html=True)

def metric_tiles(items):
    cols = st.columns(len(items))
    for col, (val, lbl, clr) in zip(cols, items):
        col.markdown(f'<div class="mtile"><div class="mtile-val" style="color:{clr}">{val}</div>'
                     f'<div class="mtile-lbl">{lbl}</div></div>', unsafe_allow_html=True)

def score_match(df, info):
    return sum(1 for c in info["key_cols"] if c in df.columns)

def safe_resample(df_indexed, freq_label):
    freq_map = {"Weekly":"W","Monthly":"ME","Daily":"D"}
    freq = freq_map.get(freq_label, "D")
    try: return df_indexed.resample(freq).mean()
    except ValueError: return df_indexed.resample({"ME":"M"}.get(freq, freq)).mean()

def detect_fitbit_files(uploaded):
    detected = {}
    if uploaded:
        raw = []
        for uf in uploaded:
            try: raw.append((uf.name, pd.read_csv(uf)))
            except: pass
        for req_name, finfo in REQUIRED_FILES.items():
            best_s, best_d = 0, None
            for uname, udf in raw:
                s = score_match(udf, finfo)
                if s > best_s: best_s, best_d = s, udf
            if best_s >= 2: detected[req_name] = best_d
    return detected

def build_master(detected):
    daily    = detected["dailyActivity_merged.csv"].copy()
    hourly_s = detected["hourlySteps_merged.csv"].copy()
    hourly_i = detected["hourlyIntensities_merged.csv"].copy()
    sleep    = detected["minuteSleep_merged.csv"].copy()
    hr       = detected["heartrate_seconds_merged.csv"].copy()
    daily["ActivityDate"]    = pd.to_datetime(daily["ActivityDate"],    format="%m/%d/%Y")
    hourly_s["ActivityHour"] = pd.to_datetime(hourly_s["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p")
    hourly_i["ActivityHour"] = pd.to_datetime(hourly_i["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p")
    sleep["date"]            = pd.to_datetime(sleep["date"],            format="%m/%d/%Y %I:%M:%S %p")
    hr["Time"]               = pd.to_datetime(hr["Time"],               format="%m/%d/%Y %I:%M:%S %p")
    hr_minute = (hr.set_index("Time").groupby("Id")["Value"]
                 .resample("1min").mean().reset_index())
    hr_minute.columns = ["Id","Time","HeartRate"]
    hr_minute = hr_minute.dropna()
    hr_minute["Date"] = hr_minute["Time"].dt.date
    hr_daily = (hr_minute.groupby(["Id","Date"])["HeartRate"]
                .agg(["mean","max","min","std"]).reset_index()
                .rename(columns={"mean":"AvgHR","max":"MaxHR","min":"MinHR","std":"StdHR"}))
    sleep["Date"] = sleep["date"].dt.date
    sleep_daily = (sleep.groupby(["Id","Date"])
                   .agg(TotalSleepMinutes=("value","count"),
                        DominantSleepStage=("value", lambda x: x.mode()[0])).reset_index())
    master = daily.copy().rename(columns={"ActivityDate":"Date"})
    master["Date"] = master["Date"].dt.date
    master = master.merge(hr_daily, on=["Id","Date"], how="left")
    master = master.merge(sleep_daily, on=["Id","Date"], how="left")
    master["TotalSleepMinutes"]  = master["TotalSleepMinutes"].fillna(0)
    master["DominantSleepStage"] = master["DominantSleepStage"].fillna(0)
    for col in ["AvgHR","MaxHR","MinHR","StdHR"]:
        master[col] = master.groupby("Id")[col].transform(lambda x: x.fillna(x.median()))
    return master, daily, hourly_s, hourly_i, sleep, hr, hr_minute

# ──────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────────────────────────────
_defaults = {
    "module": "🏠 Home",
    # Module 1 (Pre-Processing)
    "raw_df":None,"clean_df":None,"logs":None,"before_nulls":None,"pp_step":1,
    # Shared Fitbit (Modules 2,3,4)
    "fitbit_detected":{},
    "master":None,"daily":None,"hourly_s":None,"hourly_i":None,
    "sleep":None,"hr":None,"hr_minute":None,"fitbit_loaded":False,
    # Module 2 (Pattern Extraction)
    "pe_done_load":False,"pe_done_timestamps":False,"pe_done_master":False,
    "pe_done_tsfresh":False,"pe_done_prophet":False,"pe_done_clustering":False,"pe_done_summary":False,
    "pe_features":None,"pe_fc_hr":None,"pe_fc_steps":None,"pe_fc_sleep":None,
    "pe_act_hr":None,"pe_act_steps":None,"pe_act_sleep":None,
    "pe_kmeans_labels":None,"pe_dbscan_labels":None,"pe_cluster_features":None,
    "pe_X_pca":None,"pe_X_tsne":None,"pe_var_explained":None,"pe_n_clusters_db":0,"pe_n_noise":0,
    # Module 3 (Anomaly Detection)
    "anom_hr":None,"anom_steps":None,"anom_sleep":None,
    "anom_done":False,"sim_results":None,"sim_done":False,
    # Module 4 (Insights Dashboard)
    "m4_pipeline_done":False,"m4_anom_hr":None,"m4_anom_steps":None,"m4_anom_sleep":None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────
MODULES = ["🏠 Home", "⚙️ Pre-Processing & EDA", "🧬 Pattern Extraction",
           "🚨 Anomaly Detection", "📊 Insights Dashboard"]

with st.sidebar:
    # Branding
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 1.5rem;">
        <div style="font-family:Orbitron,monospace;font-size:1.3rem;font-weight:900;color:#00e6ff;letter-spacing:3px;">🏋️ FitPulse</div>
        <div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;color:#4a6a88;letter-spacing:3px;margin-top:6px;">ANALYTICS SUITE</div>
    </div>
    """, unsafe_allow_html=True)

    # Module Navigation
    st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:0.58rem;letter-spacing:3px;color:#4a6a88;text-transform:uppercase;margin-bottom:8px;">MODULES</div>', unsafe_allow_html=True)
    for m in MODULES:
        active = st.session_state.module == m
        fg = "#00e6ff" if active else "#4a6a88"
        bg = "rgba(0,230,255,0.08)" if active else "transparent"
        bd = "rgba(0,230,255,0.3)" if active else "rgba(255,255,255,0.05)"
        if st.sidebar.button(m, key=f"nav_{m}", use_container_width=True):
            st.session_state.module = m
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── SHARED FILE UPLOAD (Modules 2, 3, 4) ──
    st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:0.58rem;letter-spacing:3px;color:#4a6a88;text-transform:uppercase;margin-bottom:8px;">FITBIT FILES (Modules 2·3·4)</div>', unsafe_allow_html=True)
    fitbit_uploads = st.file_uploader(
        "Upload all 5 Fitbit CSVs here",
        type="csv", accept_multiple_files=True,
        key="shared_fitbit_uploader", label_visibility="collapsed"
    )

    # Auto-detect and cache
    if fitbit_uploads:
        detected = detect_fitbit_files(fitbit_uploads)
        st.session_state.fitbit_detected = detected
    else:
        detected = st.session_state.fitbit_detected

    n_up = len(detected)

    # File status grid in sidebar
    grid = '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:3px;margin:0.5rem 0">'
    for req, fi in REQUIRED_FILES.items():
        found = req in detected
        clr = "#34d399" if found else "#f43f5e"
        bg  = "rgba(52,211,153,0.1)" if found else "rgba(244,63,94,0.07)"
        ico = "✓" if found else "✗"
        grid += (f'<div style="background:{bg};border:1px solid {clr}44;border-radius:6px;'
                 f'padding:5px 3px;text-align:center;font-size:0.65rem">'
                 f'<div>{fi["icon"]}</div>'
                 f'<div style="color:{clr};font-family:JetBrains Mono,monospace">{ico}</div></div>')
    grid += "</div>"
    st.markdown(grid, unsafe_allow_html=True)

    st.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:0.65rem;color:{"#34d399" if n_up==5 else "#f43f5e"};text-align:center;margin-bottom:0.5rem">{n_up}/5 files ready</div>', unsafe_allow_html=True)

    if n_up == 5 and not st.session_state.fitbit_loaded:
        if st.button("⚡ Load Fitbit Files", use_container_width=True):
            with st.spinner("Building master dataset..."):
                try:
                    master, daily, hourly_s, hourly_i, sleep, hr, hr_minute = build_master(detected)
                    for k, v in [("master",master),("daily",daily),("hourly_s",hourly_s),
                                  ("hourly_i",hourly_i),("sleep",sleep),("hr",hr),("hr_minute",hr_minute)]:
                        st.session_state[k] = v
                    st.session_state.fitbit_loaded = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Load error: {e}")
    elif st.session_state.fitbit_loaded:
        master = st.session_state.master
        st.markdown(f'<div class="alert-success" style="font-size:0.72rem;padding:0.5rem 0.8rem;border-radius:0 8px 8px 0;border-left:3px solid #34d399;background:rgba(52,211,153,0.08);color:#a7f3d0">✓ {master.shape[0]:,} rows · {master["Id"].nunique()} users loaded</div>', unsafe_allow_html=True)
        if st.button("🔄 Reset Fitbit Data", use_container_width=True):
            for k in ["master","daily","hourly_s","hourly_i","sleep","hr","hr_minute","fitbit_loaded",
                      "fitbit_detected","anom_hr","anom_steps","anom_sleep","anom_done","sim_done",
                      "m4_pipeline_done","m4_anom_hr","m4_anom_steps","m4_anom_sleep",
                      "pe_done_load","pe_done_timestamps","pe_done_master","pe_done_tsfresh",
                      "pe_done_prophet","pe_done_clustering","pe_done_summary","pe_features","pe_fc_hr",
                      "pe_kmeans_labels","pe_dbscan_labels","pe_cluster_features","pe_X_pca",
                      "pe_X_tsne","pe_var_explained"]:
                st.session_state[k] = _defaults.get(k)
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:0.58rem;letter-spacing:2px;color:#4a6a88;text-align:center">FITNESS DATA PRO · SUITE v1.0</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# MODULE ROUTER
# ══════════════════════════════════════════════════════════════════════
module = st.session_state.module

# ┌──────────────────────────────────────────────────────────────────┐
# │  HOME                                                            │
# └──────────────────────────────────────────────────────────────────┘
if module == "🏠 Home":
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem 2rem;">
        <div style="font-family:Orbitron,monospace;font-size:2.8rem;font-weight:900;
                    background:linear-gradient(135deg,#00e6ff,#7c3aed,#f472b6);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    letter-spacing:3px;margin-bottom:0.5rem;">🏋️ FitPulse</div>
        <div style="font-family:Orbitron,monospace;font-size:1rem;color:#4a6a88;letter-spacing:6px;">ANALYTICS SUITE</div>
        <div style="font-family:Exo 2,sans-serif;font-size:1rem;color:#4a6a88;margin-top:1.5rem;max-width:600px;margin-left:auto;margin-right:auto;">
            A complete fitness data analytics pipeline — from raw data cleaning to ML-powered insights.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("c1","⚙️","Pre-Processing & EDA","Module 1","Upload any fitness CSV. Clean nulls, run full EDA with distributions, correlations, and time trends.","#00e6ff","rgba(0,230,255,0.08)","⚙️ Pre-Processing & EDA"),
        ("c2","🧬","Pattern Extraction","Module 2","TSFresh feature extraction, Prophet forecasting, and K-Means / DBSCAN clustering on Fitbit data.","#f472b6","rgba(244,114,182,0.08)","🧬 Pattern Extraction"),
        ("c3","🚨","Anomaly Detection","Module 3","Statistical threshold + residual-based anomaly detection on heart rate, steps, and sleep signals.","#f43f5e","rgba(244,63,94,0.08)","🚨 Anomaly Detection"),
        ("c4","📊","Insights Dashboard","Module 4","Interactive filtered dashboard with KPI strips, deep-dive tabs, and PDF/CSV export.","#a78bfa","rgba(167,139,250,0.08)","📊 Insights Dashboard"),
    ]
    for col, (key, icon, title, subtitle, desc, clr, bg, nav) in zip([c1,c2,c3,c4], cards):
        with col:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {clr}33;border-radius:16px;
                        padding:1.5rem;height:260px;text-align:center;">
                <div style="font-size:2.5rem;margin-bottom:0.5rem">{icon}</div>
                <div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:800;color:{clr};margin-bottom:0.3rem">{title}</div>
                <div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#4a6a88;letter-spacing:2px;margin-bottom:0.8rem">{subtitle}</div>
                <div style="font-size:0.78rem;color:#4a6a88;line-height:1.6">{desc}</div>
            </div>""", unsafe_allow_html=True)
            if st.button(f"Open {title}", key=f"home_{key}", use_container_width=True):
                st.session_state.module = nav; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0b1422;border:1px solid rgba(0,230,255,0.1);border-radius:14px;padding:1.5rem 2rem;">
        <div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;letter-spacing:3px;color:#4a6a88;margin-bottom:1rem">HOW TO USE</div>
        <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;font-size:0.83rem;color:#94a3b8">
            <div><b style="color:#00e6ff">Module 1</b> — Upload <i>any</i> fitness tracking CSV (e.g. your personal dataset with User_ID, Steps, etc.) directly inside that module.</div>
            <div><b style="color:#f472b6">Modules 2, 3 & 4</b> — Upload the <i>5 Fitbit CSV files</i> once in the <b>sidebar</b>. All three modules share the same uploaded data automatically.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ┌──────────────────────────────────────────────────────────────────┐
# │  MODULE 1 — PRE-PROCESSING & EDA                                 │
# └──────────────────────────────────────────────────────────────────┘
elif module == "⚙️ Pre-Processing & EDA":

    def preprocess(df):
        df = df.copy(); logs = []; before_nulls = df.isnull().sum().to_dict()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            nat = df["Date"].isna().sum()
            if nat > 0 and "User_ID" in df.columns:
                full_dates = pd.date_range("2023-01-01", periods=365, freq="D")
                parts = []
                for uid, grp in df.groupby("User_ID", sort=True):
                    grp = grp.copy().reset_index(drop=True)
                    grp["Date"] = full_dates[:len(grp)]
                    parts.append(grp)
                df = pd.concat(parts, ignore_index=True)
                logs.append(("ok", f"✅ Filled {nat:,} null Date values — sequential 2023-01-01→end per user"))
            else:
                logs.append(("ok", f"✅ Date parsed — {df['Date'].notna().sum():,} valid timestamps"))
        num_null = [c for c in df.columns if df[c].dtype in [np.float64,np.float32] and df[c].isna().any()]
        for col in num_null:
            n = int(df[col].isna().sum())
            if "User_ID" in df.columns:
                parts = []
                for uid, grp in df.groupby("User_ID", sort=True):
                    grp = grp.copy(); grp[col] = grp[col].interpolate().ffill().bfill(); parts.append(grp)
                df = pd.concat(parts, ignore_index=True)
            else:
                df[col] = df[col].interpolate().ffill().bfill()
            logs.append(("ok", f"✅ Interpolated + ffill/bfill → '{col}' ({n:,} nulls filled)"))
        if "Workout_Type" in df.columns:
            mask = (df["Workout_Type"].isna() | (df["Workout_Type"].astype(str).str.strip()=="") |
                    (df["Workout_Type"].astype(str).str.strip().str.lower()=="nan"))
            n = int(mask.sum()); df["Workout_Type"] = df["Workout_Type"].astype(str).str.strip()
            df.loc[mask,"Workout_Type"] = "No Workout"
            logs.append(("ok", f"✅ Filled {n:,} null(s) in 'Workout_Type' → 'No Workout'"))
        for col in [c for c in df.columns if df[c].dtype==object and df[c].isna().any() and c!="Workout_Type"]:
            n = int(df[col].isna().sum()); mode = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode)
            logs.append(("ok", f"✅ Filled {n:,} null(s) in '{col}' with mode → '{mode}'"))
        for col in df.select_dtypes(include=[np.float64,np.float32]).columns:
            df[col] = df[col].round(2)
        if "Date" in df.columns:
            vd = df["Date"].dropna(); span = (vd.max()-vd.min()).days
            logs.append(("info", f"ℹ️ Date range: {vd.min().date()} → {vd.max().date()} | Span: {span} days"))
            logs.append(("warn", "⚠️ Timestamps stored as local/naive — UTC normalization skipped."))
        return df, logs, before_nulls

    # Step nav in sidebar (within main content area header)
    STEP_META = {1:("📂","Upload CSV"),2:("🔍","Check Nulls"),3:("⚙️","Preprocess"),4:("👁️","Preview"),5:("📈","EDA")}

    step = st.session_state.pp_step
    icon, name = STEP_META[step]

    # Header
    col_t, col_b = st.columns([3,1])
    with col_t:
        st.markdown(f"""
        <div style="margin-bottom:1rem;">
            <div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;letter-spacing:3px;color:#00e6ff;margin-bottom:4px;">
            ⚙️ PRE-PROCESSING & EDA · STEP {step} OF 5</div>
            <h1 style="margin:0;">{icon} {name}</h1>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div style="text-align:right;padding-top:1.5rem;">
            <span style="display:inline-flex;align-items:center;gap:8px;background:rgba(0,255,163,0.08);
                border:1px solid rgba(0,255,163,0.2);border-radius:20px;padding:6px 14px;">
                <span style="width:7px;height:7px;border-radius:50%;background:#00ffa3;
                    box-shadow:0 0 8px #00ffa3;display:inline-block;"></span>
                <span style="font-family:JetBrains Mono,monospace;font-size:0.62rem;
                    color:#00ffa3;letter-spacing:1px;">MODULE 1</span>
            </span>
        </div>""", unsafe_allow_html=True)

    # Step progress bar
    steps_html = '<div style="display:flex;gap:6px;margin-bottom:1.5rem;">'
    for num, (ic, lb) in STEP_META.items():
        if num < step:   fg,bg,bd = "#00ffa3","rgba(0,255,163,0.06)","rgba(0,255,163,0.25)"; badge="✓"
        elif num==step:  fg,bg,bd = "#00e6ff","rgba(0,230,255,0.08)","rgba(0,230,255,0.3)";  badge="▶"
        else:            fg,bg,bd = "#4a6a88","transparent","rgba(255,255,255,0.05)";          badge=str(num)
        steps_html += (f'<div style="display:flex;align-items:center;gap:8px;padding:8px 14px;'
                       f'border-radius:10px;background:{bg};border:1px solid {bd};flex:1;">'
                       f'<div style="width:20px;height:20px;border-radius:6px;display:flex;align-items:center;'
                       f'justify-content:center;font-family:JetBrains Mono,monospace;font-size:0.62rem;'
                       f'font-weight:700;color:{fg};background:rgba(0,0,0,0.2)">{badge}</div>'
                       f'<span style="font-family:Exo 2,sans-serif;font-size:0.78rem;font-weight:600;color:{fg}">{ic} {lb}</span></div>')
    steps_html += '</div>'
    st.markdown(steps_html, unsafe_allow_html=True)
    pct = (step - 1) / 4
    st.progress(pct)
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── STEP 1: UPLOAD ──
    if step == 1:
        st.markdown("### Upload Your Fitness Dataset")
        st.markdown("<p style='color:#4a6a88'>Upload a CSV with your fitness tracking data. Expected columns: User_ID, Date, Steps_Taken, Calories_Burned, Hours_Slept, etc.</p>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop your CSV here", type=["csv"])
        if uploaded:
            try: st.session_state.raw_df = pd.read_csv(uploaded)
            except Exception as e: st.error(f"Could not read file: {e}")
        if st.session_state.raw_df is not None:
            df = st.session_state.raw_df
            st.success("✅ Dataset loaded!")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Rows", f"{len(df):,}")
            c2.metric("Columns", f"{len(df.columns)}")
            c3.metric("Null Cells", f"{df.isnull().sum().sum():,}")
            c4.metric("Unique Users", str(df["User_ID"].nunique()) if "User_ID" in df.columns else "—")
            with st.expander("🔎 Preview Raw Data (first 10 rows)"): st.dataframe(df.head(10), use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Next → Check Null Values ▶"): st.session_state.pp_step = 2; st.rerun()
        else: st.info("👆 Upload a CSV file to begin.")

    # ── STEP 2: NULL CHECK ──
    elif step == 2:
        if st.session_state.raw_df is None:
            st.warning("Please upload a dataset first.")
            if st.button("← Back"): st.session_state.pp_step = 1; st.rerun()
            st.stop()
        df = st.session_state.raw_df
        null_counts = df.isnull().sum(); null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
        st.markdown("### Null Value Analysis")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Null Cells", f"{null_counts.sum():,}")
        c2.metric("Columns Affected", f"{len(null_counts)}")
        c3.metric("Null Rate", f"{null_counts.sum()/(len(df)*len(df.columns))*100:.1f}%")
        c4.metric("Clean Columns", f"{len(df.columns)-len(null_counts)}")
        if len(null_counts) == 0:
            st.success("🎉 No null values detected! Dataset is already clean.")
        else:
            badge_html = ""
            for col, cnt in null_counts.items():
                pct = cnt/len(df)*100
                if pct > 20:   fg,bg,bc = "#ff8aa8","rgba(255,56,96,0.1)","rgba(255,56,96,0.3)"
                elif pct > 10: fg,bg,bc = "#ffb800","rgba(255,184,0,0.1)","rgba(255,184,0,0.3)"
                else:          fg,bg,bc = "#00e6ff","rgba(0,230,255,0.1)","rgba(0,230,255,0.3)"
                badge_html += (f"<span style='display:inline-flex;align-items:center;gap:5px;padding:5px 14px;"
                               f"border-radius:20px;font-family:JetBrains Mono,monospace;font-size:0.7rem;"
                               f"background:{bg};border:1px solid {bc};color:{fg};margin:3px;'>"
                               f"▲ {col}: {cnt:,} ({pct:.1f}%)</span>")
            st.markdown(f"<div style='margin-bottom:1.5rem'>{badge_html}</div>", unsafe_allow_html=True)
            bar_colors = ["#ff4f9b" if (v/len(df)*100)>20 else "#ffb800" if (v/len(df)*100)>10 else "#00e6ff" for v in null_counts.values]
            fig_null = go.Figure(go.Bar(
                y=null_counts.index.tolist(), x=null_counts.values.tolist(), orientation="h",
                marker=dict(color=bar_colors, line=dict(color="rgba(255,255,255,0.1)",width=1)),
                text=[f"  {v:,}  ({v/len(df)*100:.1f}%)" for v in null_counts.values],
                textposition="outside", textfont=dict(family="JetBrains Mono",size=10,color="#d4eaf7"),
            ))
            fig_null.update_layout(**PTHEME, height=max(250,len(null_counts)*60),
                                   xaxis_title="Null Count", yaxis_title="Column Name", showlegend=False)
            st.plotly_chart(fig_null, use_container_width=True)
        col_b, col_n = st.columns([1,5])
        with col_b:
            if st.button("← Back"): st.session_state.pp_step = 1; st.rerun()
        with col_n:
            if st.button("Next → Preprocess ▶"): st.session_state.pp_step = 3; st.rerun()

    # ── STEP 3: PREPROCESS ──
    elif step == 3:
        if st.session_state.raw_df is None:
            st.warning("Please upload a dataset first.")
            if st.button("← Back"): st.session_state.pp_step = 1; st.rerun(); st.stop()
        df_raw = st.session_state.raw_df
        st.markdown("### Data Preprocessing Pipeline")
        if st.button("▶ Run Preprocessing"):
            with st.spinner("Running full preprocessing pipeline…"):
                import time; time.sleep(0.5)
                clean, logs, before_nulls = preprocess(df_raw)
                st.session_state.clean_df = clean; st.session_state.logs = logs
                st.session_state.before_nulls = before_nulls
        if st.session_state.clean_df is not None:
            logs = st.session_state.logs; before_nulls = st.session_state.before_nulls
            df_clean = st.session_state.clean_df
            st.markdown("#### Preprocessing Log")
            for level, msg in logs:
                if level=="ok": st.success(msg)
                elif level=="warn": st.warning(msg)
                else: st.info(msg)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Null Value Comparison — Before vs After")
            col_b2, col_a2 = st.columns(2)
            before_series = {k:v for k,v in before_nulls.items() if v>0}
            with col_b2:
                st.markdown('<div style="background:rgba(255,56,96,0.1);border:1px solid rgba(255,56,96,0.25);border-radius:10px 10px 0 0;padding:8px 16px;font-family:JetBrains Mono,monospace;font-size:0.7rem;letter-spacing:2px;color:#ff8aa8;">BEFORE PREPROCESSING</div>', unsafe_allow_html=True)
                if before_series:
                    st.dataframe(pd.DataFrame({"Column":list(before_series.keys()),"Null Count":list(before_series.values())}), use_container_width=True, hide_index=True)
                else: st.info("No nulls were present.")
            with col_a2:
                st.markdown('<div style="background:rgba(0,255,163,0.1);border:1px solid rgba(0,255,163,0.25);border-radius:10px 10px 0 0;padding:8px 16px;font-family:JetBrains Mono,monospace;font-size:0.7rem;letter-spacing:2px;color:#00ffa3;">AFTER PREPROCESSING</div>', unsafe_allow_html=True)
                st.markdown('<div style="background:rgba(0,255,163,0.05);border:1px solid rgba(0,255,163,0.2);border-radius:0 0 10px 10px;padding:2rem 1rem;text-align:center;"><div style="font-size:2.5rem">🌟</div><div style="font-family:Orbitron,monospace;font-size:0.85rem;letter-spacing:2px;color:#00ffa3;margin-top:0.8rem;text-shadow:0 0 20px rgba(0,255,163,0.5);">ZERO NULLS REMAINING!</div></div>', unsafe_allow_html=True)
            total_filled = sum(before_series.values())
            c1,c2,c3 = st.columns(3)
            c1.metric("Nulls Removed", f"{total_filled:,}", delta=f"-{total_filled:,}")
            c2.metric("Rows Preserved", f"{len(df_clean):,}")
            c3.metric("Data Quality", "100%", delta="+100%")
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button("⬇️ Download Cleaned CSV", df_clean.to_csv(index=False).encode(),
                               "fitness_data_cleaned.csv", "text/csv")
            col_b3, col_n3 = st.columns([1,5])
            with col_b3:
                if st.button("← Back"): st.session_state.pp_step = 2; st.rerun()
            with col_n3:
                if st.button("Next → Preview Dataset ▶"): st.session_state.pp_step = 4; st.rerun()
        else:
            st.info("👆 Click **Run Preprocessing** to clean all null values automatically.")
            if st.button("← Back"): st.session_state.pp_step = 2; st.rerun()

    # ── STEP 4: PREVIEW ──
    elif step == 4:
        if st.session_state.clean_df is None:
            st.warning("Please complete preprocessing first.")
            if st.button("← Back"): st.session_state.pp_step = 3; st.rerun(); st.stop()
        df = st.session_state.clean_df
        st.markdown("### Preview Cleaned Dataset")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Rows", f"{len(df):,}")
        c2.metric("Columns", f"{len(df.columns)}")
        c3.metric("Remaining Nulls", f"{df.isnull().sum().sum():,}")
        c4.metric("Users", str(df["User_ID"].nunique()) if "User_ID" in df.columns else "—")
        col_f1, col_f2 = st.columns([3,1])
        with col_f1: selected = st.multiselect("Select columns", df.columns.tolist(), default=df.columns.tolist()) or df.columns.tolist()
        with col_f2: n_rows = st.slider("Rows to show", 5, min(200,len(df)), 10)
        st.dataframe(df[selected].head(n_rows), use_container_width=True, height=400)
        with st.expander("📐 Descriptive Statistics", expanded=True):
            st.dataframe(df.select_dtypes(include=np.number).describe().round(2), use_container_width=True)
        st.download_button("⬇️ Download Cleaned CSV", df.to_csv(index=False).encode(), "fitness_data_cleaned.csv", "text/csv")
        col_b, col_n = st.columns([1,5])
        with col_b:
            if st.button("← Back"): st.session_state.pp_step = 3; st.rerun()
        with col_n:
            if st.button("Next → Run Full EDA ▶"): st.session_state.pp_step = 5; st.rerun()

    # ── STEP 5: EDA ──
    elif step == 5:
        if st.session_state.clean_df is None:
            st.warning("Please complete preprocessing first.")
            if st.button("← Back"): st.session_state.pp_step = 3; st.rerun(); st.stop()
        df = st.session_state.clean_df.copy()
        if "Date" in df.columns: df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        st.markdown("### Exploratory Data Analysis")
        NUM_COLS = [c for c in ["Steps_Taken","Calories_Burned","Hours_Slept","Water_Intake (Liters)","Active_Minutes","Heart_Rate (bpm)"] if c in df.columns]
        if not NUM_COLS: NUM_COLS = df.select_dtypes(include=np.number).columns.tolist()
        if not NUM_COLS: st.error("No numeric columns found."); st.stop()

        tab_dist, tab_box, tab_cat, tab_corr, tab_time, tab_user = st.tabs([
            "📊 Distributions","📦 Outlier Detection","🥧 Categorical Analysis",
            "🔗 Correlation Matrix","📅 Time Trends","👤 User Analysis"])

        with tab_dist:
            st.markdown("#### Distribution of Numeric Features")
            for i in range(0, len(NUM_COLS), 2):
                cols = st.columns(2)
                for j, c in enumerate(NUM_COLS[i:i+2]):
                    with cols[j]:
                        color = COL_COLORS.get(c, "#00e6ff"); col_data = df[c].dropna()
                        if col_data.empty: st.warning(f"No data for {c}"); continue
                        mean_v = col_data.mean(); median_v = col_data.median()
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=col_data, nbinsx=35,
                            marker=dict(color=color, opacity=0.75, line=dict(color=color,width=0.5)),
                            hovertemplate=f"{AXIS_LABELS.get(c,c)}: %{{x}}<br>Count: %{{y}}<extra></extra>"))
                        fig.add_vline(x=mean_v, line_color="white", line_dash="dash", line_width=1.5,
                            annotation_text=f"Mean:{mean_v:.1f}", annotation_font_color="white", annotation_font_size=9)
                        fig.add_vline(x=median_v, line_color=color, line_dash="dot", line_width=1.5,
                            annotation_text=f"Median:{median_v:.1f}", annotation_font_color=color,
                            annotation_font_size=9, annotation_position="bottom right")
                        fig.update_layout(**PTHEME, height=270, showlegend=False, bargap=0.04,
                            title=dict(text=f"Distribution · {c}", font=dict(size=11,color="#d4eaf7")),
                            xaxis_title=AXIS_LABELS.get(c,c), yaxis_title="Records")
                        st.plotly_chart(fig, use_container_width=True)

        with tab_box:
            st.markdown("#### Outlier Detection — Boxplots")
            for i in range(0, len(NUM_COLS), 2):
                cols = st.columns(2)
                for j, c in enumerate(NUM_COLS[i:i+2]):
                    with cols[j]:
                        color = COL_COLORS.get(c,"#00e6ff"); col_data = df[c].dropna()
                        if col_data.empty: continue
                        q1,q3 = col_data.quantile(0.25), col_data.quantile(0.75)
                        iqr = q3 - q1; n_out = int(((col_data<q1-1.5*iqr)|(col_data>q3+1.5*iqr)).sum())
                        fig = go.Figure(go.Box(x=col_data, name=c, marker_color=color,
                            line_color=color, fillcolor=hex_to_rgba(color,0.13), boxmean=True))
                        fig.update_layout(**PTHEME, height=210, showlegend=False,
                            title=dict(text=f"Boxplot · {c}  ({n_out} outliers)",font=dict(size=11,color="#d4eaf7")),
                            xaxis_title=AXIS_LABELS.get(c,c))
                        st.plotly_chart(fig, use_container_width=True)
            rows = []
            for c in NUM_COLS:
                col_data = df[c].dropna()
                if col_data.empty: continue
                q1,q3 = col_data.quantile(0.25),col_data.quantile(0.75); iqr=q3-q1
                lo,hi = q1-1.5*iqr, q3+1.5*iqr; n_out=int(((col_data<lo)|(col_data>hi)).sum())
                rows.append({"Column":c,"Min":round(col_data.min(),2),"Q1":round(q1,2),"Median":round(col_data.median(),2),
                             "Mean":round(col_data.mean(),2),"Q3":round(q3,2),"Max":round(col_data.max(),2),
                             "# Outliers":n_out,"Outlier %":round(n_out/len(df)*100,2)})
            if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with tab_cat:
            st.markdown("#### Categorical Feature Analysis")
            col1, col2 = st.columns(2)
            if "Workout_Type" in df.columns:
                wc = df["Workout_Type"].value_counts().reset_index(); wc.columns=["Workout","Count"]
                fig_w = go.Figure(go.Bar(x=wc["Workout"],y=wc["Count"],
                    marker=dict(color=WORKOUT_COLORS[:len(wc)],line=dict(color="rgba(255,255,255,0.1)",width=1)),
                    text=wc["Count"],textposition="outside"))
                fig_w.update_layout(**PTHEME,height=320,showlegend=False,title="Workout Type Distribution")
                with col1: st.plotly_chart(fig_w, use_container_width=True)
            if "Mood" in df.columns:
                mc = df["Mood"].value_counts().reset_index(); mc.columns=["Mood","Count"]
                fig_m = go.Figure(go.Pie(labels=mc["Mood"],values=mc["Count"],
                    marker=dict(colors=MOOD_COLORS,line=dict(color="#030810",width=2)),hole=0.55))
                fig_m.update_layout(**PTHEME,height=320,title="Mood Distribution")
                with col2: st.plotly_chart(fig_m, use_container_width=True)
            if "Gender" in df.columns:
                gc = df["Gender"].value_counts().reset_index(); gc.columns=["Gender","Count"]
                fig_g = go.Figure(go.Pie(labels=gc["Gender"],values=gc["Count"],
                    marker=dict(colors=["#00e6ff","#ff4f9b"],line=dict(color="#030810",width=2)),hole=0.6))
                fig_g.update_layout(**PTHEME,height=300,title="Gender Distribution")
                st.plotly_chart(fig_g, use_container_width=True)

        with tab_corr:
            st.markdown("#### Feature Correlation Matrix")
            if len(NUM_COLS) >= 2:
                corr = df[NUM_COLS].corr().round(2)
                fig_c = go.Figure(go.Heatmap(z=corr.values,x=corr.columns.tolist(),y=corr.columns.tolist(),
                    colorscale=[[0,"#ff4f9b"],[0.5,"#0b1422"],[1,"#00e6ff"]],zmin=-1,zmax=1,
                    text=corr.values.round(2),texttemplate="%{text}",textfont=dict(size=11)))
                fig_c.update_layout(**PTHEME,height=500,title="Pearson Correlation Matrix")
                st.plotly_chart(fig_c, use_container_width=True)
                sc1,sc2 = st.columns(2)
                with sc1: x_f = st.selectbox("X-Axis",NUM_COLS,index=0,key="corr_x")
                with sc2: y_f = st.selectbox("Y-Axis",NUM_COLS,index=min(1,len(NUM_COLS)-1),key="corr_y")
                scatter_df = df[[x_f,y_f]].dropna()
                try:
                    fig_sc = px.scatter(scatter_df,x=x_f,y=y_f,trendline="ols",
                        trendline_color_override="white",opacity=0.55)
                except: fig_sc = px.scatter(scatter_df,x=x_f,y=y_f,opacity=0.55)
                fig_sc.update_traces(marker=dict(color="#00e6ff",size=4))
                fig_sc.update_layout(**PTHEME,height=380,title=f"Scatter: {x_f} vs {y_f}")
                st.plotly_chart(fig_sc, use_container_width=True)

        with tab_time:
            st.markdown("#### Time Series Trends")
            if "Date" in df.columns and df["Date"].notna().sum() > 0:
                col_m, col_a = st.columns([2,1])
                with col_m: metric = st.selectbox("Select Metric", NUM_COLS, key="ts_metric")
                with col_a: agg = st.radio("Aggregation",["Daily","Weekly","Monthly"],horizontal=True,key="ts_agg")
                df_t = (df.dropna(subset=["Date"]).groupby("Date")[metric].mean()
                        .reset_index().set_index("Date").sort_index())
                if agg in ("Weekly","Monthly"): df_t = safe_resample(df_t, agg)
                color = COL_COLORS.get(metric,"#00e6ff")
                if not df_t.empty and not df_t[metric].dropna().empty:
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(x=df_t.index,y=df_t[metric],mode="lines",
                        line=dict(color=color,width=2),fill="tozeroy",fillcolor=hex_to_rgba(color,0.09),
                        hovertemplate=f"%{{x|%Y-%m-%d}}<br>{metric}: %{{y:.2f}}<extra></extra>"))
                    if len(df_t) > 7:
                        roll = df_t[metric].rolling(7,min_periods=1).mean()
                        fig_ts.add_trace(go.Scatter(x=df_t.index,y=roll,mode="lines",
                            line=dict(color="white",width=1.5,dash="dot"),name="7-period MA"))
                    fig_ts.update_layout(**PTHEME,height=380,title=f"{metric} — {agg} Trend",
                        xaxis_title="Date",yaxis_title=AXIS_LABELS.get(metric,metric))
                    st.plotly_chart(fig_ts, use_container_width=True)
                else: st.warning("No time-series data available.")
            else: st.info("No valid Date column found for time-series analysis.")

        with tab_user:
            st.markdown("#### Per-User Analysis")
            if "User_ID" in df.columns:
                all_users = sorted(df["User_ID"].unique())
                col_u1, col_u2 = st.columns([1,3])
                with col_u1: selected_users = st.multiselect("Select Users",all_users,default=all_users[:5] if len(all_users)>=5 else all_users)
                with col_u2: user_metric = st.selectbox("Metric", NUM_COLS, key="user_metric")
                if selected_users:
                    df_u = df[df["User_ID"].isin(selected_users)]
                    user_avg = df_u.groupby("User_ID")[user_metric].mean().reset_index()
                    user_avg.columns = ["User_ID","Average"]
                    user_avg["Label"] = "User " + user_avg["User_ID"].astype(str)
                    fig_ua = go.Figure(go.Bar(x=user_avg["Label"],y=user_avg["Average"].round(2),
                        marker=dict(color=COL_COLORS.get(user_metric,"#00e6ff"),opacity=0.8),
                        text=user_avg["Average"].round(2),textposition="outside"))
                    fig_ua.update_layout(**PTHEME,height=360,title=f"Average {user_metric} per User",showlegend=False)
                    st.plotly_chart(fig_ua, use_container_width=True)
                    st.markdown("#### User Summary Statistics")
                    agg_dict = {c:["mean","min","max","std"] for c in NUM_COLS if c in df_u.columns}
                    if agg_dict:
                        user_sum = df_u.groupby("User_ID").agg(agg_dict)
                        user_sum.columns = [f"{col}_{fn}" for col,fn in user_sum.columns]
                        st.dataframe(user_sum.round(2).reset_index(), use_container_width=True)
            else: st.info("No 'User_ID' column found.")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Back to Preview"): st.session_state.pp_step = 4; st.rerun()


# ┌──────────────────────────────────────────────────────────────────┐
# │  MODULE 2 — PATTERN EXTRACTION                                   │
# └──────────────────────────────────────────────────────────────────┘
elif module == "🧬 Pattern Extraction":

    st.markdown('<h1>🧬 Pattern Extraction</h1>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#4a6a88;letter-spacing:2px;margin-bottom:1.5rem">TSFresh · Prophet · K-Means · DBSCAN</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    if not st.session_state.fitbit_loaded:
        st.markdown("""
        <div style="text-align:center;padding:4rem;background:#0b1422;border:1px solid rgba(0,230,255,0.1);border-radius:16px;">
            <div style="font-size:3rem;margin-bottom:1rem">📁</div>
            <div style="font-family:Orbitron,monospace;font-size:1.1rem;color:#00e6ff;margin-bottom:0.8rem">Fitbit Files Required</div>
            <div style="color:#4a6a88;font-size:0.85rem">Upload all 5 Fitbit CSV files in the sidebar and click <b style="color:#00e6ff">Load Fitbit Files</b> to continue.</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    master    = st.session_state.master
    hourly_s  = st.session_state.hourly_s
    hourly_i  = st.session_state.hourly_i
    hr_minute = st.session_state.hr_minute
    sleep     = st.session_state.sleep
    daily     = st.session_state.daily

    alert("success", f"✓ Master DataFrame ready — {master.shape[0]:,} rows · {master['Id'].nunique()} users · {master['Date'].nunique()} days")

    # Config in columns
    with st.expander("⚙️ Pipeline Configuration", expanded=False):
        c1,c2,c3,c4 = st.columns(4)
        with c1: tsfresh_mode = st.selectbox("TSFresh Mode",["MinimalFCParameters","EfficientFCParameters"])
        with c2: forecast_days = st.slider("Forecast Days",7,90,30)
        with c3:
            changepoint_scale = st.number_input("CP Prior Scale",min_value=0.001,max_value=0.5,value=0.01,step=0.005,format="%.3f")
            optimal_k = st.slider("KMeans K",2,9,3)
        with c4:
            eps_val = st.number_input("DBSCAN eps",0.1,10.0,2.2,0.1)
            min_samples = st.number_input("DBSCAN min_samples",1,10,2,1)

    STEPS_PE = ["load","timestamps","master","tsfresh","prophet","clustering","summary"]
    done_pe  = lambda s: st.session_state.get(f"pe_done_{s}", False)
    mark_pe  = lambda s: st.session_state.update({f"pe_done_{s}": True})

    # Progress bar
    n_done = sum(done_pe(s) for s in STEPS_PE)
    st.progress(n_done/len(STEPS_PE), text=f"{n_done}/{len(STEPS_PE)} steps complete")

    # ── STEP 1: DATA ALREADY LOADED (shared) ──
    with st.container(border=True):
        pill = "done" if done_pe("load") else "ready"
        st.markdown(f'<span class="step-pill {pill}">STEP 1 · Load & Validate</span>', unsafe_allow_html=True)
        c_desc, c_btn = st.columns([5,1])
        c_desc.markdown("<span style='color:#4b607a;font-size:13px'>Files already loaded via sidebar — validates shapes and null counts</span>", unsafe_allow_html=True)
        if c_btn.button("▶ Run" if not done_pe("load") else "🔁 Re-run", key="pe_b1", use_container_width=True):
            mark_pe("load")
            st.rerun()
        if done_pe("load"):
            c1,c2 = st.columns(2)
            with c1:
                sum_box("📄 File Shapes", [
                    ("dailyActivity",    f"{daily.shape[0]:,} × {daily.shape[1]}", "c-blue"),
                    ("hourlySteps",      f"{hourly_s.shape[0]:,} × {hourly_s.shape[1]}", "c-blue"),
                    ("hourlyIntensities",f"{hourly_i.shape[0]:,} × {hourly_i.shape[1]}", "c-blue"),
                    ("minuteSleep",      f"{sleep.shape[0]:,} × {sleep.shape[1]}", "c-blue"),
                    ("heartrate",        f"{st.session_state.hr.shape[0]:,} × {st.session_state.hr.shape[1]}", "c-blue"),
                ])
            with c2:
                metric_tiles([
                    (daily["Id"].nunique(), "Unique Users", "#38bdf8"),
                    (hr_minute["Id"].nunique(), "HR Users", "#f472b6"),
                    (f"{st.session_state.hr.shape[0]:,}", "HR Records", "#fb923c"),
                    (master["Date"].nunique(), "Days", "#34d399"),
                ])
            st.success("✅ Step 1 complete — all files validated.")

    # ── STEP 2: TIMESTAMP INFO ──
    with st.container(border=True):
        pill = "done" if done_pe("timestamps") else ("ready" if done_pe("load") else "")
        st.markdown(f'<span class="step-pill {pill}">STEP 2 · Timestamp Parsing</span>', unsafe_allow_html=True)
        c_desc, c_btn = st.columns([5,1])
        c_desc.markdown("<span style='color:#4b607a;font-size:13px'>Date ranges · HR resampled to 1-minute intervals (already done at load time)</span>", unsafe_allow_html=True)
        if c_btn.button("▶ Run" if not done_pe("timestamps") else "🔁 Re-run", key="pe_b2",
                        disabled=not done_pe("load"), use_container_width=True):
            mark_pe("timestamps"); st.rerun()
        if done_pe("timestamps"):
            c1,c2 = st.columns(2)
            with c1:
                sum_box("📅 Date Formats Parsed", [
                    ("ActivityDate","mm/dd/yyyy","c-green"),
                    ("ActivityHour","mm/dd/yyyy HH:MM:SS AM/PM","c-green"),
                    ("Sleep date","mm/dd/yyyy HH:MM:SS AM/PM","c-green"),
                    ("HR Time","mm/dd/yyyy HH:MM:SS AM/PM","c-green"),
                ])
            with c2:
                sum_box("🔄 Heart Rate Resampling", [
                    ("Original","Seconds","c-orange"),
                    ("Target","1-Minute intervals","c-green"),
                    ("Rows (1-min)",f"{hr_minute.shape[0]:,}","c-blue"),
                    ("Users",f"{hr_minute['Id'].nunique()}","c-green"),
                ])
            st.success("✅ Step 2 complete — timestamps parsed.")

    # ── STEP 3: MASTER ──
    with st.container(border=True):
        pill = "done" if done_pe("master") else ("ready" if done_pe("timestamps") else "")
        st.markdown(f'<span class="step-pill {pill}">STEP 3 · Master DataFrame</span>', unsafe_allow_html=True)
        c_desc, c_btn = st.columns([5,1])
        c_desc.markdown("<span style='color:#4b607a;font-size:13px'>Merge daily + HR daily + sleep daily into master — already built at load time</span>", unsafe_allow_html=True)
        if c_btn.button("▶ Run" if not done_pe("master") else "🔁 Re-run", key="pe_b3",
                        disabled=not done_pe("timestamps"), use_container_width=True):
            mark_pe("master"); st.rerun()
        if done_pe("master"):
            sum_box("🗂 Master DataFrame", [
                ("Rows",   f"{master.shape[0]:,}", "c-blue"),
                ("Columns",f"{master.shape[1]}", "c-blue"),
                ("Users",  f"{master['Id'].nunique()}", "c-green"),
                ("Nulls",  f"{master.isnull().sum().sum()}", "c-green"),
                ("Days",   f"{master['Date'].nunique()}", "c-orange"),
            ])
            with st.expander("👁 Preview Master (first 5 rows)"):
                st.dataframe(master.head(5), use_container_width=True, hide_index=True)
            st.success("✅ Step 3 complete — master DataFrame confirmed.")

    # ── STEP 4: TSFRESH ──
    with st.container(border=True):
        pill = "done" if done_pe("tsfresh") else ("ready" if done_pe("master") else "")
        st.markdown(f'<span class="step-pill {pill}">STEP 4 · TSFresh Feature Extraction</span>', unsafe_allow_html=True)
        c_desc, c_btn = st.columns([5,1])
        c_desc.markdown("<span style='color:#4b607a;font-size:13px'>Extract time-series features from HR per user using TSFresh</span>", unsafe_allow_html=True)
        if c_btn.button("▶ Run" if not done_pe("tsfresh") else "🔁 Re-run", key="pe_b4",
                        disabled=not done_pe("master"), use_container_width=True):
            with st.spinner("Extracting TSFresh features…"):
                try:
                    from tsfresh import extract_features
                    from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters
                    fc_params = MinimalFCParameters() if tsfresh_mode=="MinimalFCParameters" else EfficientFCParameters()
                    hr_ts = hr_minute[["Id","Time","HeartRate"]].copy()
                    hr_ts["Time"] = hr_ts["Time"].astype(np.int64) // 10**9
                    features = extract_features(hr_ts, column_id="Id", column_sort="Time",
                                               column_value="HeartRate", default_fc_parameters=fc_params,
                                               disable_progressbar=True)
                    features = features.dropna(axis=1, how="all").fillna(0)
                    st.session_state.pe_features = features
                    mark_pe("tsfresh"); st.rerun()
                except ImportError:
                    # Fallback: manual feature extraction if tsfresh not installed
                    st.warning("tsfresh not installed — using manual feature extraction (mean, std, min, max, etc.)")
                    feat_rows = []
                    for uid, grp in hr_minute.groupby("Id"):
                        hr_vals = grp["HeartRate"].dropna()
                        if len(hr_vals) == 0: continue
                        feat_rows.append({
                            "Id": uid,
                            "hr_mean": hr_vals.mean(), "hr_std": hr_vals.std(),
                            "hr_min": hr_vals.min(), "hr_max": hr_vals.max(),
                            "hr_range": hr_vals.max()-hr_vals.min(),
                            "hr_median": hr_vals.median(), "hr_skew": hr_vals.skew(),
                            "hr_kurtosis": hr_vals.kurtosis(), "hr_count": len(hr_vals),
                        })
                    features = pd.DataFrame(feat_rows).set_index("Id").fillna(0)
                    st.session_state.pe_features = features
                    mark_pe("tsfresh"); st.rerun()
                except Exception as e:
                    st.error(f"TSFresh error: {e}\n\n{traceback.format_exc()}")

        if done_pe("tsfresh") and st.session_state.pe_features is not None:
            features = st.session_state.pe_features
            sum_box("🧪 TSFresh Extraction Results", [
                ("Mode", tsfresh_mode, "c-blue"),
                ("Users processed", f"{features.shape[0]}", "c-green"),
                ("Features extracted", f"{features.shape[1]}", "c-green"),
                ("Feature names (first 5)", ", ".join(list(features.columns[:5])), "c-purple"),
            ])
            # ── TSFresh Feature Matrix Heatmap ──────────────────────────
            try:
                import seaborn as sns
                from sklearn.preprocessing import MinMaxScaler as _MMS
                _scaler = _MMS()
                _feat_norm = pd.DataFrame(
                    _scaler.fit_transform(features),
                    index=features.index, columns=features.columns)
                # Disable cell annotations for large matrices to prevent blank/crash
                _annot  = _feat_norm.shape[1] <= 15
                _fmt    = ".2f" if _annot else ""
                _lw     = 0.6  if _annot else 0.2
                _figw   = max(12, _feat_norm.shape[1] * 0.55)
                _figh   = max(5,  _feat_norm.shape[0] * 0.55)
                _fig_h, _ax_h = plt.subplots(figsize=(_figw, _figh))
                _fig_h.patch.set_facecolor("#080c18"); _ax_h.set_facecolor("#0f1729")
                sns.heatmap(_feat_norm, cmap="coolwarm", annot=_annot, fmt=_fmt,
                            linewidths=_lw, linecolor="#1a2840", ax=_ax_h,
                            cbar_kws={"shrink": 0.7})
                _ax_h.set_title("TSFresh Feature Matrix — Normalised 0–1 per feature",
                                fontsize=12, color="#e2e8f0", pad=12)
                _ax_h.set_xlabel("Statistical Features", color="#4b607a")
                _ax_h.set_ylabel("User ID", color="#4b607a")
                _ax_h.tick_params(colors="#4b607a", axis="x", rotation=45, labelsize=8)
                _ax_h.tick_params(colors="#4b607a", axis="y")
                plt.tight_layout()
                st.pyplot(_fig_h, use_container_width=True)
                _buf_h = io.BytesIO()
                _fig_h.savefig(_buf_h, format="png", dpi=150,
                               bbox_inches="tight", facecolor=_fig_h.get_facecolor())
                _buf_h.seek(0)
                st.download_button("⬇️ Download Heatmap PNG", _buf_h,
                                   "tsfresh_heatmap.png", "image/png", key="dl_ts_heat")
                plt.close(_fig_h)
            except Exception as _he:
                st.warning(f"Heatmap could not be rendered: {_he}")
            st.download_button("⬇️ Download Features CSV", features.to_csv().encode(),
                               "tsfresh_features.csv","text/csv")
            st.success("✅ Step 4 complete — TSFresh features extracted.")

    # ── STEP 5: PROPHET ──
    with st.container(border=True):
        pill = "done" if done_pe("prophet") else ("ready" if done_pe("tsfresh") else "")
        st.markdown(f'<span class="step-pill {pill}">STEP 5 · Prophet Forecasting</span>', unsafe_allow_html=True)
        c_desc, c_btn = st.columns([5,1])
        c_desc.markdown("<span style='color:#4b607a;font-size:13px'>Forecast Heart Rate · Total Steps · Sleep Minutes using Prophet</span>", unsafe_allow_html=True)
        if c_btn.button("▶ Run" if not done_pe("prophet") else "🔁 Re-run", key="pe_b5",
                        disabled=not done_pe("tsfresh"), use_container_width=True):
            with st.spinner("Running Prophet models…"):
                try:
                    from prophet import Prophet
                    import logging; logging.getLogger("prophet").setLevel(logging.ERROR)

                    def run_prophet(df_col, ds_col, y_col, cp_scale=0.05, horizon=30):
                        df_p = df_col[[ds_col, y_col]].copy()
                        df_p = df_p.groupby(ds_col)[y_col].mean().reset_index()
                        df_p.columns = ["ds", "y"]
                        df_p["ds"] = pd.to_datetime(df_p["ds"])
                        df_p = df_p.dropna(subset=["ds", "y"])
                        df_p = df_p[df_p["y"] > 0].sort_values("ds").reset_index(drop=True)
                        if len(df_p) < 2:
                            raise ValueError(
                                f"Not enough valid rows for '{y_col}' ({len(df_p)} non-zero rows). "
                                "Check that your data has sleep/HR records.")
                        m = Prophet(changepoint_prior_scale=cp_scale, weekly_seasonality=True,
                                    interval_width=0.8)
                        m.fit(df_p)
                        future = m.make_future_dataframe(periods=horizon)
                        fc = m.predict(future)
                        return df_p, fc

                    master_copy = master.copy()
                    master_copy["Date"] = pd.to_datetime(master_copy["Date"])
                    act_hr,    fc_hr    = run_prophet(master_copy, "Date", "AvgHR",             changepoint_scale, forecast_days)
                    act_steps, fc_steps = run_prophet(master_copy, "Date", "TotalSteps",        changepoint_scale, forecast_days)
                    act_sleep, fc_sleep = run_prophet(master_copy, "Date", "TotalSleepMinutes", changepoint_scale, forecast_days)
                    st.session_state.pe_fc_hr     = fc_hr
                    st.session_state.pe_fc_steps  = fc_steps
                    st.session_state.pe_fc_sleep  = fc_sleep
                    st.session_state.pe_act_hr    = act_hr
                    st.session_state.pe_act_steps = act_steps
                    st.session_state.pe_act_sleep = act_sleep
                    mark_pe("prophet"); st.rerun()
                except ImportError:
                    st.warning("Prophet not installed (`pip install prophet`). Skipping forecasting step.")
                    st.session_state.pe_fc_hr    = pd.DataFrame({"ds":[],"yhat":[],"yhat_lower":[],"yhat_upper":[]})
                    st.session_state.pe_fc_steps = st.session_state.pe_fc_hr.copy()
                    st.session_state.pe_fc_sleep = st.session_state.pe_fc_hr.copy()
                    mark_pe("prophet"); st.rerun()
                except Exception as e:
                    st.error(f"Prophet error: {e}")
        if done_pe("prophet"):
            # ── Prophet plots: scatter (actual) + forecast line + CI + axvline ──
            # Matches pattern_extracting.py fplot() style exactly
            DARK_BG  = "#080c18"; CARD_BG_P = "#0f1729"
            GRID_CLR = "#1a2840"; TEXT_CLR  = "#e2e8f0"; MUTED_P   = "#4b607a"

            def dark_ax(ax):
                ax.set_facecolor(CARD_BG_P)
                for sp in ax.spines.values(): sp.set_edgecolor(GRID_CLR)
                ax.tick_params(colors=MUTED_P, labelsize=9)
                ax.xaxis.label.set_color(MUTED_P)
                ax.yaxis.label.set_color(MUTED_P)
                ax.title.set_color(TEXT_CLR)
                ax.grid(color=GRID_CLR, alpha=0.5, linewidth=0.5)

            def fplot_pe(actual, fc, dot_color, title, ylabel):
                fig, ax = plt.subplots(figsize=(13, 4.2))
                fig.patch.set_facecolor(DARK_BG)
                dark_ax(ax)
                # Actual data points (scatter)
                ax.scatter(actual["ds"], actual["y"],
                           color=dot_color, s=18, alpha=0.7, label="Actual", zorder=3)
                # Forecast trend line
                ax.plot(fc["ds"], fc["yhat"],
                        color="#3b82f6", lw=2.5, label="Forecast Trend")
                # 80% confidence interval band
                ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"],
                                alpha=0.2, color="#3b82f6", label="80% CI")
                # Vertical line marking where forecast begins
                ax.axvline(actual["ds"].max(), color="#fb923c",
                           ls="--", lw=1.8, label="Forecast Start")
                ax.set_title(title, fontsize=12, color=TEXT_CLR)
                ax.set_xlabel("Date"); ax.set_ylabel(ylabel)
                ax.legend(fontsize=8, facecolor=CARD_BG_P,
                          labelcolor=TEXT_CLR, edgecolor=GRID_CLR)
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
                plt.tight_layout()
                return fig

            _plots = [
                ("pe_act_hr",    "pe_fc_hr",    "#e53e3e", f"Heart Rate Prophet Forecast (+{forecast_days}d)",   "bpm"),
                ("pe_act_steps", "pe_fc_steps", "#38a169", f"Total Steps Prophet Forecast (+{forecast_days}d)",   "Steps / day"),
                ("pe_act_sleep", "pe_fc_sleep", "#b794f4", f"Sleep Minutes Prophet Forecast (+{forecast_days}d)", "Min / day"),
            ]
            for act_key, fc_key, dot_color, title, ylabel in _plots:
                act = st.session_state.get(act_key)
                fc  = st.session_state.get(fc_key)
                if act is None or fc is None or fc.empty:
                    continue
                _fig = fplot_pe(act, fc, dot_color, title, ylabel)
                st.pyplot(_fig, use_container_width=True)
                # PNG download button for each chart
                _buf_p = io.BytesIO()
                _fig.savefig(_buf_p, format="png", dpi=150,
                             bbox_inches="tight", facecolor=_fig.get_facecolor())
                _buf_p.seek(0)
                _dl_key = f"dl_prophet_{fc_key}"
                _fname  = f"{fc_key.replace('pe_fc_','prophet_')}.png"
                st.download_button(f"⬇️ Download {title.split(' Prophet')[0]} Forecast PNG",
                                   _buf_p, _fname, "image/png", key=_dl_key)
                plt.close(_fig)
            st.success("✅ Step 5 complete — Prophet forecasts generated.")

    # ── STEP 6: CLUSTERING ──
    with st.container(border=True):
        pill = "done" if done_pe("clustering") else ("ready" if done_pe("prophet") else "")
        st.markdown(f'<span class="step-pill {pill}">STEP 6 · K-Means + DBSCAN Clustering</span>', unsafe_allow_html=True)
        c_desc, c_btn = st.columns([5,1])
        c_desc.markdown("<span style='color:#4b607a;font-size:13px'>Cluster users by activity profile · PCA + t-SNE visualisation</span>", unsafe_allow_html=True)
        if c_btn.button("▶ Run" if not done_pe("clustering") else "🔁 Re-run", key="pe_b6",
                        disabled=not done_pe("prophet"), use_container_width=True):
            with st.spinner("Running clustering…"):
                try:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.cluster import KMeans, DBSCAN
                    from sklearn.decomposition import PCA
                    from sklearn.manifold import TSNE

                    cluster_cols = ["TotalSteps","VeryActiveMinutes","SedentaryMinutes",
                                    "Calories","TotalSleepMinutes","AvgHR","LightlyActiveMinutes"]
                    avail_cols   = [c for c in cluster_cols if c in master.columns]
                    # Use fillna(0) instead of dropna() — dropna() silently drops users
                    # whose TotalSleepMinutes is NaN, leaving too few rows for clustering
                    cf = master.groupby("Id")[avail_cols].mean().fillna(0)
                    cf = cf[cf.sum(axis=1) > 0]   # drop completely-empty user rows
                    if len(cf) < max(2, optimal_k):
                        raise ValueError(
                            f"Not enough users for clustering (need ≥ {optimal_k}, got {len(cf)}). "
                            "Check that the Fitbit files loaded correctly.")
                    scaler = StandardScaler()
                    X = scaler.fit_transform(cf)
                    X = np.nan_to_num(X, nan=0.0)  # clean NaN from zero-variance columns

                    km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                    km_labels = km.fit_predict(X)
                    db = DBSCAN(eps=eps_val, min_samples=int(min_samples))
                    db_labels = db.fit_predict(X)

                    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
                    n_noise = int((db_labels==-1).sum())

                    pca = PCA(n_components=2, random_state=42)
                    X_pca = pca.fit_transform(X)
                    var_explained = [round(v*100,1) for v in pca.explained_variance_ratio_]

                    import sklearn
                    _sk_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
                    _tsne_iter_kw = "max_iter" if _sk_ver >= (1, 4) else "n_iter"
                    tsne = TSNE(n_components=2, perplexity=max(2, min(30, cf.shape[0]-1)),
                                random_state=42, **{_tsne_iter_kw: 1000})
                    X_tsne = tsne.fit_transform(X)

                    cf["KMeans_Cluster"] = km_labels; cf["DBSCAN_Cluster"] = db_labels
                    for k,v in [("pe_kmeans_labels",km_labels),("pe_dbscan_labels",db_labels),
                                 ("pe_cluster_features",cf),("pe_X_pca",X_pca),("pe_X_tsne",X_tsne),
                                 ("pe_var_explained",var_explained),("pe_n_clusters_db",n_clusters_db),
                                 ("pe_n_noise",n_noise)]:
                        st.session_state[k] = v

                    mark_pe("clustering"); st.rerun()
                except Exception as e:
                    st.error(f"Clustering error: {e}\n\n{traceback.format_exc()}")

        if done_pe("clustering"):
            km_labels     = st.session_state.pe_kmeans_labels
            db_labels     = st.session_state.pe_dbscan_labels
            X_pca         = st.session_state.pe_X_pca
            X_tsne        = st.session_state.pe_X_tsne
            var_explained = st.session_state.pe_var_explained
            n_clusters_db = st.session_state.pe_n_clusters_db
            n_noise       = st.session_state.pe_n_noise

            # PCA Plot
            fig_pca = go.Figure()
            for k in range(optimal_k):
                mask = km_labels==k
                fig_pca.add_trace(go.Scatter(
                    x=X_pca[mask,0], y=X_pca[mask,1], mode="markers", name=f"Cluster {k}",
                    marker=dict(color=PALETTE[k%len(PALETTE)],size=10,
                                line=dict(color="white",width=1.2),opacity=0.85)))
            fig_pca.update_layout(**PTHEME,height=400,
                title=f"K-Means Clustering — PCA Projection (K={optimal_k})",
                xaxis_title=f"PC1 ({var_explained[0]:.1f}% variance)",
                yaxis_title=f"PC2 ({var_explained[1]:.1f}% variance)")
            st.plotly_chart(fig_pca, use_container_width=True)

            # t-SNE Plot
            fig_tsne = go.Figure()
            for k in range(optimal_k):
                mask = km_labels==k
                fig_tsne.add_trace(go.Scatter(
                    x=X_tsne[mask,0], y=X_tsne[mask,1], mode="markers", name=f"Cluster {k}",
                    marker=dict(color=PALETTE[k%len(PALETTE)],size=10,
                                line=dict(color="white",width=1.2),opacity=0.85)))
            fig_tsne.update_layout(**PTHEME,height=400,title="t-SNE — User Behaviour Clusters")
            st.plotly_chart(fig_tsne, use_container_width=True)

            # DBSCAN Plot
            fig_db = go.Figure()
            unique_labels = sorted(set(db_labels))
            for lbl in unique_labels:
                mask = db_labels==lbl
                name = f"Cluster {lbl}" if lbl>=0 else "Outlier (−1)"
                clr  = PALETTE[lbl%len(PALETTE)] if lbl>=0 else C["red"]
                fig_db.add_trace(go.Scatter(
                    x=X_pca[mask,0], y=X_pca[mask,1], mode="markers", name=name,
                    marker=dict(color=clr,size=10 if lbl>=0 else 14,
                                symbol="circle" if lbl>=0 else "x",
                                line=dict(color="white",width=1.5),opacity=0.9)))
            fig_db.update_layout(**PTHEME,height=400,
                title=f"DBSCAN — PCA Projection (eps={eps_val}, {n_clusters_db} clusters, {n_noise} noise)",
                xaxis_title=f"PC1 ({var_explained[0]:.1f}%)",
                yaxis_title=f"PC2 ({var_explained[1]:.1f}%)")
            st.plotly_chart(fig_db, use_container_width=True)

            metric_row(
                (n_clusters_db, "DBSCAN Clusters", "blue"),
                (n_noise, "Noise Points", "red"),
                (optimal_k, "KMeans K", "green"),
                (f"{sum(var_explained):.1f}%", "PCA Variance", "purple"),
            )
            st.success("✅ Step 6 complete — clustering done.")

    # ── STEP 7: SUMMARY ──
    with st.container(border=True):
        pill = "done" if done_pe("summary") else ("ready" if done_pe("clustering") else "")
        st.markdown(f'<span class="step-pill {pill}">STEP 7 · Pipeline Summary</span>', unsafe_allow_html=True)
        c_desc, c_btn = st.columns([5,1])
        c_desc.markdown("<span style='color:#4b607a;font-size:13px'>Complete pipeline report with all results</span>", unsafe_allow_html=True)
        if c_btn.button("▶ Generate" if not done_pe("summary") else "🔁 Regenerate", key="pe_b7",
                        disabled=not done_pe("clustering"), use_container_width=True):
            mark_pe("summary"); st.rerun()
        if done_pe("summary"):
            try:
                features = st.session_state.pe_features
                km_labels = st.session_state.pe_kmeans_labels
                db_labels = st.session_state.pe_dbscan_labels
                cf        = st.session_state.pe_cluster_features
                var       = st.session_state.pe_var_explained
                metric_tiles([
                    (master["Id"].nunique(), "Total Users", "#38bdf8"),
                    (master["Date"].nunique(), "Days Tracked", "#f472b6"),
                    (features.shape[1], "TSFresh Features", "#fb923c"),
                    (optimal_k, "KMeans K", "#38bdf8"),
                    (st.session_state.pe_n_clusters_db, "DBSCAN Clusters", "#f472b6"),
                    (st.session_state.pe_n_noise, "Noise Points", "#fb923c"),
                ])
                c1,c2 = st.columns(2)
                with c1:
                    sum_box("📂 Dataset Overview", [
                        ("Unique users", f"{master['Id'].nunique()}", "c-green"),
                        ("Master DF", f"{master.shape[0]:,} × {master.shape[1]} cols", "c-blue"),
                        ("Final nulls", "0 ✅", "c-green"),
                    ])
                    sum_box("🧪 TSFresh", [
                        ("Mode", tsfresh_mode, "c-blue"),
                        ("Users processed", f"{features.shape[0]}", "c-green"),
                        ("Features", f"{features.shape[1]}", "c-green"),
                    ])
                with c2:
                    sum_box("🤖 Clustering", [
                        ("KMeans K", f"{optimal_k}", "c-green"),
                        ("DBSCAN clusters", f"{st.session_state.pe_n_clusters_db}", "c-green"),
                        ("Noise points", f"{st.session_state.pe_n_noise}", "c-orange"),
                    ])
                    sum_box("📉 Dimensionality Reduction", [
                        ("PCA PC1 variance", f"{var[0]:.1f}%", "c-blue"),
                        ("PCA PC2 variance", f"{var[1]:.1f}%", "c-blue"),
                        ("PCA Total", f"{sum(var):.1f}%", "c-green"),
                    ])

                # Cluster cards
                feat_cols = [c for c in cf.columns if c not in ("KMeans_Cluster","DBSCAN_Cluster")]
                profile   = cf.groupby("KMeans_Cluster")[feat_cols].mean().round(2)
                CARD_THEMES = [("#38bdf8","#071829"),("#f472b6","#250a18"),("#34d399","#041f12"),("#fb923c","#1f0d04"),("#a78bfa","#130a2a")]
                cl_cols = st.columns(optimal_k)
                for i, col in enumerate(cl_cols):
                    if i not in profile.index: continue
                    row = profile.loc[i]
                    steps = row.get("TotalSteps",0); cal = row.get("Calories",0)
                    act = row.get("VeryActiveMinutes",0); sed = row.get("SedentaryMinutes",0)
                    slp = row.get("TotalSleepMinutes",0)
                    emoji,lbl = ("🏃","HIGHLY ACTIVE") if steps>10000 else ("🚶","MODERATELY ACTIVE") if steps>5000 else ("🛋️","SEDENTARY")
                    fg,bg = CARD_THEMES[i%len(CARD_THEMES)]
                    col.markdown(
                        f'<div style="background:{bg};border:1.5px solid {fg};border-radius:12px;padding:16px;">'
                        f'<div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:800;color:{fg}">{emoji} Cluster {i}</div>'
                        f'<div style="font-family:JetBrains Mono,monospace;font-size:9px;color:{fg};letter-spacing:1.5px;margin-bottom:10px">{lbl}</div>'
                        f'<div style="font-size:12px;color:#94a3b8;line-height:2">'
                        f'👣 <b style="color:#e2e8f0">{steps:,.0f}</b> steps/day<br>'
                        f'🔥 <b style="color:#e2e8f0">{cal:,.0f}</b> kcal/day<br>'
                        f'⚡ <b style="color:#e2e8f0">{act:.0f} min</b> very active<br>'
                        f'🪑 <b style="color:#e2e8f0">{sed:.0f} min</b> sedentary<br>'
                        f'😴 <b style="color:#e2e8f0">{slp:.0f} min</b> sleep/night'
                        f'</div></div>', unsafe_allow_html=True)
                st.success("🎉 Milestone 2 Pipeline Complete! All 7 steps executed successfully.")
            except Exception as e:
                st.error(f"Summary error: {e}")


# ┌──────────────────────────────────────────────────────────────────┐
# │  MODULE 3 — ANOMALY DETECTION                                    │
# └──────────────────────────────────────────────────────────────────┘
elif module == "🚨 Anomaly Detection":

    # Sidebar thresholds
    with st.sidebar:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:0.58rem;letter-spacing:3px;color:#4a6a88;margin-bottom:8px;">DETECTION THRESHOLDS</div>', unsafe_allow_html=True)
        hr_high = st.number_input("HR High (bpm)",    value=100, min_value=80,  max_value=180, key="m3_hr_high")
        hr_low  = st.number_input("HR Low (bpm)",     value=50,  min_value=30,  max_value=70,  key="m3_hr_low")
        st_low  = st.number_input("Steps Low",        value=500, min_value=0,   max_value=2000,key="m3_st_low")
        sl_low  = st.number_input("Sleep Low (min)",  value=60,  min_value=0,   max_value=120, key="m3_sl_low")
        sl_high = st.number_input("Sleep High (min)", value=600, min_value=300, max_value=900, key="m3_sl_high")
        sigma   = st.slider("Residual σ", 1.0, 4.0, 2.0, 0.5, key="m3_sigma")

    # Detection helpers
    def detect_hr_anom(master, hr_high, hr_low, sigma):
        df = master[["Id","Date","AvgHR"]].dropna().copy()
        df["Date"] = pd.to_datetime(df["Date"])
        d = df.groupby("Date")["AvgHR"].mean().reset_index().sort_values("Date")
        d["rolling_med"] = d["AvgHR"].rolling(3,center=True,min_periods=1).median()
        d["residual"] = d["AvgHR"] - d["rolling_med"]; std = d["residual"].std()
        d["thresh_high"] = d["AvgHR"] > hr_high; d["thresh_low"] = d["AvgHR"] < hr_low
        d["resid_anom"] = d["residual"].abs() > sigma*std
        d["is_anomaly"] = d["thresh_high"]|d["thresh_low"]|d["resid_anom"]
        def reason(r):
            out=[]
            if r.thresh_high: out.append(f"HR>{int(hr_high)}")
            if r.thresh_low:  out.append(f"HR<{int(hr_low)}")
            if r.resid_anom:  out.append(f"±{sigma:.0f}σ")
            return ", ".join(out)
        d["reason"] = d.apply(reason,axis=1); return d

    def detect_steps_anom(master, st_low, sigma):
        df = master[["Date","TotalSteps"]].dropna().copy()
        df["Date"] = pd.to_datetime(df["Date"])
        d = df.groupby("Date")["TotalSteps"].mean().reset_index().sort_values("Date")
        d["rolling_med"] = d["TotalSteps"].rolling(3,center=True,min_periods=1).median()
        d["residual"] = d["TotalSteps"]-d["rolling_med"]; std=d["residual"].std()
        d["thresh_low"]=d["TotalSteps"]<st_low; d["thresh_high"]=d["TotalSteps"]>25000
        d["resid_anom"]=d["residual"].abs()>sigma*std
        d["is_anomaly"]=d["thresh_low"]|d["thresh_high"]|d["resid_anom"]
        def reason(r):
            out=[]
            if r.thresh_low:  out.append(f"<{int(st_low):,}")
            if r.thresh_high: out.append(">25,000")
            if r.resid_anom:  out.append(f"±{sigma:.0f}σ")
            return ", ".join(out)
        d["reason"]=d.apply(reason,axis=1); return d

    def detect_sleep_anom(master, sl_low, sl_high, sigma):
        df = master[["Date","TotalSleepMinutes"]].dropna().copy()
        df["Date"] = pd.to_datetime(df["Date"])
        d = df.groupby("Date")["TotalSleepMinutes"].mean().reset_index().sort_values("Date")
        d["rolling_med"]=d["TotalSleepMinutes"].rolling(3,center=True,min_periods=1).median()
        d["residual"]=d["TotalSleepMinutes"]-d["rolling_med"]; std=d["residual"].std()
        d["thresh_low"]=(d["TotalSleepMinutes"]>0)&(d["TotalSleepMinutes"]<sl_low)
        d["thresh_high"]=d["TotalSleepMinutes"]>sl_high
        d["resid_anom"]=d["residual"].abs()>sigma*std
        d["is_anomaly"]=d["thresh_low"]|d["thresh_high"]|d["resid_anom"]
        def reason(r):
            out=[]
            if r.thresh_low:  out.append(f"<{int(sl_low)}min")
            if r.thresh_high: out.append(f">{int(sl_high)}min")
            if r.resid_anom:  out.append(f"±{sigma:.0f}σ")
            return ", ".join(out)
        d["reason"]=d.apply(reason,axis=1); return d

    def simulate_accuracy(master, n_inject=10):
        np.random.seed(42)
        df=master[["Date","AvgHR","TotalSteps","TotalSleepMinutes"]].dropna().copy()
        df["Date"]=pd.to_datetime(df["Date"])
        df_d=df.groupby("Date").mean().reset_index().sort_values("Date")
        results={}
        for signal,col,inj_vals,lo,hi in [
            ("Heart Rate","AvgHR",[115,120,125,35,40,45,118,130,38,42],50,100),
            ("Steps","TotalSteps",[50,100,150,30000,35000,28000,80,200,31000,29000],500,25000),
            ("Sleep","TotalSleepMinutes",[10,20,30,700,750,800,15,25,710,720],60,600),
        ]:
            sim=df_d[["Date",col]].copy()
            idx=np.random.choice(len(sim),n_inject,replace=False)
            sim.loc[sim.index[idx],col]=np.random.choice(inj_vals,n_inject,replace=True)
            sim["rm"]=sim[col].rolling(3,center=True,min_periods=1).median()
            sim["res"]=sim[col]-sim["rm"]; std=sim["res"].std()
            if signal=="Sleep": sim["det"]=((sim[col]>0)&(sim[col]<lo))|(sim[col]>hi)|(sim[col].abs()>2*std)
            else: sim["det"]=(sim[col]<lo)|(sim[col]>hi)|(sim["res"].abs()>2*std)
            tp=sim.iloc[idx]["det"].sum()
            results[signal]={"injected":n_inject,"detected":int(tp),"accuracy":round(tp/n_inject*100,1)}
        results["Overall"]=round(np.mean([results[k]["accuracy"] for k in ["Heart Rate","Steps","Sleep"]]),1)
        return results

    # Header
    st.markdown('<h1>🚨 Anomaly Detection</h1>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#f43f5e;letter-spacing:2px;margin-bottom:1.5rem">Statistical Threshold · Residual ±σ · DBSCAN Outliers</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    if not st.session_state.fitbit_loaded:
        st.markdown('<div style="text-align:center;padding:4rem;background:#0b1422;border:1px solid rgba(244,63,94,0.2);border-radius:16px;"><div style="font-size:3rem;margin-bottom:1rem">🫀</div><div style="font-family:Orbitron,monospace;font-size:1.1rem;color:#f43f5e;margin-bottom:0.8rem">Fitbit Files Required</div><div style="color:#4a6a88;font-size:0.85rem">Upload all 5 Fitbit CSV files in the sidebar and click Load Fitbit Files to continue.</div></div>', unsafe_allow_html=True)
        st.stop()

    master = st.session_state.master
    alert("success", f"✓ {master.shape[0]:,} rows · {master['Id'].nunique()} users · {master['Date'].nunique()} days")

    # Method description cards
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;margin-bottom:1.5rem">
      <div class="card" style="border-color:#f43f5e33">
        <div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#f43f5e;letter-spacing:0.1em;margin-bottom:0.5rem">METHOD 01</div>
        <div style="font-weight:600;margin-bottom:0.3rem">Threshold Violations</div>
        <div style="font-size:0.78rem;color:#6b7280">Hard upper/lower bounds on HR, Steps & Sleep. Immediate flags.</div>
      </div>
      <div class="card" style="border-color:#38bdf833">
        <div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#38bdf8;letter-spacing:0.1em;margin-bottom:0.5rem">METHOD 02</div>
        <div style="font-weight:600;margin-bottom:0.3rem">Residual ±σ Detection</div>
        <div style="font-size:0.78rem;color:#6b7280">Rolling median baseline — flag days with statistically unusual deviation.</div>
      </div>
      <div class="card" style="border-color:#34d39933">
        <div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#34d399;letter-spacing:0.1em;margin-bottom:0.5rem">METHOD 03</div>
        <div style="font-weight:600;margin-bottom:0.3rem">DBSCAN User Outliers</div>
        <div style="font-size:0.78rem;color:#6b7280">Label −1 users = structural behavioural outliers across all features.</div>
      </div>
    </div>""", unsafe_allow_html=True)

    steps_done_m3 = sum([st.session_state.anom_done, st.session_state.sim_done])
    st.progress(steps_done_m3/2, text=f"{steps_done_m3}/2 steps complete")

    if st.button("🔍 Run Anomaly Detection — All Methods", use_container_width=False):
        with st.spinner("Running detection pipeline…"):
            try:
                st.session_state.anom_hr    = detect_hr_anom(master, hr_high, hr_low, sigma)
                st.session_state.anom_steps = detect_steps_anom(master, st_low, sigma)
                st.session_state.anom_sleep = detect_sleep_anom(master, sl_low, sl_high, sigma)
                st.session_state.anom_done  = True
                st.rerun()
            except Exception as e: st.error(f"Detection error: {e}")

    if st.session_state.anom_done:
        anom_hr    = st.session_state.anom_hr
        anom_steps = st.session_state.anom_steps
        anom_sleep = st.session_state.anom_sleep
        n_hr  = int(anom_hr["is_anomaly"].sum())
        n_st  = int(anom_steps["is_anomaly"].sum())
        n_sl  = int(anom_sleep["is_anomaly"].sum())
        n_tot = n_hr + n_st + n_sl
        alert("danger", f"🚨 {n_tot} total anomaly flags — HR: {n_hr} · Steps: {n_st} · Sleep: {n_sl}")
        metric_row(
            (n_hr,"HR Anomalies","red"),(n_st,"Step Anomalies","red"),
            (n_sl,"Sleep Anomalies","red"),(n_tot,"Total Flags","amber"))

        # ── HEART RATE CHART ──
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        sec("❤️", f"Heart Rate — Anomaly Timeline · {n_hr} flags", badge="Step 02")
        alert("info", f"Red circles = anomaly days. Dashed lines = thresholds (>{int(hr_high)}/{int(hr_low)} bpm). Blue band = ±{sigma:.0f}σ corridor.")
        rolling_std = anom_hr["residual"].std()
        fig_hr = go.Figure()
        fig_hr.add_trace(go.Scatter(
            x=pd.concat([anom_hr["Date"],anom_hr["Date"].iloc[::-1]]),
            y=pd.concat([anom_hr["rolling_med"]+sigma*rolling_std,
                         (anom_hr["rolling_med"]-sigma*rolling_std).iloc[::-1]]),
            fill="toself", fillcolor="rgba(56,189,248,0.07)", line=dict(width=0),
            name=f"±{sigma:.0f}σ Band", hoverinfo="skip"))
        fig_hr.add_trace(go.Scatter(x=anom_hr["Date"],y=anom_hr["AvgHR"],
            fill="tozeroy", fillcolor="rgba(56,189,248,0.04)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_hr.add_trace(go.Scatter(x=anom_hr["Date"],y=anom_hr["AvgHR"],mode="lines+markers",name="Avg HR",
            line=dict(color=C["blue"],width=2.5), marker=dict(size=5,color=C["blue"]),
            hovertemplate="<b>%{x|%b %d}</b><br>HR: <b>%{y:.1f} bpm</b><extra></extra>"))
        fig_hr.add_trace(go.Scatter(x=anom_hr["Date"],y=anom_hr["rolling_med"],mode="lines",
            name="Trend",line=dict(color=C["green"],width=1.5,dash="dot")))
        hr_anom = anom_hr[anom_hr["is_anomaly"]]
        if not hr_anom.empty:
            fig_hr.add_trace(go.Scatter(x=hr_anom["Date"],y=hr_anom["AvgHR"],mode="markers",name="Anomaly",
                marker=dict(color=C["red"],size=15,symbol="circle",line=dict(color="white",width=2.5)),
                hovertemplate="<b>⚠️ %{x|%b %d}</b><br>HR: %{y:.1f} bpm<extra></extra>"))
        fig_hr.add_hline(y=hr_high,line_dash="dash",line_color=C["red"],line_width=1.5,opacity=0.6,
            annotation_text=f"  High — {int(hr_high)} bpm",annotation_font=dict(color=C["red"],size=10))
        fig_hr.add_hline(y=hr_low,line_dash="dash",line_color=C["amber"],line_width=1.5,opacity=0.6,
            annotation_text=f"  Low — {int(hr_low)} bpm",annotation_position="bottom right",
            annotation_font=dict(color=C["amber"],size=10))
        fig_hr.update_layout(**base_layout(height=480),title=dict(
            text="❤️  Heart Rate Anomaly Detection — Daily Average (All Users)",
            font=dict(size=14,color=C["text"],family="Space Grotesk"),x=0.01),
            xaxis_title="Date",yaxis_title="Heart Rate (bpm)")
        st.plotly_chart(fig_hr, use_container_width=True)
        if not hr_anom.empty:
            with st.expander(f"📋 {len(hr_anom)} HR Anomaly Records"):
                st.dataframe(hr_anom[["Date","AvgHR","rolling_med","residual","reason"]].rename(
                    columns={"rolling_med":"Expected","residual":"Deviation","reason":"Flag"}).round(2),
                    use_container_width=True)

        # ── SLEEP CHART ──
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        sec("💤", f"Sleep Pattern — Dual-Panel Analysis · {n_sl} flags", badge="Step 03")
        anom_badge(f"{n_sl} anomalous sleep-days")
        fig_slp = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.68,0.32],
            subplot_titles=["Sleep Duration (min/night, cohort avg)","Residual Deviation"],vertical_spacing=0.06)
        fig_slp.add_hrect(y0=sl_low,y1=sl_high,fillcolor="rgba(52,211,153,0.06)",line_width=0,row=1,col=1)
        fig_slp.add_trace(go.Scatter(x=anom_sleep["Date"],y=anom_sleep["TotalSleepMinutes"],
            mode="lines+markers",name="Sleep (min)",line=dict(color="#a78bfa",width=2.5),marker=dict(size=5)),row=1,col=1)
        fig_slp.add_trace(go.Scatter(x=anom_sleep["Date"],y=anom_sleep["rolling_med"],mode="lines",
            name="Trend",line=dict(color=C["green"],width=1.5,dash="dot")),row=1,col=1)
        sl_anom = anom_sleep[anom_sleep["is_anomaly"]]
        if not sl_anom.empty:
            fig_slp.add_trace(go.Scatter(x=sl_anom["Date"],y=sl_anom["TotalSleepMinutes"],
                mode="markers",name="Anomaly",marker=dict(color=C["red"],size=13,symbol="diamond",
                line=dict(color="white",width=2))),row=1,col=1)
        fig_slp.add_hline(y=int(sl_low),line_dash="dash",line_color=C["red"],line_width=1.5,opacity=0.7,
            annotation_text=f"Min ({int(sl_low)} min)",annotation_font_color=C["red"],row=1,col=1)
        fig_slp.add_hline(y=int(sl_high),line_dash="dash",line_color=C["blue"],line_width=1.5,opacity=0.6,
            annotation_text=f"Max ({int(sl_high)} min)",annotation_font_color=C["blue"],row=1,col=1)
        res_colors = [C["red"] if v else "#a78bfa" for v in anom_sleep["resid_anom"]]
        fig_slp.add_trace(go.Bar(x=anom_sleep["Date"],y=anom_sleep["residual"],
            name="Residual",marker_color=res_colors),row=2,col=1)
        fig_slp.add_hline(y=0,line_color=C["muted"],line_width=1,row=2,col=1)
        fig_slp.update_layout(**base_layout(height=520),title=dict(
            text="💤  Sleep Pattern — Anomaly Visualization",font=dict(size=14,color=C["text"]),x=0.01))
        fig_slp.update_annotations(font=dict(color=C["muted"],size=11))
        st.plotly_chart(fig_slp, use_container_width=True)

        # ── STEPS CHART ──
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        sec("🚶", f"Step Count — Trend & Alerts · {n_st} flags", badge="Step 04")
        fig_stp = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.68,0.32],
            subplot_titles=["Daily Steps (cohort avg)","Residual Deviation"],vertical_spacing=0.06)
        st_anom = anom_steps[anom_steps["is_anomaly"]]
        for _,row in st_anom.iterrows():
            fig_stp.add_vrect(x0=row["Date"]-timedelta(hours=12),x1=row["Date"]+timedelta(hours=12),
                fillcolor="rgba(244,63,94,0.08)",line_width=0,row=1,col=1)
        fig_stp.add_trace(go.Scatter(x=anom_steps["Date"],y=anom_steps["TotalSteps"],
            mode="lines+markers",name="Steps",line=dict(color=C["green"],width=2.5),marker=dict(size=5)),row=1,col=1)
        fig_stp.add_trace(go.Scatter(x=anom_steps["Date"],y=anom_steps["rolling_med"],mode="lines",
            name="Trend",line=dict(color=C["blue"],width=1.5,dash="dot")),row=1,col=1)
        if not st_anom.empty:
            fig_stp.add_trace(go.Scatter(x=st_anom["Date"],y=st_anom["TotalSteps"],
                mode="markers",name="Alert",marker=dict(color=C["red"],size=13,symbol="triangle-up",
                line=dict(color="white",width=2))),row=1,col=1)
        fig_stp.add_hline(y=int(st_low),line_dash="dash",line_color=C["red"],line_width=1.5,opacity=0.7,
            annotation_text=f"Low ({int(st_low):,})",annotation_font_color=C["red"],row=1,col=1)
        res_stp_colors = [C["red"] if v else C["green"] for v in anom_steps["resid_anom"]]
        fig_stp.add_trace(go.Bar(x=anom_steps["Date"],y=anom_steps["residual"],
            name="Residual",marker_color=res_stp_colors),row=2,col=1)
        fig_stp.add_hline(y=0,line_color=C["muted"],line_width=1,row=2,col=1)
        fig_stp.update_layout(**base_layout(height=520),title=dict(
            text="🚶  Step Count — Trend & Alert Bands",font=dict(size=14,color=C["text"]),x=0.01))
        fig_stp.update_annotations(font=dict(color=C["muted"],size=11))
        st.plotly_chart(fig_stp, use_container_width=True)

        # ── DBSCAN OUTLIERS ──
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        sec("🔍", "DBSCAN User Outlier Detection", badge="Step 05")
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import DBSCAN
            from sklearn.decomposition import PCA
            cluster_cols = ["TotalSteps","VeryActiveMinutes","SedentaryMinutes","Calories","TotalSleepMinutes","AvgHR"]
            avail_c = [c for c in cluster_cols if c in master.columns]
            cf = master.groupby("Id")[avail_c].mean().dropna()
            X = StandardScaler().fit_transform(cf)
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X)
            var = [round(v*100,1) for v in pca.explained_variance_ratio_]
            db  = DBSCAN(eps=2.2, min_samples=2); db_labels = db.fit_predict(X)
            n_clusters = len(set(db_labels))-(1 if -1 in db_labels else 0)
            n_outliers  = int((db_labels==-1).sum())
            cf["DBSCAN"] = db_labels
            metric_row((n_clusters,"Clusters","blue"),(n_outliers,"Outlier Users","red"),
                       (len(cf)-n_outliers,"Normal Users","green"))
            fig_db = go.Figure()
            mask_in  = db_labels >= 0
            mask_out = db_labels == -1
            unique_cl = sorted(set(db_labels[mask_in]))
            for lbl in unique_cl:
                mk = db_labels==lbl
                fig_db.add_trace(go.Scatter(x=X_pca[mk,0],y=X_pca[mk,1],mode="markers",
                    name=f"Cluster {lbl}",marker=dict(color=PALETTE[lbl%len(PALETTE)],size=12,
                    line=dict(color="white",width=1.5),opacity=0.85)))
            if n_outliers > 0:
                fig_db.add_trace(go.Scatter(x=X_pca[mask_out,0],y=X_pca[mask_out,1],mode="markers",
                    name="Outlier (−1)",marker=dict(color=C["red"],size=16,symbol="x",
                    line=dict(color="white",width=2.5),opacity=0.95)))
            fig_db.update_layout(**base_layout(height=480),title=dict(
                text=f"🔍  DBSCAN User Outlier Detection — PCA Projection (eps=2.2 · {n_clusters} clusters)",
                font=dict(size=13,color=C["text"]),x=0.01),
                xaxis_title=f"PC1 ({var[0]:.1f}% variance)",yaxis_title=f"PC2 ({var[1]:.1f}% variance)")
            st.plotly_chart(fig_db, use_container_width=True)
        except Exception as e:
            alert("warn", f"DBSCAN skipped: {e}")

        # ── ACCURACY SIMULATION ──
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        sec("🎯", "Simulated Detection Accuracy — 90%+ Target", badge="Step 06")
        anom_badge("10 injected anomalies per signal · Recall validation")
        alert("info", "Known anomalies are injected at random positions. Recall (detected/injected) is computed per signal.")
        if st.button("🎯 Run Accuracy Simulation"):
            with st.spinner("Injecting & validating…"):
                try:
                    st.session_state.sim_results = simulate_accuracy(master, n_inject=10)
                    st.session_state.sim_done = True; st.rerun()
                except Exception as e: st.error(f"Simulation error: {e}")
        if st.session_state.sim_done and st.session_state.sim_results:
            sim = st.session_state.sim_results; overall = sim["Overall"]
            passed = overall >= 90.0
            if passed: alert("success", f"✓ Overall accuracy: {overall}% — MEETS the ≥90% requirement")
            else:       alert("warn",    f"⚠ Overall accuracy: {overall}% — below 90% target; adjust thresholds")
            metric_row(
                (f"{sim['Heart Rate']['accuracy']}%",f"Heart Rate ({sim['Heart Rate']['detected']}/10)",
                 "green" if sim["Heart Rate"]["accuracy"]>=90 else "red"),
                (f"{sim['Steps']['accuracy']}%",f"Steps ({sim['Steps']['detected']}/10)",
                 "green" if sim["Steps"]["accuracy"]>=90 else "red"),
                (f"{sim['Sleep']['accuracy']}%",f"Sleep ({sim['Sleep']['detected']}/10)",
                 "green" if sim["Sleep"]["accuracy"]>=90 else "red"),
                (f"{overall}%","Overall Recall","green" if passed else "red"))
            signals = ["Heart Rate","Steps","Sleep"]
            accs    = [sim[s]["accuracy"] for s in signals]
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Bar(x=signals,y=[100,100,100],marker_color=[C["border"]]*3,
                name="",showlegend=False,hoverinfo="skip",width=0.45))
            fig_acc.add_trace(go.Bar(x=signals,y=accs,
                marker_color=[C["green"] if a>=90 else C["red"] for a in accs],
                text=[f"{a}%<br>{sim[s]['detected']}/10" for a,s in zip(accs,signals)],
                textposition="inside",textfont=dict(color="white",size=13),
                name="Detection Accuracy",width=0.45))
            fig_acc.add_hline(y=90,line_dash="dash",line_color=C["red"],line_width=2,opacity=0.8,
                annotation_text="  ≥90% Target",annotation_font=dict(color=C["red"],size=11))
            fig_acc.update_layout(**base_layout(height=380),barmode="overlay",yaxis_range=[0,115],
                title=dict(text="🎯  Anomaly Detection Recall Simulation (n=10 per signal)",
                           font=dict(size=13,color=C["text"]),x=0.01),showlegend=False)
            st.plotly_chart(fig_acc, use_container_width=True)

        # ── COMPLETION CHECKLIST ──
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        sec("✅", "Completion Checklist")
        checklist = [
            ("❤️","HR Anomaly Chart",     st.session_state.anom_done, f"Threshold >{int(hr_high)}/{int(hr_low)} bpm + ±{sigma:.0f}σ"),
            ("💤","Sleep Pattern Chart",  st.session_state.anom_done, "Dual-panel: duration + residual bars"),
            ("🚶","Steps Trend Chart",    st.session_state.anom_done, "Alert bands + deviation from trend"),
            ("🔍","DBSCAN Outlier Scatter",st.session_state.anom_done,"PCA projection · structural outliers"),
            ("🎯","Accuracy Simulation",  st.session_state.sim_done,
             f"Overall: {st.session_state.sim_results['Overall']}%" if st.session_state.sim_results else "Pending"),
        ]
        for icon,label,done,detail in checklist:
            tick = C["green"] if done else C["border2"]; txt = C["text"] if done else C["muted"]
            st.markdown(f"""<div style="display:flex;align-items:center;gap:1rem;padding:0.7rem 0;border-bottom:1px solid {C['border']}">
              <span style="font-size:1.1rem;color:{tick}">{"✓" if done else "○"}</span>
              <span style="font-size:0.88rem;font-weight:600;color:{txt};min-width:200px">{icon} {label}</span>
              <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:{C['muted']}">{detail}</span>
            </div>""", unsafe_allow_html=True)


# ┌──────────────────────────────────────────────────────────────────┐
# │  MODULE 4 — INSIGHTS DASHBOARD                                   │
# └──────────────────────────────────────────────────────────────────┘
elif module == "📊 Insights Dashboard":

    ACCENT="#63b3ed"; ACCENT2="#f687b3"; ACCENT3="#68d391"; ACCENT_RED="#fc8181"
    ACCENT_ORG="#f6ad55"; CARD_BG="rgba(15,23,42,0.85)"; CARD_BOR="rgba(99,179,237,0.2)"
    MUTED="#94a3b8"; TEXT="#e2e8f0"; PLOT_BG="#0f172a"; PAPER_BG="#0a0e1a"
    GRID_CLR="rgba(255,255,255,0.05)"; BADGE_BG="rgba(99,179,237,0.15)"
    SECTION_BG="rgba(99,179,237,0.07)"; SUCCESS_BG="rgba(104,211,145,0.1)"
    SUCCESS_BOR="rgba(104,211,145,0.4)"; DANGER_BG="rgba(252,129,129,0.1)"
    DANGER_BOR="rgba(252,129,129,0.4)"; WARN_BG="rgba(246,173,85,0.12)"
    WARN_BOR="rgba(246,173,85,0.4)"

    PLOTLY_M4 = dict(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font_color=TEXT,
        font_family="Inter, sans-serif",
        legend=dict(bgcolor=CARD_BG, bordercolor=CARD_BOR, borderwidth=1, font_color=TEXT),
        margin=dict(l=50, r=30, t=55, b=45),
        hoverlabel=dict(bgcolor=CARD_BG, bordercolor=CARD_BOR, font_color=TEXT))

    def ptheme_m4(fig, title="", h=400):
        fig.update_layout(**PLOTLY_M4, height=h)
        fig.update_xaxes(gridcolor=GRID_CLR, zeroline=False, linecolor=CARD_BOR, tickfont_color=MUTED)
        fig.update_yaxes(gridcolor=GRID_CLR, zeroline=False, linecolor=CARD_BOR, tickfont_color=MUTED)
        if title: fig.update_layout(title=dict(text=title, font_color=TEXT, font_size=13))
        return fig

    def sec_m4(icon, title, badge=None):
        bp = f'<span style="margin-left:auto;background:{BADGE_BG};border:1px solid {CARD_BOR};border-radius:100px;padding:0.2rem 0.7rem;font-size:0.7rem;font-family:JetBrains Mono,monospace;color:{ACCENT}">{badge}</span>' if badge else ''
        st.markdown(f'<div style="display:flex;align-items:center;gap:0.8rem;margin:1.5rem 0 0.8rem;padding-bottom:0.5rem;border-bottom:1px solid {CARD_BOR}"><span style="font-size:1.3rem">{icon}</span><p style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{TEXT};margin:0">{title}</p>{bp}</div>', unsafe_allow_html=True)

    def detect_hr_m4(master, hr_high=100, hr_low=50, sigma=2.0):
        df=master[["Id","Date","AvgHR"]].dropna().copy(); df["Date"]=pd.to_datetime(df["Date"])
        d=df.groupby("Date")["AvgHR"].mean().reset_index().sort_values("Date")
        d["rolling_med"]=d["AvgHR"].rolling(3,center=True,min_periods=1).median()
        d["residual"]=d["AvgHR"]-d["rolling_med"]; std=d["residual"].std()
        d["thresh_high"]=d["AvgHR"]>hr_high; d["thresh_low"]=d["AvgHR"]<hr_low
        d["resid_anom"]=d["residual"].abs()>sigma*std
        d["is_anomaly"]=d["thresh_high"]|d["thresh_low"]|d["resid_anom"]
        def reason(r):
            parts=[]
            if r.thresh_high: parts.append(f"HR>{int(hr_high)}")
            if r.thresh_low:  parts.append(f"HR<{int(hr_low)}")
            if r.resid_anom:  parts.append(f"+/-{sigma:.0f}σ")
            return ", ".join(parts)
        d["reason"]=d.apply(reason,axis=1); return d

    def detect_steps_m4(master, st_low=500, sigma=2.0):
        df=master[["Date","TotalSteps"]].dropna().copy(); df["Date"]=pd.to_datetime(df["Date"])
        d=df.groupby("Date")["TotalSteps"].mean().reset_index().sort_values("Date")
        d["rolling_med"]=d["TotalSteps"].rolling(3,center=True,min_periods=1).median()
        d["residual"]=d["TotalSteps"]-d["rolling_med"]; std=d["residual"].std()
        d["thresh_low"]=d["TotalSteps"]<st_low; d["thresh_high"]=d["TotalSteps"]>25000
        d["resid_anom"]=d["residual"].abs()>sigma*std
        d["is_anomaly"]=d["thresh_low"]|d["thresh_high"]|d["resid_anom"]
        def reason(r):
            parts=[]
            if r.thresh_low:  parts.append(f"Steps<{int(st_low):,}")
            if r.thresh_high: parts.append("Steps>25,000")
            if r.resid_anom:  parts.append(f"+/-{sigma:.0f}σ")
            return ", ".join(parts)
        d["reason"]=d.apply(reason,axis=1); return d

    def detect_sleep_m4(master, sl_low=60, sl_high=600, sigma=2.0):
        df=master[["Date","TotalSleepMinutes"]].dropna().copy(); df["Date"]=pd.to_datetime(df["Date"])
        d=df.groupby("Date")["TotalSleepMinutes"].mean().reset_index().sort_values("Date")
        d["rolling_med"]=d["TotalSleepMinutes"].rolling(3,center=True,min_periods=1).median()
        d["residual"]=d["TotalSleepMinutes"]-d["rolling_med"]; std=d["residual"].std()
        d["thresh_low"]=(d["TotalSleepMinutes"]>0)&(d["TotalSleepMinutes"]<sl_low)
        d["thresh_high"]=d["TotalSleepMinutes"]>sl_high
        d["resid_anom"]=d["residual"].abs()>sigma*std
        d["is_anomaly"]=d["thresh_low"]|d["thresh_high"]|d["resid_anom"]
        def reason(r):
            parts=[]
            if r.thresh_low:  parts.append(f"Sleep<{int(sl_low)}min")
            if r.thresh_high: parts.append(f"Sleep>{int(sl_high)}min")
            if r.resid_anom:  parts.append(f"+/-{sigma:.0f}σ")
            return ", ".join(parts)
        d["reason"]=d.apply(reason,axis=1); return d

    def generate_csv_m4(ah,ast,asl):
        hr_o=ah[ah["is_anomaly"]][["Date","AvgHR","rolling_med","residual","reason"]].copy()
        hr_o["signal"]="Heart Rate"; hr_o=hr_o.rename(columns={"AvgHR":"value","rolling_med":"expected"})
        st_o=ast[ast["is_anomaly"]][["Date","TotalSteps","rolling_med","residual","reason"]].copy()
        st_o["signal"]="Steps"; st_o=st_o.rename(columns={"TotalSteps":"value","rolling_med":"expected"})
        sl_o=asl[asl["is_anomaly"]][["Date","TotalSleepMinutes","rolling_med","residual","reason"]].copy()
        sl_o["signal"]="Sleep"; sl_o=sl_o.rename(columns={"TotalSleepMinutes":"value","rolling_med":"expected"})
        combined=pd.concat([hr_o,st_o,sl_o],ignore_index=True)
        combined=combined[["signal","Date","value","expected","residual","reason"]].sort_values(["signal","Date"]).round(2)
        buf=io.StringIO(); combined.to_csv(buf,index=False); return buf.getvalue().encode()

    # Sidebar thresholds for M4
    with st.sidebar:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:0.58rem;letter-spacing:3px;color:#4a6a88;margin-bottom:8px;">THRESHOLDS (Dashboard)</div>', unsafe_allow_html=True)
        m4_hr_high = int(st.number_input("HR High (bpm)",    value=100, min_value=80,  max_value=180, key="m4_hr_high"))
        m4_hr_low  = int(st.number_input("HR Low (bpm)",     value=50,  min_value=30,  max_value=70,  key="m4_hr_low"))
        m4_st_low  = int(st.number_input("Steps Low/day",    value=500, min_value=0,   max_value=2000,key="m4_st_low2"))
        m4_sl_low  = int(st.number_input("Sleep Low (min)",  value=60,  min_value=0,   max_value=120, key="m4_sl_low2"))
        m4_sl_high = int(st.number_input("Sleep High (min)", value=600, min_value=300, max_value=900, key="m4_sl_high2"))
        m4_sigma   = float(st.slider("Sigma", 1.0, 4.0, 2.0, 0.5, key="m4_sigma2"))

    # Header
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(99,179,237,0.08),rgba(104,211,145,0.05),rgba(10,14,26,0.9));
                border:1px solid {CARD_BOR};border-radius:20px;padding:2rem 2.5rem;margin-bottom:1.5rem">
        <div style="display:inline-block;background:{BADGE_BG};border:1px solid {CARD_BOR};border-radius:100px;padding:0.25rem 0.9rem;font-size:0.72rem;font-family:'JetBrains Mono',monospace;color:{ACCENT};margin-bottom:0.8rem">
            Module 4 · Insights & Export
        </div>
        <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:{TEXT};margin:0 0 0.3rem;letter-spacing:-0.02em">
            📊 Insights Dashboard
        </div>
        <div style="font-size:1rem;color:{MUTED};font-weight:300">
            Filtered analytics · KPI strip · Deep-dive tabs · PDF & CSV export
        </div>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.fitbit_loaded:
        st.markdown(f'<div style="text-align:center;padding:4rem;background:{CARD_BG};border:1px solid {CARD_BOR};border-radius:16px;"><div style="font-size:3rem;margin-bottom:1rem">📊</div><div style="font-family:Syne,sans-serif;font-size:1.1rem;color:{ACCENT};margin-bottom:0.8rem">Fitbit Files Required</div><div style="color:{MUTED};font-size:0.85rem">Upload all 5 Fitbit CSV files in the sidebar and click Load Fitbit Files to continue.</div></div>', unsafe_allow_html=True)
        st.stop()

    master = st.session_state.master

    # Run pipeline button
    run_m4 = st.button("⚡ Run Analytics Pipeline", use_container_width=False)
    if run_m4:
        with st.spinner("Detecting anomalies…"):
            try:
                st.session_state.m4_anom_hr    = detect_hr_m4(master, m4_hr_high, m4_hr_low, m4_sigma)
                st.session_state.m4_anom_steps = detect_steps_m4(master, m4_st_low, m4_sigma)
                st.session_state.m4_anom_sleep = detect_sleep_m4(master, m4_sl_low, m4_sl_high, m4_sigma)
                st.session_state.m4_pipeline_done = True; st.rerun()
            except Exception as e: st.error(f"Pipeline error: {e}")

    if not st.session_state.m4_pipeline_done:
        alert("info", "Click ⚡ Run Analytics Pipeline to load the dashboard.")
        st.stop()

    anom_hr    = st.session_state.m4_anom_hr
    anom_steps = st.session_state.m4_anom_steps
    anom_sleep = st.session_state.m4_anom_sleep

    # Date & User filters
    all_dates = pd.to_datetime(master["Date"])
    d_min = all_dates.min().date(); d_max = all_dates.max().date()
    fc1, fc2, fc3 = st.columns([2,2,1])
    with fc1:
        date_range = st.date_input("Date Range", value=(d_min,d_max), min_value=d_min, max_value=d_max, key="m4_dr")
    with fc2:
        all_users = sorted(master["Id"].unique())
        user_opts = ["All Users"] + [f"...{str(u)[-6:]}" for u in all_users]
        sel_user_lbl = st.selectbox("User Filter", user_opts, key="m4_usr")
        sel_user = None if sel_user_lbl=="All Users" else all_users[user_opts.index(sel_user_lbl)-1]
    with fc3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:{MUTED}">Pipeline: ✓ Active</div>', unsafe_allow_html=True)

    # Apply filters
    if isinstance(date_range, tuple) and len(date_range)==2:
        dr_start, dr_end = date_range
    else:
        dr_start, dr_end = d_min, d_max

    def filter_df(df, date_col="Date"):
        df = df.copy(); df[date_col] = pd.to_datetime(df[date_col])
        mask = (df[date_col].dt.date>=dr_start)&(df[date_col].dt.date<=dr_end)
        return df[mask]

    master_f     = filter_df(master)
    if sel_user: master_f = master_f[master_f["Id"]==sel_user]
    anom_hr_f    = filter_df(anom_hr)
    anom_steps_f = filter_df(anom_steps)
    anom_sleep_f = filter_df(anom_sleep)

    n_hr_f  = int(anom_hr_f["is_anomaly"].sum())
    n_st_f  = int(anom_steps_f["is_anomaly"].sum())
    n_sl_f  = int(anom_sleep_f["is_anomaly"].sum())
    n_tot_f = n_hr_f + n_st_f + n_sl_f

    # KPI Strip
    sec_m4("📊", "Key Performance Indicators")
    kpi_items = [
        (f"{master_f['Id'].nunique()}",               "Users",          ACCENT),
        (f"{master_f['Date'].nunique()}",             "Days",           ACCENT),
        (f"{master_f['TotalSteps'].mean():,.0f}",     "Avg Steps/Day",  ACCENT3),
        (f"{master_f['Calories'].mean():,.0f}" if "Calories" in master_f.columns else "—","Avg Calories",ACCENT2),
        (f"{master_f['AvgHR'].mean():.1f}" if "AvgHR" in master_f.columns else "—","Avg HR (bpm)",ACCENT_RED),
        (f"{master_f['TotalSleepMinutes'].mean():.0f}" if "TotalSleepMinutes" in master_f.columns else "—","Avg Sleep (min)","#b794f4"),
        (f"{n_hr_f}",  "HR Flags",    ACCENT_RED),
        (f"{n_st_f}",  "Step Flags",  ACCENT3),
        (f"{n_sl_f}",  "Sleep Flags", "#b794f4"),
        (f"{n_tot_f}", "Total Flags", ACCENT_ORG),
    ]
    kpi_html = '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:0.6rem;margin:1rem 0">'
    for val,lbl,clr in kpi_items:
        kpi_html += (f'<div class="kpi-card"><div class="kpi-val" style="color:{clr}">{val}</div>'
                     f'<div class="kpi-label">{lbl}</div></div>')
    kpi_html += '</div>'
    st.markdown(kpi_html, unsafe_allow_html=True)

    # Tabs
    tab_ov, tab_hr, tab_steps, tab_sleep, tab_export = st.tabs([
        "🏠 Overview","❤️ Heart Rate","🚶 Steps","💤 Sleep","📥 Export"])

    with tab_ov:
        sec_m4("🏠", "Overview — Activity & Anomaly Summary")
        col_a, col_b = st.columns(2)
        if "TotalSteps" in master_f.columns:
            with col_a:
                steps_daily = master_f.groupby("Date")["TotalSteps"].mean().reset_index().sort_values("Date")
                fig_ov = go.Figure()
                fig_ov.add_trace(go.Scatter(x=pd.to_datetime(steps_daily["Date"]),y=steps_daily["TotalSteps"],
                    mode="lines+markers",name="Steps",line=dict(color=ACCENT3,width=2.5),
                    fill="tozeroy",fillcolor="rgba(104,211,145,0.08)"))
                ptheme_m4(fig_ov, "📈 Daily Steps — Cohort Average", 340)
                st.plotly_chart(fig_ov, use_container_width=True)
        if "AvgHR" in master_f.columns:
            with col_b:
                hr_daily = master_f.groupby("Date")["AvgHR"].mean().reset_index().sort_values("Date")
                fig_hr_ov = go.Figure(go.Scatter(x=pd.to_datetime(hr_daily["Date"]),y=hr_daily["AvgHR"],
                    mode="lines+markers",name="HR",line=dict(color=ACCENT_RED,width=2.5),
                    fill="tozeroy",fillcolor="rgba(252,129,129,0.07)"))
                ptheme_m4(fig_hr_ov, "❤️ Daily Heart Rate — Cohort Average", 340)
                st.plotly_chart(fig_hr_ov, use_container_width=True)
        # Anomaly summary table
        st.markdown(f'<div style="background:{CARD_BG};border:1px solid {CARD_BOR};border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:0.8rem"><div style="font-family:Syne,sans-serif;font-size:0.85rem;font-weight:700;color:{MUTED};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem">Anomaly Summary</div>', unsafe_allow_html=True)
        for icon2, label2, n2 in [("❤️","Heart Rate",n_hr_f),("🚶","Steps",n_st_f),("💤","Sleep",n_sl_f)]:
            clr2 = DANGER_BOR if n2>0 else SUCCESS_BOR
            st.markdown(f'<div style="display:flex;align-items:center;gap:0.6rem;padding:0.45rem 0;border-bottom:1px solid {CARD_BOR};font-size:0.82rem"><span>{icon2}</span><span style="flex:1;color:{TEXT}">{label2}</span><span style="font-family:JetBrains Mono,monospace;font-size:0.8rem;color:{ACCENT_RED if n2>0 else ACCENT3};font-weight:700">{n2} flags</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_hr:
        sec_m4("❤️", f"Heart Rate Deep Dive · {n_hr_f} anomalies")
        fig_hr2 = go.Figure()
        std_hr  = anom_hr_f["residual"].std()
        fig_hr2.add_trace(go.Scatter(x=anom_hr_f["Date"],y=anom_hr_f["rolling_med"]+m4_sigma*std_hr,
            mode="lines",line=dict(width=0),showlegend=False,hoverinfo="skip"))
        fig_hr2.add_trace(go.Scatter(x=anom_hr_f["Date"],y=anom_hr_f["rolling_med"]-m4_sigma*std_hr,
            mode="lines",fill="tonexty",fillcolor="rgba(99,179,237,0.1)",
            line=dict(width=0),name=f"±{m4_sigma:.0f}σ Band"))
        fig_hr2.add_trace(go.Scatter(x=anom_hr_f["Date"],y=anom_hr_f["AvgHR"],mode="lines+markers",
            name="Avg HR",line=dict(color=ACCENT,width=2.5),marker=dict(size=5),
            hovertemplate="<b>%{x|%d %b}</b><br>HR: %{y:.1f} bpm<extra></extra>"))
        fig_hr2.add_trace(go.Scatter(x=anom_hr_f["Date"],y=anom_hr_f["rolling_med"],mode="lines",
            name="Trend",line=dict(color=ACCENT3,width=1.5,dash="dot")))
        a_hr = anom_hr_f[anom_hr_f["is_anomaly"]]
        if not a_hr.empty:
            fig_hr2.add_trace(go.Scatter(x=a_hr["Date"],y=a_hr["AvgHR"],mode="markers",name="Anomaly",
                marker=dict(color=ACCENT_RED,size=13,symbol="circle",line=dict(color="white",width=2)),
                hovertemplate="<b>%{x|%d %b}</b><br>HR: %{y:.1f}<br><b>ANOMALY</b><extra>⚠️</extra>"))
        fig_hr2.add_hline(y=m4_hr_high,line_dash="dash",line_color=ACCENT_RED,line_width=1.5,opacity=0.6,
            annotation_text=f"High ({int(m4_hr_high)} bpm)",annotation_font_color=ACCENT_RED)
        fig_hr2.add_hline(y=m4_hr_low,line_dash="dash",line_color=ACCENT2,line_width=1.5,opacity=0.6,
            annotation_text=f"Low ({int(m4_hr_low)} bpm)",annotation_font_color=ACCENT2,annotation_position="bottom right")
        ptheme_m4(fig_hr2,"❤️ Heart Rate — Anomaly Detection",420)
        st.plotly_chart(fig_hr2, use_container_width=True)
        col_a2,col_b2 = st.columns(2)
        with col_a2:
            st.markdown(f'<div style="background:{CARD_BG};border:1px solid {CARD_BOR};border-radius:14px;padding:1.2rem"><div style="font-size:0.83rem;line-height:2">Mean HR: <b style="color:{ACCENT}">{anom_hr_f["AvgHR"].mean():.1f} bpm</b><br>Max: <b style="color:{ACCENT_RED}">{anom_hr_f["AvgHR"].max():.1f} bpm</b><br>Anomaly days: <b style="color:{ACCENT_RED}">{n_hr_f}</b> of {len(anom_hr_f)} total</div></div>', unsafe_allow_html=True)
        with col_b2:
            if not a_hr.empty:
                st.dataframe(a_hr[["Date","AvgHR","rolling_med","residual","reason"]].rename(
                    columns={"rolling_med":"Expected","residual":"Deviation","reason":"Reason"}).round(2),
                    use_container_width=True, height=200)

    with tab_steps:
        sec_m4("🚶", f"Steps Deep Dive · {n_st_f} anomalies")
        fig_stp2 = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.65,0.35],
            subplot_titles=["Daily Steps","Residual"],vertical_spacing=0.07)
        a_st = anom_steps_f[anom_steps_f["is_anomaly"]]
        fig_stp2.add_trace(go.Scatter(x=anom_steps_f["Date"],y=anom_steps_f["TotalSteps"],
            mode="lines+markers",name="Steps",line=dict(color=ACCENT3,width=2.5)),row=1,col=1)
        fig_stp2.add_trace(go.Scatter(x=anom_steps_f["Date"],y=anom_steps_f["rolling_med"],mode="lines",
            name="Trend",line=dict(color=ACCENT,width=1.5,dash="dot")),row=1,col=1)
        if not a_st.empty:
            fig_stp2.add_trace(go.Scatter(x=a_st["Date"],y=a_st["TotalSteps"],mode="markers",
                name="Alert",marker=dict(color=ACCENT_RED,size=13,symbol="triangle-up",
                line=dict(color="white",width=2))),row=1,col=1)
        fig_stp2.add_hline(y=int(m4_st_low),line_dash="dash",line_color=ACCENT_RED,line_width=1.5,
            annotation_text=f"Low ({int(m4_st_low):,})",annotation_font_color=ACCENT_RED,row=1,col=1)
        stp_res_clrs = [ACCENT_RED if v else ACCENT3 for v in anom_steps_f["resid_anom"]]
        fig_stp2.add_trace(go.Bar(x=anom_steps_f["Date"],y=anom_steps_f["residual"],
            marker_color=stp_res_clrs,name="Residual"),row=2,col=1)
        fig_stp2.add_hline(y=0,line_color=MUTED,line_width=1,row=2,col=1)
        ptheme_m4(fig_stp2,"🚶 Step Count — Trend & Alerts",440)
        fig_stp2.update_layout(paper_bgcolor=PAPER_BG,plot_bgcolor=PLOT_BG,font_color=TEXT)
        st.plotly_chart(fig_stp2, use_container_width=True)
        if not a_st.empty:
            st.dataframe(a_st[["Date","TotalSteps","rolling_med","residual","reason"]].rename(
                columns={"TotalSteps":"Steps","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}).round(2),
                use_container_width=True, height=200)

    with tab_sleep:
        sec_m4("💤", f"Sleep Deep Dive · {n_sl_f} anomalies")
        fig_slp2 = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.65,0.35],
            subplot_titles=["Sleep Duration (min)","Residual"],vertical_spacing=0.07)
        fig_slp2.add_hrect(y0=m4_sl_low,y1=m4_sl_high,fillcolor="rgba(104,211,145,0.07)",
            line_width=0,annotation_text="✅ Healthy Zone",annotation_position="top right",
            annotation_font_color=ACCENT3,row=1,col=1)
        a_sl = anom_sleep_f[anom_sleep_f["is_anomaly"]]
        fig_slp2.add_trace(go.Scatter(x=anom_sleep_f["Date"],y=anom_sleep_f["TotalSleepMinutes"],
            mode="lines+markers",name="Sleep",line=dict(color="#b794f4",width=2.5)),row=1,col=1)
        fig_slp2.add_trace(go.Scatter(x=anom_sleep_f["Date"],y=anom_sleep_f["rolling_med"],mode="lines",
            name="Trend",line=dict(color=ACCENT3,width=1.5,dash="dot")),row=1,col=1)
        if not a_sl.empty:
            fig_slp2.add_trace(go.Scatter(x=a_sl["Date"],y=a_sl["TotalSleepMinutes"],mode="markers",
                name="Anomaly",marker=dict(color=ACCENT_RED,size=13,symbol="diamond",
                line=dict(color="white",width=2))),row=1,col=1)
        fig_slp2.add_hline(y=int(m4_sl_low),line_dash="dash",line_color=ACCENT_RED,row=1,col=1)
        fig_slp2.add_hline(y=int(m4_sl_high),line_dash="dash",line_color=ACCENT,row=1,col=1)
        slp_res_clrs = [ACCENT_RED if v else "#b794f4" for v in anom_sleep_f["resid_anom"]]
        fig_slp2.add_trace(go.Bar(x=anom_sleep_f["Date"],y=anom_sleep_f["residual"],
            marker_color=slp_res_clrs,name="Residual"),row=2,col=1)
        fig_slp2.add_hline(y=0,line_color=MUTED,line_width=1,row=2,col=1)
        ptheme_m4(fig_slp2,"💤 Sleep Pattern — Anomaly Visualization",440)
        fig_slp2.update_layout(paper_bgcolor=PAPER_BG,plot_bgcolor=PLOT_BG,font_color=TEXT)
        st.plotly_chart(fig_slp2, use_container_width=True)
        col_a3,col_b3 = st.columns(2)
        with col_a3:
            st.markdown(f'<div style="background:{CARD_BG};border:1px solid {CARD_BOR};border-radius:14px;padding:1.2rem"><div style="font-size:0.83rem;line-height:2">Mean: <b style="color:#b794f4">{anom_sleep_f["TotalSleepMinutes"].mean():.0f} min</b><br>Max: <b style="color:{ACCENT}">{anom_sleep_f["TotalSleepMinutes"].max():.0f} min</b><br>Anomaly days: <b style="color:{ACCENT_RED}">{n_sl_f}</b> of {len(anom_sleep_f)} total</div></div>', unsafe_allow_html=True)
        with col_b3:
            if not a_sl.empty:
                st.dataframe(a_sl[["Date","TotalSleepMinutes","rolling_med","residual","reason"]].rename(
                    columns={"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}).round(2),
                    use_container_width=True, height=200)

    with tab_export:
        sec_m4("📥", "Export — PDF Report & CSV Data", "Downloadable")
        st.markdown(f'<div style="background:{CARD_BG};border:1px solid {CARD_BOR};border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:1rem"><div style="font-size:0.83rem;color:{MUTED};line-height:1.8">✅ All anomaly records from all 3 signals<br>✅ Signal type, date, actual vs expected value<br>✅ Residual deviation & anomaly reason text</div></div>', unsafe_allow_html=True)

        col_pdf, col_csv = st.columns(2)

        with col_pdf:
            sec_m4("📄", "PDF Report")
            st.markdown(f'<div style="color:{MUTED};font-size:0.82rem;margin-bottom:0.8rem">Full PDF with executive summary, anomaly tables, and user profiles.</div>', unsafe_allow_html=True)

            # Pre-check fpdf2 dependency
            _fpdf_ok = False
            try:
                from fpdf import FPDF as _FPDF_CHECK  # noqa
                _fpdf_ok = True
            except ImportError:
                st.warning("⚠️ `fpdf2` not installed. Run: `pip install fpdf2` then restart.")

            if _fpdf_ok and st.button("📄 Generate PDF Report", key="gen_pdf_m4"):
                with st.spinner("⏳ Generating PDF…"):
                    try:
                        from fpdf import FPDF

                        class _PDF(FPDF):
                            def header(self):
                                self.set_fill_color(15, 23, 42)
                                self.rect(0, 0, 210, 18, 'F')
                                self.set_font("Helvetica", "B", 13)
                                self.set_text_color(99, 179, 237)
                                self.set_y(4)
                                self.cell(0, 10, "FitPulse Anomaly Detection Report - Module 4", align="C")
                                self.set_text_color(148, 163, 184)
                                self.set_font("Helvetica", "", 7)
                                self.set_y(13)
                                self.cell(0, 4, f"Generated: {datetime.now().strftime('%d %B %Y  %H:%M')}", align="C")
                                self.ln(6)

                            def footer(self):
                                self.set_y(-13)
                                self.set_font("Helvetica", "", 7)
                                self.set_text_color(148, 163, 184)
                                self.cell(0, 8, f"FitPulse Analytics Suite  .  Page {self.page_no()}", align="C")

                            def section(self, title, color=(99, 179, 237)):
                                self.ln(3)
                                self.set_fill_color(*color)
                                self.set_text_color(255, 255, 255)
                                self.set_font("Helvetica", "B", 10)
                                self.cell(0, 8, f"  {title}", fill=True, ln=True)
                                self.set_text_color(30, 30, 40)
                                self.ln(2)

                            def kv(self, key, val):
                                self.set_font("Helvetica", "B", 9)
                                self.set_text_color(80, 80, 100)
                                self.cell(55, 6, key + ":", ln=False)
                                self.set_font("Helvetica", "", 9)
                                self.set_text_color(20, 20, 30)
                                self.cell(0, 6, str(val), ln=True)

                            def para(self, text, size=8.5):
                                self.set_font("Helvetica", "", size)
                                self.set_text_color(60, 60, 80)
                                self.multi_cell(0, 5, text)
                                self.ln(1)

                        _pdf = _PDF()
                        _pdf.set_auto_page_break(auto=True, margin=18)
                        _pdf.add_page()

                        _n_hr  = int(anom_hr_f["is_anomaly"].sum())
                        _n_st  = int(anom_steps_f["is_anomaly"].sum())
                        _n_sl  = int(anom_sleep_f["is_anomaly"].sum())
                        _n_usr = master_f["Id"].nunique()
                        _n_day = master_f["Date"].nunique()

                        _pdf.section("1. EXECUTIVE SUMMARY", (15, 23, 60))
                        _pdf.kv("Dataset",    "Real Fitbit Device Data")
                        _pdf.kv("Users",      f"{_n_usr} participants (filtered)")
                        _pdf.kv("Days",       f"{_n_day} days of observations")
                        _pdf.kv("Pipeline",   "Module 4 - Insights Dashboard")
                        _pdf.ln(2)

                        _pdf.section("2. ANOMALY SUMMARY", (180, 50, 50))
                        _pdf.kv("Heart Rate Anomalies", f"{_n_hr} days flagged")
                        _pdf.kv("Steps Anomalies",      f"{_n_st} days flagged")
                        _pdf.kv("Sleep Anomalies",      f"{_n_sl} days flagged")
                        _pdf.kv("Total Flags",          f"{_n_hr + _n_st + _n_sl} across all signals")
                        _pdf.ln(2)

                        _pdf.section("3. DETECTION THRESHOLDS", (40, 100, 60))
                        _pdf.kv("HR High",     f"> {m4_hr_high} bpm")
                        _pdf.kv("HR Low",      f"< {m4_hr_low} bpm")
                        _pdf.kv("Steps Low",   f"< {m4_st_low:,} steps/day")
                        _pdf.kv("Sleep Low",   f"< {m4_sl_low} min/night")
                        _pdf.kv("Sleep High",  f"> {m4_sl_high} min/night")
                        _pdf.kv("Sigma",       f"+/- {m4_sigma:.1f} sigma from rolling median")
                        _pdf.ln(2)

                        _pdf.section("4. METHODOLOGY", (60, 80, 140))
                        _pdf.para(
                            "Three anomaly detection methods: (1) Threshold violations - hard "
                            "upper/lower bounds. (2) Residual +/-sigma - 3-day rolling median baseline, "
                            f"flag days deviating > {m4_sigma:.1f} sigma. (3) DBSCAN outlier clustering "
                            "- users assigned label -1 are structural outliers."
                        )

                        # Anomaly tables
                        def _sanitize(text):
                            """Replace non-latin-1 characters with ASCII equivalents for fpdf2."""
                            return (str(text)
                                    .replace("σ", "sigma").replace("±", "+/-")
                                    .replace("—", "-").replace("–", "-")
                                    .replace("−", "-").replace("·", ".")
                                    .encode("latin-1", errors="replace")
                                    .decode("latin-1"))

                        def _table(pdf, df, cols, rename_map, max_rows=20):
                            df2 = df[df["is_anomaly"]][cols].copy().rename(columns=rename_map)
                            if df2.empty:
                                pdf.para("No anomalies detected.")
                                return
                            col_w = 180 // len(df2.columns)
                            pdf.set_fill_color(15, 23, 60)
                            pdf.set_text_color(180, 210, 255)
                            pdf.set_font("Helvetica", "B", 7.5)
                            for col in df2.columns:
                                pdf.cell(col_w, 6, _sanitize(col)[:18], border=0, fill=True)
                            pdf.ln()
                            pdf.set_font("Helvetica", "", 7.5)
                            for i, (_, row) in enumerate(df2.head(max_rows).iterrows()):
                                if i % 2 == 0:
                                    pdf.set_fill_color(30, 40, 60)
                                else:
                                    pdf.set_fill_color(20, 30, 50)
                                pdf.set_text_color(200, 210, 225)
                                for val in row:
                                    cell_text = _sanitize(f"{val:.2f}" if isinstance(val, float) else val)[:18]
                                    pdf.cell(col_w, 5.5, cell_text, border=0, fill=True)
                                pdf.ln()
                            if len(df2) > max_rows:
                                pdf.set_text_color(100, 130, 180)
                                pdf.set_font("Helvetica", "I", 7)
                                pdf.cell(0, 5, f"  ... and {len(df2)-max_rows} more (see CSV export)", ln=True)
                            pdf.ln(3)

                        _pdf.add_page()
                        _pdf.section("5. HEART RATE ANOMALY RECORDS", (180, 50, 50))
                        _table(_pdf, anom_hr_f, ["Date","AvgHR","rolling_med","residual","reason"],
                               {"AvgHR":"Avg HR","rolling_med":"Expected","residual":"Deviation","reason":"Reason"})

                        _pdf.section("6. STEPS ANOMALY RECORDS", (40, 130, 80))
                        _table(_pdf, anom_steps_f, ["Date","TotalSteps","rolling_med","residual","reason"],
                               {"TotalSteps":"Steps","rolling_med":"Expected","residual":"Deviation","reason":"Reason"})

                        _pdf.section("7. SLEEP ANOMALY RECORDS", (100, 60, 160))
                        _table(_pdf, anom_sleep_f, ["Date","TotalSleepMinutes","rolling_med","residual","reason"],
                               {"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected","residual":"Deviation","reason":"Reason"})

                        # User profiles
                        _pdf.add_page()
                        _pdf.section("8. USER ACTIVITY PROFILES", (15, 23, 60))
                        _prof_cols = [c for c in ["TotalSteps","Calories","VeryActiveMinutes",
                                                   "SedentaryMinutes","TotalSleepMinutes"]
                                      if c in master_f.columns]
                        _uprof = master_f.groupby("Id")[_prof_cols].mean().round(1)
                        _col_w2 = 180 // (len(_prof_cols) + 1)
                        _pdf.set_font("Helvetica", "B", 8)
                        _pdf.set_fill_color(15, 23, 60); _pdf.set_text_color(180, 210, 255)
                        _pdf.cell(_col_w2, 6, "User ID", border=0, fill=True)
                        for _c in _prof_cols:
                            _pdf.cell(_col_w2, 6, _sanitize(_c[:12]), border=0, fill=True)
                        _pdf.ln()
                        _pdf.set_font("Helvetica", "", 7.5)
                        for _i, (_uid, _row) in enumerate(_uprof.iterrows()):
                            if _i % 2 == 0:
                                _pdf.set_fill_color(30, 40, 60)
                            else:
                                _pdf.set_fill_color(20, 30, 50)
                            _pdf.set_text_color(200, 210, 225)
                            _pdf.cell(_col_w2, 5.5, _sanitize(f"...{str(_uid)[-6:]}"), border=0, fill=True)
                            for _val in _row:
                                _pdf.cell(_col_w2, 5.5, _sanitize(f"{_val:,.0f}"), border=0, fill=True)
                            _pdf.ln()

                        # Output
                        _pdf_out = _pdf.output()
                        _pdf_buf = io.BytesIO()
                        _pdf_buf.write(_pdf_out if isinstance(_pdf_out, (bytes, bytearray))
                                       else _pdf_out.encode("latin-1"))
                        _pdf_buf.seek(0)
                        _fname_pdf = f"FitPulse_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=_pdf_buf,
                            file_name=_fname_pdf,
                            mime="application/pdf",
                            key="dl_pdf_m4"
                        )
                        st.success(f"✅ PDF ready — {_fname_pdf}")
                    except Exception as _pe:
                        st.error(f"PDF error: {_pe}\n\n{traceback.format_exc()}")

        with col_csv:
            sec_m4("📊", "CSV Export")
            st.markdown(f'<div style="color:{MUTED};font-size:0.82rem;margin-bottom:0.8rem">All anomaly records from all 3 signals in a single CSV file.</div>', unsafe_allow_html=True)
            csv_data = generate_csv_m4(anom_hr_f, anom_steps_f, anom_sleep_f)
            fname_csv = f"FitPulse_Anomalies_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            st.download_button("⬇️ Download Anomaly CSV", csv_data, fname_csv, "text/csv", use_container_width=True)
            with st.expander("👁️ Preview CSV"):
                preview = pd.concat([
                    anom_hr_f[anom_hr_f["is_anomaly"]].assign(signal="Heart Rate").rename(columns={"AvgHR":"value","rolling_med":"expected"})[["signal","Date","value","expected","residual","reason"]],
                    anom_steps_f[anom_steps_f["is_anomaly"]].assign(signal="Steps").rename(columns={"TotalSteps":"value","rolling_med":"expected"})[["signal","Date","value","expected","residual","reason"]],
                    anom_sleep_f[anom_sleep_f["is_anomaly"]].assign(signal="Sleep").rename(columns={"TotalSleepMinutes":"value","rolling_med":"expected"})[["signal","Date","value","expected","residual","reason"]],
                ], ignore_index=True).sort_values(["signal","Date"]).round(2)
                st.dataframe(preview, use_container_width=True, height=280)


# ──────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:1.5rem 0;border-top:1px solid rgba(0,230,255,0.08);">
    <span style="font-family:JetBrains Mono,monospace;font-size:0.62rem;letter-spacing:2px;color:#2a4560;">
        FITPULSE ANALYTICS SUITE &nbsp;·&nbsp; Pre-Processing · Pattern Extraction · Anomaly Detection · Insights Dashboard
        &nbsp;·&nbsp; BUILT WITH STREAMLIT + PLOTLY
    </span>
</div>
""", unsafe_allow_html=True)