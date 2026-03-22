"""
╔══════════════════════════════════════════════════════════════════╗
║   FITNESS DATA PRO  —  UNIFIED PIPELINE                         ║
║   Pre Processing  +  Pattern Extracting                         ║
║   Run:  streamlit run merged_pipeline.py                        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import io, warnings, traceback, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fitness Data Pro — Unified Pipeline",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# GLOBAL CSS  (merged + launcher styles)
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap');

/* ── Base ── */
html,body,.stApp,[data-testid="stAppViewContainer"]{
    background:#030810 !important; color:#d4eaf7 !important;
    font-family:'Exo 2',sans-serif !important;
}
[data-testid="stMain"]{background:transparent !important;}
[data-testid="stAppViewContainer"]::before{
    content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
    background-image:
        linear-gradient(rgba(0,230,255,.025) 1px,transparent 1px),
        linear-gradient(90deg,rgba(0,230,255,.025) 1px,transparent 1px);
    background-size:55px 55px;
}
[data-testid="stSidebar"]{
    background:#040a14 !important;
    border-right:1px solid rgba(0,230,255,.12) !important;
}
section[data-testid="stSidebar"] *{color:#d4eaf7;}

/* ── Typography ── */
h1{font-family:'Orbitron',monospace !important;font-size:1.9rem !important;
   letter-spacing:2px !important;color:#fff !important;}
h2{font-family:'Orbitron',monospace !important;font-size:1.1rem !important;
   letter-spacing:1.5px !important;color:#00e6ff !important;}
h3{font-family:'Exo 2',sans-serif !important;font-weight:600 !important;color:#d4eaf7 !important;}
p,li{color:#d4eaf7 !important;}

/* ── Metrics ── */
[data-testid="stMetric"]{
    background:#0b1422 !important;border:1px solid rgba(0,230,255,.14) !important;
    border-radius:14px !important;padding:1.2rem 1.4rem !important;
    box-shadow:0 6px 30px rgba(0,0,0,.5) !important;transition:transform .2s;
}
[data-testid="stMetric"]:hover{transform:translateY(-3px);}
[data-testid="stMetricValue"]{
    font-family:'Orbitron',monospace !important;font-size:2rem !important;
    font-weight:900 !important;color:#00e6ff !important;
    text-shadow:0 0 22px rgba(0,230,255,.5) !important;
}
[data-testid="stMetricLabel"]{
    font-family:'JetBrains Mono',monospace !important;font-size:.62rem !important;
    letter-spacing:2.5px !important;text-transform:uppercase !important;color:#4a6a88 !important;
}
[data-testid="stMetricDelta"]{font-family:'JetBrains Mono',monospace !important;font-size:.72rem !important;}

/* ── Buttons ── */
.stButton>button{
    background:linear-gradient(135deg,#00c8ff,#0055cc) !important;
    color:#fff !important;border:none !important;border-radius:10px !important;
    font-family:'Exo 2',sans-serif !important;font-weight:600 !important;
    font-size:.9rem !important;padding:.65rem 1.8rem !important;
    box-shadow:0 4px 22px rgba(0,200,255,.38) !important;
    letter-spacing:.5px !important;transition:all .25s !important;
}
.stButton>button:hover{
    background:linear-gradient(135deg,#00d8ff,#0066ff) !important;
    box-shadow:0 8px 32px rgba(0,200,255,.55) !important;
    transform:translateY(-2px) !important;
}

/* ── Sidebar nav button overrides ── */
[data-testid="stSidebar"] .stButton>button{
    font-size:.78rem !important;
    padding:.5rem .8rem !important;
    letter-spacing:.5px !important;
}
/* M1 nav button */
[data-testid="stSidebar"] div[data-testid="column"]:first-child .stButton>button{
    background:linear-gradient(135deg,rgba(0,230,255,.15),rgba(0,100,200,.2)) !important;
    border:1px solid rgba(0,230,255,.4) !important;
    box-shadow:0 0 16px rgba(0,230,255,.15) !important;
}
[data-testid="stSidebar"] div[data-testid="column"]:first-child .stButton>button:hover{
    background:linear-gradient(135deg,rgba(0,230,255,.28),rgba(0,100,200,.35)) !important;
    border-color:#00e6ff !important;
    box-shadow:0 0 24px rgba(0,230,255,.35) !important;
}
/* M2 nav button */
[data-testid="stSidebar"] div[data-testid="column"]:last-child .stButton>button{
    background:linear-gradient(135deg,rgba(244,114,182,.15),rgba(168,85,247,.2)) !important;
    border:1px solid rgba(244,114,182,.4) !important;
    box-shadow:0 0 16px rgba(244,114,182,.15) !important;
}
[data-testid="stSidebar"] div[data-testid="column"]:last-child .stButton>button:hover{
    background:linear-gradient(135deg,rgba(244,114,182,.28),rgba(168,85,247,.35)) !important;
    border-color:#f472b6 !important;
    box-shadow:0 0 24px rgba(244,114,182,.35) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"]{
    background:rgba(0,230,255,.03) !important;
    border:2px dashed rgba(0,230,255,.3) !important;
    border-radius:16px !important;padding:1.5rem !important;
}
[data-testid="stFileUploader"]:hover{
    border-color:rgba(0,230,255,.7) !important;
    background:rgba(0,230,255,.06) !important;
}

/* ── Expander ── */
[data-testid="stExpander"]{
    background:#0b1422 !important;
    border:1px solid rgba(0,230,255,.12) !important;border-radius:12px !important;
}

/* ── Select boxes ── */
.stSelectbox>div>div,.stMultiSelect>div>div{
    background:#0b1422 !important;border-color:rgba(0,230,255,.22) !important;
    border-radius:10px !important;color:#d4eaf7 !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"]>div{
    background:linear-gradient(90deg,#7c3aed,#00e6ff,#00ffa3) !important;
    border-radius:4px !important;box-shadow:0 0 12px rgba(0,230,255,.4) !important;
}
[data-testid="stProgressBar"]{background:rgba(255,255,255,.05) !important;border-radius:4px !important;}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"]{
    background:#0b1422 !important;border-radius:12px 12px 0 0 !important;
    border-bottom:1px solid rgba(0,230,255,.15) !important;gap:4px !important;padding:4px !important;
}
[data-testid="stTabs"] [role="tab"]{
    font-family:'Exo 2',sans-serif !important;font-weight:600 !important;
    font-size:.85rem !important;color:#4a6a88 !important;
    border-radius:8px !important;padding:8px 18px !important;border:none !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{
    background:rgba(0,230,255,.1) !important;color:#00e6ff !important;
    border:1px solid rgba(0,230,255,.28) !important;box-shadow:0 0 14px rgba(0,230,255,.12) !important;
}
[data-testid="stTabContent"]{
    background:#0b1422 !important;border:1px solid rgba(0,230,255,.1) !important;
    border-top:none !important;border-radius:0 0 12px 12px !important;padding:1.5rem !important;
}

/* ── Misc ── */
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:rgba(0,230,255,.18);border-radius:3px;}
hr{border-color:rgba(0,230,255,.1) !important;margin:1rem 0 !important;}
[data-testid="stAlert"]{border-radius:12px !important;font-family:'JetBrains Mono',monospace !important;font-size:.8rem !important;}
#MainMenu,footer{visibility:hidden;}
div[data-testid="stDecoration"]{display:none;}

/* ── M2 component classes ── */
.hero-title{
    font-family:'Syne',sans-serif;font-size:2.1rem;font-weight:800;
    background:linear-gradient(135deg,#38bdf8 30%,#f472b6 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    line-height:1.15;margin-bottom:3px;
}
.hero-sub{font-family:'JetBrains Mono',monospace;font-size:11px;color:#4b607a;letter-spacing:1.2px;}
.sec-lbl{
    font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:2.5px;
    text-transform:uppercase;color:#4b607a;border-bottom:1px solid #1a2840;
    padding-bottom:6px;margin:26px 0 14px;
}
.chip{
    display:inline-flex;align-items:center;gap:5px;padding:4px 11px;border-radius:5px;
    font-family:'JetBrains Mono',monospace;font-size:11px;margin:3px;
    border:1px solid #1e2d45;background:#0f1c30;color:#4b607a;
}
.chip.ok{border-color:#22c55e;background:rgba(34,197,94,.08);color:#22c55e;}
.step-bar{display:flex;align-items:center;gap:10px;padding:13px 20px;background:#111f36;border-bottom:1px solid #1a2840;}
.step-pill{
    font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:1.5px;
    text-transform:uppercase;padding:3px 9px;border-radius:4px;
    background:#1a2840;color:#4b607a;white-space:nowrap;
}
.step-pill.done{background:rgba(34,197,94,.15);color:#22c55e;}
.step-pill.ready{background:rgba(56,189,248,.15);color:#38bdf8;}
.step-bar-title{font-family:'Syne',sans-serif;font-size:.97rem;font-weight:700;color:#e2e8f0;flex:1;}
.step-flag{font-family:'JetBrains Mono',monospace;font-size:11px;}
.step-flag.done{color:#22c55e;} .step-flag.ready{color:#38bdf8;} .step-flag.locked{color:#4b607a;}
.sum-box{background:#0d1526;border:1px solid #1a2840;border-radius:10px;padding:16px 20px;margin-bottom:10px;}
.sum-box-title{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#4b607a;margin-bottom:10px;}
.sum-row{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid #1a2840;font-size:13px;}
.sum-row:last-child{border-bottom:none;}
.sum-key{color:#94a3b8;} .sum-val{font-family:'JetBrains Mono',monospace;color:#e2e8f0;font-weight:600;}
.c-green{color:#22c55e !important;} .c-blue{color:#38bdf8 !important;} .c-pink{color:#f472b6 !important;}
.c-orange{color:#fb923c !important;} .c-purple{color:#a78bfa !important;} .c-teal{color:#34d399 !important;}
.mtile{background:#0d1526;border:1px solid #1a2840;border-radius:10px;padding:14px 16px;text-align:center;}
.mtile-val{font-family:'Syne',sans-serif;font-size:1.75rem;font-weight:800;}
.mtile-lbl{font-family:'JetBrains Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:1.5px;color:#4b607a;margin-top:3px;}
.ss-badge{
    display:inline-block;background:rgba(251,146,60,.12);border:1px solid rgba(251,146,60,.35);
    color:#fb923c;font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:1px;
    padding:3px 10px;border-radius:4px;margin-bottom:8px;
}

/* ── Launcher cards ── */
.launch-card{
    border-radius:20px;padding:2.2rem 2rem;cursor:pointer;
    transition:transform .25s,box-shadow .25s;text-align:center;
}
.launch-card:hover{transform:translateY(-6px);}
.launch-card-m1{
    background:linear-gradient(145deg,rgba(0,230,255,.07),rgba(0,200,255,.03));
    border:1.5px solid rgba(0,230,255,.35);
}
.launch-card-m1:hover{box-shadow:0 0 40px rgba(0,230,255,.25);}
.launch-card-m2{
    background:linear-gradient(145deg,rgba(244,114,182,.07),rgba(168,85,247,.03));
    border:1.5px solid rgba(244,114,182,.35);
}
.launch-card-m2:hover{box-shadow:0 0 40px rgba(244,114,182,.25);}
.launch-card.active-m1{
    border-color:#00e6ff;
    box-shadow:0 0 48px rgba(0,230,255,.35),inset 0 0 30px rgba(0,230,255,.05);
}
.launch-card.active-m2{
    border-color:#f472b6;
    box-shadow:0 0 48px rgba(244,114,182,.35),inset 0 0 30px rgba(244,114,182,.05);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SHARED CONSTANTS  (M1)
# ══════════════════════════════════════════════════════════════════
PTHEME = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color="#d4eaf7", size=11),
    xaxis=dict(gridcolor="rgba(0,230,255,.06)", zerolinecolor="rgba(0,230,255,.1)",
               linecolor="rgba(0,230,255,.1)", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="rgba(0,230,255,.06)", zerolinecolor="rgba(0,230,255,.1)",
               linecolor="rgba(0,230,255,.1)", tickfont=dict(size=10)),
    margin=dict(l=10, r=10, t=45, b=10),
)
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

# M2 palette / dark-fig
DARK_BG="#080c18"; CARD_BG="#0f1729"; GRID_CLR="#1a2840"; TEXT_CLR="#e2e8f0"; MUTED="#4b607a"
PALETTE=["#38bdf8","#f472b6","#34d399","#fb923c","#a78bfa","#f87171","#4fd1c5"]


# ══════════════════════════════════════════════════════════════════
# SHARED HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def hex_to_rgba(h, a=.13):
    h=h.lstrip("#"); r,g,b=int(h[:2],16),int(h[2:4],16),int(h[4:],16)
    return f"rgba({r},{g},{b},{a})"

def safe_resample(df_idx, freq_lbl):
    freq_map={"Weekly":"W","Monthly":"ME","Daily":"D"}
    freq=freq_map.get(freq_lbl,"D")
    try: return df_idx.resample(freq).mean()
    except ValueError:
        fb={"ME":"M","W":"W","D":"D"}; return df_idx.resample(fb.get(freq,freq)).mean()

def dark_ax(ax):
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_CLR)
    ax.tick_params(colors=MUTED,labelsize=9)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT_CLR); ax.grid(color=GRID_CLR,alpha=.5,linewidth=.5)

def dark_fig(w=13,h=4):
    fig,ax=plt.subplots(figsize=(w,h)); fig.patch.set_facecolor(DARK_BG); dark_ax(ax); return fig,ax

def fig_to_png(fig):
    buf=io.BytesIO()
    fig.savefig(buf,format="png",dpi=180,bbox_inches="tight",facecolor=fig.get_facecolor())
    buf.seek(0); return buf

# M2 UI helpers
def sum_box(title, rows):
    inner="".join(
        f'<div class="sum-row"><span class="sum-key">{k}</span>'
        f'<span class="sum-val {c}">{v}</span></div>' for k,v,c in rows)
    st.markdown(f'<div class="sum-box"><div class="sum-box-title">{title}</div>{inner}</div>',
                unsafe_allow_html=True)

def metric_tiles(items):
    cols=st.columns(len(items))
    for col,(val,lbl,clr) in zip(cols,items):
        col.markdown(f'<div class="mtile"><div class="mtile-val" style="color:{clr}">{val}</div>'
                     f'<div class="mtile-lbl">{lbl}</div></div>',unsafe_allow_html=True)

def ss_badge(label="📸 Screenshot This"):
    st.markdown(f'<div class="ss-badge">{label}</div>',unsafe_allow_html=True)

# M1 preprocessing
def preprocess(df: pd.DataFrame):
    df=df.copy(); logs=[]; before_nulls=df.isnull().sum().to_dict()
    if "Date" in df.columns:
        df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
        nat=df["Date"].isna().sum()
        if nat>0 and "User_ID" in df.columns:
            full_dates=pd.date_range("2023-01-01",periods=365,freq="D"); parts=[]
            for uid,grp in df.groupby("User_ID",sort=True):
                grp=grp.copy().reset_index(drop=True); grp["Date"]=full_dates[:len(grp)]; parts.append(grp)
            df=pd.concat(parts,ignore_index=True)
            logs.append(("ok",f"✅  Filled {nat:,} null Date values — assigned sequential dates per user"))
        else:
            logs.append(("ok",f"✅  Date column parsed — {df['Date'].notna().sum():,} valid timestamps"))
    num_null_cols=[c for c in df.columns if df[c].dtype in [np.float64,np.float32] and df[c].isna().any()]
    for col in num_null_cols:
        n=int(df[col].isna().sum())
        if "User_ID" in df.columns:
            parts=[]
            for uid,grp in df.groupby("User_ID",sort=True):
                grp=grp.copy(); grp[col]=grp[col].interpolate(method="linear").ffill().bfill(); parts.append(grp)
            df=pd.concat(parts,ignore_index=True)
        else:
            df[col]=df[col].interpolate(method="linear").ffill().bfill()
        logs.append(("ok",f"✅  Interpolated (linear)+ffill/bfill → '{col}' ({n:,} nulls filled)"))
    if "Workout_Type" in df.columns:
        mm=(df["Workout_Type"].isna()|(df["Workout_Type"].astype(str).str.strip()=="")|
            (df["Workout_Type"].astype(str).str.strip().str.lower()=="nan"))
        n=int(mm.sum()); df["Workout_Type"]=df["Workout_Type"].astype(str).str.strip()
        df.loc[mm,"Workout_Type"]="No Workout"
        logs.append(("ok",f"✅  Filled {n:,} null(s) in 'Workout_Type' → 'No Workout'"))
    cat_null=[c for c in df.columns if df[c].dtype==object and df[c].isna().any() and c!="Workout_Type"]
    for col in cat_null:
        n=int(df[col].isna().sum()); mode=df[col].mode()[0] if not df[col].mode().empty else "Unknown"
        df[col]=df[col].fillna(mode)
        logs.append(("ok",f"✅  Filled {n:,} null(s) in '{col}' with mode → '{mode}'"))
    for col in df.select_dtypes(include=[np.float64,np.float32]).columns:
        df[col]=df[col].round(2)
    if "Date" in df.columns:
        vd=df["Date"].dropna(); span=(vd.max()-vd.min()).days
        logs.append(("info",f"ℹ️  Date range: {vd.min().date()} → {vd.max().date()} | Span: {span} days"))
        logs.append(("warn","⚠️  Timestamps stored as local/naive — UTC normalization skipped."))
    return df,logs,before_nulls


# ══════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════
# Launcher state
if "active_module" not in st.session_state:
    st.session_state.active_module = None   # None | "m1" | "m2"

# ── M1 state ──
for k in ["m1_raw_df","m1_clean_df","m1_logs","m1_before_nulls","m1_step"]:
    if k not in st.session_state: st.session_state[k]=None
if st.session_state.m1_step is None: st.session_state.m1_step=1

M1_STEP_META={1:("📂","Upload CSV"),2:("🔍","Check Nulls"),
              3:("⚙️","Preprocess"),4:("👁️","Preview"),5:("📈","Run EDA")}

# ── M2 state ──
M2_STEPS=["load","timestamps","master","tsfresh","prophet","clustering","summary"]
_m2_defaults={
    "m2_daily":None,"m2_hourly_s":None,"m2_hourly_i":None,
    "m2_sleep":None,"m2_hr":None,"m2_hr_minute":None,
    "m2_master":None,"m2_features":None,
    "m2_kmeans_labels":None,"m2_dbscan_labels":None,
    "m2_cluster_features":None,"m2_X_pca":None,"m2_X_tsne":None,
    "m2_var_explained":None,"m2_n_clusters_db":0,"m2_n_noise":0,"m2_fc_hr":None,
    **{f"m2_done_{s}":False for s in M2_STEPS},
}
for k,v in _m2_defaults.items():
    if k not in st.session_state: st.session_state[k]=v

def m2_done(s): return st.session_state[f"m2_done_{s}"]
def m2_mark(s): st.session_state[f"m2_done_{s}"]=True
def m2_reset_from(step):
    idx=M2_STEPS.index(step)
    for s in M2_STEPS[idx:]: st.session_state[f"m2_done_{s}"]=False
    wipe={"load":["m2_daily","m2_hourly_s","m2_hourly_i","m2_sleep","m2_hr"],
          "timestamps":["m2_hr_minute"],"master":["m2_master"],"tsfresh":["m2_features"],
          "prophet":["m2_fc_hr"],
          "clustering":["m2_kmeans_labels","m2_dbscan_labels","m2_cluster_features",
                        "m2_X_pca","m2_X_tsne","m2_var_explained"]}
    for s in M2_STEPS[idx:]:
        for k in wipe.get(s,[]): st.session_state[k]=None

def m2_sstate(step,prereq=None):
    if m2_done(step): return "done"
    if prereq is None or m2_done(prereq): return "ready"
    return "locked"

def m2_step_header(num,icon,title,step,prereq=None):
    st_=m2_sstate(step,prereq)
    pill={"done":"done","ready":"ready","locked":""}[st_]
    flag={"done":"✅ DONE","ready":"● READY","locked":"🔒 LOCKED"}[st_]
    st.markdown(
        f'<div class="step-bar"><span class="step-pill {pill}">STEP {num}</span>'
        f'<span style="font-size:1.15rem">{icon}</span>'
        f'<span class="step-bar-title">{title}</span>'
        f'<span class="step-flag {st_ if st_!="ready" else "ready"}">{flag}</span></div>',
        unsafe_allow_html=True)
    return st_


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:.5rem 0 1.5rem;">
        <div style="font-family:Orbitron,monospace;font-size:1.2rem;font-weight:900;
                    color:#fff;letter-spacing:3px;">🏋️ FITNESS</div>
        <div style="font-family:Orbitron,monospace;font-size:.75rem;color:#00e6ff;
                    letter-spacing:5px;margin-top:4px;">DATA PRO</div>
        <div style="font-family:JetBrains Mono,monospace;font-size:.6rem;color:#4a6a88;
                    letter-spacing:2px;margin-top:8px;">UNIFIED PIPELINE v3.0</div>
    </div><hr>""", unsafe_allow_html=True)

    mod = st.session_state.active_module

    # ══════════════════════════════════════════════════════════════
    # ── MODULE NAVIGATION BUTTONS ────────────────────────────────
    # ══════════════════════════════════════════════════════════════
    st.markdown(
        '<div style="font-family:JetBrains Mono,monospace;font-size:.6rem;'
        'letter-spacing:2px;color:#4a6a88;margin-bottom:8px;">⚡ QUICK NAVIGATE</div>',
        unsafe_allow_html=True)

    _nav_col1, _nav_col2 = st.columns(2)

    with _nav_col1:
        _m1_is_active = mod == "m1"
        _m1_btn_label = "⚙️ Pre Processing ●" if _m1_is_active else "⚙️ Pre Processing"
        if st.button(
            _m1_btn_label,
            key="sb_nav_m1",
            help="Pre Processing — Upload, Clean, EDA",
            use_container_width=True,
        ):
            st.session_state.active_module = "m1"
            st.rerun()

    with _nav_col2:
        _m2_is_active = mod == "m2"
        _m2_btn_label = "🔬 Pattern Extracting ●" if _m2_is_active else "🔬 Pattern Extracting"
        if st.button(
            _m2_btn_label,
            key="sb_nav_m2",
            help="Pattern Extracting — TSFresh, Prophet, Clustering",
            use_container_width=True,
        ):
            st.session_state.active_module = "m2"
            st.rerun()

    # Home / Launcher button — shown only when inside a module
    if mod in ("m1", "m2"):
        if st.button("🏠  Launcher Home", key="sb_nav_home", use_container_width=True):
            st.session_state.active_module = None
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Module status indicators ──
    st.markdown(
        '<div style="font-family:JetBrains Mono,monospace;font-size:.6rem;'
        'letter-spacing:2px;color:#4a6a88;margin-bottom:6px;">MODULE STATUS</div>',
        unsafe_allow_html=True)

    for lbl,key,color in [
        ("⚙️ Pre Processing","m1","#00e6ff"),
        ("🔬 Pattern Extracting","m2","#f472b6"),
    ]:
        active = (mod==key)
        bg  = f"rgba(0,0,0,.0)" if not active else \
              ("rgba(0,230,255,.08)" if key=="m1" else "rgba(244,114,182,.08)")
        bdc = color if active else "rgba(255,255,255,.06)"
        dot = f'<span style="width:7px;height:7px;border-radius:50%;background:{color};'\
              f'box-shadow:0 0 8px {color};display:inline-block;margin-right:6px;"></span>' if active else ""
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;padding:9px 14px;'
            f'border-radius:10px;margin-bottom:4px;background:{bg};border:1px solid {bdc};">'
            f'{dot}<span style="font-family:Exo 2,sans-serif;font-weight:{"700" if active else "400"};'
            f'font-size:.85rem;color:{"#fff" if active else "#4a6a88"};">{lbl}</span></div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Context-sensitive progress ──
    if mod=="m1":
        pct=(st.session_state.m1_step-1)/4
        st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:.6rem;'
                    'letter-spacing:2px;color:#4a6a88;margin-bottom:6px;">M1 PIPELINE PROGRESS</div>',
                    unsafe_allow_html=True)
        st.progress(pct)
        for num,(icon,label) in M1_STEP_META.items():
            active=st.session_state.m1_step==num; done=st.session_state.m1_step>num
            if active: fg,bg,bd,badge="#00e6ff","rgba(0,230,255,.08)","rgba(0,230,255,.3)","▶"
            elif done: fg,bg,bd,badge="#00ffa3","rgba(0,255,163,.06)","rgba(0,255,163,.25)","✓"
            else: fg,bg,bd,badge="#4a6a88","transparent","rgba(255,255,255,.05)",str(num)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;padding:8px 12px;'
                f'border-radius:8px;margin-bottom:3px;background:{bg};border:1px solid {bd};">'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:.6rem;'
                f'font-weight:700;color:{fg};width:16px;text-align:center;">{badge}</span>'
                f'<span style="font-family:Exo 2,sans-serif;font-size:.8rem;color:{fg};">'
                f'{icon} {label}</span></div>', unsafe_allow_html=True)

    elif mod=="m2":
        n_done=sum(m2_done(s) for s in M2_STEPS)
        st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:.6rem;'
                    'letter-spacing:2px;color:#4a6a88;margin-bottom:6px;">M2 PIPELINE PROGRESS</div>',
                    unsafe_allow_html=True)
        st.progress(n_done/len(M2_STEPS), text=f"{n_done}/{len(M2_STEPS)} steps")
        for s in M2_STEPS:
            clr="#22c55e" if m2_done(s) else "#4b607a"; ic="✅" if m2_done(s) else "⬜"
            st.markdown(f'<span style="font-family:JetBrains Mono,monospace;font-size:11px;'
                        f'color:{clr}">{ic} {s.replace("_"," ").title()}</span>',
                        unsafe_allow_html=True)
        # M2 configuration knobs
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:.6rem;'
                    'letter-spacing:2px;color:#4a6a88;margin-bottom:8px;">⚙️ M2 CONFIGURATION</div>',
                    unsafe_allow_html=True)
        st.markdown("**🧪 TSFresh**")
        tsfresh_mode=st.selectbox("Feature Parameters",["MinimalFCParameters","EfficientFCParameters"],
            help="Minimal ~9 features; Efficient ~72 features")
        st.markdown("**📈 Prophet**")
        forecast_days=st.slider("Forecast Horizon (days)",7,90,30)
        changepoint_scale=st.number_input("Changepoint Prior Scale",min_value=.001,max_value=.5,
            value=.01,step=.005,format="%.3f")
        st.markdown("**🤖 KMeans**")
        optimal_k=st.slider("Number of Clusters (K)",2,9,3)
        st.markdown("**🔵 DBSCAN**")
        eps_val=st.number_input("Epsilon (eps)",.1,10.0,2.2,.1)
        min_samples=st.number_input("Min Samples",1,10,2,1)
    else:
        # Defaults so variables exist regardless of active module
        tsfresh_mode="MinimalFCParameters"; forecast_days=30; changepoint_scale=.01
        optimal_k=3; eps_val=2.2; min_samples=2

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:.6rem;'
                'letter-spacing:2px;color:#4a6a88;margin-bottom:8px;">QUICK ACTIONS</div>',
                unsafe_allow_html=True)
    if st.button("🔄  Reset Everything"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()
    if mod=="m1" and st.button("↩  Reset M1 Pipeline"):
        for k in ["m1_raw_df","m1_clean_df","m1_logs","m1_before_nulls"]: st.session_state[k]=None
        st.session_state.m1_step=1; st.rerun()
    if mod=="m2" and st.button("↩  Reset M2 Pipeline"):
        for k,v in _m2_defaults.items(): st.session_state[k]=v; st.rerun()


# ══════════════════════════════════════════════════════════════════
# ██████████████████████  LAUNCHER PAGE  ██████████████████████████
# ══════════════════════════════════════════════════════════════════
def render_launcher():
    # Header
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 .5rem;">
        <div style="font-family:Orbitron,monospace;font-size:.65rem;letter-spacing:4px;
                    color:#00e6ff;opacity:.7;margin-bottom:.5rem;">FITNESS DATA PRO</div>
        <h1 style="margin:0;font-size:2.4rem;letter-spacing:3px;">UNIFIED PIPELINE</h1>
        <p style="font-family:JetBrains Mono,monospace;font-size:.72rem;color:#4a6a88;
                  letter-spacing:2px;margin-top:.5rem;">
            SELECT A MODULE BELOW TO BEGIN YOUR ANALYSIS
        </p>
    </div>""", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Two big launcher cards ──
    col1, col2 = st.columns(2, gap="large")

    with col1:
        is_active = st.session_state.active_module == "m1"
        active_cls = "active-m1" if is_active else ""
        st.markdown(f"""
        <div class="launch-card launch-card-m1 {active_cls}">
            <div style="font-size:3.5rem;margin-bottom:1rem;line-height:1;">⚙️</div>
            <div style="font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;
                        color:#00e6ff;letter-spacing:1px;margin-bottom:.5rem;">
                Pre Processing
            </div>
            <div style="font-family:JetBrains Mono,monospace;font-size:.68rem;
                        color:#4a6a88;letter-spacing:1.2px;line-height:2;margin-bottom:1.2rem;">
                UPLOAD CSV · NULL ANALYSIS · CLEAN DATA<br>
                PREVIEW DATASET · FULL EDA (6 TABS)
            </div>
            <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:6px;margin-bottom:1rem;">
                {"".join(f'<span style="font-family:JetBrains Mono,monospace;font-size:9px;padding:3px 10px;border-radius:4px;background:rgba(0,230,255,.1);border:1px solid rgba(0,230,255,.25);color:#00e6ff;">{t}</span>' for t in ["Plotly Charts","Null Heatmap","Correlation Matrix","Time Trends","User Analysis"])}
            </div>
            {"<div style='font-family:JetBrains Mono,monospace;font-size:.65rem;color:#00ffa3;letter-spacing:1.5px;'>● CURRENTLY ACTIVE</div>" if is_active else ""}
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚙️  Launch Pre Processing", key="btn_m1", use_container_width=True):
            st.session_state.active_module = "m1"; st.rerun()

    with col2:
        is_active = st.session_state.active_module == "m2"
        active_cls = "active-m2" if is_active else ""
        st.markdown(f"""
        <div class="launch-card launch-card-m2 {active_cls}">
            <div style="font-size:3.5rem;margin-bottom:1rem;line-height:1;">🔬</div>
            <div style="font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;
                        color:#f472b6;letter-spacing:1px;margin-bottom:.5rem;">
                Pattern Extracting
            </div>
            <div style="font-family:JetBrains Mono,monospace;font-size:.68rem;
                        color:#4a6a88;letter-spacing:1.2px;line-height:2;margin-bottom:1.2rem;">
                TSFRESH FEATURES · PROPHET FORECASTING<br>
                KMEANS · DBSCAN · PCA · t-SNE · CLUSTER PROFILES
            </div>
            <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:6px;margin-bottom:1rem;">
                {"".join(f'<span style="font-family:JetBrains Mono,monospace;font-size:9px;padding:3px 10px;border-radius:4px;background:rgba(244,114,182,.1);border:1px solid rgba(244,114,182,.25);color:#f472b6;">{t}</span>' for t in ["TSFresh","Prophet","KMeans","DBSCAN","PCA","t-SNE"])}
            </div>
            {"<div style='font-family:JetBrains Mono,monospace;font-size:.65rem;color:#f472b6;letter-spacing:1.5px;'>● CURRENTLY ACTIVE</div>" if is_active else ""}
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔬  Launch Pattern Extracting", key="btn_m2", use_container_width=True):
            st.session_state.active_module = "m2"; st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 .5rem;">
        <span style="font-family:JetBrains Mono,monospace;font-size:.62rem;
            letter-spacing:2px;color:#2a4560;">
            FITNESS DATA PRO &nbsp;·&nbsp; UNIFIED PIPELINE &nbsp;·&nbsp;
            STREAMLIT + PLOTLY + TSFRESH + PROPHET + SKLEARN
        </span>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ██████████████  MODULE 1 — PRE PROCESSING  ██████████████████████
# ══════════════════════════════════════════════════════════════════
def render_m1():
    # ── Sub-header ──
    icon,name=M1_STEP_META[st.session_state.m1_step]
    col_t,col_b=st.columns([3,1])
    with col_t:
        st.markdown(f"""
        <div style="margin-bottom:1rem;">
            <div style="font-family:JetBrains Mono,monospace;font-size:.65rem;
                 letter-spacing:3px;color:#00e6ff;margin-bottom:4px;opacity:.8;">
                 ⚙️ PRE PROCESSING — STEP {st.session_state.m1_step} OF 5</div>
            <h1 style="margin:0;">{icon} {name}</h1>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div style="text-align:right;padding-top:1.2rem;">
            <span style="display:inline-flex;align-items:center;gap:8px;
                background:rgba(0,255,163,.08);border:1px solid rgba(0,255,163,.2);
                border-radius:20px;padding:6px 16px;">
                <span style="width:7px;height:7px;border-radius:50%;background:#00ffa3;
                    box-shadow:0 0 8px #00ffa3;display:inline-block;"></span>
                <span style="font-family:JetBrains Mono,monospace;font-size:.65rem;
                    color:#00ffa3;letter-spacing:1px;">PIPELINE ACTIVE</span>
            </span>
        </div>""", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Back-to-launcher button
    if st.button("← Back to Launcher"):
        st.session_state.active_module=None; st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)

    step = st.session_state.m1_step

    # ── STEP 1 : UPLOAD ──────────────────────────────────────────
    if step==1:
        st.markdown("### Upload Your Fitness Dataset")
        st.markdown("<p style='color:#4a6a88;'>Upload a CSV file with your fitness tracking data.</p>",
                    unsafe_allow_html=True)
        uploaded=st.file_uploader("Drop your CSV here or click to browse",type=["csv"],
            help="Expected: User_ID, Date, Steps_Taken, Calories_Burned, Hours_Slept, etc.")
        if uploaded:
            try: st.session_state.m1_raw_df=pd.read_csv(uploaded)
            except Exception as e: st.error(f"Could not read file: {e}")
        if st.session_state.m1_raw_df is not None:
            df=st.session_state.m1_raw_df
            st.success("✅  Dataset loaded successfully!")
            c1,c2,c3,c4=st.columns(4)
            c1.metric("📊 Total Rows",f"{len(df):,}")
            c2.metric("📋 Columns",f"{len(df.columns)}")
            c3.metric("⚠️ Null Cells",f"{df.isnull().sum().sum():,}")
            c4.metric("👥 Unique Users",str(df["User_ID"].nunique()) if "User_ID" in df.columns else "—")
            with st.expander("🔎 Preview Raw Data (first 10 rows)"):
                st.dataframe(df.head(10),use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Next → Check Null Values  ▶"):
                st.session_state.m1_step=2; st.rerun()
        else:
            st.info("👆  Upload a CSV file to begin.")

    # ── STEP 2 : NULL CHECK ──────────────────────────────────────
    elif step==2:
        if st.session_state.m1_raw_df is None:
            st.warning("Please upload a dataset first.")
            if st.button("← Back to Upload"): st.session_state.m1_step=1; st.rerun()
            st.stop()
        df=st.session_state.m1_raw_df
        null_counts=df.isnull().sum(); null_counts=null_counts[null_counts>0].sort_values(ascending=False)
        st.markdown("### Null Value Analysis")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Total Null Cells",f"{null_counts.sum():,}")
        c2.metric("Columns Affected",f"{len(null_counts)}")
        c3.metric("Overall Null Rate",f"{null_counts.sum()/(len(df)*len(df.columns))*100:.1f}%")
        c4.metric("Clean Columns",f"{len(df.columns)-len(null_counts)}")
        st.markdown("<br>", unsafe_allow_html=True)
        if len(null_counts)==0:
            st.success("🎉  No null values detected! Dataset is already clean.")
        else:
            badge_html=""
            for col,cnt in null_counts.items():
                pct=cnt/len(df)*100
                if pct>20: fg,bg,bc="#ff8aa8","rgba(255,56,96,.1)","rgba(255,56,96,.3)"
                elif pct>10: fg,bg,bc="#ffb800","rgba(255,184,0,.1)","rgba(255,184,0,.3)"
                else: fg,bg,bc="#00e6ff","rgba(0,230,255,.1)","rgba(0,230,255,.3)"
                badge_html+=(f"<span style='display:inline-flex;align-items:center;gap:5px;"
                             f"padding:5px 14px;border-radius:20px;font-family:JetBrains Mono,monospace;"
                             f"font-size:.7rem;background:{bg};border:1px solid {bc};color:{fg};margin:3px;'>"
                             f"▲ {col}: {cnt:,} ({pct:.1f}%)</span>")
            st.markdown(f"<div style='margin-bottom:1.5rem;'>{badge_html}</div>",unsafe_allow_html=True)
            bar_colors=["#ff4f9b" if (v/len(df)*100)>20 else "#ffb800" if (v/len(df)*100)>10 else "#00e6ff"
                        for v in null_counts.values]
            fig_null=go.Figure(go.Bar(y=null_counts.index.tolist(),x=null_counts.values.tolist(),
                orientation="h",marker=dict(color=bar_colors,line=dict(color="rgba(255,255,255,.1)",width=1)),
                text=[f"  {v:,}  ({v/len(df)*100:.1f}%)" for v in null_counts.values],
                textposition="outside",textfont=dict(family="JetBrains Mono",size=10,color="#d4eaf7"),
                hovertemplate="<b>%{y}</b><br>Null Count: %{x:,}<extra></extra>"))
            fig_null.update_layout(**PTHEME,height=max(250,len(null_counts)*60),
                                   xaxis_title="Null Count",yaxis_title="Column Name",showlegend=False)
            st.plotly_chart(fig_null,use_container_width=True)
            tab1,tab2=st.tabs(["📋 Summary Table","🗺️ Null Heatmap"])
            with tab1:
                tbl=pd.DataFrame({"Column":null_counts.index,"Null Count":null_counts.values,
                    "Null %":(null_counts.values/len(df)*100).round(2),
                    "Data Type":[str(df[c].dtype) for c in null_counts.index],
                    "Severity":["🔴 High" if v/len(df)>.2 else "🟡 Medium" if v/len(df)>.1 else "🔵 Low"
                                for v in null_counts.values]})
                st.dataframe(tbl,use_container_width=True,hide_index=True)
            with tab2:
                sample=df[null_counts.index.tolist()].head(100).isnull().astype(int)
                fig_heat=go.Figure(go.Heatmap(z=sample.T.values,x=[str(i) for i in sample.index],
                    y=sample.columns.tolist(),colorscale=[[0,"#0b1422"],[1,"#ff4f9b"]],showscale=False,
                    hovertemplate="Row %{x}<br>Column: %{y}<br>Is Null: %{z}<extra></extra>"))
                fig_heat.update_layout(**PTHEME,height=320,title="Null Pattern — first 100 rows",
                                       xaxis_title="Row Index",yaxis_title="Column Name")
                st.plotly_chart(fig_heat,use_container_width=True)
        col_b,col_n=st.columns([1,5])
        with col_b:
            if st.button("← Back"): st.session_state.m1_step=1; st.rerun()
        with col_n:
            if st.button("Next → Preprocess Data  ▶"): st.session_state.m1_step=3; st.rerun()

    # ── STEP 3 : PREPROCESS ──────────────────────────────────────
    elif step==3:
        if st.session_state.m1_raw_df is None:
            st.warning("Please upload a dataset first.")
            if st.button("← Back to Upload"): st.session_state.m1_step=1; st.rerun()
            st.stop()
        df_raw=st.session_state.m1_raw_df
        st.markdown("### Data Preprocessing Pipeline")
        if st.button("▶  Run Preprocessing"):
            with st.spinner("Running full preprocessing pipeline …"):
                time.sleep(.4)
                clean,logs,before_nulls=preprocess(df_raw)
                st.session_state.m1_clean_df=clean
                st.session_state.m1_logs=logs
                st.session_state.m1_before_nulls=before_nulls
        if st.session_state.m1_clean_df is not None:
            logs=st.session_state.m1_logs; before_nulls=st.session_state.m1_before_nulls
            df_clean=st.session_state.m1_clean_df
            st.markdown("#### 📋 Preprocessing Log")
            for level,msg in logs:
                if level=="ok": st.success(msg)
                elif level=="warn": st.warning(msg)
                else: st.info(msg)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Null Value Comparison — Before vs After")
            col_b,col_a=st.columns(2)
            before_series={k:v for k,v in before_nulls.items() if v>0}
            with col_b:
                st.markdown('<div style="background:rgba(255,56,96,.1);border:1px solid rgba(255,56,96,.25);'
                            'border-radius:10px 10px 0 0;padding:8px 16px;font-family:JetBrains Mono,monospace;'
                            'font-size:.7rem;letter-spacing:2px;color:#ff8aa8;">BEFORE PREPROCESSING</div>',
                            unsafe_allow_html=True)
                if before_series:
                    st.dataframe(pd.DataFrame({"Column":list(before_series.keys()),
                                               "Null Count":list(before_series.values())}),
                                 use_container_width=True,hide_index=True)
                else: st.info("No nulls were present.")
            with col_a:
                st.markdown('<div style="background:rgba(0,255,163,.1);border:1px solid rgba(0,255,163,.25);'
                            'border-radius:10px 10px 0 0;padding:8px 16px;font-family:JetBrains Mono,monospace;'
                            'font-size:.7rem;letter-spacing:2px;color:#00ffa3;">AFTER PREPROCESSING</div>',
                            unsafe_allow_html=True)
                st.markdown('<div style="background:rgba(0,255,163,.05);border:1px solid rgba(0,255,163,.2);'
                            'border-radius:0 0 10px 10px;padding:3rem 1rem;text-align:center;">'
                            '<div style="font-size:3rem;">🌟</div>'
                            '<div style="font-family:Orbitron,monospace;font-size:.85rem;letter-spacing:2px;'
                            'color:#00ffa3;margin-top:1rem;text-shadow:0 0 20px rgba(0,255,163,.5);">'
                            'ZERO NULLS REMAINING!</div></div>',unsafe_allow_html=True)
            total_filled=sum(before_series.values())
            c1,c2,c3=st.columns(3)
            c1.metric("Nulls Removed",f"{total_filled:,}",delta=f"-{total_filled:,}")
            c2.metric("Rows Preserved",f"{len(df_clean):,}")
            c3.metric("Data Quality","100%",delta="+100%")
            csv_bytes=df_clean.to_csv(index=False).encode()
            st.download_button("⬇️  Download Cleaned CSV",data=csv_bytes,
                               file_name="fitness_data_fully_cleaned.csv",mime="text/csv")
            col_b2,col_n2=st.columns([1,5])
            with col_b2:
                if st.button("← Back"): st.session_state.m1_step=2; st.rerun()
            with col_n2:
                if st.button("Next → Preview Dataset  ▶"): st.session_state.m1_step=4; st.rerun()
        else:
            st.info("👆  Click **Run Preprocessing** to clean all null values automatically.")
            if st.button("← Back"): st.session_state.m1_step=2; st.rerun()

    # ── STEP 4 : PREVIEW ─────────────────────────────────────────
    elif step==4:
        if st.session_state.m1_clean_df is None:
            st.warning("Please complete preprocessing first.")
            if st.button("← Back to Preprocessing"): st.session_state.m1_step=3; st.rerun()
            st.stop()
        df=st.session_state.m1_clean_df
        st.markdown("### Preview Cleaned Dataset")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Total Rows",f"{len(df):,}"); c2.metric("Columns",f"{len(df.columns)}")
        c3.metric("Remaining Nulls",f"{df.isnull().sum().sum():,}")
        c4.metric("Users",str(df["User_ID"].nunique()) if "User_ID" in df.columns else "—")
        col_f1,col_f2=st.columns([3,1])
        with col_f1:
            all_cols=df.columns.tolist()
            selected=st.multiselect("Select columns to display",all_cols,default=all_cols)
            if not selected: selected=all_cols
        with col_f2:
            n_rows=st.slider("Rows to show",5,min(200,len(df)),10)
        st.dataframe(df[selected].head(n_rows),use_container_width=True,height=400)
        with st.expander("📐 Descriptive Statistics",expanded=True):
            st.dataframe(df.select_dtypes(include=np.number).describe().round(2),use_container_width=True)
        cat_cols=df.select_dtypes(include="object").columns.tolist()
        if cat_cols:
            with st.expander("🏷️ Categorical Value Counts"):
                for c in cat_cols:
                    st.markdown(f"**{c}**"); vc=df[c].value_counts().reset_index()
                    vc.columns=[c,"Count"]; st.dataframe(vc,use_container_width=True,hide_index=True)
        csv_bytes=df.to_csv(index=False).encode()
        st.download_button("⬇️  Download Cleaned CSV",data=csv_bytes,
                           file_name="fitness_data_fully_cleaned.csv",mime="text/csv")
        col_b,col_n=st.columns([1,5])
        with col_b:
            if st.button("← Back"): st.session_state.m1_step=3; st.rerun()
        with col_n:
            if st.button("Next → Run Full EDA  ▶"): st.session_state.m1_step=5; st.rerun()

    # ── STEP 5 : EDA ─────────────────────────────────────────────
    elif step==5:
        if st.session_state.m1_clean_df is None:
            st.warning("Please complete preprocessing first.")
            if st.button("← Back to Preprocessing"): st.session_state.m1_step=3; st.rerun()
            st.stop()
        df=st.session_state.m1_clean_df.copy()
        if "Date" in df.columns: df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
        st.markdown("### Exploratory Data Analysis")
        NUM_COLS=[c for c in ["Steps_Taken","Calories_Burned","Hours_Slept",
                              "Water_Intake (Liters)","Active_Minutes","Heart_Rate (bpm)"]
                  if c in df.columns]
        if not NUM_COLS: NUM_COLS=df.select_dtypes(include=np.number).columns.tolist()
        if not NUM_COLS: st.error("No numeric columns found for EDA."); st.stop()

        tab_dist,tab_box,tab_cat,tab_corr,tab_time,tab_user=st.tabs([
            "📊 Distributions","📦 Outlier Detection","🥧 Categorical",
            "🔗 Correlation","📅 Time Trends","👤 User Analysis"])

        # Tab 1: Distributions
        with tab_dist:
            st.markdown("#### Distribution of Numeric Features")
            for i in range(0,len(NUM_COLS),2):
                cols=st.columns(2)
                for j,c in enumerate(NUM_COLS[i:i+2]):
                    with cols[j]:
                        color=COL_COLORS.get(c,"#00e6ff"); x_lbl=AXIS_LABELS.get(c,c)
                        col_data=df[c].dropna()
                        if col_data.empty: st.warning(f"No data for: {c}"); continue
                        fig=go.Figure()
                        fig.add_trace(go.Histogram(x=col_data,nbinsx=35,
                            marker=dict(color=color,opacity=.75,line=dict(color=color,width=.5)),
                            hovertemplate=f"{x_lbl}: %{{x}}<br>Count: %{{y}}<extra></extra>"))
                        fig.add_vline(x=col_data.mean(),line_color="white",line_dash="dash",line_width=1.5,
                                      annotation_text=f"Mean:{col_data.mean():.1f}",
                                      annotation_font_color="white",annotation_font_size=9)
                        fig.add_vline(x=col_data.median(),line_color=color,line_dash="dot",line_width=1.5,
                                      annotation_text=f"Med:{col_data.median():.1f}",
                                      annotation_font_color=color,annotation_font_size=9,
                                      annotation_position="bottom right")
                        fig.update_layout(**PTHEME,height=270,showlegend=False,bargap=.04,
                                          title=dict(text=f"Distribution · {c}",font=dict(size=11,color="#d4eaf7")),
                                          xaxis_title=x_lbl,yaxis_title="Records")
                        st.plotly_chart(fig,use_container_width=True)

        # Tab 2: Outlier Detection
        with tab_box:
            st.markdown("#### Outlier Detection — Boxplots")
            for i in range(0,len(NUM_COLS),2):
                cols=st.columns(2)
                for j,c in enumerate(NUM_COLS[i:i+2]):
                    with cols[j]:
                        color=COL_COLORS.get(c,"#00e6ff"); col_data=df[c].dropna()
                        if col_data.empty: continue
                        q1,q3=col_data.quantile(.25),col_data.quantile(.75); iqr=q3-q1
                        n_out=int(((col_data<q1-1.5*iqr)|(col_data>q3+1.5*iqr)).sum())
                        fig=go.Figure(go.Box(x=col_data,name=c,marker_color=color,line_color=color,
                            fillcolor=hex_to_rgba(color,.13),boxmean=True))
                        fig.update_layout(**PTHEME,height=210,showlegend=False,
                                          title=dict(text=f"Boxplot · {c}  ({n_out} outliers)",
                                                     font=dict(size=11,color="#d4eaf7")))
                        st.plotly_chart(fig,use_container_width=True)
            rows=[]
            for c in NUM_COLS:
                cd=df[c].dropna()
                if cd.empty: continue
                q1,q3=cd.quantile(.25),cd.quantile(.75); iqr=q3-q1; lo,hi=q1-1.5*iqr,q3+1.5*iqr
                n_out=int(((cd<lo)|(cd>hi)).sum())
                rows.append({"Column":c,"Min":round(cd.min(),2),"Q1":round(q1,2),
                             "Median":round(cd.median(),2),"Mean":round(cd.mean(),2),"Q3":round(q3,2),
                             "Max":round(cd.max(),2),"IQR":round(iqr,2),"# Outliers":n_out,
                             "Outlier %":round(n_out/len(df)*100,2)})
            if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

        # Tab 3: Categorical
        with tab_cat:
            st.markdown("#### Categorical Feature Analysis")
            col1,col2=st.columns(2)
            if "Workout_Type" in df.columns:
                wc=df["Workout_Type"].value_counts().reset_index(); wc.columns=["Workout","Count"]
                fig_w=go.Figure(go.Bar(x=wc["Workout"],y=wc["Count"],
                    marker=dict(color=WORKOUT_COLORS[:len(wc)],line=dict(color="rgba(255,255,255,.1)",width=1)),
                    text=wc["Count"],textposition="outside"))
                fig_w.update_layout(**PTHEME,height=320,showlegend=False,title="Workout Type Distribution")
                with col1: st.plotly_chart(fig_w,use_container_width=True)
            if "Mood" in df.columns:
                mc=df["Mood"].value_counts().reset_index(); mc.columns=["Mood","Count"]
                fig_m=go.Figure(go.Pie(labels=mc["Mood"],values=mc["Count"],
                    marker=dict(colors=MOOD_COLORS,line=dict(color="#030810",width=2)),hole=.55))
                fig_m.update_layout(**PTHEME,height=320,showlegend=True,title="Mood Distribution")
                with col2: st.plotly_chart(fig_m,use_container_width=True)
            if "Gender" in df.columns:
                gc=df["Gender"].value_counts().reset_index(); gc.columns=["Gender","Count"]
                fig_g=go.Figure(go.Pie(labels=gc["Gender"],values=gc["Count"],
                    marker=dict(colors=["#00e6ff","#ff4f9b"],line=dict(color="#030810",width=2)),hole=.6))
                fig_g.update_layout(**PTHEME,height=300,showlegend=True,title="Gender Distribution")
                st.plotly_chart(fig_g,use_container_width=True)
            if "Gender" in df.columns and "Workout_Type" in df.columns:
                wg=df.groupby(["Gender","Workout_Type"]).size().reset_index(name="Count")
                fig_wg=px.bar(wg,x="Workout_Type",y="Count",color="Gender",barmode="group",
                    color_discrete_map={"Male":"#00e6ff","Female":"#ff4f9b"},title="Workout by Gender")
                fig_wg.update_layout(**PTHEME,height=320)
                st.plotly_chart(fig_wg,use_container_width=True)

        # Tab 4: Correlation
        with tab_corr:
            st.markdown("#### Feature Correlation Matrix")
            if len(NUM_COLS)>=2:
                corr=df[NUM_COLS].corr().round(2)
                fig_c=go.Figure(go.Heatmap(z=corr.values,x=corr.columns.tolist(),y=corr.columns.tolist(),
                    colorscale=[[0.,"#ff4f9b"],[.5,"#0b1422"],[1.,"#00e6ff"]],zmin=-1,zmax=1,
                    text=corr.values.round(2),texttemplate="%{text}",textfont=dict(size=11),
                    hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.2f}<extra></extra>"))
                fig_c.update_layout(**PTHEME,height=500,title="Pearson Correlation Matrix")
                st.plotly_chart(fig_c,use_container_width=True)
                sc1,sc2,sc3=st.columns(3)
                with sc1: x_feat=st.selectbox("X-Axis",NUM_COLS,index=0)
                with sc2: y_feat=st.selectbox("Y-Axis",NUM_COLS,index=min(1,len(NUM_COLS)-1))
                with sc3: color_feat=st.selectbox("Color By",["None","Gender","Mood","Workout_Type"],index=0)
                s_cols=[x_feat,y_feat]; color_col=None if color_feat=="None" else color_feat
                if color_col and color_col in df.columns: s_cols.append(color_col)
                scatter_df=df[list(dict.fromkeys(s_cols))].dropna()
                try:
                    fig_sc=px.scatter(scatter_df,x=x_feat,y=y_feat,color=color_col,
                        trendline="ols",trendline_color_override="#ffffff",opacity=.55)
                except Exception:
                    fig_sc=px.scatter(scatter_df,x=x_feat,y=y_feat,color=color_col,opacity=.55)
                fig_sc.update_traces(marker=dict(size=4))
                fig_sc.update_layout(**PTHEME,height=380,title=f"Scatter: {x_feat} vs {y_feat}")
                st.plotly_chart(fig_sc,use_container_width=True)

        # Tab 5: Time Trends
        with tab_time:
            st.markdown("#### Time Series Trends")
            if "Date" in df.columns and df["Date"].notna().sum()>0:
                col_m,col_a=st.columns([2,1])
                with col_m: metric=st.selectbox("Select Metric",NUM_COLS,key="ts_metric")
                with col_a: agg=st.radio("Aggregation",["Daily","Weekly","Monthly"],horizontal=True)
                df_t=(df.dropna(subset=["Date"]).groupby("Date")[metric].mean()
                      .reset_index().set_index("Date").sort_index())
                if agg in ("Weekly","Monthly"): df_t=safe_resample(df_t,agg)
                color=COL_COLORS.get(metric,"#00e6ff")
                if not df_t.empty and not df_t[metric].dropna().empty:
                    fig_ts=go.Figure()
                    fig_ts.add_trace(go.Scatter(x=df_t.index,y=df_t[metric],mode="lines",
                        line=dict(color=color,width=2),fill="tozeroy",fillcolor=hex_to_rgba(color,.09)))
                    if len(df_t)>7:
                        roll=df_t[metric].rolling(7,min_periods=1).mean()
                        fig_ts.add_trace(go.Scatter(x=df_t.index,y=roll,mode="lines",
                            line=dict(color="#fff",width=1.5,dash="dot"),name="7-period Rolling Avg"))
                    fig_ts.update_layout(**PTHEME,height=380,title=f"{agg} Trend · {metric}",
                                         xaxis_title="Date",yaxis_title=AXIS_LABELS.get(metric,metric))
                    st.plotly_chart(fig_ts,use_container_width=True)
                # Monthly heatmap
                df_h=(df.dropna(subset=["Date"]).groupby("Date")[NUM_COLS].mean().sort_index())
                df_monthly=safe_resample(df_h,"Monthly").dropna(how="all")
                if not df_monthly.empty:
                    try: mlbls=df_monthly.index.strftime("%b %Y").tolist()
                    except AttributeError: mlbls=[str(i) for i in df_monthly.index]
                    mn=df_monthly.min(); mx=df_monthly.max(); dn=(mx-mn).replace(0,1)
                    df_norm=(df_monthly-mn)/dn
                    fig_mh=go.Figure(go.Heatmap(z=df_norm.T.values,x=mlbls,y=NUM_COLS,
                        colorscale=[[0,"#0b1422"],[.5,"#7c3aed"],[1,"#00e6ff"]],
                        text=df_monthly.T.round(1).values,texttemplate="%{text}",textfont=dict(size=9)))
                    fig_mh.update_layout(**PTHEME,height=340,title="Monthly Average — Normalised")
                    st.plotly_chart(fig_mh,use_container_width=True)
            else:
                st.info("No valid Date column found for time series analysis.")

        # Tab 6: User Analysis
        with tab_user:
            st.markdown("#### Per-User Analysis")
            if "User_ID" in df.columns:
                all_users=sorted(df["User_ID"].unique())
                cu1,cu2=st.columns([1,3])
                with cu1: sel_users=st.multiselect("Select Users",all_users,
                                                    default=all_users[:5] if len(all_users)>=5 else all_users)
                with cu2: u_metric=st.selectbox("Metric",NUM_COLS,key="user_metric")
                if sel_users:
                    df_u=df[df["User_ID"].isin(sel_users)]
                    u_avg=df_u.groupby("User_ID")[u_metric].mean().reset_index()
                    u_avg.columns=["User_ID","Average"]
                    if "Full Name" in df.columns:
                        nm=df.drop_duplicates("User_ID").set_index("User_ID")["Full Name"]
                        u_avg["Label"]=u_avg["User_ID"].map(nm).fillna("User "+u_avg["User_ID"].astype(str))
                    else:
                        u_avg["Label"]="User "+u_avg["User_ID"].astype(str)
                    fig_ua=go.Figure(go.Bar(x=u_avg["Label"],y=u_avg["Average"].round(2),
                        marker=dict(color=COL_COLORS.get(u_metric,"#00e6ff"),opacity=.8),
                        text=u_avg["Average"].round(2),textposition="outside"))
                    fig_ua.update_layout(**PTHEME,height=360,title=f"Average {u_metric} per User")
                    st.plotly_chart(fig_ua,use_container_width=True)
                    agg_d={c:["mean","min","max","std"] for c in NUM_COLS if c in df_u.columns}
                    if agg_d:
                        u_sum=df_u.groupby("User_ID").agg(agg_d)
                        u_sum.columns=[f"{c}_{f}" for c,f in u_sum.columns]
                        st.dataframe(u_sum.round(2).reset_index(),use_container_width=True)
                else:
                    st.info("Select at least one user to view analysis.")
            else:
                st.info("No 'User_ID' column found.")

        # ── M1 Complete → M2 handoff banner ──────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(0,230,255,.07),rgba(244,114,182,.07));
                    border:1.5px solid rgba(0,230,255,.3);border-radius:18px;padding:2rem 2.2rem;
                    margin-top:1.5rem;position:relative;overflow:hidden;">
            <div style="position:absolute;top:-30px;right:-30px;width:160px;height:160px;
                        border-radius:50%;background:radial-gradient(circle,rgba(244,114,182,.12),transparent);
                        pointer-events:none;"></div>
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:1rem;">
                <span style="font-size:2rem;">🎉</span>
                <div>
                    <div style="font-family:Orbitron,monospace;font-size:.65rem;letter-spacing:3px;
                                color:#00ffa3;margin-bottom:3px;">PRE PROCESSING COMPLETE</div>
                    <div style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:#fff;">
                        Pre Processing Pipeline — Done!
                    </div>
                </div>
            </div>
            <div style="font-family:JetBrains Mono,monospace;font-size:.72rem;color:#4a6a88;
                        line-height:2;margin-bottom:1.2rem;">
                ✅ &nbsp;Dataset uploaded &nbsp;·&nbsp; ✅ &nbsp;Nulls analysed
                &nbsp;·&nbsp; ✅ &nbsp;Data cleaned &nbsp;·&nbsp; ✅ &nbsp;EDA complete
            </div>
            <div style="font-family:JetBrains Mono,monospace;font-size:.68rem;color:#94a3b8;
                        margin-bottom:1.4rem;">
                Your dataset is fully preprocessed and ready for advanced pattern extraction.
                Continue to <b style="color:#f472b6;">Pattern Extracting</b> to run TSFresh feature
                extraction, Prophet forecasting, KMeans &amp; DBSCAN clustering, PCA and t-SNE.
            </div>
            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                <span style="font-family:JetBrains Mono,monospace;font-size:9px;padding:3px 10px;
                            border-radius:4px;background:rgba(244,114,182,.1);border:1px solid rgba(244,114,182,.3);
                            color:#f472b6;">🧪 TSFresh</span>
                <span style="font-family:JetBrains Mono,monospace;font-size:9px;padding:3px 10px;
                            border-radius:4px;background:rgba(244,114,182,.1);border:1px solid rgba(244,114,182,.3);
                            color:#f472b6;">📈 Prophet</span>
                <span style="font-family:JetBrains Mono,monospace;font-size:9px;padding:3px 10px;
                            border-radius:4px;background:rgba(244,114,182,.1);border:1px solid rgba(244,114,182,.3);
                            color:#f472b6;">🤖 KMeans</span>
                <span style="font-family:JetBrains Mono,monospace;font-size:9px;padding:3px 10px;
                            border-radius:4px;background:rgba(244,114,182,.1);border:1px solid rgba(244,114,182,.3);
                            color:#f472b6;">🔵 DBSCAN</span>
                <span style="font-family:JetBrains Mono,monospace;font-size:9px;padding:3px 10px;
                            border-radius:4px;background:rgba(244,114,182,.1);border:1px solid rgba(244,114,182,.3);
                            color:#f472b6;">📉 PCA + t-SNE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Action row: back button + big M2 launch button
        col_back, col_next = st.columns([1, 3])
        with col_back:
            if st.button("← Back to Preview", key="m1_back_from_eda"):
                st.session_state.m1_step=4; st.rerun()
        with col_next:
            st.markdown("""
            <style>
            div[data-testid="stHorizontalBlock"] div[data-testid="column"]:last-child .stButton>button{
                background:linear-gradient(135deg,#c026d3,#f472b6,#a855f7) !important;
                font-size:1rem !important;padding:.85rem 2rem !important;
                box-shadow:0 6px 30px rgba(244,114,182,.45) !important;
                letter-spacing:1px !important;
            }
            div[data-testid="stHorizontalBlock"] div[data-testid="column"]:last-child .stButton>button:hover{
                box-shadow:0 10px 40px rgba(244,114,182,.65) !important;
                transform:translateY(-3px) !important;
            }
            </style>
            """, unsafe_allow_html=True)
            if st.button("🔬  Continue to Pattern Extracting  →",
                         key="m1_to_m2_launch", use_container_width=True):
                st.session_state.active_module = "m2"; st.rerun()


# ══════════════════════════════════════════════════════════════════
# ███████████████  MODULE 2 — PATTERN EXTRACTING  █████████████████
# ══════════════════════════════════════════════════════════════════
def render_m2():
    # Header
    st.markdown(
        '<div class="hero-title">🔬 Pattern Extracting Pipeline</div>'
        '<div class="hero-sub">TSFRESH · PROPHET · KMEANS · DBSCAN · PCA · t-SNE</div>',
        unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("← Back to Launcher"):
        st.session_state.active_module=None; st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)

    # ── File upload ──────────────────────────────────────────────
    st.markdown('<div class="sec-lbl">01 · Upload Fitbit Dataset Files</div>',unsafe_allow_html=True)
    REQUIRED={"dailyActivity_merged.csv":"Daily Activity","hourlySteps_merged.csv":"Hourly Steps",
               "hourlyIntensities_merged.csv":"Hourly Intensities",
               "minuteSleep_merged.csv":"Minute Sleep","heartrate_seconds_merged.csv":"Heart Rate"}
    uploaded=st.file_uploader("Select all 5 Fitbit CSV files at once (Ctrl+Click / Cmd+Click)",
                               type=["csv"],accept_multiple_files=True,key="m2_uploader")
    file_map={f.name:f for f in uploaded} if uploaded else {}
    all_uploaded=all(n in file_map for n in REQUIRED)
    chips="".join(f'<span class="chip {"ok" if n in file_map else ""}">{"✅" if n in file_map else "⬜"} {lbl}</span>'
                  for n,lbl in REQUIRED.items())
    st.markdown(chips,unsafe_allow_html=True)
    if not all_uploaded:
        miss=[lbl for n,lbl in REQUIRED.items() if n not in file_map]
        st.info(f"⏳  Waiting for: **{', '.join(miss)}**")
    else:
        st.success("✅  All 5 files uploaded — run each step below in order.")

    st.markdown('<div class="sec-lbl">02 · Pipeline Steps — Click Each Step to Execute</div>',
                unsafe_allow_html=True)

    # ── STEP 1 : LOAD & VALIDATE ─────────────────────────────────
    with st.container(border=True):
        m2_step_header(1,"📂","Load & Validate All Files","load")
        dc,bc=st.columns([5,1])
        dc.markdown("<span style='color:#4b607a;font-size:13px'>Read all 5 CSVs · Report shape · Null audit</span>",
                    unsafe_allow_html=True)
        run1=bc.button("▶ Run" if not m2_done("load") else "🔁 Re-run",
                       key="m2b1",disabled=not all_uploaded,use_container_width=True)
        if run1: m2_reset_from("load")
        if m2_done("load") or run1:
            with st.spinner("Loading files…"):
                try:
                    daily=pd.read_csv(file_map["dailyActivity_merged.csv"])
                    hourly_s=pd.read_csv(file_map["hourlySteps_merged.csv"])
                    hourly_i=pd.read_csv(file_map["hourlyIntensities_merged.csv"])
                    sleep_df=pd.read_csv(file_map["minuteSleep_merged.csv"])
                    hr_df=pd.read_csv(file_map["heartrate_seconds_merged.csv"])
                    st.session_state.m2_daily=daily; st.session_state.m2_hourly_s=hourly_s
                    st.session_state.m2_hourly_i=hourly_i; st.session_state.m2_sleep=sleep_df
                    st.session_state.m2_hr=hr_df
                    c1,c2=st.columns(2)
                    with c1:
                        sum_box("📄 File Shapes",[
                            ("dailyActivity_merged",f"{daily.shape[0]:,} × {daily.shape[1]}","c-blue"),
                            ("hourlySteps_merged",f"{hourly_s.shape[0]:,} × {hourly_s.shape[1]}","c-blue"),
                            ("hourlyIntensities_merged",f"{hourly_i.shape[0]:,} × {hourly_i.shape[1]}","c-blue"),
                            ("minuteSleep_merged",f"{sleep_df.shape[0]:,} × {sleep_df.shape[1]}","c-blue"),
                            ("heartrate_seconds_merged",f"{hr_df.shape[0]:,} × {hr_df.shape[1]}","c-blue"),
                        ])
                    with c2:
                        sum_box("🔍 Null Value Audit",[
                            (nm,f"{d.isnull().sum().sum()} nulls {'✅' if d.isnull().sum().sum()==0 else '⚠️'}",
                             "c-green" if d.isnull().sum().sum()==0 else "c-orange")
                            for nm,d in [("dailyActivity",daily),("hourlySteps",hourly_s),
                                         ("hourlyIntensities",hourly_i),("minuteSleep",sleep_df),("heartrate",hr_df)]])
                    metric_tiles([(daily["Id"].nunique(),"Unique Users (daily)","#38bdf8"),
                                  (hr_df["Id"].nunique(),"Unique Users (HR)","#f472b6"),
                                  (sleep_df["Id"].nunique(),"Unique Users (sleep)","#34d399"),
                                  (f"{hr_df.shape[0]:,}","Total HR Records","#fb923c")])
                    with st.expander("👁 Preview dailyActivity (top 5 rows)"):
                        st.dataframe(daily.head(5),use_container_width=True,hide_index=True)
                    m2_mark("load"); st.success("✅ Step 1 complete — all 5 files loaded.")
                except Exception as e: st.error(f"❌ {e}\n\n{traceback.format_exc()}")

    # ── STEP 2 : TIMESTAMP PARSING ───────────────────────────────
    with st.container(border=True):
        m2_step_header(2,"⏱","Timestamp Parsing & Time Normalisation","timestamps","load")
        dc,bc=st.columns([5,1])
        dc.markdown("<span style='color:#4b607a;font-size:13px'>Parse all date/time columns · Resample HR seconds → 1-min</span>",
                    unsafe_allow_html=True)
        run2=bc.button("▶ Run" if not m2_done("timestamps") else "🔁 Re-run",
                       key="m2b2",disabled=not m2_done("load"),use_container_width=True)
        if run2: m2_reset_from("timestamps")
        if m2_done("timestamps") or run2:
            with st.spinner("Parsing timestamps…"):
                try:
                    daily=st.session_state.m2_daily.copy(); hourly_s=st.session_state.m2_hourly_s.copy()
                    hourly_i=st.session_state.m2_hourly_i.copy(); sleep_df=st.session_state.m2_sleep.copy()
                    hr_df=st.session_state.m2_hr.copy()
                    daily["ActivityDate"]=pd.to_datetime(daily["ActivityDate"],format="%m/%d/%Y")
                    hourly_s["ActivityHour"]=pd.to_datetime(hourly_s["ActivityHour"],format="%m/%d/%Y %I:%M:%S %p")
                    hourly_i["ActivityHour"]=pd.to_datetime(hourly_i["ActivityHour"],format="%m/%d/%Y %I:%M:%S %p")
                    sleep_df["date"]=pd.to_datetime(sleep_df["date"],format="%m/%d/%Y %I:%M:%S %p")
                    hr_df["Time"]=pd.to_datetime(hr_df["Time"],format="%m/%d/%Y %I:%M:%S %p")
                    hr_minute=(hr_df.set_index("Time").groupby("Id")["Value"]
                               .resample("1min").mean().reset_index())
                    hr_minute.columns=["Id","Time","HeartRate"]; hr_minute=hr_minute.dropna()
                    freq_check=(hourly_s.groupby("Id")["ActivityHour"].diff().dropna().dt.total_seconds()/3600)
                    st.session_state.m2_daily=daily; st.session_state.m2_hourly_s=hourly_s
                    st.session_state.m2_hourly_i=hourly_i; st.session_state.m2_sleep=sleep_df
                    st.session_state.m2_hr=hr_df; st.session_state.m2_hr_minute=hr_minute
                    c1,c2=st.columns(2)
                    with c1:
                        sum_box("📅 Parsed Date Formats",[
                            ("ActivityDate","%m/%d/%Y","c-green"),
                            ("ActivityHour","%m/%d/%Y %I:%M:%S %p","c-green"),
                            ("minuteSleep · date","%m/%d/%Y %I:%M:%S %p","c-green"),
                            ("heartrate · Time","%m/%d/%Y %I:%M:%S %p","c-green")])
                        sum_box("📆 Date Ranges",[
                            ("Daily start",str(daily["ActivityDate"].min().date()),"c-blue"),
                            ("Daily end",str(daily["ActivityDate"].max().date()),"c-blue"),
                            ("Span",f"{(daily['ActivityDate'].max()-daily['ActivityDate'].min()).days} days","c-orange")])
                    with c2:
                        sum_box("🔄 HR Resampling",[
                            ("Original","Seconds","c-orange"),("Target","1-Minute intervals","c-green"),
                            ("Rows before",f"{hr_df.shape[0]:,}","c-blue"),("Rows after",f"{hr_minute.shape[0]:,}","c-green"),
                            ("Reduction",f"~{hr_df.shape[0]/hr_minute.shape[0]:.1f}×","c-pink")])
                        sum_box("📊 Frequency Verification",[
                            ("Hourly median interval",f"{freq_check.median():.1f} h","c-green"),
                            ("Exact 1-h accuracy",f"{(freq_check==1.0).mean()*100:.1f}%","c-green"),
                            ("Sleep stages","1=Light · 2=Deep · 3=REM","c-purple")])
                    metric_tiles([(f"{hr_minute.shape[0]:,}","HR Rows (1-min)","#38bdf8"),
                                  (f"{hr_df.shape[0]:,}","HR Rows (original)","#f472b6"),
                                  (f"{(daily['ActivityDate'].max()-daily['ActivityDate'].min()).days}d","Date Span","#34d399"),
                                  (f"{(freq_check==1.0).mean()*100:.1f}%","Hourly Accuracy","#fb923c")])
                    m2_mark("timestamps"); st.success("✅ Step 2 complete — timestamps parsed, HR resampled.")
                except Exception as e: st.error(f"❌ {e}\n\n{traceback.format_exc()}")

    # ── STEP 3 : MASTER DATAFRAME ────────────────────────────────
    with st.container(border=True):
        m2_step_header(3,"🔗","Aggregate & Build Master DataFrame","master","timestamps")
        dc,bc=st.columns([5,1])
        dc.markdown("<span style='color:#4b607a;font-size:13px'>Daily HR stats · Daily sleep totals · Merge all → master · Fill nulls</span>",
                    unsafe_allow_html=True)
        run3=bc.button("▶ Run" if not m2_done("master") else "🔁 Re-run",
                       key="m2b3",disabled=not m2_done("timestamps"),use_container_width=True)
        if run3: m2_reset_from("master")
        if m2_done("master") or run3:
            with st.spinner("Building master dataframe…"):
                try:
                    daily=st.session_state.m2_daily.copy(); sleep_df=st.session_state.m2_sleep.copy()
                    hr_minute=st.session_state.m2_hr_minute.copy()
                    hr_minute["Date"]=hr_minute["Time"].dt.date
                    hr_daily=(hr_minute.groupby(["Id","Date"])["HeartRate"]
                              .agg(["mean","max","min","std"]).reset_index()
                              .rename(columns={"mean":"AvgHR","max":"MaxHR","min":"MinHR","std":"StdHR"}))
                    sleep_df["Date"]=sleep_df["date"].dt.date
                    sleep_daily=(sleep_df.groupby(["Id","Date"])
                                 .agg(TotalSleepMinutes=("value","count"),
                                      DominantSleepStage=("value",lambda x:x.mode()[0])).reset_index())
                    master=daily.rename(columns={"ActivityDate":"Date"}).copy()
                    master["Date"]=master["Date"].dt.date
                    master=master.merge(hr_daily,on=["Id","Date"],how="left")
                    master=master.merge(sleep_daily,on=["Id","Date"],how="left")
                    master["TotalSleepMinutes"]=master["TotalSleepMinutes"].fillna(0)
                    master["DominantSleepStage"]=master["DominantSleepStage"].fillna(0)
                    for col in ["AvgHR","MaxHR","MinHR","StdHR"]:
                        master[col]=master.groupby("Id")[col].transform(lambda x:x.fillna(x.median()))
                    st.session_state.m2_master=master
                    c1,c2=st.columns(2)
                    with c1:
                        sum_box("🗂️ Master DataFrame Info",[
                            ("Total rows",f"{master.shape[0]:,}","c-blue"),("Total columns",f"{master.shape[1]}","c-blue"),
                            ("Unique users",f"{master['Id'].nunique()}","c-green"),
                            ("Remaining nulls",f"{master.isnull().sum().sum()} ✅","c-green"),
                            ("HR null strategy","Per-user median fill","c-orange"),("Sleep null strategy","Fill with 0","c-orange")])
                        sum_box("❤️ Heart Rate (daily avg)",[
                            ("Mean AvgHR",f"{master['AvgHR'].mean():.1f} bpm","c-pink"),
                            ("Min AvgHR",f"{master['AvgHR'].min():.1f} bpm","c-blue"),
                            ("Max AvgHR",f"{master['AvgHR'].max():.1f} bpm","c-orange"),
                            ("Peak MaxHR",f"{master['MaxHR'].max():.1f} bpm","c-orange")])
                    with c2:
                        sum_box("🚶 Activity Stats",[
                            ("Avg daily steps",f"{master['TotalSteps'].mean():,.0f}","c-blue"),
                            ("Max daily steps",f"{master['TotalSteps'].max():,.0f}","c-green"),
                            ("Avg calories/day",f"{master['Calories'].mean():,.0f} kcal","c-orange"),
                            ("Avg very active min",f"{master['VeryActiveMinutes'].mean():.1f}","c-green"),
                            ("Avg sedentary min",f"{master['SedentaryMinutes'].mean():.1f}","c-pink")])
                        sum_box("😴 Sleep Stats",[
                            ("Avg sleep/night",f"{master['TotalSleepMinutes'].mean():.1f} min","c-purple"),
                            ("Max sleep/night",f"{master['TotalSleepMinutes'].max():.0f} min","c-blue"),
                            ("Records with sleep",f"{(master['TotalSleepMinutes']>0).sum()}","c-green")])
                    metric_tiles([(f"{master.shape[0]:,}","Master Rows","#38bdf8"),
                                  (master["Id"].nunique(),"Users","#f472b6"),
                                  (master.shape[1],"Columns","#34d399"),
                                  (master.isnull().sum().sum(),"Remaining Nulls","#22c55e")])
                    with st.expander("👁 Master DataFrame — First 20 Rows"):
                        st.dataframe(master[["Id","Date","TotalSteps","Calories","AvgHR",
                                             "TotalSleepMinutes","VeryActiveMinutes","SedentaryMinutes"]].head(20),
                                     use_container_width=True,hide_index=True)
                    m2_mark("master"); st.success(f"✅ Step 3 complete — {master.shape[0]:,} × {master.shape[1]} cols.")
                except Exception as e: st.error(f"❌ {e}\n\n{traceback.format_exc()}")

    # ── STEP 4 : TSFRESH ─────────────────────────────────────────
    with st.container(border=True):
        m2_step_header(4,"🧪","TSFresh Feature Extraction","tsfresh","master")
        dc,bc=st.columns([5,1])
        dc.markdown(f"<span style='color:#4b607a;font-size:13px'>Extract statistical features from minute-level HR · Mode: <b>{tsfresh_mode}</b></span>",
                    unsafe_allow_html=True)
        run4=bc.button("▶ Run" if not m2_done("tsfresh") else "🔁 Re-run",
                       key="m2b4",disabled=not m2_done("master"),use_container_width=True)
        if run4: m2_reset_from("tsfresh")
        if m2_done("tsfresh") or run4:
            with st.spinner("Running TSFresh — may take 30–60 seconds…"):
                try:
                    from tsfresh import extract_features
                    from tsfresh.feature_extraction import MinimalFCParameters,EfficientFCParameters
                    from sklearn.preprocessing import MinMaxScaler
                    hr_minute=st.session_state.m2_hr_minute.copy()
                    ts_hr=(hr_minute[["Id","Time","HeartRate"]].dropna().sort_values(["Id","Time"])
                           .rename(columns={"Id":"id","Time":"time","HeartRate":"value"}))
                    fc_params=(MinimalFCParameters() if tsfresh_mode=="MinimalFCParameters" else EfficientFCParameters())
                    features=extract_features(ts_hr,column_id="id",column_sort="time",column_value="value",
                                              default_fc_parameters=fc_params,disable_progressbar=True)
                    features=features.dropna(axis=1,how="all"); st.session_state.m2_features=features
                    c1,c2=st.columns(2)
                    with c1:
                        sum_box("🧪 Extraction Details",[("Mode",tsfresh_mode,"c-blue"),
                            ("Input signal","Minute-level heart rate","c-blue"),
                            ("Users processed",f"{features.shape[0]}","c-green"),
                            ("Features extracted",f"{features.shape[1]}","c-green"),
                            ("NaN values remain","0 (dropped automatically)","c-green")])
                    with c2:
                        sum_box("📐 Feature Names",[(f"{i+1}.",col,"c-purple") for i,col in enumerate(features.columns)])
                    metric_tiles([(features.shape[0],"Users Processed","#38bdf8"),
                                  (features.shape[1],"Features Extracted","#f472b6"),(0,"NaN Values","#22c55e")])
                    ss_badge("📸 SCREENSHOT 1 — TSFresh Feature Matrix Heatmap")
                    scaler_vis=MinMaxScaler()
                    features_norm=pd.DataFrame(scaler_vis.fit_transform(features),
                                               index=features.index,columns=features.columns)
                    fig_h,ax_h=plt.subplots(figsize=(14,max(5,features_norm.shape[0]*.5)))
                    fig_h.patch.set_facecolor(DARK_BG); ax_h.set_facecolor(CARD_BG)
                    sns.heatmap(features_norm,cmap="coolwarm",annot=True,fmt=".2f",
                                linewidths=.6,linecolor=GRID_CLR,ax=ax_h,cbar_kws={"shrink":.7})
                    ax_h.set_title("TSFresh Feature Matrix — Real Fitbit HR\n(Normalised 0–1)",
                                   fontsize=13,color=TEXT_CLR,pad=14)
                    ax_h.tick_params(colors=MUTED); plt.tight_layout()
                    st.pyplot(fig_h,use_container_width=True)
                    st.download_button("⬇️ Download Screenshot 1 — TSFresh Heatmap (PNG)",
                                       data=fig_to_png(fig_h),file_name="screenshot1_tsfresh_heatmap.png",
                                       mime="image/png",key="dl_heatmap")
                    plt.close(fig_h)
                    buf=io.StringIO(); features.to_csv(buf)
                    st.download_button("⬇️ Download tsfresh_features.csv",buf.getvalue(),
                                       "tsfresh_features.csv","text/csv",key="dl_tsfresh_csv")
                    m2_mark("tsfresh"); st.success(f"✅ Step 4 complete — {features.shape[1]} features extracted.")
                except Exception as e: st.error(f"❌ {e}\n\n{traceback.format_exc()}")

    # ── STEP 5 : PROPHET ─────────────────────────────────────────
    with st.container(border=True):
        m2_step_header(5,"📈","Prophet Trend Forecasting","prophet","tsfresh")
        dc,bc=st.columns([5,1])
        dc.markdown(f"<span style='color:#4b607a;font-size:13px'>Fit Prophet on HR · Steps · Sleep · {forecast_days}-day forecast · 80% CI</span>",
                    unsafe_allow_html=True)
        run5=bc.button("▶ Run" if not m2_done("prophet") else "🔁 Re-run",
                       key="m2b5",disabled=not m2_done("tsfresh"),use_container_width=True)
        if run5: m2_reset_from("prophet")
        if m2_done("prophet") or run5:
            with st.spinner("Fitting 3 Prophet models…"):
                try:
                    from prophet import Prophet
                    hr_minute=st.session_state.m2_hr_minute.copy()
                    daily=st.session_state.m2_daily.copy()
                    master=st.session_state.m2_master.copy()
                    hr_minute["Date"]=pd.to_datetime(hr_minute["Time"]).dt.normalize()
                    master["Date"]=pd.to_datetime(master["Date"])
                    def fit_prophet(df,date_col,val_col,cp=.1):
                        agg=df.groupby(date_col)[val_col].mean().reset_index()
                        agg.columns=["ds","y"]; agg["ds"]=pd.to_datetime(agg["ds"],errors="coerce")
                        agg=agg.dropna().sort_values("ds")
                        m=Prophet(daily_seasonality=False,weekly_seasonality=True,yearly_seasonality=False,
                                  interval_width=.80,changepoint_prior_scale=cp,changepoint_range=.8)
                        m.fit(agg); fc=m.predict(m.make_future_dataframe(periods=forecast_days))
                        return agg,m,fc
                    act_hr,mod_hr,fc_hr=fit_prophet(hr_minute,"Date","HeartRate",changepoint_scale)
                    act_st,mod_st,fc_st=fit_prophet(daily,"ActivityDate","TotalSteps",.1)
                    act_sl,mod_sl,fc_sl=fit_prophet(master,"Date","TotalSleepMinutes",.1)
                    st.session_state.m2_fc_hr=fc_hr
                    c1,c2,c3=st.columns(3)
                    with c1:
                        sum_box("💓 Heart Rate Model",[("Metric","Heart Rate (bpm)","c-blue"),
                            ("Changepoint scale",f"{changepoint_scale}","c-orange"),
                            ("Forecast horizon",f"{forecast_days} days","c-green"),
                            ("Actual HR range",f"{act_hr['y'].min():.1f}–{act_hr['y'].max():.1f} bpm","c-pink"),
                            ("Weekly seasonality","✅ Enabled","c-green")])
                    with c2:
                        sum_box("🚶 Steps Model",[("Metric","Total Steps/day","c-blue"),
                            ("Forecast horizon",f"{forecast_days} days","c-green"),
                            ("Avg actual steps",f"{act_st['y'].mean():,.0f}","c-pink"),
                            ("Weekly seasonality","✅ Enabled","c-green")])
                    with c3:
                        sum_box("😴 Sleep Model",[("Metric","Sleep minutes/day","c-blue"),
                            ("Forecast horizon",f"{forecast_days} days","c-green"),
                            ("Avg actual sleep",f"{act_sl['y'].mean():.1f} min","c-pink"),
                            ("Weekly seasonality","✅ Enabled","c-green")])
                    metric_tiles([("3","Models Fitted","#38bdf8"),(f"{forecast_days}d","Forecast Horizon","#f472b6"),
                                  ("80%","Confidence Interval","#34d399"),(f"{changepoint_scale}","CP Scale","#fb923c")])
                    def fplot(actual,fc,color,title,ylabel):
                        fig,ax=dark_fig(13,4.2)
                        ax.scatter(actual["ds"],actual["y"],color=color,s=18,alpha=.7,label="Actual",zorder=3)
                        ax.plot(fc["ds"],fc["yhat"],color="#3b82f6",lw=2.5,label="Forecast Trend")
                        ax.fill_between(fc["ds"],fc["yhat_lower"],fc["yhat_upper"],alpha=.2,color="#3b82f6",label="80% CI")
                        ax.axvline(actual["ds"].max(),color="#fb923c",ls="--",lw=1.8,label="Forecast Start")
                        ax.set_title(title,fontsize=12,color=TEXT_CLR); ax.set_xlabel("Date"); ax.set_ylabel(ylabel)
                        ax.legend(fontsize=8,facecolor=CARD_BG,labelcolor=TEXT_CLR,edgecolor=GRID_CLR)
                        plt.setp(ax.get_xticklabels(),rotation=30,ha="right"); plt.tight_layout(); return fig
                    ss_badge("📸 SCREENSHOT 2 — Heart Rate Forecast")
                    fig1=fplot(act_hr,fc_hr,"#e53e3e",f"Heart Rate Prophet Forecast (+{forecast_days}d)","bpm")
                    st.pyplot(fig1,use_container_width=True)
                    st.download_button("⬇️ Download Screenshot 2 — HR Forecast (PNG)",data=fig_to_png(fig1),
                                       file_name="screenshot2_prophet_heartrate.png",mime="image/png",key="dl_hr")
                    plt.close(fig1)
                    st.markdown("##### 📐 Prophet Components — Heart Rate")
                    fc_comp=mod_hr.plot_components(fc_hr); fc_comp.patch.set_facecolor(CARD_BG)
                    for ax_c in fc_comp.get_axes():
                        ax_c.set_facecolor(CARD_BG); ax_c.tick_params(colors=MUTED); ax_c.title.set_color(TEXT_CLR)
                    st.pyplot(fc_comp,use_container_width=True)
                    st.download_button("⬇️ Download Prophet Components (PNG)",data=fig_to_png(fc_comp),
                                       file_name="prophet_hr_components.png",mime="image/png",key="dl_comp")
                    plt.close(fc_comp)
                    ss_badge("📸 SCREENSHOT 3 — Total Steps Forecast")
                    fig2=fplot(act_st,fc_st,"#38a169",f"Total Steps Prophet Forecast (+{forecast_days}d)","Steps/day")
                    st.pyplot(fig2,use_container_width=True)
                    st.download_button("⬇️ Download Screenshot 3 — Steps Forecast (PNG)",data=fig_to_png(fig2),
                                       file_name="screenshot3_prophet_steps.png",mime="image/png",key="dl_st")
                    plt.close(fig2)
                    ss_badge("📸 SCREENSHOT 4 — Sleep Minutes Forecast")
                    fig3=fplot(act_sl,fc_sl,"#b794f4",f"Sleep Minutes Prophet Forecast (+{forecast_days}d)","Min/day")
                    st.pyplot(fig3,use_container_width=True)
                    st.download_button("⬇️ Download Screenshot 4 — Sleep Forecast (PNG)",data=fig_to_png(fig3),
                                       file_name="screenshot4_prophet_sleep.png",mime="image/png",key="dl_sl")
                    plt.close(fig3)
                    m2_mark("prophet"); st.success(f"✅ Step 5 complete — 3 Prophet models · {forecast_days}-day forecasts.")
                except Exception as e: st.error(f"❌ {e}\n\n{traceback.format_exc()}")

    # ── STEP 6 : CLUSTERING ──────────────────────────────────────
    with st.container(border=True):
        m2_step_header(6,"🤖","KMeans & DBSCAN Clustering","clustering","prophet")
        dc,bc=st.columns([5,1])
        dc.markdown(f"<span style='color:#4b607a;font-size:13px'>Scale · Elbow · KMeans K={optimal_k} · DBSCAN eps={eps_val} · PCA · t-SNE</span>",
                    unsafe_allow_html=True)
        run6=bc.button("▶ Run" if not m2_done("clustering") else "🔁 Re-run",
                       key="m2b6",disabled=not m2_done("prophet"),use_container_width=True)
        if run6: m2_reset_from("clustering")
        if m2_done("clustering") or run6:
            with st.spinner("Running clustering pipeline…"):
                try:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.cluster import KMeans,DBSCAN
                    from sklearn.decomposition import PCA
                    from sklearn.manifold import TSNE
                    master=st.session_state.m2_master.copy()
                    cluster_cols=["TotalSteps","Calories","VeryActiveMinutes","FairlyActiveMinutes",
                                  "LightlyActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]
                    cluster_features=master.groupby("Id")[cluster_cols].mean().round(3).dropna()
                    scaler=StandardScaler(); X_scaled=scaler.fit_transform(cluster_features)
                    inertias=[KMeans(n_clusters=k,random_state=42,n_init=10).fit(X_scaled).inertia_ for k in range(2,10)]
                    ss_badge("📸 SCREENSHOT — KMeans Elbow Curve")
                    fig_el,ax_el=dark_fig(8,3.5)
                    ax_el.plot(range(2,10),inertias,"o-",color="#63b3ed",lw=2.5,ms=9,markerfacecolor="#f687b3")
                    ax_el.axvline(optimal_k,color="#fb923c",ls="--",lw=1.8,label=f"Selected K={optimal_k}")
                    ax_el.set_title("KMeans Elbow Curve",fontsize=12,color=TEXT_CLR)
                    ax_el.set_xlabel("K"); ax_el.set_ylabel("Inertia")
                    ax_el.legend(fontsize=8,facecolor=CARD_BG,labelcolor=TEXT_CLR,edgecolor=GRID_CLR)
                    plt.tight_layout(); st.pyplot(fig_el,use_container_width=True); plt.close(fig_el)
                    kmeans=KMeans(n_clusters=optimal_k,random_state=42,n_init=10)
                    kmeans_labels=kmeans.fit_predict(X_scaled); cluster_features["KMeans_Cluster"]=kmeans_labels
                    dbscan=DBSCAN(eps=eps_val,min_samples=int(min_samples))
                    dbscan_labels=dbscan.fit_predict(X_scaled); cluster_features["DBSCAN_Cluster"]=dbscan_labels
                    n_clusters_db=len(set(dbscan_labels))-(1 if -1 in dbscan_labels else 0)
                    n_noise=list(dbscan_labels).count(-1)
                    pca=PCA(n_components=2,random_state=42); X_pca=pca.fit_transform(X_scaled)
                    var_explained=pca.explained_variance_ratio_*100
                    tsne=TSNE(n_components=2,random_state=42,perplexity=min(30,len(X_scaled)-1),max_iter=1000)
                    X_tsne=tsne.fit_transform(X_scaled)
                    st.session_state.m2_kmeans_labels=kmeans_labels; st.session_state.m2_dbscan_labels=dbscan_labels
                    st.session_state.m2_cluster_features=cluster_features; st.session_state.m2_X_pca=X_pca
                    st.session_state.m2_X_tsne=X_tsne; st.session_state.m2_var_explained=var_explained
                    st.session_state.m2_n_clusters_db=n_clusters_db; st.session_state.m2_n_noise=n_noise
                    c1,c2,c3=st.columns(3)
                    with c1:
                        sum_box("⚙️ Scaling",[("Users",f"{cluster_features.shape[0]}","c-blue"),
                            ("Features used",f"{len(cluster_cols)}","c-blue"),("Scaling","StandardScaler z-score","c-orange")])
                    with c2:
                        km_dist={i:int((np.array(kmeans_labels)==i).sum()) for i in range(optimal_k)}
                        sum_box("🎯 KMeans Results",[("K selected",f"{optimal_k}","c-blue"),
                            *[(f"Cluster {ci}",f"{cnt} users","c-green") for ci,cnt in km_dist.items()]])
                    with c3:
                        sum_box("🔵 DBSCAN Results",[("Epsilon",f"{eps_val}","c-blue"),
                            ("Min samples",f"{int(min_samples)}","c-blue"),("Clusters",f"{n_clusters_db}","c-green"),
                            ("Noise",f"{n_noise} users","c-orange"),
                            ("Noise %",f"{n_noise/len(dbscan_labels)*100:.1f}%","c-orange")])
                    c1b,c2b=st.columns(2)
                    with c1b:
                        sum_box("📉 PCA",[("PC1 variance",f"{var_explained[0]:.1f}%","c-blue"),
                            ("PC2 variance",f"{var_explained[1]:.1f}%","c-blue"),
                            ("Total explained",f"{sum(var_explained):.1f}%","c-green"),
                            ("Dims in→out",f"{X_scaled.shape[1]} → 2","c-orange")])
                    with c2b:
                        sum_box("🌐 t-SNE",[("Perplexity",f"{min(30,len(X_scaled)-1)}","c-blue"),
                            ("Iterations","1000","c-blue"),("Output shape",f"{X_tsne.shape[0]} × 2","c-green")])
                    metric_tiles([(cluster_features.shape[0],"Users Clustered","#38bdf8"),
                                  (optimal_k,"KMeans Clusters","#f472b6"),(n_clusters_db,"DBSCAN Clusters","#34d399"),
                                  (n_noise,"Noise Points","#fb923c"),(f"{sum(var_explained):.1f}%","PCA Variance","#a78bfa")])
                    def scatter(X2d,labels,title,noise_red=False):
                        fig,ax=dark_fig(8,6)
                        for lbl in sorted(set(labels)):
                            mask=np.array(labels)==lbl
                            if noise_red and lbl==-1:
                                ax.scatter(X2d[mask,0],X2d[mask,1],c="red",marker="x",s=150,label="Noise",alpha=.9,lw=2)
                            else:
                                ax.scatter(X2d[mask,0],X2d[mask,1],c=PALETTE[lbl%len(PALETTE)],
                                           label=f"Cluster {lbl}",s=120,alpha=.85,edgecolors="white",lw=.7)
                        ax.set_title(title,fontsize=11,color=TEXT_CLR)
                        ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)" if "PCA" in title else "Dim 1")
                        ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)" if "PCA" in title else "Dim 2")
                        ax.legend(title="Cluster",fontsize=8,facecolor=CARD_BG,labelcolor=TEXT_CLR,edgecolor=GRID_CLR)
                        plt.tight_layout(); return fig
                    ss_badge("📸 SCREENSHOT 5 — KMeans PCA Scatter")
                    fig_km=scatter(X_pca,kmeans_labels,f"KMeans — PCA Projection  (K={optimal_k})")
                    st.pyplot(fig_km,use_container_width=True)
                    st.download_button("⬇️ Download Screenshot 5 — KMeans PCA (PNG)",data=fig_to_png(fig_km),
                                       file_name="screenshot5_kmeans_pca.png",mime="image/png",key="dl_km_pca")
                    plt.close(fig_km)
                    ss_badge("📸 SCREENSHOT 6 — DBSCAN PCA Scatter")
                    fig_db=scatter(X_pca,dbscan_labels,f"DBSCAN — PCA Projection  (eps={eps_val})",noise_red=True)
                    st.pyplot(fig_db,use_container_width=True)
                    st.download_button("⬇️ Download Screenshot 6 — DBSCAN PCA (PNG)",data=fig_to_png(fig_db),
                                       file_name="screenshot6_dbscan_pca.png",mime="image/png",key="dl_db_pca")
                    plt.close(fig_db)
                    ss_badge("📸 SCREENSHOT 7 — t-SNE Projection (KMeans + DBSCAN)")
                    fig_ts,axes_t=plt.subplots(1,2,figsize=(14,6)); fig_ts.patch.set_facecolor(DARK_BG)
                    for ax_t,lbls,ttl in zip(axes_t,[kmeans_labels,dbscan_labels],
                                             [f"KMeans — t-SNE (K={optimal_k})",f"DBSCAN — t-SNE (eps={eps_val})"]):
                        dark_ax(ax_t)
                        for lbl in sorted(set(lbls)):
                            mask=np.array(lbls)==lbl
                            if lbl==-1: ax_t.scatter(X_tsne[mask,0],X_tsne[mask,1],c="red",marker="x",s=150,label="Noise",alpha=.9,lw=2)
                            else: ax_t.scatter(X_tsne[mask,0],X_tsne[mask,1],c=PALETTE[lbl%len(PALETTE)],
                                               label=f"Cluster {lbl}",s=120,alpha=.85,edgecolors="white",lw=.7)
                        ax_t.set_title(ttl,fontsize=11,color=TEXT_CLR)
                        ax_t.set_xlabel("t-SNE dim 1"); ax_t.set_ylabel("t-SNE dim 2")
                        ax_t.legend(title="Cluster",fontsize=8,facecolor=CARD_BG,labelcolor=TEXT_CLR,edgecolor=GRID_CLR)
                    plt.tight_layout(); st.pyplot(fig_ts,use_container_width=True)
                    st.download_button("⬇️ Download Screenshot 7 — t-SNE (PNG)",data=fig_to_png(fig_ts),
                                       file_name="screenshot7_tsne.png",mime="image/png",key="dl_tsne")
                    plt.close(fig_ts)
                    feat_cols=[c for c in cluster_features.columns if c not in ("KMeans_Cluster","DBSCAN_Cluster")]
                    profile=cluster_features.groupby("KMeans_Cluster")[feat_cols].mean().round(2)
                    st.dataframe(profile,use_container_width=True)
                    ss_badge("📸 SCREENSHOT 8 — Cluster Profile Bar Chart")
                    fig_pr,ax_pr=dark_fig(12,4)
                    profile[["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]].plot(
                        kind="bar",ax=ax_pr,colormap="Set2",edgecolor="white",width=.72)
                    ax_pr.set_title("Cluster Profiles — Key Feature Averages",fontsize=12,color=TEXT_CLR)
                    ax_pr.set_xlabel("Cluster"); ax_pr.set_ylabel("Mean Value")
                    ax_pr.set_xticklabels([f"Cluster {i}" for i in range(optimal_k)],rotation=0)
                    ax_pr.legend(bbox_to_anchor=(1.02,1),title="Feature",facecolor=CARD_BG,labelcolor=TEXT_CLR,edgecolor=GRID_CLR)
                    plt.tight_layout(); st.pyplot(fig_pr,use_container_width=True)
                    st.download_button("⬇️ Download Screenshot 8 — Cluster Profiles (PNG)",data=fig_to_png(fig_pr),
                                       file_name="screenshot8_cluster_profiles.png",mime="image/png",key="dl_profiles")
                    plt.close(fig_pr)
                    st.markdown("##### 🏷️ Cluster Interpretation Cards")
                    cl_cols_=st.columns(optimal_k)
                    CARD_THEMES=[("#38bdf8","#071829"),("#f472b6","#250a18"),("#34d399","#041f12"),
                                 ("#fb923c","#1f0d04"),("#a78bfa","#130a2a")]
                    for i,col in enumerate(cl_cols_):
                        if i not in profile.index: continue
                        row=profile.loc[i]
                        steps,act,sed,cal,slp=(row["TotalSteps"],row["VeryActiveMinutes"],row["SedentaryMinutes"],row["Calories"],row["TotalSleepMinutes"])
                        emoji,lbl=(("🏃","HIGHLY ACTIVE") if steps>10000 else ("🚶","MODERATELY ACTIVE") if steps>5000 else ("🛋️","SEDENTARY"))
                        fg,bg=CARD_THEMES[i%len(CARD_THEMES)]
                        col.markdown(
                            f'<div style="background:{bg};border:1.5px solid {fg};border-radius:12px;padding:16px;">'
                            f'<div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:800;color:{fg}">{emoji} Cluster {i}</div>'
                            f'<div style="font-family:JetBrains Mono,monospace;font-size:9px;letter-spacing:1.5px;color:{fg};margin-bottom:10px">{lbl}</div>'
                            f'<div style="font-size:12px;color:#94a3b8;line-height:2">'
                            f'👣 <b style="color:#e2e8f0">{steps:,.0f}</b> steps/day<br>🔥 <b style="color:#e2e8f0">{cal:,.0f}</b> kcal/day<br>'
                            f'⚡ <b style="color:#e2e8f0">{act:.0f} min</b> very active<br>🪑 <b style="color:#e2e8f0">{sed:.0f} min</b> sedentary<br>'
                            f'😴 <b style="color:#e2e8f0">{slp:.0f} min</b> sleep/night</div></div>',unsafe_allow_html=True)
                    m2_mark("clustering"); st.success(f"✅ Step 6 complete — KMeans: {optimal_k} clusters | DBSCAN: {n_clusters_db} clusters, {n_noise} noise.")
                except Exception as e: st.error(f"❌ {e}\n\n{traceback.format_exc()}")

    # ── STEP 7 : FINAL SUMMARY ───────────────────────────────────
    with st.container(border=True):
        m2_step_header(7,"📊","Pattern Extracting — Final Summary","summary","clustering")
        dc,bc=st.columns([5,1])
        dc.markdown("<span style='color:#4b607a;font-size:13px'>All results compiled · Download full report</span>",
                    unsafe_allow_html=True)
        run7=bc.button("▶ Run" if not m2_done("summary") else "🔁 Re-run",
                       key="m2b7",disabled=not m2_done("clustering"),use_container_width=True)
        if run7: m2_reset_from("summary")
        if m2_done("summary") or run7:
            try:
                master=st.session_state.m2_master; features=st.session_state.m2_features
                kmeans_labels=st.session_state.m2_kmeans_labels; dbscan_labels=st.session_state.m2_dbscan_labels
                n_clusters_db=st.session_state.m2_n_clusters_db; n_noise=st.session_state.m2_n_noise
                var_explained=st.session_state.m2_var_explained; cluster_features=st.session_state.m2_cluster_features
                fc_hr=st.session_state.m2_fc_hr
                st.markdown("---")
                st.markdown('<div style="font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;'
                            'background:linear-gradient(135deg,#38bdf8,#f472b6);'
                            '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
                            'margin-bottom:14px">📊 Pattern Extracting — Complete Results</div>',
                            unsafe_allow_html=True)
                metric_tiles([(master["Id"].nunique(),"Total Users","#38bdf8"),("31","Days Tracked","#f472b6"),
                              ("Mar–Apr 2016","Study Period","#34d399"),(features.shape[1],"TSFresh Features","#fb923c"),
                              ("3","Prophet Models","#a78bfa"),(optimal_k,"KMeans Clusters","#38bdf8"),
                              (n_clusters_db,"DBSCAN Clusters","#f472b6"),(n_noise,"Noise Points","#fb923c")])
                st.markdown("---")
                c1,c2=st.columns(2)
                with c1:
                    sum_box("📂 Dataset Overview",[("Source","Real Fitbit wearable device","c-blue"),
                        ("Unique users",f"{master['Id'].nunique()}","c-green"),("Date span","31 days · March–April 2016","c-blue"),
                        ("Master DF",f"{master.shape[0]:,} × {master.shape[1]} cols","c-blue"),("Final null values","0 ✅","c-green"),
                        ("HR granularity","1-minute intervals","c-orange"),("Sleep granularity","1-minute · 3 stages","c-orange")])
                    sum_box("🧪 TSFresh Feature Extraction",[("Mode",tsfresh_mode,"c-blue"),
                        ("Source signal","Heart rate (1-min)","c-blue"),("Users processed",f"{features.shape[0]}","c-green"),
                        ("Features extracted",f"{features.shape[1]}","c-green"),
                        ("Feature names",", ".join(features.columns),"c-purple"),("Output CSV","tsfresh_features.csv ✅","c-green")])
                    sum_box("📉 Dimensionality Reduction",[("PCA — PC1 variance",f"{var_explained[0]:.1f}%","c-blue"),
                        ("PCA — PC2 variance",f"{var_explained[1]:.1f}%","c-blue"),
                        ("PCA — Total",f"{sum(var_explained):.1f}%","c-green"),
                        ("t-SNE perplexity",f"{min(30,cluster_features.shape[0]-1)}","c-purple"),("t-SNE iterations","1000","c-purple")])
                with c2:
                    sum_box("📈 Prophet Forecasting",[("Models fitted","3  (Heart Rate · Steps · Sleep)","c-blue"),
                        ("Forecast horizon",f"{forecast_days} days","c-green"),("Confidence band","80% interval","c-purple"),
                        ("Weekly seasonality","Enabled on all 3 models","c-orange"),
                        ("HR changepoint scale",f"{changepoint_scale}","c-orange"),
                        ("HR forecast peak",f"{fc_hr['yhat'].max():.1f} bpm","c-pink"),
                        ("HR forecast floor",f"{fc_hr['yhat'].min():.1f} bpm","c-pink")])
                    sum_box("🤖 Clustering Results",[("Users clustered",f"{cluster_features.shape[0]}","c-blue"),
                        ("Features used","7 (activity + sleep)","c-blue"),("KMeans K",f"{optimal_k}","c-green"),
                        ("KMeans distribution","  |  ".join(f"C{i}:{int((np.array(kmeans_labels)==i).sum())}" for i in range(optimal_k)),"c-green"),
                        ("DBSCAN epsilon",f"{eps_val}","c-orange"),("DBSCAN min_samples",f"{int(min_samples)}","c-orange"),
                        ("DBSCAN clusters",f"{n_clusters_db}","c-green"),
                        ("DBSCAN noise",f"{n_noise} users ({n_noise/len(dbscan_labels)*100:.1f}%)","c-orange")])
                st.markdown("---")
                feat_cols=[c for c in cluster_features.columns if c not in ("KMeans_Cluster","DBSCAN_Cluster")]
                profile=cluster_features.groupby("KMeans_Cluster")[feat_cols].mean().round(2)
                cl_cols_=st.columns(optimal_k)
                CARD_THEMES=[("#38bdf8","#071829"),("#f472b6","#250a18"),("#34d399","#041f12"),
                             ("#fb923c","#1f0d04"),("#a78bfa","#130a2a")]
                for i,col in enumerate(cl_cols_):
                    if i not in profile.index: continue
                    row=profile.loc[i]
                    steps,act,sed,cal,slp=(row["TotalSteps"],row["VeryActiveMinutes"],row["SedentaryMinutes"],row["Calories"],row["TotalSleepMinutes"])
                    emoji,lbl=(("🏃","HIGHLY ACTIVE") if steps>10000 else ("🚶","MODERATELY ACTIVE") if steps>5000 else ("🛋️","SEDENTARY"))
                    fg,bg=CARD_THEMES[i%len(CARD_THEMES)]
                    col.markdown(
                        f'<div style="background:{bg};border:1.5px solid {fg};border-radius:12px;padding:16px;">'
                        f'<div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:800;color:{fg}">{emoji} Cluster {i}</div>'
                        f'<div style="font-family:JetBrains Mono,monospace;font-size:9px;letter-spacing:1.5px;color:{fg};margin-bottom:10px">{lbl}</div>'
                        f'<div style="font-size:12px;color:#94a3b8;line-height:2">'
                        f'👣 <b style="color:#e2e8f0">{steps:,.0f}</b> steps/day<br>🔥 <b style="color:#e2e8f0">{cal:,.0f}</b> kcal/day<br>'
                        f'⚡ <b style="color:#e2e8f0">{act:.0f} min</b> very active<br>🪑 <b style="color:#e2e8f0">{sed:.0f} min</b> sedentary<br>'
                        f'😴 <b style="color:#e2e8f0">{slp:.0f} min</b> sleep/night</div></div>',unsafe_allow_html=True)
                dist_str=" | ".join(f"C{i}:{int((np.array(kmeans_labels)==i).sum())}" for i in range(optimal_k))
                report=(f"{'='*60}\n   PATTERN EXTRACTING SUMMARY — REAL FITBIT DATA\n{'='*60}\n\n"
                        f"DATASET\n  Users:{master['Id'].nunique()}  Date span:31 days (Mar–Apr 2016)\n"
                        f"  Master DF:{master.shape[0]:,} rows × {master.shape[1]} cols  Nulls:0\n\n"
                        f"TSFRESH\n  Mode:{tsfresh_mode}  Users:{features.shape[0]}  Features:{features.shape[1]}\n"
                        f"  Names:{', '.join(features.columns)}\n\n"
                        f"PROPHET\n  Models:Heart Rate · Total Steps · Sleep Minutes\n"
                        f"  Horizon:{forecast_days} days  CI:80%  CP scale:{changepoint_scale}\n\n"
                        f"KMEANS\n  K:{optimal_k}  Distribution:{dist_str}\n\n"
                        f"DBSCAN\n  eps:{eps_val}  min_samples:{int(min_samples)}  Clusters:{n_clusters_db}  Noise:{n_noise} ({n_noise/len(dbscan_labels)*100:.1f}%)\n\n"
                        f"PCA\n  PC1:{var_explained[0]:.1f}%  PC2:{var_explained[1]:.1f}%  Total:{sum(var_explained):.1f}%\n{'='*60}\n")
                st.markdown("---")
                st.download_button("⬇️  Download Full Summary Report (.txt)",report,
                                   "pattern_extracting_summary.txt","text/plain")
                m2_mark("summary"); st.success("🎉 Pattern Extracting Pipeline Complete! All 7 steps done.")
            except Exception as e: st.error(f"❌ {e}\n\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════
# ROUTER  — decide which page to render
# ══════════════════════════════════════════════════════════════════
mod = st.session_state.active_module

if mod is None:
    render_launcher()
elif mod == "m1":
    render_m1()
elif mod == "m2":
    render_m2()

# ── Footer ──
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:1.5rem 0;border-top:1px solid rgba(0,230,255,.08);">
    <span style="font-family:JetBrains Mono,monospace;font-size:.62rem;letter-spacing:2px;color:#2a4560;">
        FITNESS DATA PRO &nbsp;·&nbsp; PRE PROCESSING + PATTERN EXTRACTING &nbsp;·&nbsp;
        BUILT WITH STREAMLIT + PLOTLY + TSFRESH + PROPHET + SKLEARN
    </span>
</div>""", unsafe_allow_html=True)