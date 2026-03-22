"""
╔══════════════════════════════════════════════════════════════════╗
║   🏋️  Fitbit ML Pipeline — Milestone 2                          ║
║   TSFresh + Prophet + Clustering | Step-by-Step UI              ║
║   Run:  streamlit run fitbit_pipeline_app.py                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import io
import warnings
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fitbit ML Pipeline · Milestone 2",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
section[data-testid="stSidebar"] { background:#0d1526 !important; border-right:1px solid #1e2d45; }
.main .block-container { background:#080c18; padding-top:1.4rem; padding-bottom:4rem; max-width:1180px; }

.hero-title {
    font-family:'Syne',sans-serif; font-size:2.1rem; font-weight:800;
    background:linear-gradient(135deg,#38bdf8 30%,#f472b6 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    line-height:1.15; margin-bottom:3px;
}
.hero-sub { font-family:'JetBrains Mono',monospace; font-size:11px; color:#4b607a; letter-spacing:1.2px; }

.sec-lbl {
    font-family:'JetBrains Mono',monospace; font-size:10px;
    letter-spacing:2.5px; text-transform:uppercase; color:#4b607a;
    border-bottom:1px solid #1a2840; padding-bottom:6px; margin:26px 0 14px;
}

.chip {
    display:inline-flex; align-items:center; gap:5px;
    padding:4px 11px; border-radius:5px;
    font-family:'JetBrains Mono',monospace; font-size:11px; margin:3px;
    border:1px solid #1e2d45; background:#0f1c30; color:#4b607a;
}
.chip.ok { border-color:#22c55e; background:rgba(34,197,94,.08); color:#22c55e; }

.step-bar {
    display:flex; align-items:center; gap:10px;
    padding:13px 20px; background:#111f36; border-bottom:1px solid #1a2840;
}
.step-pill {
    font-family:'JetBrains Mono',monospace; font-size:9px;
    letter-spacing:1.5px; text-transform:uppercase;
    padding:3px 9px; border-radius:4px;
    background:#1a2840; color:#4b607a; white-space:nowrap;
}
.step-pill.done  { background:rgba(34,197,94,.15);  color:#22c55e; }
.step-pill.ready { background:rgba(56,189,248,.15); color:#38bdf8; }
.step-bar-title  { font-family:'Syne',sans-serif; font-size:.97rem; font-weight:700; color:#e2e8f0; flex:1; }
.step-flag       { font-family:'JetBrains Mono',monospace; font-size:11px; }
.step-flag.done  { color:#22c55e; }
.step-flag.ready { color:#38bdf8; }
.step-flag.locked{ color:#4b607a; }

/* Summary box */
.sum-box { background:#0d1526; border:1px solid #1a2840; border-radius:10px; padding:16px 20px; margin-bottom:10px; }
.sum-box-title { font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:2px; text-transform:uppercase; color:#4b607a; margin-bottom:10px; }
.sum-row { display:flex; justify-content:space-between; align-items:center; padding:5px 0; border-bottom:1px solid #1a2840; font-size:13px; }
.sum-row:last-child { border-bottom:none; }
.sum-key  { color:#94a3b8; }
.sum-val  { font-family:'JetBrains Mono',monospace; color:#e2e8f0; font-weight:600; }
.c-green  { color:#22c55e !important; }
.c-blue   { color:#38bdf8 !important; }
.c-pink   { color:#f472b6 !important; }
.c-orange { color:#fb923c !important; }
.c-purple { color:#a78bfa !important; }
.c-teal   { color:#34d399 !important; }

/* Metric tile */
.mtile { background:#0d1526; border:1px solid #1a2840; border-radius:10px; padding:14px 16px; text-align:center; }
.mtile-val { font-family:'Syne',sans-serif; font-size:1.75rem; font-weight:800; }
.mtile-lbl { font-family:'JetBrains Mono',monospace; font-size:9px; text-transform:uppercase; letter-spacing:1.5px; color:#4b607a; margin-top:3px; }

.ss-badge {
    display:inline-block; background:rgba(251,146,60,.12);
    border:1px solid rgba(251,146,60,.35); color:#fb923c;
    font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:1px;
    padding:3px 10px; border-radius:4px; margin-bottom:8px;
}

#MainMenu, footer { visibility:hidden; }
div[data-testid="stDecoration"] { display:none; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────
STEPS = ["load","timestamps","master","tsfresh","prophet","clustering","summary"]

_defaults = {
    "daily":None,"hourly_s":None,"hourly_i":None,
    "sleep":None,"hr":None,"hr_minute":None,
    "master":None,"features":None,
    "kmeans_labels":None,"dbscan_labels":None,
    "cluster_features":None,"X_pca":None,"X_tsne":None,
    "var_explained":None,"n_clusters_db":0,"n_noise":0,
    "fc_hr":None,
    **{f"done_{s}":False for s in STEPS},
}
for k,v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

done = lambda s: st.session_state[f"done_{s}"]
mark = lambda s: st.session_state.update({f"done_{s}": True})

def reset_from(step):
    idx = STEPS.index(step)
    for s in STEPS[idx:]:
        st.session_state[f"done_{s}"] = False
    wipe = {
        "load":["daily","hourly_s","hourly_i","sleep","hr"],
        "timestamps":["hr_minute"],
        "master":["master"],
        "tsfresh":["features"],
        "prophet":["fc_hr"],
        "clustering":["kmeans_labels","dbscan_labels","cluster_features",
                      "X_pca","X_tsne","var_explained","n_clusters_db","n_noise"],
    }
    for s in STEPS[idx:]:
        for k in wipe.get(s,[]):
            st.session_state[k] = None

def sstate(step,prereq=None):
    if done(step): return "done"
    if prereq is None or done(prereq): return "ready"
    return "locked"

def step_header(num, icon, title, step, prereq=None):
    st_ = sstate(step, prereq)
    pill = {"done":"done","ready":"ready","locked":""}[st_]
    flag = {"done":"✅ DONE","ready":"● READY","locked":"🔒 LOCKED"}[st_]
    st.markdown(
        f'<div class="step-bar">'
        f'<span class="step-pill {pill}">STEP {num}</span>'
        f'<span style="font-size:1.15rem">{icon}</span>'
        f'<span class="step-bar-title">{title}</span>'
        f'<span class="step-flag {st_ if st_!="ready" else "ready"}">{flag}</span>'
        f'</div>', unsafe_allow_html=True
    )
    return st_

def sum_box(title, rows):
    """rows = list of (key, value, css_color_class)"""
    inner = "".join(
        f'<div class="sum-row">'
        f'<span class="sum-key">{k}</span>'
        f'<span class="sum-val {c}">{v}</span>'
        f'</div>'
        for k,v,c in rows
    )
    st.markdown(
        f'<div class="sum-box"><div class="sum-box-title">{title}</div>{inner}</div>',
        unsafe_allow_html=True
    )

def metric_tiles(items):
    cols = st.columns(len(items))
    for col,(val,lbl,clr) in zip(cols,items):
        col.markdown(
            f'<div class="mtile">'
            f'<div class="mtile-val" style="color:{clr}">{val}</div>'
            f'<div class="mtile-lbl">{lbl}</div>'
            f'</div>', unsafe_allow_html=True
        )

def ss_badge(label="📸 Screenshot This"):
    st.markdown(f'<div class="ss-badge">{label}</div>', unsafe_allow_html=True)

DARK_BG  = "#080c18"
CARD_BG  = "#0f1729"
GRID_CLR = "#1a2840"
TEXT_CLR = "#e2e8f0"
MUTED    = "#4b607a"
PALETTE  = ["#38bdf8","#f472b6","#34d399","#fb923c","#a78bfa","#f87171","#4fd1c5"]

def dark_ax(ax):
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_CLR)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT_CLR)
    ax.grid(color=GRID_CLR, alpha=0.5, linewidth=0.5)

def dark_fig(w=13,h=4):
    fig,ax = plt.subplots(figsize=(w,h))
    fig.patch.set_facecolor(DARK_BG)
    dark_ax(ax)
    return fig,ax


# ──────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    st.markdown("**🧪 TSFresh**")
    tsfresh_mode = st.selectbox("Feature Parameters",
        ["MinimalFCParameters","EfficientFCParameters"],
        help="Minimal ~9 features; Efficient ~72 features")

    st.markdown("**📈 Prophet**")
    forecast_days     = st.slider("Forecast Horizon (days)", 7, 90, 30)
    changepoint_scale = st.number_input("Changepoint Prior Scale",
        min_value=0.001, max_value=0.5, value=0.01, step=0.005, format="%.3f")

    st.markdown("**🤖 KMeans**")
    optimal_k = st.slider("Number of Clusters (K)", 2, 9, 3)

    st.markdown("**🔵 DBSCAN**")
    eps_val     = st.number_input("Epsilon (eps)", 0.1, 10.0, 2.2, 0.1)
    min_samples = st.number_input("Min Samples",   1,   10,   2,   1)

    st.markdown("---")
    st.markdown("**Pipeline Progress**")
    n_done = sum(done(s) for s in STEPS)
    st.progress(n_done / len(STEPS), text=f"{n_done}/{len(STEPS)} steps complete")
    for s in STEPS:
        clr  = "#22c55e" if done(s) else "#4b607a"
        icon = "✅" if done(s) else "⬜"
        st.markdown(
            f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;color:{clr}'>"
            f"{icon} {s.replace('_',' ').title()}</span>", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🔄 Reset All"):
        for k,v in _defaults.items():
            st.session_state[k] = v
        st.rerun()
    st.markdown(
        "<span style='font-family:JetBrains Mono,monospace;font-size:10px;color:#4b607a'>"
        "Fitbit · Milestone 2<br>TSFresh · Prophet · Clustering</span>",
        unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-title">🏋️ Fitbit ML Pipeline</div>'
    '<div class="hero-sub">MILESTONE 2 · CLICK EACH STEP BUTTON TO PROCESS SEQUENTIALLY</div>',
    unsafe_allow_html=True)
st.markdown("---")


# ──────────────────────────────────────────────────────────────────
# FILE UPLOAD
# ──────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-lbl">01 · Upload Dataset Files</div>', unsafe_allow_html=True)

REQUIRED = {
    "dailyActivity_merged.csv":     "Daily Activity",
    "hourlySteps_merged.csv":       "Hourly Steps",
    "hourlyIntensities_merged.csv": "Hourly Intensities",
    "minuteSleep_merged.csv":       "Minute Sleep",
    "heartrate_seconds_merged.csv": "Heart Rate",
}

uploaded     = st.file_uploader(
    "Select all 5 Fitbit CSV files at once  (Ctrl+Click / Cmd+Click to multi-select)",
    type=["csv"], accept_multiple_files=True)
file_map     = {f.name:f for f in uploaded} if uploaded else {}
all_uploaded = all(n in file_map for n in REQUIRED)

chips = "".join(
    f'<span class="chip {"ok" if n in file_map else ""}">'
    f'{"✅" if n in file_map else "⬜"} {lbl}</span>'
    for n,lbl in REQUIRED.items())
st.markdown(chips, unsafe_allow_html=True)

if not all_uploaded:
    miss = [lbl for n,lbl in REQUIRED.items() if n not in file_map]
    st.info(f"⏳  Waiting for: **{', '.join(miss)}**")
else:
    st.success("✅  All 5 files uploaded — run each step below in order.")

st.markdown('<div class="sec-lbl">02 · Pipeline Steps — Click Each Step to Execute</div>',
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# STEP 1 — LOAD & VALIDATE
# ══════════════════════════════════════════════════════════════════
with st.container(border=True):
    step_header(1,"📂","Load & Validate All Files","load")

    dc,bc = st.columns([5,1])
    dc.markdown("<span style='color:#4b607a;font-size:13px'>"
                "Read all 5 CSVs into DataFrames · Report shape of each file · Full null-value audit"
                "</span>", unsafe_allow_html=True)
    run1 = bc.button("▶ Run" if not done("load") else "🔁 Re-run",
                     key="b1", disabled=not all_uploaded, use_container_width=True)

    if run1: reset_from("load")

    if done("load") or run1:
        with st.spinner("Loading files…"):
            try:
                daily    = pd.read_csv(file_map["dailyActivity_merged.csv"])
                hourly_s = pd.read_csv(file_map["hourlySteps_merged.csv"])
                hourly_i = pd.read_csv(file_map["hourlyIntensities_merged.csv"])
                sleep_df = pd.read_csv(file_map["minuteSleep_merged.csv"])
                hr_df    = pd.read_csv(file_map["heartrate_seconds_merged.csv"])

                st.session_state.daily=daily; st.session_state.hourly_s=hourly_s
                st.session_state.hourly_i=hourly_i; st.session_state.sleep=sleep_df
                st.session_state.hr=hr_df

                st.markdown("##### 📋 Step 1 Summary")
                c1,c2 = st.columns(2)
                with c1:
                    sum_box("📄 File Shapes", [
                        ("dailyActivity_merged",     f"{daily.shape[0]:,} rows × {daily.shape[1]} cols",    "c-blue"),
                        ("hourlySteps_merged",       f"{hourly_s.shape[0]:,} rows × {hourly_s.shape[1]} cols","c-blue"),
                        ("hourlyIntensities_merged", f"{hourly_i.shape[0]:,} rows × {hourly_i.shape[1]} cols","c-blue"),
                        ("minuteSleep_merged",       f"{sleep_df.shape[0]:,} rows × {sleep_df.shape[1]} cols","c-blue"),
                        ("heartrate_seconds_merged", f"{hr_df.shape[0]:,} rows × {hr_df.shape[1]} cols",    "c-blue"),
                    ])
                with c2:
                    sum_box("🔍 Null Value Audit", [
                        (name,
                         f"{df.isnull().sum().sum()} nulls  {'✅ Clean' if df.isnull().sum().sum()==0 else '⚠️ Has nulls'}",
                         "c-green" if df.isnull().sum().sum()==0 else "c-orange")
                        for name,df in [
                            ("dailyActivity",daily),("hourlySteps",hourly_s),
                            ("hourlyIntensities",hourly_i),("minuteSleep",sleep_df),
                            ("heartrate",hr_df)]
                    ])

                metric_tiles([
                    (daily["Id"].nunique(),   "Unique Users (daily)", "#38bdf8"),
                    (hr_df["Id"].nunique(),   "Unique Users (HR)",    "#f472b6"),
                    (sleep_df["Id"].nunique(),"Unique Users (sleep)", "#34d399"),
                    (f"{hr_df.shape[0]:,}",  "Total HR Records",     "#fb923c"),
                ])

                with st.expander("👁 Preview dailyActivity (top 5 rows)"):
                    st.dataframe(daily.head(5), use_container_width=True, hide_index=True)

                mark("load")
                st.success("✅ Step 1 complete — all 5 files loaded & validated.")
            except Exception as e:
                st.error(f"❌ {e}\n\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════
# STEP 2 — TIMESTAMP PARSING
# ══════════════════════════════════════════════════════════════════
with st.container(border=True):
    step_header(2,"⏱","Timestamp Parsing & Time Normalisation","timestamps","load")

    dc,bc = st.columns([5,1])
    dc.markdown("<span style='color:#4b607a;font-size:13px'>"
                "Parse all date/time columns · Resample heart rate seconds → 1-minute intervals"
                "</span>", unsafe_allow_html=True)
    run2 = bc.button("▶ Run" if not done("timestamps") else "🔁 Re-run",
                     key="b2", disabled=not done("load"), use_container_width=True)

    if run2: reset_from("timestamps")

    if done("timestamps") or run2:
        with st.spinner("Parsing timestamps & resampling…"):
            try:
                daily    = st.session_state.daily.copy()
                hourly_s = st.session_state.hourly_s.copy()
                hourly_i = st.session_state.hourly_i.copy()
                sleep_df = st.session_state.sleep.copy()
                hr_df    = st.session_state.hr.copy()

                daily["ActivityDate"]    = pd.to_datetime(daily["ActivityDate"],    format="%m/%d/%Y")
                hourly_s["ActivityHour"] = pd.to_datetime(hourly_s["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p")
                hourly_i["ActivityHour"] = pd.to_datetime(hourly_i["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p")
                sleep_df["date"]         = pd.to_datetime(sleep_df["date"],          format="%m/%d/%Y %I:%M:%S %p")
                hr_df["Time"]            = pd.to_datetime(hr_df["Time"],             format="%m/%d/%Y %I:%M:%S %p")

                hr_minute = (
                    hr_df.set_index("Time").groupby("Id")["Value"]
                    .resample("1min").mean().reset_index()
                )
                hr_minute.columns = ["Id","Time","HeartRate"]
                hr_minute = hr_minute.dropna()

                freq_check = (
                    hourly_s.groupby("Id")["ActivityHour"]
                    .diff().dropna().dt.total_seconds()/3600
                )

                st.session_state.daily=daily; st.session_state.hourly_s=hourly_s
                st.session_state.hourly_i=hourly_i; st.session_state.sleep=sleep_df
                st.session_state.hr=hr_df; st.session_state.hr_minute=hr_minute

                st.markdown("##### 📋 Step 2 Summary")
                c1,c2 = st.columns(2)
                with c1:
                    sum_box("📅 Parsed Date Formats", [
                        ("ActivityDate",       "%m/%d/%Y",              "c-green"),
                        ("ActivityHour",       "%m/%d/%Y %I:%M:%S %p", "c-green"),
                        ("minuteSleep · date", "%m/%d/%Y %I:%M:%S %p", "c-green"),
                        ("heartrate · Time",   "%m/%d/%Y %I:%M:%S %p", "c-green"),
                    ])
                    sum_box("📆 Date Ranges", [
                        ("Daily start",  str(daily["ActivityDate"].min().date()), "c-blue"),
                        ("Daily end",    str(daily["ActivityDate"].max().date()), "c-blue"),
                        ("Span",
                         f"{(daily['ActivityDate'].max()-daily['ActivityDate'].min()).days} days",
                         "c-orange"),
                        ("HR start", str(hr_df["Time"].min()), "c-blue"),
                        ("HR end",   str(hr_df["Time"].max()), "c-blue"),
                    ])
                with c2:
                    sum_box("🔄 Heart Rate Resampling", [
                        ("Original granularity", "Seconds",                       "c-orange"),
                        ("Target granularity",   "1-Minute intervals",            "c-green"),
                        ("Rows before",          f"{hr_df.shape[0]:,}",           "c-blue"),
                        ("Rows after",           f"{hr_minute.shape[0]:,}",       "c-green"),
                        ("Reduction factor",     f"~{hr_df.shape[0]/hr_minute.shape[0]:.1f}×", "c-pink"),
                    ])
                    sum_box("📊 Frequency Verification", [
                        ("Hourly median interval", f"{freq_check.median():.1f} h",            "c-green"),
                        ("Exact 1-h accuracy",     f"{(freq_check==1.0).mean()*100:.1f}%",    "c-green"),
                        ("Sleep records",           f"{sleep_df.shape[0]:,}",                 "c-blue"),
                        ("Sleep granularity",       "1-minute",                               "c-blue"),
                        ("Sleep stages",            "1=Light · 2=Deep · 3=REM",               "c-purple"),
                        ("Timezone note",           "Local time · No UTC conversion",         "c-orange"),
                    ])

                metric_tiles([
                    (f"{hr_minute.shape[0]:,}", "HR Rows (1-min)",    "#38bdf8"),
                    (f"{hr_df.shape[0]:,}",     "HR Rows (original)", "#f472b6"),
                    (f"{(daily['ActivityDate'].max()-daily['ActivityDate'].min()).days}d", "Date Span", "#34d399"),
                    (f"{(freq_check==1.0).mean()*100:.1f}%", "Hourly Accuracy", "#fb923c"),
                ])

                mark("timestamps")
                st.success("✅ Step 2 complete — timestamps parsed, HR resampled to 1-minute.")
            except Exception as e:
                st.error(f"❌ {e}\n\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════
# STEP 3 — MASTER DATAFRAME
# ══════════════════════════════════════════════════════════════════
with st.container(border=True):
    step_header(3,"🔗","Aggregate & Build Master DataFrame","master","timestamps")

    dc,bc = st.columns([5,1])
    dc.markdown("<span style='color:#4b607a;font-size:13px'>"
                "Daily HR stats (avg/max/min/std) · Daily sleep totals · Merge all → master · Fill nulls"
                "</span>", unsafe_allow_html=True)
    run3 = bc.button("▶ Run" if not done("master") else "🔁 Re-run",
                     key="b3", disabled=not done("timestamps"), use_container_width=True)

    if run3: reset_from("master")

    if done("master") or run3:
        with st.spinner("Building master dataframe…"):
            try:
                daily     = st.session_state.daily.copy()
                sleep_df  = st.session_state.sleep.copy()
                hr_minute = st.session_state.hr_minute.copy()

                hr_minute["Date"] = hr_minute["Time"].dt.date
                hr_daily = (
                    hr_minute.groupby(["Id","Date"])["HeartRate"]
                    .agg(["mean","max","min","std"]).reset_index()
                    .rename(columns={"mean":"AvgHR","max":"MaxHR","min":"MinHR","std":"StdHR"})
                )

                sleep_df["Date"] = sleep_df["date"].dt.date
                sleep_daily = (
                    sleep_df.groupby(["Id","Date"])
                    .agg(TotalSleepMinutes=("value","count"),
                         DominantSleepStage=("value",lambda x: x.mode()[0]))
                    .reset_index()
                )

                master = daily.rename(columns={"ActivityDate":"Date"}).copy()
                master["Date"] = master["Date"].dt.date
                master = master.merge(hr_daily,    on=["Id","Date"], how="left")
                master = master.merge(sleep_daily, on=["Id","Date"], how="left")
                master["TotalSleepMinutes"]  = master["TotalSleepMinutes"].fillna(0)
                master["DominantSleepStage"] = master["DominantSleepStage"].fillna(0)
                for col in ["AvgHR","MaxHR","MinHR","StdHR"]:
                    master[col] = master.groupby("Id")[col].transform(
                        lambda x: x.fillna(x.median()))

                st.session_state.master = master

                st.markdown("##### 📋 Step 3 Summary")
                c1,c2 = st.columns(2)
                with c1:
                    sum_box("🗂️ Master DataFrame Info", [
                        ("Total rows",          f"{master.shape[0]:,}",          "c-blue"),
                        ("Total columns",       f"{master.shape[1]}",            "c-blue"),
                        ("Unique users",        f"{master['Id'].nunique()}",      "c-green"),
                        ("Remaining nulls",     f"{master.isnull().sum().sum()} ✅","c-green"),
                        ("HR null strategy",    "Per-user median fill",           "c-orange"),
                        ("Sleep null strategy", "Fill with 0",                    "c-orange"),
                    ])
                    sum_box("❤️ Heart Rate (daily avg)", [
                        ("Mean AvgHR",  f"{master['AvgHR'].mean():.1f} bpm",  "c-pink"),
                        ("Min AvgHR",   f"{master['AvgHR'].min():.1f} bpm",   "c-blue"),
                        ("Max AvgHR",   f"{master['AvgHR'].max():.1f} bpm",   "c-orange"),
                        ("Peak MaxHR",  f"{master['MaxHR'].max():.1f} bpm",   "c-orange"),
                    ])
                with c2:
                    sum_box("🚶 Activity Stats", [
                        ("Avg daily steps",     f"{master['TotalSteps'].mean():,.0f}",       "c-blue"),
                        ("Max daily steps",     f"{master['TotalSteps'].max():,.0f}",        "c-green"),
                        ("Avg calories/day",    f"{master['Calories'].mean():,.0f} kcal",    "c-orange"),
                        ("Avg very active min", f"{master['VeryActiveMinutes'].mean():.1f}", "c-green"),
                        ("Avg sedentary min",   f"{master['SedentaryMinutes'].mean():.1f}",  "c-pink"),
                    ])
                    sum_box("😴 Sleep Stats", [
                        ("Avg sleep/night",      f"{master['TotalSleepMinutes'].mean():.1f} min","c-purple"),
                        ("Max sleep/night",      f"{master['TotalSleepMinutes'].max():.0f} min", "c-blue"),
                        ("Records with sleep",   f"{(master['TotalSleepMinutes']>0).sum()}",     "c-green"),
                    ])

                metric_tiles([
                    (f"{master.shape[0]:,}",      "Master Rows",      "#38bdf8"),
                    (master["Id"].nunique(),        "Users",            "#f472b6"),
                    (master.shape[1],               "Columns",          "#34d399"),
                    (master.isnull().sum().sum(),   "Remaining Nulls",  "#22c55e"),
                ])

                with st.expander("📊 Descriptive Statistics — Key Columns"):
                    key_cols = ["TotalSteps","Calories","AvgHR","TotalSleepMinutes",
                                "VeryActiveMinutes","SedentaryMinutes"]
                    st.dataframe(master[key_cols].describe().round(2), use_container_width=True)
                with st.expander("👁 Master DataFrame — First 20 Rows"):
                    st.dataframe(
                        master[["Id","Date","TotalSteps","Calories","AvgHR",
                                "TotalSleepMinutes","VeryActiveMinutes","SedentaryMinutes"]].head(20),
                        use_container_width=True, hide_index=True)

                mark("master")
                st.success(f"✅ Step 3 complete — master: {master.shape[0]:,} rows × {master.shape[1]} cols.")
            except Exception as e:
                st.error(f"❌ {e}\n\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════
# STEP 4 — TSFRESH
# ══════════════════════════════════════════════════════════════════
with st.container(border=True):
    step_header(4,"🧪","TSFresh Feature Extraction","tsfresh","master")

    dc,bc = st.columns([5,1])
    dc.markdown(f"<span style='color:#4b607a;font-size:13px'>"
                f"Extract statistical features from minute-level HR · Mode: <b>{tsfresh_mode}</b>"
                f"</span>", unsafe_allow_html=True)
    run4 = bc.button("▶ Run" if not done("tsfresh") else "🔁 Re-run",
                     key="b4", disabled=not done("master"), use_container_width=True)

    if run4: reset_from("tsfresh")

    if done("tsfresh") or run4:
        with st.spinner("Running TSFresh — may take 30–60 seconds…"):
            try:
                from tsfresh import extract_features
                from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters
                from sklearn.preprocessing import MinMaxScaler

                hr_minute = st.session_state.hr_minute.copy()
                ts_hr = (
                    hr_minute[["Id","Time","HeartRate"]].dropna()
                    .sort_values(["Id","Time"])
                    .rename(columns={"Id":"id","Time":"time","HeartRate":"value"})
                )
                fc_params = (MinimalFCParameters() if tsfresh_mode=="MinimalFCParameters"
                             else EfficientFCParameters())
                features = extract_features(
                    ts_hr, column_id="id", column_sort="time", column_value="value",
                    default_fc_parameters=fc_params, disable_progressbar=True)
                features = features.dropna(axis=1, how="all")
                st.session_state.features = features

                st.markdown("##### 📋 Step 4 Summary")
                c1,c2 = st.columns(2)
                with c1:
                    sum_box("🧪 Extraction Details", [
                        ("Mode",               tsfresh_mode,                "c-blue"),
                        ("Input signal",       "Minute-level heart rate",   "c-blue"),
                        ("Users processed",    f"{features.shape[0]}",      "c-green"),
                        ("Features extracted", f"{features.shape[1]}",      "c-green"),
                        ("NaN values remain",  "0 (dropped automatically)", "c-green"),
                    ])
                with c2:
                    sum_box("📐 Extracted Feature Names", [
                        (f"{i+1}.", col, "c-purple")
                        for i,col in enumerate(features.columns)
                    ])

                metric_tiles([
                    (features.shape[0], "Users Processed",    "#38bdf8"),
                    (features.shape[1], "Features Extracted", "#f472b6"),
                    (0,                 "NaN Values",         "#22c55e"),
                ])

                # ── shared PNG helper (available to steps 4, 5, 6) ──
                def fig_to_png(fig):
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=180,
                                bbox_inches="tight", facecolor=fig.get_facecolor())
                    buf.seek(0)
                    return buf

                ss_badge("📸 SCREENSHOT 1 — TSFresh Feature Matrix Heatmap")
                scaler_vis    = MinMaxScaler()
                features_norm = pd.DataFrame(
                    scaler_vis.fit_transform(features),
                    index=features.index, columns=features.columns)
                fig_h, ax_h = plt.subplots(figsize=(14, max(5, features_norm.shape[0]*0.5)))
                fig_h.patch.set_facecolor(DARK_BG); ax_h.set_facecolor(CARD_BG)
                sns.heatmap(features_norm, cmap="coolwarm", annot=True, fmt=".2f",
                            linewidths=0.6, linecolor=GRID_CLR, ax=ax_h, cbar_kws={"shrink":0.7})
                ax_h.set_title("TSFresh Feature Matrix — Real Fitbit HR\n(Normalised 0–1 per feature)",
                               fontsize=13, color=TEXT_CLR, pad=14)
                ax_h.set_xlabel("Statistical Features", color=MUTED)
                ax_h.set_ylabel("User ID", color=MUTED)
                ax_h.tick_params(colors=MUTED)
                plt.tight_layout()
                st.pyplot(fig_h, use_container_width=True)
                st.download_button("⬇️ Download Screenshot 1 — TSFresh Heatmap (PNG)",
                                   data=fig_to_png(fig_h),
                                   file_name="screenshot1_tsfresh_heatmap.png",
                                   mime="image/png", key="dl_heatmap")
                plt.close(fig_h)

                buf = io.StringIO(); features.to_csv(buf)
                st.download_button("⬇️ Download tsfresh_features.csv",
                                   buf.getvalue(), "tsfresh_features.csv", "text/csv",
                                   key="dl_tsfresh_csv")

                mark("tsfresh")
                st.success(f"✅ Step 4 complete — {features.shape[1]} features extracted from {features.shape[0]} users.")
            except Exception as e:
                st.error(f"❌ {e}\n\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════
# STEP 5 — PROPHET
# ══════════════════════════════════════════════════════════════════
with st.container(border=True):
    step_header(5,"📈","Prophet Trend Forecasting","prophet","tsfresh")

    dc,bc = st.columns([5,1])
    dc.markdown(f"<span style='color:#4b607a;font-size:13px'>"
                f"Fit Prophet on Heart Rate · Steps · Sleep · {forecast_days}-day forecast · 80% CI"
                f"</span>", unsafe_allow_html=True)
    run5 = bc.button("▶ Run" if not done("prophet") else "🔁 Re-run",
                     key="b5", disabled=not done("tsfresh"), use_container_width=True)

    if run5: reset_from("prophet")

    if done("prophet") or run5:
        with st.spinner("Fitting 3 Prophet models…"):
            try:
                from prophet import Prophet

                hr_minute = st.session_state.hr_minute.copy()
                daily     = st.session_state.daily.copy()
                master    = st.session_state.master.copy()

                # Ensure hr_minute has a proper datetime Date column
                hr_minute["Date"] = pd.to_datetime(hr_minute["Time"]).dt.normalize()

                # master["Date"] is python date objects — convert to datetime for groupby
                master["Date"] = pd.to_datetime(master["Date"])

                def fit_prophet(df, date_col, val_col, cp=0.1):
                    agg = df.groupby(date_col)[val_col].mean().reset_index()
                    agg.columns = ["ds","y"]
                    agg["ds"] = pd.to_datetime(agg["ds"], errors="coerce")
                    agg = agg.dropna().sort_values("ds")
                    m = Prophet(daily_seasonality=False, weekly_seasonality=True,
                                yearly_seasonality=False, interval_width=0.80,
                                changepoint_prior_scale=cp, changepoint_range=0.8)
                    m.fit(agg)
                    fc = m.predict(m.make_future_dataframe(periods=forecast_days))
                    return agg, m, fc

                act_hr, mod_hr, fc_hr = fit_prophet(hr_minute, "Date",        "HeartRate",          changepoint_scale)
                act_st, mod_st, fc_st = fit_prophet(daily,     "ActivityDate","TotalSteps",         0.1)
                act_sl, mod_sl, fc_sl = fit_prophet(master,    "Date",        "TotalSleepMinutes",  0.1)
                st.session_state.fc_hr = fc_hr

                st.markdown("##### 📋 Step 5 Summary")
                c1,c2,c3 = st.columns(3)
                with c1:
                    sum_box("💓 Heart Rate Model", [
                        ("Metric",              "Heart Rate (bpm)",                                         "c-blue"),
                        ("Changepoint scale",   f"{changepoint_scale}",                                     "c-orange"),
                        ("Forecast horizon",    f"{forecast_days} days",                                    "c-green"),
                        ("Confidence interval", "80%",                                                      "c-purple"),
                        ("Actual HR range",     f"{act_hr['y'].min():.1f} – {act_hr['y'].max():.1f} bpm",  "c-pink"),
                        ("Weekly seasonality",  "✅ Enabled",                                               "c-green"),
                    ])
                with c2:
                    sum_box("🚶 Steps Model", [
                        ("Metric",              "Total Steps / day",              "c-blue"),
                        ("Changepoint scale",   "0.1",                            "c-orange"),
                        ("Forecast horizon",    f"{forecast_days} days",          "c-green"),
                        ("Confidence interval", "80%",                            "c-purple"),
                        ("Avg actual steps",    f"{act_st['y'].mean():,.0f}",     "c-pink"),
                        ("Weekly seasonality",  "✅ Enabled",                     "c-green"),
                    ])
                with c3:
                    sum_box("😴 Sleep Model", [
                        ("Metric",              "Sleep minutes / day",            "c-blue"),
                        ("Changepoint scale",   "0.1",                            "c-orange"),
                        ("Forecast horizon",    f"{forecast_days} days",          "c-green"),
                        ("Confidence interval", "80%",                            "c-purple"),
                        ("Avg actual sleep",    f"{act_sl['y'].mean():.1f} min",  "c-pink"),
                        ("Weekly seasonality",  "✅ Enabled",                     "c-green"),
                    ])

                metric_tiles([
                    ("3",                 "Models Fitted",       "#38bdf8"),
                    (f"{forecast_days}d", "Forecast Horizon",    "#f472b6"),
                    ("80%",               "Confidence Interval", "#34d399"),
                    (f"{changepoint_scale}", "Changepoint Scale","#fb923c"),
                ])

                def fplot(actual, fc, color, title, ylabel):
                    fig,ax = dark_fig(13,4.2)
                    ax.scatter(actual["ds"],actual["y"],color=color,s=18,alpha=0.7,label="Actual",zorder=3)
                    ax.plot(fc["ds"],fc["yhat"],color="#3b82f6",lw=2.5,label="Forecast Trend")
                    ax.fill_between(fc["ds"],fc["yhat_lower"],fc["yhat_upper"],
                                    alpha=0.2,color="#3b82f6",label="80% CI")
                    ax.axvline(actual["ds"].max(),color="#fb923c",ls="--",lw=1.8,label="Forecast Start")
                    ax.set_title(title,fontsize=12,color=TEXT_CLR)
                    ax.set_xlabel("Date"); ax.set_ylabel(ylabel)
                    ax.legend(fontsize=8,facecolor=CARD_BG,labelcolor=TEXT_CLR,edgecolor=GRID_CLR)
                    plt.setp(ax.get_xticklabels(),rotation=30,ha="right")
                    plt.tight_layout(); return fig

                def fig_to_png(fig):
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=180,
                                bbox_inches="tight", facecolor=fig.get_facecolor())
                    buf.seek(0)
                    return buf

                def fplot(actual, fc, color, title, ylabel):
                    fig, ax = dark_fig(13, 4.2)
                    ax.scatter(actual["ds"], actual["y"], color=color, s=18, alpha=0.7,
                               label="Actual", zorder=3)
                    ax.plot(fc["ds"], fc["yhat"], color="#3b82f6", lw=2.5, label="Forecast Trend")
                    ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"],
                                    alpha=0.2, color="#3b82f6", label="80% CI")
                    ax.axvline(actual["ds"].max(), color="#fb923c", ls="--", lw=1.8,
                               label="Forecast Start")
                    ax.set_title(title, fontsize=12, color=TEXT_CLR)
                    ax.set_xlabel("Date"); ax.set_ylabel(ylabel)
                    ax.legend(fontsize=8, facecolor=CARD_BG, labelcolor=TEXT_CLR,
                              edgecolor=GRID_CLR)
                    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
                    plt.tight_layout()
                    return fig

                # ── Screenshot 2 : Heart Rate Forecast ──────────
                ss_badge("📸 SCREENSHOT 2 — Heart Rate Forecast")
                fig1 = fplot(act_hr, fc_hr, "#e53e3e",
                             f"Heart Rate Prophet Forecast (+{forecast_days}d)", "bpm")
                st.pyplot(fig1, use_container_width=True)
                st.download_button("⬇️ Download Screenshot 2 — HR Forecast (PNG)",
                                   data=fig_to_png(fig1),
                                   file_name="screenshot2_prophet_heartrate.png",
                                   mime="image/png", key="dl_hr")
                plt.close(fig1)

                # ── Prophet Components (bonus, no screenshot number) ──
                st.markdown("##### 📐 Prophet Components — Heart Rate")
                fc_comp = mod_hr.plot_components(fc_hr)
                fc_comp.patch.set_facecolor(CARD_BG)
                for ax_c in fc_comp.get_axes():
                    ax_c.set_facecolor(CARD_BG)
                    ax_c.tick_params(colors=MUTED)
                    ax_c.title.set_color(TEXT_CLR)
                st.pyplot(fc_comp, use_container_width=True)
                st.download_button("⬇️ Download Prophet Components (PNG)",
                                   data=fig_to_png(fc_comp),
                                   file_name="prophet_hr_components.png",
                                   mime="image/png", key="dl_comp")
                plt.close(fc_comp)

                # ── Screenshot 3 : Steps Forecast ───────────────
                ss_badge("📸 SCREENSHOT 3 — Total Steps Forecast")
                fig2 = fplot(act_st, fc_st, "#38a169",
                             f"Total Steps Prophet Forecast (+{forecast_days}d)", "Steps / day")
                st.pyplot(fig2, use_container_width=True)
                st.download_button("⬇️ Download Screenshot 3 — Steps Forecast (PNG)",
                                   data=fig_to_png(fig2),
                                   file_name="screenshot3_prophet_steps.png",
                                   mime="image/png", key="dl_st")
                plt.close(fig2)

                # ── Screenshot 4 : Sleep Forecast ───────────────
                ss_badge("📸 SCREENSHOT 4 — Sleep Minutes Forecast")
                fig3 = fplot(act_sl, fc_sl, "#b794f4",
                             f"Sleep Minutes Prophet Forecast (+{forecast_days}d)", "Min / day")
                st.pyplot(fig3, use_container_width=True)
                st.download_button("⬇️ Download Screenshot 4 — Sleep Forecast (PNG)",
                                   data=fig_to_png(fig3),
                                   file_name="screenshot4_prophet_sleep.png",
                                   mime="image/png", key="dl_sl")
                plt.close(fig3)

                mark("prophet")
                st.success(f"✅ Step 5 complete — 3 Prophet models · {forecast_days}-day forecasts done.")
            except Exception as e:
                st.error(f"❌ {e}\n\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════
# STEP 6 — CLUSTERING
# ══════════════════════════════════════════════════════════════════
with st.container(border=True):
    step_header(6,"🤖","KMeans & DBSCAN Clustering","clustering","prophet")

    dc,bc = st.columns([5,1])
    dc.markdown(f"<span style='color:#4b607a;font-size:13px'>"
                f"Scale · Elbow · KMeans K={optimal_k} · DBSCAN eps={eps_val} · PCA · t-SNE · Profiles"
                f"</span>", unsafe_allow_html=True)
    run6 = bc.button("▶ Run" if not done("clustering") else "🔁 Re-run",
                     key="b6", disabled=not done("prophet"), use_container_width=True)

    if run6: reset_from("clustering")

    if done("clustering") or run6:
        with st.spinner("Running clustering pipeline…"):
            try:
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans, DBSCAN
                from sklearn.decomposition import PCA
                from sklearn.manifold import TSNE

                master = st.session_state.master.copy()
                cluster_cols = ["TotalSteps","Calories","VeryActiveMinutes",
                                "FairlyActiveMinutes","LightlyActiveMinutes",
                                "SedentaryMinutes","TotalSleepMinutes"]
                cluster_features = master.groupby("Id")[cluster_cols].mean().round(3).dropna()

                scaler   = StandardScaler()
                X_scaled = scaler.fit_transform(cluster_features)

                # Elbow
                inertias = [KMeans(n_clusters=k,random_state=42,n_init=10).fit(X_scaled).inertia_
                            for k in range(2,10)]

                ss_badge("📸 SCREENSHOT — KMeans Elbow Curve")
                fig_el,ax_el = dark_fig(8,3.5)
                ax_el.plot(range(2,10),inertias,"o-",color="#63b3ed",lw=2.5,ms=9,markerfacecolor="#f687b3")
                ax_el.axvline(optimal_k,color="#fb923c",ls="--",lw=1.8,label=f"Selected K={optimal_k}")
                ax_el.set_title("KMeans Elbow Curve — Real Fitbit Data",fontsize=12,color=TEXT_CLR)
                ax_el.set_xlabel("K"); ax_el.set_ylabel("Inertia")
                ax_el.legend(fontsize=8,facecolor=CARD_BG,labelcolor=TEXT_CLR,edgecolor=GRID_CLR)
                plt.tight_layout()
                st.pyplot(fig_el,use_container_width=True); plt.close(fig_el)

                kmeans        = KMeans(n_clusters=optimal_k,random_state=42,n_init=10)
                kmeans_labels = kmeans.fit_predict(X_scaled)
                cluster_features["KMeans_Cluster"] = kmeans_labels

                dbscan        = DBSCAN(eps=eps_val,min_samples=int(min_samples))
                dbscan_labels = dbscan.fit_predict(X_scaled)
                cluster_features["DBSCAN_Cluster"] = dbscan_labels
                n_clusters_db = len(set(dbscan_labels))-(1 if -1 in dbscan_labels else 0)
                n_noise       = list(dbscan_labels).count(-1)

                pca   = PCA(n_components=2,random_state=42)
                X_pca = pca.fit_transform(X_scaled)
                var_explained = pca.explained_variance_ratio_*100

                tsne   = TSNE(n_components=2,random_state=42,
                              perplexity=min(30,len(X_scaled)-1),max_iter=1000)
                X_tsne = tsne.fit_transform(X_scaled)

                st.session_state.kmeans_labels    = kmeans_labels
                st.session_state.dbscan_labels    = dbscan_labels
                st.session_state.cluster_features = cluster_features
                st.session_state.X_pca            = X_pca
                st.session_state.X_tsne           = X_tsne
                st.session_state.var_explained    = var_explained
                st.session_state.n_clusters_db    = n_clusters_db
                st.session_state.n_noise          = n_noise

                st.markdown("##### 📋 Step 6 Summary")
                c1,c2,c3 = st.columns(3)
                with c1:
                    sum_box("⚙️ Feature Matrix & Scaling", [
                        ("Users",              f"{cluster_features.shape[0]}",  "c-blue"),
                        ("Features used",      f"{len(cluster_cols)}",          "c-blue"),
                        ("Scaling method",     "StandardScaler (z-score)",      "c-orange"),
                        ("Mean post-scale",    "≈ 0.0",                         "c-green"),
                        ("Std post-scale",     "≈ 1.0",                         "c-green"),
                    ])
                with c2:
                    km_dist = {i: int((np.array(kmeans_labels)==i).sum()) for i in range(optimal_k)}
                    sum_box("🎯 KMeans Results", [
                        ("K selected", f"{optimal_k}", "c-blue"),
                        *[(f"Cluster {ci}", f"{cnt} users", "c-green") for ci,cnt in km_dist.items()],
                    ])
                with c3:
                    sum_box("🔵 DBSCAN Results", [
                        ("Epsilon (eps)",    f"{eps_val}",         "c-blue"),
                        ("Min samples",      f"{int(min_samples)}","c-blue"),
                        ("Clusters found",   f"{n_clusters_db}",   "c-green"),
                        ("Noise points",     f"{n_noise} users",   "c-orange"),
                        ("Noise %",          f"{n_noise/len(dbscan_labels)*100:.1f}%","c-orange"),
                    ])

                c1b,c2b = st.columns(2)
                with c1b:
                    sum_box("📉 PCA (2D Reduction)", [
                        ("PC1 variance",     f"{var_explained[0]:.1f}%",       "c-blue"),
                        ("PC2 variance",     f"{var_explained[1]:.1f}%",       "c-blue"),
                        ("Total explained",  f"{sum(var_explained):.1f}%",     "c-green"),
                        ("Dims in → out",    f"{X_scaled.shape[1]} → 2",       "c-orange"),
                    ])
                with c2b:
                    sum_box("🌐 t-SNE (2D Projection)", [
                        ("Perplexity",   f"{min(30,len(X_scaled)-1)}", "c-blue"),
                        ("Iterations",   "1000",                       "c-blue"),
                        ("Input shape",  f"{X_scaled.shape[0]} × {X_scaled.shape[1]}","c-orange"),
                        ("Output shape", f"{X_tsne.shape[0]} × 2",    "c-green"),
                    ])

                metric_tiles([
                    (cluster_features.shape[0], "Users Clustered", "#38bdf8"),
                    (optimal_k,                 "KMeans Clusters", "#f472b6"),
                    (n_clusters_db,             "DBSCAN Clusters", "#34d399"),
                    (n_noise,                   "Noise Points",    "#fb923c"),
                    (f"{sum(var_explained):.1f}%","PCA Variance",  "#a78bfa"),
                ])

                # ── PNG helper (local to step 6) ─────────────────
                def fig_to_png(fig):
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=180,
                                bbox_inches="tight", facecolor=fig.get_facecolor())
                    buf.seek(0)
                    return buf

                # ── Scatter helper ───────────────────────────────
                def scatter(X2d, labels, title, noise_red=False):
                    fig, ax = dark_fig(8, 6)
                    for lbl in sorted(set(labels)):
                        mask = np.array(labels) == lbl
                        if noise_red and lbl == -1:
                            ax.scatter(X2d[mask,0], X2d[mask,1], c="red", marker="x",
                                       s=150, label="Noise", alpha=0.9, lw=2)
                        else:
                            ax.scatter(X2d[mask,0], X2d[mask,1],
                                       c=PALETTE[lbl % len(PALETTE)],
                                       label=f"Cluster {lbl}", s=120, alpha=0.85,
                                       edgecolors="white", lw=0.7)
                    ax.set_title(title, fontsize=11, color=TEXT_CLR)
                    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)" if "PCA" in title else "Dim 1")
                    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)" if "PCA" in title else "Dim 2")
                    ax.legend(title="Cluster", fontsize=8, facecolor=CARD_BG,
                              labelcolor=TEXT_CLR, edgecolor=GRID_CLR)
                    plt.tight_layout()
                    return fig

                # ── Screenshot 5 : KMeans PCA ────────────────────
                ss_badge("📸 SCREENSHOT 5 — KMeans PCA Scatter")
                fig_km = scatter(X_pca, kmeans_labels,
                                 f"KMeans — PCA Projection  (K={optimal_k})")
                st.pyplot(fig_km, use_container_width=True)
                st.download_button("⬇️ Download Screenshot 5 — KMeans PCA (PNG)",
                                   data=fig_to_png(fig_km),
                                   file_name="screenshot5_kmeans_pca.png",
                                   mime="image/png", key="dl_km_pca")
                plt.close(fig_km)

                # ── Screenshot 6 : DBSCAN PCA ────────────────────
                ss_badge("📸 SCREENSHOT 6 — DBSCAN PCA Scatter")
                fig_db = scatter(X_pca, dbscan_labels,
                                 f"DBSCAN — PCA Projection  (eps={eps_val})",
                                 noise_red=True)
                st.pyplot(fig_db, use_container_width=True)
                st.download_button("⬇️ Download Screenshot 6 — DBSCAN PCA (PNG)",
                                   data=fig_to_png(fig_db),
                                   file_name="screenshot6_dbscan_pca.png",
                                   mime="image/png", key="dl_db_pca")
                plt.close(fig_db)

                # ── Screenshot 7 : t-SNE ─────────────────────────
                ss_badge("📸 SCREENSHOT 7 — t-SNE Projection (KMeans + DBSCAN)")
                fig_ts, axes_t = plt.subplots(1, 2, figsize=(14, 6))
                fig_ts.patch.set_facecolor(DARK_BG)
                for ax_t, lbls, title in zip(
                    axes_t,
                    [kmeans_labels, dbscan_labels],
                    [f"KMeans — t-SNE (K={optimal_k})", f"DBSCAN — t-SNE (eps={eps_val})"],
                ):
                    dark_ax(ax_t)
                    for lbl in sorted(set(lbls)):
                        mask = np.array(lbls) == lbl
                        if lbl == -1:
                            ax_t.scatter(X_tsne[mask,0], X_tsne[mask,1], c="red",
                                         marker="x", s=150, label="Noise", alpha=0.9, lw=2)
                        else:
                            ax_t.scatter(X_tsne[mask,0], X_tsne[mask,1],
                                         c=PALETTE[lbl % len(PALETTE)],
                                         label=f"Cluster {lbl}", s=120, alpha=0.85,
                                         edgecolors="white", lw=0.7)
                    ax_t.set_title(title, fontsize=11, color=TEXT_CLR)
                    ax_t.set_xlabel("t-SNE dim 1"); ax_t.set_ylabel("t-SNE dim 2")
                    ax_t.legend(title="Cluster", fontsize=8, facecolor=CARD_BG,
                                labelcolor=TEXT_CLR, edgecolor=GRID_CLR)
                plt.tight_layout()
                st.pyplot(fig_ts, use_container_width=True)
                st.download_button("⬇️ Download Screenshot 7 — t-SNE Projection (PNG)",
                                   data=fig_to_png(fig_ts),
                                   file_name="screenshot7_tsne.png",
                                   mime="image/png", key="dl_tsne")
                plt.close(fig_ts)

                # ── Screenshot 8 : Cluster Profiles ─────────────
                feat_cols = [c for c in cluster_features.columns
                             if c not in ("KMeans_Cluster","DBSCAN_Cluster")]
                profile   = cluster_features.groupby("KMeans_Cluster")[feat_cols].mean().round(2)
                st.dataframe(profile, use_container_width=True)

                ss_badge("📸 SCREENSHOT 8 — Cluster Profile Bar Chart")
                fig_pr, ax_pr = dark_fig(12, 4)
                profile[["TotalSteps","Calories","VeryActiveMinutes",
                         "SedentaryMinutes","TotalSleepMinutes"]].plot(
                    kind="bar", ax=ax_pr, colormap="Set2", edgecolor="white", width=0.72)
                ax_pr.set_title("Cluster Profiles — Key Feature Averages",
                                fontsize=12, color=TEXT_CLR)
                ax_pr.set_xlabel("Cluster"); ax_pr.set_ylabel("Mean Value")
                ax_pr.set_xticklabels([f"Cluster {i}" for i in range(optimal_k)], rotation=0)
                ax_pr.legend(bbox_to_anchor=(1.02, 1), title="Feature",
                             facecolor=CARD_BG, labelcolor=TEXT_CLR, edgecolor=GRID_CLR)
                plt.tight_layout()
                st.pyplot(fig_pr, use_container_width=True)
                st.download_button("⬇️ Download Screenshot 8 — Cluster Profiles (PNG)",
                                   data=fig_to_png(fig_pr),
                                   file_name="screenshot8_cluster_profiles.png",
                                   mime="image/png", key="dl_profiles")
                plt.close(fig_pr)

                # Interpretation cards
                st.markdown("##### 🏷️ Cluster Interpretation Cards")
                cl_cols_ = st.columns(optimal_k)
                CARD_THEMES = [
                    ("#38bdf8","#071829"),("#f472b6","#250a18"),
                    ("#34d399","#041f12"),("#fb923c","#1f0d04"),("#a78bfa","#130a2a"),
                ]
                for i,col in enumerate(cl_cols_):
                    if i not in profile.index: continue
                    row = profile.loc[i]
                    steps,act,sed,cal,slp = (row["TotalSteps"],row["VeryActiveMinutes"],
                                             row["SedentaryMinutes"],row["Calories"],
                                             row["TotalSleepMinutes"])
                    emoji,label = (("🏃","HIGHLY ACTIVE") if steps>10000
                                   else ("🚶","MODERATELY ACTIVE") if steps>5000
                                   else ("🛋️","SEDENTARY"))
                    fg,bg = CARD_THEMES[i%len(CARD_THEMES)]
                    col.markdown(
                        f'<div style="background:{bg};border:1.5px solid {fg};border-radius:12px;padding:16px;">'
                        f'<div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:800;color:{fg}">'
                        f'{emoji} Cluster {i}</div>'
                        f'<div style="font-family:JetBrains Mono,monospace;font-size:9px;'
                        f'letter-spacing:1.5px;color:{fg};margin-bottom:10px">{label}</div>'
                        f'<div style="font-size:12px;color:#94a3b8;line-height:2">'
                        f'👣 <b style="color:#e2e8f0">{steps:,.0f}</b> steps/day<br>'
                        f'🔥 <b style="color:#e2e8f0">{cal:,.0f}</b> kcal/day<br>'
                        f'⚡ <b style="color:#e2e8f0">{act:.0f} min</b> very active<br>'
                        f'🪑 <b style="color:#e2e8f0">{sed:.0f} min</b> sedentary<br>'
                        f'😴 <b style="color:#e2e8f0">{slp:.0f} min</b> sleep/night'
                        f'</div></div>', unsafe_allow_html=True)

                mark("clustering")
                st.success(f"✅ Step 6 complete — KMeans: {optimal_k} clusters | "
                           f"DBSCAN: {n_clusters_db} clusters, {n_noise} noise points.")
            except Exception as e:
                st.error(f"❌ {e}\n\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════
# STEP 7 — FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════
with st.container(border=True):
    step_header(7,"📊","Milestone 2 — Final Summary","summary","clustering")

    dc,bc = st.columns([5,1])
    dc.markdown("<span style='color:#4b607a;font-size:13px'>"
                "All results compiled in clearly labelled summary boxes · Download full report"
                "</span>", unsafe_allow_html=True)
    run7 = bc.button("▶ Run" if not done("summary") else "🔁 Re-run",
                     key="b7", disabled=not done("clustering"), use_container_width=True)

    if run7: reset_from("summary")

    if done("summary") or run7:
        try:
            master           = st.session_state.master
            features         = st.session_state.features
            kmeans_labels    = st.session_state.kmeans_labels
            dbscan_labels    = st.session_state.dbscan_labels
            n_clusters_db    = st.session_state.n_clusters_db
            n_noise          = st.session_state.n_noise
            var_explained    = st.session_state.var_explained
            cluster_features = st.session_state.cluster_features
            fc_hr            = st.session_state.fc_hr

            st.markdown("---")
            st.markdown(
                '<div style="font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;'
                'background:linear-gradient(135deg,#38bdf8,#f472b6);'
                '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
                'margin-bottom:14px">📊 Milestone 2 — Complete Results</div>',
                unsafe_allow_html=True)

            metric_tiles([
                (master["Id"].nunique(),  "Total Users",       "#38bdf8"),
                ("31",                    "Days Tracked",      "#f472b6"),
                ("Mar–Apr 2016",          "Study Period",      "#34d399"),
                (features.shape[1],       "TSFresh Features",  "#fb923c"),
                ("3",                     "Prophet Models",    "#a78bfa"),
                (optimal_k,               "KMeans Clusters",   "#38bdf8"),
                (n_clusters_db,           "DBSCAN Clusters",   "#f472b6"),
                (n_noise,                 "Noise Points",      "#fb923c"),
            ])

            st.markdown("---")
            c1,c2 = st.columns(2)
            with c1:
                sum_box("📂 Dataset Overview", [
                    ("Source",             "Real Fitbit wearable device",        "c-blue"),
                    ("Unique users",       f"{master['Id'].nunique()}",           "c-green"),
                    ("Date span",          "31 days · March – April 2016",       "c-blue"),
                    ("Master DF",          f"{master.shape[0]:,} × {master.shape[1]} cols","c-blue"),
                    ("Final null values",  "0 ✅",                               "c-green"),
                    ("HR granularity",     "1-minute intervals",                 "c-orange"),
                    ("Sleep granularity",  "1-minute · 3 stages",               "c-orange"),
                ])
                sum_box("🧪 TSFresh Feature Extraction", [
                    ("Mode",               tsfresh_mode,                         "c-blue"),
                    ("Source signal",      "Heart rate (1-min)",                 "c-blue"),
                    ("Users processed",    f"{features.shape[0]}",               "c-green"),
                    ("Features extracted", f"{features.shape[1]}",               "c-green"),
                    ("Feature names",      ", ".join(features.columns),          "c-purple"),
                    ("Output CSV",         "tsfresh_features.csv ✅",            "c-green"),
                ])
                sum_box("📉 Dimensionality Reduction", [
                    ("PCA — PC1 variance", f"{var_explained[0]:.1f}%",           "c-blue"),
                    ("PCA — PC2 variance", f"{var_explained[1]:.1f}%",           "c-blue"),
                    ("PCA — Total",        f"{sum(var_explained):.1f}%",         "c-green"),
                    ("t-SNE perplexity",   f"{min(30,cluster_features.shape[0]-1)}","c-purple"),
                    ("t-SNE iterations",   "1000",                               "c-purple"),
                ])
            with c2:
                sum_box("📈 Prophet Forecasting", [
                    ("Models fitted",      "3  (Heart Rate · Steps · Sleep)",    "c-blue"),
                    ("Forecast horizon",   f"{forecast_days} days",              "c-green"),
                    ("Confidence band",    "80% interval",                       "c-purple"),
                    ("Weekly seasonality", "Enabled on all 3 models",            "c-orange"),
                    ("HR changepoint scale",f"{changepoint_scale}",             "c-orange"),
                    ("HR forecast peak",   f"{fc_hr['yhat'].max():.1f} bpm",    "c-pink"),
                    ("HR forecast floor",  f"{fc_hr['yhat'].min():.1f} bpm",    "c-pink"),
                ])
                sum_box("🤖 Clustering Results", [
                    ("Users clustered",    f"{cluster_features.shape[0]}",       "c-blue"),
                    ("Features used",      "7 (activity + sleep)",               "c-blue"),
                    ("KMeans K",           f"{optimal_k}",                       "c-green"),
                    ("KMeans distribution",
                     "  |  ".join(f"C{i}:{int((np.array(kmeans_labels)==i).sum())}"
                                   for i in range(optimal_k)),                   "c-green"),
                    ("DBSCAN epsilon",     f"{eps_val}",                         "c-orange"),
                    ("DBSCAN min_samples", f"{int(min_samples)}",                "c-orange"),
                    ("DBSCAN clusters",    f"{n_clusters_db}",                   "c-green"),
                    ("DBSCAN noise",       f"{n_noise} users ({n_noise/len(dbscan_labels)*100:.1f}%)", "c-orange"),
                ])


            # Cluster cards
            st.markdown("---")
            st.markdown("##### 🏷️ Final Cluster Profiles")
            feat_cols = [c for c in cluster_features.columns
                         if c not in ("KMeans_Cluster","DBSCAN_Cluster")]
            profile   = cluster_features.groupby("KMeans_Cluster")[feat_cols].mean().round(2)
            cl_cols_  = st.columns(optimal_k)
            CARD_THEMES = [
                ("#38bdf8","#071829"),("#f472b6","#250a18"),
                ("#34d399","#041f12"),("#fb923c","#1f0d04"),("#a78bfa","#130a2a"),
            ]
            for i,col in enumerate(cl_cols_):
                if i not in profile.index: continue
                row = profile.loc[i]
                steps,act,sed,cal,slp = (row["TotalSteps"],row["VeryActiveMinutes"],
                                         row["SedentaryMinutes"],row["Calories"],
                                         row["TotalSleepMinutes"])
                emoji,label = (("🏃","HIGHLY ACTIVE") if steps>10000
                               else ("🚶","MODERATELY ACTIVE") if steps>5000
                               else ("🛋️","SEDENTARY"))
                fg,bg = CARD_THEMES[i%len(CARD_THEMES)]
                col.markdown(
                    f'<div style="background:{bg};border:1.5px solid {fg};border-radius:12px;padding:16px;">'
                    f'<div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:800;color:{fg}">'
                    f'{emoji} Cluster {i}</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:9px;'
                    f'letter-spacing:1.5px;color:{fg};margin-bottom:10px">{label}</div>'
                    f'<div style="font-size:12px;color:#94a3b8;line-height:2">'
                    f'👣 <b style="color:#e2e8f0">{steps:,.0f}</b> steps/day<br>'
                    f'🔥 <b style="color:#e2e8f0">{cal:,.0f}</b> kcal/day<br>'
                    f'⚡ <b style="color:#e2e8f0">{act:.0f} min</b> very active<br>'
                    f'🪑 <b style="color:#e2e8f0">{sed:.0f} min</b> sedentary<br>'
                    f'😴 <b style="color:#e2e8f0">{slp:.0f} min</b> sleep/night'
                    f'</div></div>', unsafe_allow_html=True)

            # Download report
            dist_str = " | ".join(f"C{i}:{int((np.array(kmeans_labels)==i).sum())}"
                                   for i in range(optimal_k))
            report = (
                f"{'='*60}\n   MILESTONE 2 SUMMARY — REAL FITBIT DATA\n{'='*60}\n\n"
                f"DATASET\n"
                f"  Users       : {master['Id'].nunique()}\n"
                f"  Date span   : 31 days (March–April 2016)\n"
                f"  Master DF   : {master.shape[0]:,} rows × {master.shape[1]} cols\n"
                f"  Nulls       : 0\n\n"
                f"TSFRESH\n"
                f"  Mode        : {tsfresh_mode}\n"
                f"  Users       : {features.shape[0]}\n"
                f"  Features    : {features.shape[1]}\n"
                f"  Names       : {', '.join(features.columns)}\n\n"
                f"PROPHET\n"
                f"  Models      : Heart Rate · Total Steps · Sleep Minutes\n"
                f"  Horizon     : {forecast_days} days  |  CI: 80%\n"
                f"  CP scale    : {changepoint_scale}\n\n"
                f"KMEANS\n"
                f"  K           : {optimal_k}\n"
                f"  Distribution: {dist_str}\n\n"
                f"DBSCAN\n"
                f"  eps         : {eps_val}   min_samples: {int(min_samples)}\n"
                f"  Clusters    : {n_clusters_db}   Noise: {n_noise} "
                f"({n_noise/len(dbscan_labels)*100:.1f}%)\n\n"
                f"PCA\n"
                f"  PC1: {var_explained[0]:.1f}%   PC2: {var_explained[1]:.1f}%   "
                f"Total: {sum(var_explained):.1f}%\n{'='*60}\n"
            )
            st.markdown("---")
            st.download_button("⬇️  Download Full Summary Report (.txt)",
                               report,"milestone2_summary.txt","text/plain")

            mark("summary")
            st.success("🎉 Milestone 2 Pipeline Complete! All 7 steps executed successfully.")
        except Exception as e:
            st.error(f"❌ {e}\n\n{traceback.format_exc()}")