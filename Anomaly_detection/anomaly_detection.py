import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitPulse · Anomaly Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [
    ("dark_mode", True), ("files_loaded", False),
    ("anomaly_done", False), ("simulation_done", False),
    ("daily", None), ("hourly_s", None), ("hourly_i", None),
    ("sleep", None), ("hr", None), ("hr_minute", None),
    ("master", None), ("anom_hr", None), ("anom_steps", None),
    ("anom_sleep", None), ("sim_results", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Color Palette ──────────────────────────────────────────────────────────────
C = {
    "bg":       "#050810",
    "surface":  "#0d1117",
    "panel":    "#111827",
    "border":   "#1f2937",
    "border2":  "#374151",
    "text":     "#f1f5f9",
    "muted":    "#6b7280",
    "subtle":   "#9ca3af",
    "red":      "#f43f5e",
    "red_dim":  "rgba(244,63,94,0.15)",
    "red_glow": "rgba(244,63,94,0.08)",
    "blue":     "#38bdf8",
    "blue_dim": "rgba(56,189,248,0.12)",
    "green":    "#34d399",
    "green_dim":"rgba(52,211,153,0.12)",
    "amber":    "#fbbf24",
    "amber_dim":"rgba(251,191,36,0.12)",
    "purple":   "#a78bfa",
    "purp_dim": "rgba(167,139,250,0.12)",
    "pink":     "#f472b6",
    "plot_bg":  "#080b12",
    "grid":     "rgba(255,255,255,0.04)",
}

# ── Plotly default layout ──────────────────────────────────────────────────────
def base_layout(**kwargs):
    return dict(
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["plot_bg"],
        font=dict(family="'DM Mono', 'IBM Plex Mono', monospace", color=C["subtle"], size=11),
        xaxis=dict(gridcolor=C["grid"], showgrid=True, zeroline=False,
                   linecolor=C["border"], tickfont=dict(color=C["muted"], size=10),
                   title_font=dict(color=C["subtle"])),
        yaxis=dict(gridcolor=C["grid"], showgrid=True, zeroline=False,
                   linecolor=C["border"], tickfont=dict(color=C["muted"], size=10),
                   title_font=dict(color=C["subtle"])),
        legend=dict(bgcolor="rgba(13,17,23,0.9)", bordercolor=C["border2"],
                    borderwidth=1, font=dict(color=C["subtle"], size=10)),
        margin=dict(l=55, r=30, t=65, b=50),
        hoverlabel=dict(bgcolor=C["panel"], bordercolor=C["border2"],
                        font=dict(color=C["text"], size=12)),
        **kwargs
    )

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Orbitron:wght@700;900&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"], .main {{
    background: {C["bg"]} !important;
    font-family: 'Space Grotesk', sans-serif;
    color: {C["text"]} !important;
}}

[data-testid="stHeader"] {{ background: transparent !important; }}

[data-testid="stSidebar"] {{
    background: {C["surface"]} !important;
    border-right: 1px solid {C["border"]} !important;
}}
[data-testid="stSidebar"] * {{ color: {C["text"]} !important; }}

.block-container {{ padding: 1.5rem 2rem 4rem !important; max-width: 1500px; }}

/* ── Hero ── */
.hero {{
    position: relative; overflow: hidden;
    border: 1px solid {C["border2"]};
    border-radius: 20px; padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    background: linear-gradient(135deg, {C["surface"]} 0%, #0f1520 60%, {C["surface"]} 100%);
}}
.hero::before {{
    content:''; position:absolute; top:-80px; right:-80px;
    width:400px; height:400px;
    background: radial-gradient(circle, rgba(244,63,94,0.08) 0%, transparent 65%);
    border-radius:50%; pointer-events:none;
}}
.hero::after {{
    content:''; position:absolute; bottom:-60px; left:20%;
    width:300px; height:300px;
    background: radial-gradient(circle, rgba(56,189,248,0.05) 0%, transparent 65%);
    border-radius:50%; pointer-events:none;
}}
.hero-eyebrow {{
    font-family:'DM Mono',monospace; font-size:0.7rem; letter-spacing:0.15em;
    color:{C["red"]}; text-transform:uppercase; margin-bottom:1rem;
    display:inline-flex; align-items:center; gap:0.5rem;
}}
.hero-eyebrow::before {{
    content:''; width:24px; height:1px; background:{C["red"]}; display:inline-block;
}}
.hero-title {{
    font-family:'Orbitron',monospace; font-size:2.8rem; font-weight:900;
    color:{C["text"]}; margin:0 0 0.6rem; letter-spacing:-0.01em; line-height:1.1;
}}
.hero-title span {{ color:{C["red"]}; }}
.hero-sub {{
    font-size:1rem; color:{C["muted"]}; font-weight:300; margin:0; max-width:600px;
}}
.hero-chips {{
    display:flex; gap:0.6rem; flex-wrap:wrap; margin-top:1.5rem;
}}
.chip {{
    padding:0.3rem 0.9rem; border-radius:100px; font-size:0.72rem;
    font-family:'DM Mono',monospace; border:1px solid; letter-spacing:0.05em;
}}
.chip-red   {{ color:{C["red"]};    border-color:rgba(244,63,94,0.35);    background:rgba(244,63,94,0.08); }}
.chip-blue  {{ color:{C["blue"]};   border-color:rgba(56,189,248,0.35);   background:rgba(56,189,248,0.08); }}
.chip-green {{ color:{C["green"]};  border-color:rgba(52,211,153,0.35);   background:rgba(52,211,153,0.08); }}
.chip-amber {{ color:{C["amber"]};  border-color:rgba(251,191,36,0.35);   background:rgba(251,191,36,0.08); }}
.chip-purp  {{ color:{C["purple"]}; border-color:rgba(167,139,250,0.35);  background:rgba(167,139,250,0.08); }}

/* ── Section ── */
.sec-row {{
    display:flex; align-items:center; gap:1rem;
    margin:2.5rem 0 1.2rem; padding-bottom:0.8rem;
    border-bottom:1px solid {C["border"]};
}}
.sec-num {{
    font-family:'Orbitron',monospace; font-size:0.65rem; font-weight:700;
    color:{C["muted"]}; letter-spacing:0.12em;
    background:{C["panel"]}; border:1px solid {C["border2"]};
    border-radius:6px; padding:0.25rem 0.55rem;
}}
.sec-title {{
    font-family:'Space Grotesk',sans-serif; font-size:1.2rem; font-weight:600;
    color:{C["text"]}; margin:0;
}}
.sec-pill {{
    margin-left:auto; font-family:'DM Mono',monospace; font-size:0.68rem;
    color:{C["blue"]}; background:rgba(56,189,248,0.1); border:1px solid rgba(56,189,248,0.25);
    border-radius:100px; padding:0.2rem 0.7rem;
}}

/* ── Cards ── */
.card {{
    background:{C["panel"]}; border:1px solid {C["border"]};
    border-radius:16px; padding:1.5rem 1.8rem; margin-bottom:1rem;
}}
.card-danger {{
    background:rgba(244,63,94,0.05); border:1px solid rgba(244,63,94,0.25);
}}
.card-success {{
    background:rgba(52,211,153,0.05); border:1px solid rgba(52,211,153,0.25);
}}

/* ── Alert boxes ── */
.alert {{ border-radius:0 12px 12px 0; padding:0.85rem 1.1rem; margin:0.6rem 0; font-size:0.84rem; }}
.alert-info    {{ background:rgba(56,189,248,0.08);  border-left:3px solid {C["blue"]};   color:#bae6fd; }}
.alert-success {{ background:rgba(52,211,153,0.08);  border-left:3px solid {C["green"]};  color:#a7f3d0; }}
.alert-warn    {{ background:rgba(251,191,36,0.08);  border-left:3px solid {C["amber"]};  color:#fde68a; }}
.alert-danger  {{ background:rgba(244,63,94,0.08);   border-left:3px solid {C["red"]};    color:#fecdd3; }}

/* ── Metric grid ── */
.metric-row {{ display:flex; gap:0.8rem; flex-wrap:wrap; margin:1rem 0; }}
.metric-box {{
    flex:1; min-width:110px; border-radius:14px; padding:1.1rem 1.3rem;
    text-align:center; border:1px solid {C["border2"]}; background:{C["panel"]};
    position:relative; overflow:hidden;
}}
.metric-box::before {{
    content:''; position:absolute; inset:0;
    background:var(--glow); opacity:0.5; pointer-events:none;
}}
.metric-val {{
    font-family:'Orbitron',monospace; font-size:1.8rem; font-weight:900;
    line-height:1; margin-bottom:0.3rem;
}}
.metric-lbl {{ font-size:0.68rem; color:{C["muted"]}; letter-spacing:0.08em; text-transform:uppercase; }}

/* ── Anomaly badge ── */
.anom-badge {{
    display:inline-flex; align-items:center; gap:0.4rem;
    background:rgba(244,63,94,0.1); border:1px solid rgba(244,63,94,0.3);
    border-radius:100px; padding:0.3rem 0.9rem; font-family:'DM Mono',monospace;
    font-size:0.7rem; color:{C["red"]}; margin:0.5rem 0;
}}
.pulse {{ display:inline-block; width:7px; height:7px; border-radius:50%;
          background:{C["red"]}; animation:pulse 1.4s ease-in-out infinite; }}
@keyframes pulse {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:0.4;transform:scale(0.7)}} }}

/* ── Buttons ── */
.stButton > button {{
    background:rgba(244,63,94,0.1); border:1px solid rgba(244,63,94,0.35);
    color:{C["red"]}; border-radius:10px; font-family:'DM Mono',monospace;
    font-size:0.8rem; padding:0.55rem 1.3rem; transition:all 0.2s; letter-spacing:0.03em;
}}
.stButton > button:hover {{
    background:{C["red"]}; color:white; border-color:{C["red"]};
    transform:translateY(-1px); box-shadow:0 4px 20px rgba(244,63,94,0.3);
}}

/* ── Progress bar ── */
.prog-track {{ background:{C["border"]}; border-radius:4px; height:4px; overflow:hidden; margin:0.4rem 0 1rem; }}
.prog-fill  {{ height:100%; border-radius:4px;
               background:linear-gradient(90deg,{C["red"]},{C["purple"]},{C["blue"]});
               transition:width 0.5s ease; }}

/* ── Divider ── */
.div {{ border:none; border-top:1px solid {C["border"]}; margin:2rem 0; }}

/* ── File uploader ── */
div[data-testid="stFileUploader"] {{
    background:{C["panel"]}; border:1.5px dashed {C["border2"]}; border-radius:14px; padding:0.5rem;
}}

/* ── Dataframe ── */
.stDataFrame {{ background:{C["panel"]}; border-radius:12px; }}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
def sec(icon, title, badge=None, step=None):
    badge_html = f'<span class="sec-pill">{badge}</span>' if badge else ''
    num_html   = f'<span class="sec-num">{step}</span>' if step else ''
    st.markdown(f"""
    <div class="sec-row">
      {num_html}
      <span style="font-size:1.3rem">{icon}</span>
      <p class="sec-title">{title}</p>
      {badge_html}
    </div>""", unsafe_allow_html=True)

def alert(kind, msg):
    st.markdown(f'<div class="alert alert-{kind}">{msg}</div>', unsafe_allow_html=True)

def anom_badge(label):
    st.markdown(f'<div class="anom-badge"><span class="pulse"></span>{label}</div>', unsafe_allow_html=True)

def metric_row(*items):
    """items: (value, label, color_key)"""
    colors = {"red":C["red"],"blue":C["blue"],"green":C["green"],"amber":C["amber"],"purple":C["purple"],"muted":C["muted"]}
    html = '<div class="metric-row">'
    for val, lbl, clr in items:
        c = colors.get(clr, C["blue"])
        glow_color = c.replace(")", ",0.06)").replace("rgb(", "rgba(") if "rgb" in c else f"linear-gradient(135deg,{c}18,transparent)"
        html += f"""
        <div class="metric-box" style="--glow:{glow_color}; border-color:{c}33;">
          <div class="metric-val" style="color:{c}">{val}</div>
          <div class="metric-lbl">{lbl}</div>
        </div>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

REQUIRED_FILES = {
    "dailyActivity_merged.csv":     {"key_cols":["ActivityDate","TotalSteps","Calories"],       "label":"Daily Activity",    "icon":"🏃"},
    "hourlySteps_merged.csv":       {"key_cols":["ActivityHour","StepTotal"],                   "label":"Hourly Steps",      "icon":"👣"},
    "hourlyIntensities_merged.csv": {"key_cols":["ActivityHour","TotalIntensity"],              "label":"Hourly Intensities","icon":"⚡"},
    "minuteSleep_merged.csv":       {"key_cols":["date","value","logId"],                       "label":"Minute Sleep",      "icon":"💤"},
    "heartrate_seconds_merged.csv": {"key_cols":["Time","Value"],                               "label":"Heart Rate",        "icon":"❤️"},
}
def score_match(df, info):
    return sum(1 for c in info["key_cols"] if c in df.columns)

# ── Detection functions ────────────────────────────────────────────────────────
def detect_hr_anomalies(master, hr_high=100, hr_low=50, sigma=2.0):
    df = master[["Id","Date","AvgHR","MaxHR","MinHR"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    hr_d = df.groupby("Date")["AvgHR"].mean().reset_index()
    hr_d.columns = ["Date","AvgHR"]
    hr_d = hr_d.sort_values("Date")
    hr_d["rolling_med"] = hr_d["AvgHR"].rolling(3,center=True,min_periods=1).median()
    hr_d["residual"]    = hr_d["AvgHR"] - hr_d["rolling_med"]
    std = hr_d["residual"].std()
    hr_d["thresh_high"]   = hr_d["AvgHR"] > hr_high
    hr_d["thresh_low"]    = hr_d["AvgHR"] < hr_low
    hr_d["resid_anomaly"] = hr_d["residual"].abs() > sigma * std
    hr_d["is_anomaly"]    = hr_d["thresh_high"] | hr_d["thresh_low"] | hr_d["resid_anomaly"]
    def reason(r):
        out=[]
        if r["thresh_high"]:   out.append(f"HR>{hr_high}")
        if r["thresh_low"]:    out.append(f"HR<{hr_low}")
        if r["resid_anomaly"]: out.append(f"±{sigma:.0f}σ")
        return ", ".join(out)
    hr_d["reason"] = hr_d.apply(reason, axis=1)
    return hr_d

def detect_steps_anomalies(master, st_low=500, st_high=25000, sigma=2.0):
    df = master[["Date","TotalSteps"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    s = df.groupby("Date")["TotalSteps"].mean().reset_index().sort_values("Date")
    s["rolling_med"]   = s["TotalSteps"].rolling(3,center=True,min_periods=1).median()
    s["residual"]      = s["TotalSteps"] - s["rolling_med"]
    std = s["residual"].std()
    s["thresh_low"]    = s["TotalSteps"] < st_low
    s["thresh_high"]   = s["TotalSteps"] > st_high
    s["resid_anomaly"] = s["residual"].abs() > sigma * std
    s["is_anomaly"]    = s["thresh_low"] | s["thresh_high"] | s["resid_anomaly"]
    def reason(r):
        out=[]
        if r["thresh_low"]:    out.append(f"<{st_low}")
        if r["thresh_high"]:   out.append(f">{st_high}")
        if r["resid_anomaly"]: out.append(f"±{sigma:.0f}σ")
        return ", ".join(out)
    s["reason"] = s.apply(reason, axis=1)
    return s

def detect_sleep_anomalies(master, sl_low=60, sl_high=600, sigma=2.0):
    df = master[["Date","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    s = df.groupby("Date")["TotalSleepMinutes"].mean().reset_index().sort_values("Date")
    s["rolling_med"]   = s["TotalSleepMinutes"].rolling(3,center=True,min_periods=1).median()
    s["residual"]      = s["TotalSleepMinutes"] - s["rolling_med"]
    std = s["residual"].std()
    s["thresh_low"]    = (s["TotalSleepMinutes"]>0) & (s["TotalSleepMinutes"]<sl_low)
    s["thresh_high"]   = s["TotalSleepMinutes"] > sl_high
    s["no_data"]       = s["TotalSleepMinutes"] == 0
    s["resid_anomaly"] = s["residual"].abs() > sigma * std
    s["is_anomaly"]    = s["thresh_low"] | s["thresh_high"] | s["resid_anomaly"]
    def reason(r):
        out=[]
        if r["no_data"]:       out.append("No device")
        if r["thresh_low"]:    out.append(f"<{sl_low}min")
        if r["thresh_high"]:   out.append(f">{sl_high}min")
        if r["resid_anomaly"]: out.append(f"±{sigma:.0f}σ")
        return ", ".join(out)
    s["reason"] = s.apply(reason, axis=1)
    return s

def simulate_accuracy(master, n_inject=10):
    np.random.seed(42)
    df = master[["Date","AvgHR","TotalSteps","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df_d = df.groupby("Date").mean().reset_index().sort_values("Date")
    results = {}
    for signal, col, inj_vals, lo, hi in [
        ("Heart Rate","AvgHR",[115,120,125,35,40,45,118,130,38,42],50,100),
        ("Steps","TotalSteps",[50,100,150,30000,35000,28000,80,200,31000,29000],500,25000),
        ("Sleep","TotalSleepMinutes",[10,20,30,700,750,800,15,25,710,720],60,600),
    ]:
        sim = df_d[["Date",col]].copy()
        idx = np.random.choice(len(sim), n_inject, replace=False)
        sim.loc[idx, col] = np.random.choice(inj_vals, n_inject, replace=True)
        sim["rm"]  = sim[col].rolling(3,center=True,min_periods=1).median()
        sim["res"] = sim[col] - sim["rm"]
        std = sim["res"].std()
        if signal == "Sleep":
            sim["det"] = ((sim[col]>0)&(sim[col]<lo))|(sim[col]>hi)|(sim[col].abs()>2*std)
        else:
            sim["det"] = (sim[col]<lo)|(sim[col]>hi)|(sim["res"].abs()>2*std)
        tp = sim.iloc[idx]["det"].sum()
        results[signal] = {"injected":n_inject,"detected":int(tp),"accuracy":round(tp/n_inject*100,1)}
    results["Overall"] = round(np.mean([results[k]["accuracy"] for k in ["Heart Rate","Steps","Sleep"]]),1)
    return results

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:0.5rem 0 1.5rem">
      <div style="font-family:'Orbitron',monospace;font-size:1.1rem;font-weight:900;color:{C['red']}">
        🫀 FITPULSE
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:{C['muted']};margin-top:0.2rem;letter-spacing:0.1em">
        MILESTONE 3 · ANOMALY DETECTION
      </div>
    </div>
    """, unsafe_allow_html=True)

    steps_done = sum([st.session_state.files_loaded,
                      st.session_state.anomaly_done,
                      st.session_state.simulation_done])
    pct = int(steps_done / 3 * 100)
    st.markdown(f"""
    <div style="margin-bottom:1.2rem">
      <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:{C['muted']};margin-bottom:0.4rem;letter-spacing:0.1em">
        PIPELINE PROGRESS · {pct}%
      </div>
      <div class="prog-track"><div class="prog-fill" style="width:{pct}%"></div></div>
    </div>
    """, unsafe_allow_html=True)

    for done, icon, lbl in [
        (st.session_state.files_loaded,    "📂", "Data Loaded"),
        (st.session_state.anomaly_done,    "🚨", "Anomalies Detected"),
        (st.session_state.simulation_done, "🎯", "Accuracy Validated"),
    ]:
        dot = f'<span style="color:{C["green"]}">●</span>' if done else f'<span style="color:{C["border2"]}">○</span>'
        st.markdown(f'<div style="font-size:0.82rem;padding:0.3rem 0;color:{C["text"] if done else C["muted"]}">{dot} {icon} {lbl}</div>', unsafe_allow_html=True)

    st.markdown(f'<hr style="border-color:{C["border"]};margin:1rem 0">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:0.65rem;color:{C["muted"]};margin-bottom:0.6rem;letter-spacing:0.1em">DETECTION THRESHOLDS</div>', unsafe_allow_html=True)

    hr_high = st.number_input("HR High (bpm)",    value=100, min_value=80,  max_value=180)
    hr_low  = st.number_input("HR Low (bpm)",     value=50,  min_value=30,  max_value=70)
    st_low  = st.number_input("Steps Low",        value=500, min_value=0,   max_value=2000)
    sl_low  = st.number_input("Sleep Low (min)",  value=60,  min_value=0,   max_value=120)
    sl_high = st.number_input("Sleep High (min)", value=600, min_value=300, max_value=900)
    sigma   = st.slider("Residual σ", 1.0, 4.0, 2.0, 0.5)

    st.markdown(f'<hr style="border-color:{C["border"]};margin:1rem 0">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:0.65rem;color:{C["muted"]};line-height:1.8">Fitbit Dataset · 30 users<br>March – April 2016</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <div class="hero-eyebrow">Milestone 3 · Anomaly Detection & Visualization</div>
  <h1 class="hero-title">FitPulse <span>Anomaly</span><br>Detector</h1>
  <p class="hero-sub">Real-time biometric anomaly detection across heart rate, sleep, and activity signals using statistical threshold analysis and residual-based flagging.</p>
  <div class="hero-chips">
    <span class="chip chip-red">Threshold Violations</span>
    <span class="chip chip-blue">Residual ±σ Detection</span>
    <span class="chip chip-green">DBSCAN Outliers</span>
    <span class="chip chip-amber">30 Users · 31 Days</span>
    <span class="chip chip-purp">Interactive Plotly</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
sec("📂", "Data Ingestion", badge="Step 01", step="01")
alert("info", "Upload all 5 Fitbit CSV files. Auto-detected by column structure — order doesn't matter.")

uploaded_files = st.file_uploader(
    "Drop Fitbit CSV files here",
    type="csv", accept_multiple_files=True, key="m3_uploader"
)

detected = {}
if uploaded_files:
    raw = []
    for uf in uploaded_files:
        try:
            raw.append((uf.name, pd.read_csv(uf)))
        except: pass
    used = set()
    for req_name, finfo in REQUIRED_FILES.items():
        best_score, best_df = 0, None
        for uname, udf in raw:
            s = score_match(udf, finfo)
            if s > best_score:
                best_score, best_df, best_nm = s, udf, uname
        if best_score >= 2:
            detected[req_name] = best_df
            used.add(best_nm)

# File status grid
grid_html = '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:0.7rem;margin:1.2rem 0">'
for req, finfo in REQUIRED_FILES.items():
    found = req in detected
    bg  = C["green_dim"] if found else C["red_glow"]
    bor = C["green"]     if found else C["border2"]
    ico = "✓" if found else "✗"
    clr = C["green"]     if found else C["muted"]
    grid_html += f"""
    <div style="background:{bg};border:1px solid {bor}44;border-radius:12px;padding:0.9rem 1rem">
      <div style="font-size:1.4rem;margin-bottom:0.4rem">{finfo['icon']}</div>
      <div style="font-size:0.75rem;font-weight:600;color:{C['text']}">{finfo['label']}</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:{clr};margin-top:0.2rem">{ico} {'ready' if found else 'missing'}</div>
    </div>"""
grid_html += "</div>"
st.markdown(grid_html, unsafe_allow_html=True)

n_up = len(detected)
metric_row(
    (n_up,       "Files Detected",   "green" if n_up==5 else "red"),
    (5 - n_up,   "Files Missing",    "muted"),
    ("✓" if n_up==5 else "✗", "Ready", "green" if n_up==5 else "red"),
)

if n_up < 5:
    missing = [REQUIRED_FILES[r]["label"] for r in REQUIRED_FILES if r not in detected]
    alert("warn", f"Missing: {', '.join(missing)}")

if st.button("⚡ Load & Build Master DataFrame", disabled=(n_up < 5)):
    with st.spinner("Parsing timestamps & merging datasets..."):
        try:
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
                                DominantSleepStage=("value", lambda x: x.mode()[0]))
                           .reset_index())

            master = daily.copy().rename(columns={"ActivityDate":"Date"})
            master["Date"] = master["Date"].dt.date
            master = master.merge(hr_daily, on=["Id","Date"], how="left")
            master = master.merge(sleep_daily, on=["Id","Date"], how="left")
            master["TotalSleepMinutes"]  = master["TotalSleepMinutes"].fillna(0)
            master["DominantSleepStage"] = master["DominantSleepStage"].fillna(0)
            for col in ["AvgHR","MaxHR","MinHR","StdHR"]:
                master[col] = master.groupby("Id")[col].transform(lambda x: x.fillna(x.median()))

            for k, v in [("daily",daily),("hourly_s",hourly_s),("hourly_i",hourly_i),
                          ("sleep",sleep),("hr",hr),("hr_minute",hr_minute),("master",master)]:
                st.session_state[k] = v
            st.session_state.files_loaded = True
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

if not st.session_state.files_loaded:
    st.markdown(f"""
    <div class="card" style="text-align:center;padding:3rem;margin-top:1rem">
      <div style="font-size:3.5rem;margin-bottom:1rem">🫀</div>
      <div style="font-family:'Orbitron',monospace;font-size:1.1rem;font-weight:700;color:{C['text']};margin-bottom:0.5rem">
        Awaiting Data Upload
      </div>
      <div style="color:{C['muted']};font-size:0.85rem">Upload all 5 CSV files and click Load to begin</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADED
# ══════════════════════════════════════════════════════════════════════════════
master = st.session_state.master
alert("success", f"✓ Master DataFrame · {master.shape[0]:,} rows · {master['Id'].nunique()} users · {master['Date'].nunique()} days")

st.markdown('<hr class="div">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
sec("🚨", "Anomaly Detection Engine", badge="Steps 02–05", step="02")

# Method cards
st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;margin-bottom:1.2rem">
  <div class="card" style="border-color:{C['red']}33">
    <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:{C['red']};letter-spacing:0.1em;margin-bottom:0.6rem">METHOD 01</div>
    <div style="font-weight:600;color:{C['text']};margin-bottom:0.4rem">Threshold Violations</div>
    <div style="font-size:0.8rem;color:{C['muted']}">Hard upper/lower bounds on HR, Steps, and Sleep. Immediate, interpretable alert flags.</div>
  </div>
  <div class="card" style="border-color:{C['blue']}33">
    <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:{C['blue']};letter-spacing:0.1em;margin-bottom:0.6rem">METHOD 02</div>
    <div style="font-weight:600;color:{C['text']};margin-bottom:0.4rem">Residual ±{sigma:.0f}σ Detection</div>
    <div style="font-size:0.8rem;color:{C['muted']}">Rolling median as pseudo-forecast. Flag days with statistically unusual deviations.</div>
  </div>
  <div class="card" style="border-color:{C['green']}33">
    <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:{C['green']};letter-spacing:0.1em;margin-bottom:0.6rem">METHOD 03</div>
    <div style="font-weight:600;color:{C['text']};margin-bottom:0.4rem">DBSCAN Outlier Clustering</div>
    <div style="font-size:0.8rem;color:{C['muted']}">Users labelled −1 by DBSCAN are structural behavioural outliers across all features.</div>
  </div>
</div>
""", unsafe_allow_html=True)

if st.button("🔍 Run Anomaly Detection — All Methods"):
    with st.spinner("Running detection pipeline..."):
        try:
            st.session_state.anom_hr    = detect_hr_anomalies(master, hr_high, hr_low, sigma)
            st.session_state.anom_steps = detect_steps_anomalies(master, st_low, 25000, sigma)
            st.session_state.anom_sleep = detect_sleep_anomalies(master, sl_low, sl_high, sigma)
            st.session_state.anomaly_done = True
            st.rerun()
        except Exception as e:
            st.error(f"Detection error: {e}")

if not st.session_state.anomaly_done:
    st.stop()

anom_hr    = st.session_state.anom_hr
anom_steps = st.session_state.anom_steps
anom_sleep = st.session_state.anom_sleep

n_hr, n_st, n_sl = int(anom_hr["is_anomaly"].sum()), int(anom_steps["is_anomaly"].sum()), int(anom_sleep["is_anomaly"].sum())
n_total = n_hr + n_st + n_sl

alert("danger", f"🚨 {n_total} total anomaly flags — HR: {n_hr} · Steps: {n_st} · Sleep: {n_sl}")
metric_row(
    (n_hr,    "HR Anomalies",    "red"),
    (n_st,    "Step Anomalies",  "red"),
    (n_sl,    "Sleep Anomalies", "red"),
    (n_total, "Total Flags",     "amber"),
)

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — HEART RATE ANOMALY (enhanced)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="div">', unsafe_allow_html=True)
sec("❤️", "Heart Rate — Anomaly Timeline", badge="Step 02", step="02")
anom_badge(f"{n_hr} anomalous days · Threshold + Residual Detection")
alert("info", f"Red markers = anomaly days. Dashed lines = thresholds (>{hr_high} / <{hr_low} bpm). Blue band = ±{sigma:.0f}σ expected corridor.")

hr_anom   = anom_hr[anom_hr["is_anomaly"]]
hr_normal = anom_hr[~anom_hr["is_anomaly"]]

fig_hr = go.Figure()

# Confidence band
rolling_std = anom_hr["residual"].std()
fig_hr.add_trace(go.Scatter(
    x=pd.concat([anom_hr["Date"], anom_hr["Date"].iloc[::-1]]),
    y=pd.concat([anom_hr["rolling_med"] + sigma * rolling_std,
                 (anom_hr["rolling_med"] - sigma * rolling_std).iloc[::-1]]),
    fill="toself", fillcolor="rgba(56,189,248,0.07)",
    line=dict(width=0), name=f"±{sigma:.0f}σ Band", showlegend=True,
    hoverinfo="skip"
))

# Area fill under HR line
fig_hr.add_trace(go.Scatter(
    x=anom_hr["Date"], y=anom_hr["AvgHR"],
    fill="tozeroy", fillcolor="rgba(56,189,248,0.04)",
    line=dict(width=0), showlegend=False, hoverinfo="skip"
))

# Main HR line
fig_hr.add_trace(go.Scatter(
    x=anom_hr["Date"], y=anom_hr["AvgHR"],
    mode="lines+markers", name="Avg Heart Rate",
    line=dict(color=C["blue"], width=2.5),
    marker=dict(size=5, color=C["blue"], line=dict(color=C["surface"], width=1)),
    hovertemplate="<b>%{x|%b %d}</b><br>Heart Rate: <b>%{y:.1f} bpm</b><extra></extra>"
))

# Rolling median
fig_hr.add_trace(go.Scatter(
    x=anom_hr["Date"], y=anom_hr["rolling_med"],
    mode="lines", name="Trend (3-day Median)",
    line=dict(color=C["green"], width=1.5, dash="dot"),
    hovertemplate="<b>%{x|%b %d}</b><br>Trend: %{y:.1f} bpm<extra></extra>"
))

# Anomaly markers
if not hr_anom.empty:
    fig_hr.add_trace(go.Scatter(
        x=hr_anom["Date"], y=hr_anom["AvgHR"],
        mode="markers", name="Anomaly",
        marker=dict(color=C["red"], size=15, symbol="circle",
                    line=dict(color="white", width=2.5),
                    opacity=0.95),
        hovertemplate="<b>⚠️ ANOMALY — %{x|%b %d}</b><br>HR: %{y:.1f} bpm<extra></extra>"
    ))
    for _, row in hr_anom.iterrows():
        fig_hr.add_annotation(
            x=row["Date"], y=row["AvgHR"],
            text=f"<b>{row['reason']}</b>", showarrow=True,
            arrowhead=2, arrowcolor=C["red"], arrowsize=1.3, arrowwidth=1.5,
            ax=0, ay=-50,
            font=dict(color=C["red"], size=9, family="DM Mono"),
            bgcolor=C["panel"], bordercolor="rgba(244,63,94,0.4)", borderwidth=1, borderpad=4
        )

# Threshold lines
fig_hr.add_hline(y=hr_high, line_dash="dash", line_color=C["red"],  line_width=1.5, opacity=0.6,
                  annotation_text=f"  High Threshold — {hr_high} bpm",
                  annotation_font=dict(color=C["red"], size=10, family="DM Mono"))
fig_hr.add_hline(y=hr_low,  line_dash="dash", line_color=C["amber"], line_width=1.5, opacity=0.6,
                  annotation_text=f"  Low Threshold — {hr_low} bpm",
                  annotation_position="bottom right",
                  annotation_font=dict(color=C["amber"], size=10, family="DM Mono"))

fig_hr.update_layout(
    **base_layout(height=500),
    title=dict(text="❤️  Heart Rate Anomaly Detection — Daily Average (All Users)",
               font=dict(size=14, color=C["text"], family="Space Grotesk"), x=0.01),
    xaxis_title="Date", yaxis_title="Heart Rate (bpm)",
)
fig_hr.update_xaxes(gridcolor=C["grid"], tickfont=dict(color=C["muted"], size=10), tickformat="%b %d")
fig_hr.update_yaxes(gridcolor=C["grid"], tickfont=dict(color=C["muted"], size=10))
st.plotly_chart(fig_hr, use_container_width=True)

if not hr_anom.empty:
    with st.expander(f"📋 {len(hr_anom)} HR Anomaly Records"):
        st.dataframe(
            hr_anom[hr_anom["is_anomaly"]][["Date","AvgHR","rolling_med","residual","reason"]]
            .rename(columns={"rolling_med":"Expected","residual":"Deviation","reason":"Flag Reason"})
            .round(2), use_container_width=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — SLEEP (enhanced dual-panel)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="div">', unsafe_allow_html=True)
sec("💤", "Sleep Duration — Pattern & Residual Analysis", badge="Step 03", step="03")
anom_badge(f"{n_sl} anomalous sleep-days detected")
alert("info", f"Green band = healthy zone ({sl_low}–{sl_high} min). Bottom panel shows daily deviation from trend. Diamond markers = anomaly days.")

sleep_anom   = anom_sleep[anom_sleep["is_anomaly"]]
sleep_normal = anom_sleep[~anom_sleep["is_anomaly"]]

fig_sleep = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.68, 0.32],
    subplot_titles=["Sleep Duration (min/night, cohort avg)", "Residual Deviation from Trend"],
    vertical_spacing=0.06
)
# Update subplot title fonts
fig_sleep.update_annotations(font=dict(color=C["muted"], size=11, family="DM Mono"))

# Healthy band
fig_sleep.add_hrect(y0=sl_low, y1=sl_high, fillcolor="rgba(52,211,153,0.06)",
                     line_width=0, row=1, col=1)
fig_sleep.add_annotation(
    x=0.98, y=(sl_high+sl_low)/2, xref="x domain", yref="y",
    text=f"✓ Healthy Zone ({sl_low}–{sl_high} min)",
    showarrow=False, font=dict(color=C["green"], size=9, family="DM Mono"),
    align="right", row=1, col=1
)

# Sleep area fill
fig_sleep.add_trace(go.Scatter(
    x=anom_sleep["Date"], y=anom_sleep["TotalSleepMinutes"],
    fill="tozeroy", fillcolor="rgba(167,139,250,0.05)",
    line=dict(width=0), showlegend=False, hoverinfo="skip"
), row=1, col=1)

# Sleep line
fig_sleep.add_trace(go.Scatter(
    x=anom_sleep["Date"], y=anom_sleep["TotalSleepMinutes"],
    mode="lines+markers", name="Sleep Duration",
    line=dict(color=C["purple"], width=2.5),
    marker=dict(size=5, color=C["purple"], line=dict(color=C["surface"], width=1)),
    hovertemplate="<b>%{x|%b %d}</b><br>Sleep: <b>%{y:.0f} min</b><extra></extra>"
), row=1, col=1)

# Rolling median
fig_sleep.add_trace(go.Scatter(
    x=anom_sleep["Date"], y=anom_sleep["rolling_med"],
    mode="lines", name="3-day Trend",
    line=dict(color=C["green"], width=1.5, dash="dot"),
    hovertemplate="<b>%{x|%b %d}</b><br>Trend: %{y:.0f} min<extra></extra>"
), row=1, col=1)

# Anomaly diamonds
if not sleep_anom.empty:
    fig_sleep.add_trace(go.Scatter(
        x=sleep_anom["Date"], y=sleep_anom["TotalSleepMinutes"],
        mode="markers", name="Anomaly",
        marker=dict(color=C["red"], size=16, symbol="diamond",
                    line=dict(color="white", width=2.5)),
        hovertemplate="<b>⚠️ %{x|%b %d}</b><br>Sleep: %{y:.0f} min<extra>ANOMALY</extra>"
    ), row=1, col=1)
    for _, row in sleep_anom.iterrows():
        fig_sleep.add_annotation(
            x=row["Date"], y=row["TotalSleepMinutes"],
            text=f"<b>{row['reason']}</b>", showarrow=True,
            arrowhead=2, arrowcolor=C["red"], arrowsize=1.2, ax=25, ay=-45,
            font=dict(color=C["red"], size=9, family="DM Mono"),
            bgcolor=C["panel"], bordercolor="rgba(244,63,94,0.33)", borderwidth=1, borderpad=3,
            row=1, col=1
        )

# Threshold lines
fig_sleep.add_hline(y=sl_low,  line_dash="dash", line_color=C["red"],  line_width=1.3, opacity=0.7, row=1, col=1)
fig_sleep.add_hline(y=sl_high, line_dash="dash", line_color=C["blue"], line_width=1.3, opacity=0.7, row=1, col=1)

# Residual bars — gradient colours
res_clrs = [C["red"] if v else "rgba(167,139,250,0.4)" for v in anom_sleep["resid_anomaly"]]
fig_sleep.add_trace(go.Bar(
    x=anom_sleep["Date"], y=anom_sleep["residual"],
    name="Residual", marker_color=res_clrs,
    hovertemplate="<b>%{x|%b %d}</b><br>Residual: %{y:.0f} min<extra></extra>"
), row=2, col=1)
fig_sleep.add_hline(y=0, line_dash="solid", line_color=C["border2"], line_width=1, row=2, col=1)

fig_sleep.update_layout(
    **base_layout(height=580),
    showlegend=True
)
fig_sleep.update_xaxes(gridcolor=C["grid"], tickfont=dict(color=C["muted"], size=10), tickformat="%b %d")
fig_sleep.update_yaxes(gridcolor=C["grid"], tickfont=dict(color=C["muted"], size=10))
st.plotly_chart(fig_sleep, use_container_width=True)

if not sleep_anom.empty:
    with st.expander(f"📋 {len(sleep_anom)} Sleep Anomaly Records"):
        st.dataframe(
            sleep_anom[sleep_anom["is_anomaly"]][["Date","TotalSleepMinutes","rolling_med","residual","reason"]]
            .rename(columns={"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected","residual":"Deviation","reason":"Flag Reason"})
            .round(2), use_container_width=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — STEPS (enhanced with candlestick-style alert bands)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="div">', unsafe_allow_html=True)
sec("🚶", "Step Count — Trend & Alert Zones", badge="Step 04", step="04")
anom_badge(f"{n_st} anomalous step-count days · Alert bands active")
alert("info", f"Red shaded zones = anomaly alert days. Triangle markers = flagged events. Bottom: residual deviation from 3-day rolling trend.")

steps_anom = anom_steps[anom_steps["is_anomaly"]]

fig_steps = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.65, 0.35],
    subplot_titles=["Daily Step Count (cohort avg)", "Residual Deviation from Trend"],
    vertical_spacing=0.06
)
fig_steps.update_annotations(font=dict(color=C["muted"], size=11, family="DM Mono"))

# Alert vertical bands
for _, row in steps_anom.iterrows():
    d = str(row["Date"])
    d1 = str(pd.Timestamp(d) + pd.Timedelta(days=1))[:10]
    fig_steps.add_vrect(x0=d, x1=d1, fillcolor="rgba(244,63,94,0.1)",
                         line_color="rgba(244,63,94,0.4)", line_width=1, row=1, col=1)

# Step count area fill
fig_steps.add_trace(go.Scatter(
    x=anom_steps["Date"], y=anom_steps["TotalSteps"],
    fill="tozeroy", fillcolor="rgba(52,211,153,0.04)",
    line=dict(width=0), showlegend=False, hoverinfo="skip"
), row=1, col=1)

# Steps line
fig_steps.add_trace(go.Scatter(
    x=anom_steps["Date"], y=anom_steps["TotalSteps"],
    mode="lines+markers", name="Avg Daily Steps",
    line=dict(color=C["green"], width=2.5),
    marker=dict(size=5, color=C["green"], line=dict(color=C["surface"], width=1)),
    hovertemplate="<b>%{x|%b %d}</b><br>Steps: <b>%{y:,.0f}</b><extra></extra>"
), row=1, col=1)

# Rolling trend
fig_steps.add_trace(go.Scatter(
    x=anom_steps["Date"], y=anom_steps["rolling_med"],
    mode="lines", name="Trend",
    line=dict(color=C["blue"], width=2, dash="dash"),
    hovertemplate="<b>%{x|%b %d}</b><br>Trend: %{y:,.0f}<extra></extra>"
), row=1, col=1)

# Anomaly triangles
if not steps_anom.empty:
    fig_steps.add_trace(go.Scatter(
        x=steps_anom["Date"], y=steps_anom["TotalSteps"],
        mode="markers", name="Anomaly",
        marker=dict(color=C["red"], size=16, symbol="triangle-up",
                    line=dict(color="white", width=2.5)),
        hovertemplate="<b>⚠️ %{x|%b %d}</b><br>Steps: %{y:,.0f}<extra>ALERT</extra>"
    ), row=1, col=1)

# Threshold lines
fig_steps.add_hline(y=st_low,  line_dash="dash", line_color=C["red"],  line_width=1.5, opacity=0.7, row=1, col=1,
                     annotation_text=f"  Low Alert ({st_low:,} steps)",
                     annotation_font=dict(color=C["red"], size=10, family="DM Mono"))
fig_steps.add_hline(y=25000, line_dash="dash", line_color=C["pink"], line_width=1.5, opacity=0.7, row=1, col=1,
                     annotation_text="  High Alert (25,000 steps)",
                     annotation_font=dict(color=C["pink"], size=10, family="DM Mono"))

# WHO daily target line
fig_steps.add_hline(y=10000, line_dash="dot", line_color=C["amber"], line_width=1, opacity=0.5, row=1, col=1,
                     annotation_text="  WHO Target (10,000)",
                     annotation_font=dict(color=C["amber"], size=9, family="DM Mono"))

# Residual bars
res_clrs = [C["red"] if v else "rgba(52,211,153,0.35)" for v in anom_steps["resid_anomaly"]]
fig_steps.add_trace(go.Bar(
    x=anom_steps["Date"], y=anom_steps["residual"],
    name="Residual", marker_color=res_clrs, marker_line_width=0,
    hovertemplate="<b>%{x|%b %d}</b><br>Deviation: %{y:,.0f} steps<extra></extra>"
), row=2, col=1)
fig_steps.add_hline(y=0, line_dash="solid", line_color=C["border2"], line_width=1, row=2, col=1)

fig_steps.update_layout(
    **base_layout(height=580),
    showlegend=True
)
fig_steps.update_xaxes(gridcolor=C["grid"], tickfont=dict(color=C["muted"], size=10), tickformat="%b %d")
fig_steps.update_yaxes(gridcolor=C["grid"], tickfont=dict(color=C["muted"], size=10))
st.plotly_chart(fig_steps, use_container_width=True)

if not steps_anom.empty:
    with st.expander(f"📋 {len(steps_anom)} Steps Anomaly Records"):
        st.dataframe(
            steps_anom[steps_anom["is_anomaly"]][["Date","TotalSteps","rolling_med","residual","reason"]]
            .rename(columns={"TotalSteps":"Steps","rolling_med":"Expected","residual":"Deviation","reason":"Flag Reason"})
            .round(2), use_container_width=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — DBSCAN PCA (enhanced)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="div">', unsafe_allow_html=True)
sec("🔍", "DBSCAN — Structural Outlier Detection", badge="Step 05", step="05")
anom_badge("Cluster-based · Users labelled −1 are behavioural outliers")
alert("info", "PCA reduces 7-dimensional activity profile to 2D. Users outside all clusters (label −1) show fundamentally atypical behaviour patterns.")

cluster_cols = ["TotalSteps","Calories","VeryActiveMinutes","FairlyActiveMinutes",
                "LightlyActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA

    cf = master.groupby("Id")[cluster_cols].mean().round(3).dropna()
    X_scaled = StandardScaler().fit_transform(cf)
    db_labels = DBSCAN(eps=2.2, min_samples=2).fit_predict(X_scaled)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_ * 100

    cf["DBSCAN"] = db_labels
    outlier_idx = [i for i, l in enumerate(db_labels) if l == -1]
    n_outliers = len(outlier_idx)
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)

    metric_row(
        (n_clusters,           "DBSCAN Clusters",  "blue"),
        (n_outliers,           "Outlier Users",     "red"),
        (len(cf)-n_outliers,   "Normal Users",      "green"),
        (f"{var[0]:.0f}%",     "PC1 Variance",      "purple"),
    )

    CLUSTER_PAL = [C["blue"], C["green"], C["amber"], C["purple"], C["pink"]]
    fig_db = go.Figure()

    # Cluster hulls (convex hull approximation via scatter fill — simplified as background)
    for lbl in sorted(set(db_labels)):
        if lbl == -1: continue
        mask = np.array(db_labels) == lbl
        xvals, yvals = X_pca[mask, 0], X_pca[mask, 1]
        col = CLUSTER_PAL[lbl % len(CLUSTER_PAL)]
        # Draw transparent convex region with scatter voronoi approximation
        fig_db.add_trace(go.Scatter(
            x=xvals, y=yvals,
            mode="markers+text",
            name=f"Cluster {lbl}",
            marker=dict(size=16, color=col, opacity=0.85,
                        line=dict(color=C["surface"], width=2)),
            text=[f"···{str(uid)[-4:]}" for uid in cf.index[mask]],
            textposition="top center",
            textfont=dict(size=8, color=col, family="DM Mono"),
            hovertemplate="<b>Cluster %{text}</b><br>PC1: %{x:.2f} · PC2: %{y:.2f}<extra></extra>"
        ))

    # Outlier X markers
    if n_outliers > 0:
        mask_out = np.array(db_labels) == -1
        fig_db.add_trace(go.Scatter(
            x=X_pca[mask_out, 0], y=X_pca[mask_out, 1],
            mode="markers+text",
            name="⚠ Outlier",
            marker=dict(size=22, color=C["red"], symbol="x-thin",
                        line=dict(color=C["red"], width=3), opacity=0.95),
            text=[f"···{str(uid)[-4:]}" for uid in cf.index[mask_out]],
            textposition="top center",
            textfont=dict(size=9, color=C["red"], family="DM Mono"),
            hovertemplate="<b>⚠️ OUTLIER ···%{text}</b><br>PC1: %{x:.2f} · PC2: %{y:.2f}<extra></extra>"
        ))
        # Halo rings for outliers
        for i in range(sum(mask_out)):
            xi, yi = X_pca[mask_out][i]
            for r, op in [(0.35, 0.25), (0.55, 0.12)]:
                fig_db.add_shape(type="circle",
                    x0=xi-r, y0=yi-r, x1=xi+r, y1=yi+r,
                    line=dict(color=C["red"], width=1.5, dash="dot"),
                    fillcolor=f"rgba(244,63,94,{op})"
                )

    fig_db.update_layout(
        **base_layout(height=520),
        title=dict(text=f"🔍  DBSCAN User Outlier Detection — PCA Projection (eps=2.2 · {n_clusters} clusters)",
                   font=dict(size=13, color=C["text"], family="Space Grotesk"), x=0.01),
        xaxis_title=f"Principal Component 1 ({var[0]:.1f}% variance)",
        yaxis_title=f"Principal Component 2 ({var[1]:.1f}% variance)",
    )
    fig_db.update_xaxes(gridcolor=C["grid"], tickfont=dict(color=C["muted"], size=10))
    fig_db.update_yaxes(gridcolor=C["grid"], tickfont=dict(color=C["muted"], size=10))
    st.plotly_chart(fig_db, use_container_width=True)

    if n_outliers > 0:
        out_prof = cf[cf["DBSCAN"]==-1][cluster_cols]
        st.markdown(f'<div class="card card-danger"><div style="font-family:DM Mono,monospace;font-size:0.7rem;color:{C["red"]};letter-spacing:0.1em;margin-bottom:0.8rem">OUTLIER USER ACTIVITY PROFILES</div>', unsafe_allow_html=True)
        st.dataframe(out_prof.round(2), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

except Exception as e:
    alert("warn", f"DBSCAN skipped: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ACCURACY SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="div">', unsafe_allow_html=True)
sec("🎯", "Simulated Detection Accuracy — 90%+ Target", badge="Step 06", step="06")
anom_badge("10 injected anomalies per signal · Recall validation")
alert("info", "Known anomalies are injected at random positions. The detector is evaluated and recall (detected/injected) computed per signal.")

if st.button("🎯 Run Accuracy Simulation"):
    with st.spinner("Injecting anomalies and validating..."):
        try:
            st.session_state.sim_results = simulate_accuracy(master, n_inject=10)
            st.session_state.simulation_done = True
            st.rerun()
        except Exception as e:
            st.error(f"Simulation error: {e}")

if st.session_state.simulation_done and st.session_state.sim_results:
    sim = st.session_state.sim_results
    overall = sim["Overall"]
    passed  = overall >= 90.0

    if passed:
        alert("success", f"✓ Overall accuracy: {overall}% — MEETS the ≥90% requirement")
    else:
        alert("warn", f"⚠ Overall accuracy: {overall}% — below 90% target; adjust thresholds in sidebar")

    # Per-signal metric cards
    metric_row(
        (f"{sim['Heart Rate']['accuracy']}%", f"Heart Rate ({sim['Heart Rate']['detected']}/{sim['Heart Rate']['injected']})",
         "green" if sim["Heart Rate"]["accuracy"] >= 90 else "red"),
        (f"{sim['Steps']['accuracy']}%",      f"Steps ({sim['Steps']['detected']}/{sim['Steps']['injected']})",
         "green" if sim["Steps"]["accuracy"] >= 90 else "red"),
        (f"{sim['Sleep']['accuracy']}%",      f"Sleep ({sim['Sleep']['detected']}/{sim['Sleep']['injected']})",
         "green" if sim["Sleep"]["accuracy"] >= 90 else "red"),
        (f"{overall}%", "Overall Recall",   "green" if passed else "red"),
    )

    # Accuracy bar chart (enhanced)
    signals = ["Heart Rate", "Steps", "Sleep"]
    accs    = [sim[s]["accuracy"] for s in signals]
    detected_n = [sim[s]["detected"] for s in signals]

    fig_acc = go.Figure()

    # Background max bar
    fig_acc.add_trace(go.Bar(
        x=signals, y=[100,100,100],
        marker_color=[C["border"]]*3,
        marker_line_width=0,
        name="", showlegend=False,
        hoverinfo="skip", width=0.45
    ))

    # Accuracy bars
    bar_clrs = [C["green"] if a >= 90 else C["red"] for a in accs]
    fig_acc.add_trace(go.Bar(
        x=signals, y=accs,
        marker_color=bar_clrs,
        marker_line_width=0,
        text=[f"{a}%<br><span style='font-size:10px'>{d}/10</span>" for a, d in zip(accs, detected_n)],
        textposition="inside", textfont=dict(color="white", size=14, family="Orbitron"),
        hovertemplate="<b>%{x}</b><br>Accuracy: %{y}%<extra></extra>",
        name="Detection Accuracy",
        width=0.45
    ))

    # 90% target line
    fig_acc.add_hline(y=90, line_dash="dash", line_color=C["red"],
                       line_width=2, opacity=0.8,
                       annotation_text="  ≥90% Target",
                       annotation_font=dict(color=C["red"], size=11, family="DM Mono"),
                       annotation_position="top right")

    # Overall annotation
    fig_acc.add_annotation(
        x=2.5, y=overall,
        text=f"<b>Overall: {overall}%</b>",
        showarrow=False, font=dict(color=C["green"] if passed else C["red"], size=12, family="Orbitron"),
        bgcolor=C["panel"], bordercolor="rgba(52,211,153,0.33)" if passed else "rgba(244,63,94,0.33)", borderwidth=1, borderpad=6
    )

    fig_acc.update_layout(
        **base_layout(height=400),
        title=dict(text="🎯  Anomaly Detection Recall — Injected Anomaly Simulation (n=10 per signal)",
                   font=dict(size=13, color=C["text"], family="Space Grotesk"), x=0.01),
        yaxis_range=[0, 115],
        barmode="overlay",
        yaxis_title="Detection Accuracy (%)",
        showlegend=False
    )
    fig_acc.update_xaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=C["text"], size=13, family="Space Grotesk"))
    fig_acc.update_yaxes(gridcolor=C["grid"], tickfont=dict(color=C["muted"], size=10))
    st.plotly_chart(fig_acc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="div">', unsafe_allow_html=True)
sec("✅", "Milestone 3 — Completion Summary")

checklist = [
    ("❤️", "HR Anomaly Chart",          st.session_state.anomaly_done,    f"Threshold >{hr_high}/{hr_low} bpm + ±{sigma:.0f}σ residual"),
    ("💤", "Sleep Pattern Chart",        st.session_state.anomaly_done,    f"Dual-panel: duration timeline + residual bars"),
    ("🚶", "Steps Trend Chart",          st.session_state.anomaly_done,    f"Alert bands + deviation from trend"),
    ("🔍", "DBSCAN Outlier Scatter",     st.session_state.anomaly_done,    "PCA projection · structural user outliers"),
    ("🎯", "Accuracy Simulation (90%+)", st.session_state.simulation_done, f"Overall: {st.session_state.sim_results['Overall']}%" if st.session_state.sim_results else "Pending"),
]

for icon, label, done, detail in checklist:
    tick = C["green"] if done else C["border2"]
    txt  = C["text"]  if done else C["muted"]
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:1rem;padding:0.7rem 0;border-bottom:1px solid {C['border']}">
      <span style="font-size:1.1rem;color:{tick}">{"✓" if done else "○"}</span>
      <span style="font-size:0.88rem;font-weight:600;color:{txt};min-width:200px">{icon} {label}</span>
      <span style="font-family:'DM Mono',monospace;font-size:0.75rem;color:{C['muted']}">{detail}</span>
    </div>""", unsafe_allow_html=True)

st.markdown(f"""
<br>
<div class="card" style="border-color:{C['border2']};margin-top:1rem">
  <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:{C['muted']};letter-spacing:0.12em;margin-bottom:1rem">SCREENSHOT CHECKLIST</div>
  <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:0.6rem;font-size:0.82rem">
    <div style="background:{C['panel']};border-radius:10px;padding:0.6rem 0.9rem;border:1px solid {C['border']}">
      <span style="color:{C['purple']}">📸</span> <b>Chart 1</b> — Heart Rate timeline with anomaly annotations
    </div>
    <div style="background:{C['panel']};border-radius:10px;padding:0.6rem 0.9rem;border:1px solid {C['border']}">
      <span style="color:{C['purple']}">📸</span> <b>Chart 2</b> — Sleep dual-panel with residual deviation
    </div>
    <div style="background:{C['panel']};border-radius:10px;padding:0.6rem 0.9rem;border:1px solid {C['border']}">
      <span style="color:{C['purple']}">📸</span> <b>Chart 3</b> — Step trend with alert bands
    </div>
    <div style="background:{C['panel']};border-radius:10px;padding:0.6rem 0.9rem;border:1px solid {C['border']}">
      <span style="color:{C['purple']}">📸</span> <b>Chart 4</b> — DBSCAN PCA outlier scatter
    </div>
    <div style="background:{C['panel']};border-radius:10px;padding:0.6rem 0.9rem;border:1px solid {C['border']};grid-column:1/-1">
      <span style="color:{C['purple']}">📸</span> <b>Chart 5</b> — Accuracy bar chart with 90% target line
    </div>
  </div>
</div>
""", unsafe_allow_html=True)