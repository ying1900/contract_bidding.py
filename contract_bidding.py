from __future__ import annotations

import csv
import io
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Contract Bid Optimizer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --blue:       #1e88e5;
    --blue-dark:  #1565c0;
    --blue-light: #e3f2fd;
    --navy:       #1e3a8a;
    --text:       #1e293b;
    --subtext:    #475569;
    --muted:      #94a3b8;
    --border:     #e2e8f0;
    --bg:         #f0f2f6;
    --card:       #ffffff;
    --red:        #ef4444;
    --green:      #22c55e;
    --amber:      #f59e0b;
    --purple:     #8b5cf6;
    --teal:       #14b8a6;
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg); }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }

.header-container {
    display: flex; align-items: center; gap: 20px;
    margin-bottom: 28px;
    background: linear-gradient(90deg, #1e3a8a 0%, #1e88e5 100%);
    padding: 30px 36px; border-radius: 15px; color: white;
}
.header-text h1 {
    font-size: 28px; font-weight: 800; color: white !important;
    margin: 0 0 6px 0; letter-spacing: -0.02em;
}
.header-text p { font-size: 13px; opacity: 0.88; margin: 0; line-height: 1.5; }

[data-testid="stMetric"] {
    background: white !important; padding: 18px 16px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
    border-left: 5px solid var(--blue) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 12px !important; font-weight: 700 !important;
    color: var(--subtext) !important; text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stMetricValue"] {
    font-size: 24px !important; font-weight: 800 !important;
    color: var(--navy) !important;
}
[data-testid="stMetricDelta"] { font-size: 12px !important; font-weight: 600 !important; }

.panel-title {
    font-size: 12px; font-weight: 700; color: var(--navy);
    margin-bottom: 14px; text-transform: uppercase; letter-spacing: 0.07em;
    border-bottom: 2px solid var(--blue-light); padding-bottom: 9px;
}
.sec-lbl {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--muted); margin: 0 0 8px 0;
}
.rec-banner {
    background: #eff6ff; border: 1.5px solid var(--blue);
    border-left: 5px solid var(--blue);
    border-radius: 10px; padding: 16px 20px; margin-bottom: 20px;
}
.rec-banner.green { background:#f0fdf4; border-color:var(--green); border-left-color:var(--green); }
.rec-banner.amber { background:#fffbeb; border-color:var(--amber); border-left-color:var(--amber); }
.rec-banner.red   { background:#fef2f2; border-color:var(--red);   border-left-color:var(--red);   }
.rec-title { font-size: 15px; font-weight: 700; color: var(--navy); margin-bottom: 5px; }
.rec-body  { font-size: 13px; color: var(--subtext); line-height: 1.6; }

.brow {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 14px; border-radius: 8px; margin: 3px 0;
    background: #f8fafc; font-size: 13px; font-weight: 500;
    border: 1px solid var(--border);
}
.alert-info { background:var(--blue-light); border:1px solid #90caf9; border-radius:8px; padding:9px 13px; font-size:12px; color:#1565c0; font-weight:600; margin:8px 0; }
.alert-warn { background:#fffbeb; border:1px solid #fcd34d; border-radius:8px; padding:9px 13px; font-size:12px; color:#92400e; font-weight:600; margin:8px 0; }

hr { border: none; border-top: 1px solid var(--border); margin: 16px 0; }

div.stButton > button {
    border-radius: 8px !important; font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important; font-size: 14px !important;
    padding: 10px 0 !important; transition: all 0.15s !important;
}
div.stButton > button[kind="primary"] {
    background: var(--blue) !important; border-color: var(--blue) !important; color: white !important;
}
div.stButton > button[kind="primary"]:hover {
    background: var(--blue-dark) !important; border-color: var(--blue-dark) !important;
}
div[data-testid="stNumberInput"] input {
    background: white !important; border: 1.5px solid var(--border) !important;
    border-radius: 8px !important; color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important;
}
label {
    font-size: 11px !important; font-weight: 700 !important;
    color: var(--subtext) !important; letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
.stSlider > div > div > div { background: var(--blue) !important; }
.stTabs [data-baseweb="tab-list"] {
    gap: 3px; background: #f0f2f6; border-radius: 10px;
    padding: 4px; border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important; font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important; font-size: 12px !important;
    color: var(--subtext) !important; padding: 7px 14px !important;
}
.stTabs [aria-selected="true"] { background: var(--blue) !important; color: white !important; }
[data-testid="stDataFrame"] { border-radius: 10px !important; }
</style>
"""

# ═══════════════════════════════════════════════════════════════════
# DATA LAYER
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HistoricalCostData:
    prep_costs: list[float]
    project_costs: list[float]
    row_count: int
    source_name: str


def _read_source(source) -> tuple[str, str]:
    if hasattr(source, "read"):
        raw = source.read()
        name = getattr(source, "name", "uploaded_file")
        if isinstance(raw, bytes):
            text = raw.decode("utf-8-sig")
        else:
            text = raw.encode("utf-8", errors="ignore").decode("utf-8-sig")
        return text, name
    path = Path(source)
    return path.read_text(encoding="utf-8-sig"), path.name


def load_historical_costs(source) -> HistoricalCostData:
    text, source_name = _read_source(source)
    rows = list(csv.reader(io.StringIO(text)))

    if not rows:
        raise ValueError("The CSV file is empty.")
    if len(rows[0]) < 2:
        raise ValueError("The CSV must include at least two columns.")

    prep_costs: list[float] = []
    project_costs: list[float] = []

    for index, row in enumerate(rows[1:], start=2):
        if not row or all(not cell.strip() for cell in row):
            continue

        prep_raw = row[0].strip() if len(row) > 0 else ""
        project_raw = row[1].strip() if len(row) > 1 else ""

        if not prep_raw:
            raise ValueError(f"Row {index}: bid preparation cost is required.")
        try:
            prep_value = float(prep_raw)
        except ValueError as exc:
            raise ValueError(f"Row {index}: invalid bid preparation cost '{prep_raw}'.") from exc

        prep_costs.append(prep_value)

        if project_raw:
            try:
                project_costs.append(float(project_raw))
            except ValueError as exc:
                raise ValueError(
                    f"Row {index}: invalid project completion cost '{project_raw}'."
                ) from exc

    if not prep_costs:
        raise ValueError("The CSV does not contain any usable historical rows.")
    if not project_costs:
        raise ValueError("The CSV must contain at least one non-blank project completion cost.")

    return HistoricalCostData(
        prep_costs=prep_costs,
        project_costs=project_costs,
        row_count=len(prep_costs),
        source_name=source_name,
    )


# ═══════════════════════════════════════════════════════════════════
# SIMULATION LAYER
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CompetitorConfig:
    name: str
    participation_probability: float
    min_bid: float
    mode_bid: float
    max_bid: float
    guaranteed: bool = False

    def validate(self) -> None:
        if self.guaranteed and self.participation_probability != 1.0:
            raise ValueError(f"{self.name}: guaranteed competitors must have 100% participation.")
        if not 0.0 <= self.participation_probability <= 1.0:
            raise ValueError(f"{self.name}: participation probability must be between 0 and 1.")
        if self.min_bid > self.mode_bid or self.mode_bid > self.max_bid:
            raise ValueError(
                f"{self.name}: bids must satisfy min ≤ mode ≤ max for triangular sampling."
            )


@dataclass(frozen=True)
class ScenarioConfig:
    simulation_count: int
    project_cost_floor: float
    candidate_bid_min: float
    candidate_bid_max: float
    candidate_bid_step: float
    competitors: list[CompetitorConfig]
    random_seed: int | None = None

    def validate(self) -> None:
        if self.simulation_count <= 0:
            raise ValueError("Simulation count must be positive.")
        if self.project_cost_floor < 0:
            raise ValueError("Project cost floor cannot be negative.")
        if self.candidate_bid_step <= 0:
            raise ValueError("Candidate bid step must be positive.")
        if self.candidate_bid_min > self.candidate_bid_max:
            raise ValueError("Candidate bid minimum cannot exceed the maximum.")
        if not self.competitors:
            raise ValueError("At least one competitor configuration is required.")
        for competitor in self.competitors:
            competitor.validate()


@dataclass(frozen=True)
class CandidateResult:
    bid_amount: float
    win_probability: float
    expected_revenue: float
    expected_total_cost: float
    expected_profit: float
    probability_of_loss: float
    profit_p5: float
    profit_p50: float
    profit_p95: float
    simulated_profits: list[float]


@dataclass(frozen=True)
class AnalysisResult:
    recommendation: str
    best_bid: float
    best_result: CandidateResult
    candidate_results: list[CandidateResult]


@dataclass(frozen=True)
class Trial:
    prep_cost: float
    completion_cost: float
    competitor_bids: list[float]


def generate_candidate_bids(minimum: float, maximum: float, step: float) -> list[float]:
    values: list[float] = []
    current = minimum
    while current <= maximum + (step / 1000):
        values.append(round(current, 2))
        current += step
    return values


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _build_trials(history: HistoricalCostData, scenario: ScenarioConfig) -> list[Trial]:
    rng = random.Random(scenario.random_seed)
    trials: list[Trial] = []
    for _ in range(scenario.simulation_count):
        prep_cost = rng.choice(history.prep_costs)
        completion_cost = max(rng.choice(history.project_costs), scenario.project_cost_floor)
        competitor_bids: list[float] = []
        for competitor in scenario.competitors:
            participates = competitor.guaranteed or rng.random() <= competitor.participation_probability
            if participates:
                bid = rng.triangular(competitor.min_bid, competitor.max_bid, competitor.mode_bid)
                competitor_bids.append(round(bid, 2))
        trials.append(Trial(prep_cost=prep_cost, completion_cost=completion_cost, competitor_bids=competitor_bids))
    return trials


def _score_bid(candidate_bid: float, trials: list[Trial]) -> CandidateResult:
    profits: list[float] = []
    revenues: list[float] = []
    total_costs: list[float] = []
    win_shares: list[float] = []
    for trial in trials:
        all_bids = [candidate_bid, *trial.competitor_bids]
        lowest_bid = min(all_bids)
        tie_count = sum(1 for b in all_bids if b == lowest_bid)
        win_share = (1.0 / tie_count) if candidate_bid == lowest_bid else 0.0
        revenue = win_share * candidate_bid
        total_cost = trial.prep_cost + win_share * trial.completion_cost
        profit = revenue - total_cost
        revenues.append(revenue)
        total_costs.append(total_cost)
        profits.append(profit)
        win_shares.append(win_share)
    n = len(trials)
    return CandidateResult(
        bid_amount=candidate_bid,
        win_probability=sum(win_shares) / n,
        expected_revenue=sum(revenues) / n,
        expected_total_cost=sum(total_costs) / n,
        expected_profit=sum(profits) / n,
        probability_of_loss=sum(1 for v in profits if v < 0) / n,
        profit_p5=_percentile(profits, 0.05),
        profit_p50=_percentile(profits, 0.50),
        profit_p95=_percentile(profits, 0.95),
        simulated_profits=profits,
    )


def analyze_scenario(history: HistoricalCostData, scenario: ScenarioConfig) -> AnalysisResult:
    scenario.validate()
    candidate_bids = generate_candidate_bids(
        scenario.candidate_bid_min, scenario.candidate_bid_max, scenario.candidate_bid_step
    )
    trials = _build_trials(history, scenario)
    results = [_score_bid(b, trials) for b in candidate_bids]
    best_result = max(results, key=lambda r: r.expected_profit)
    return AnalysisResult(
        recommendation="Bid" if best_result.expected_profit > 0 else "No Bid",
        best_bid=best_result.bid_amount,
        best_result=best_result,
        candidate_results=results,
    )


# ═══════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════

def summarize_history(history: HistoricalCostData) -> dict[str, float]:
    return {
        "historical_rows": history.row_count,
        "won_project_rows": len(history.project_costs),
        "mean_prep_cost": statistics.mean(history.prep_costs),
        "mean_project_cost": statistics.mean(history.project_costs),
    }


def build_export_csv(
    analysis: AnalysisResult, scenario: ScenarioConfig, history: HistoricalCostData
) -> str:
    rows: list[dict[str, Any]] = [
        {"record_type": "summary", "metric": "recommendation",    "value": analysis.recommendation,                        "source": history.source_name},
        {"record_type": "summary", "metric": "recommended_bid",   "value": analysis.best_bid,                              "source": history.source_name},
        {"record_type": "summary", "metric": "expected_profit",   "value": round(analysis.best_result.expected_profit, 2), "source": history.source_name},
        {"record_type": "summary", "metric": "win_probability",   "value": round(analysis.best_result.win_probability, 4), "source": history.source_name},
        {"record_type": "summary", "metric": "probability_of_loss","value": round(analysis.best_result.probability_of_loss, 4),"source": history.source_name},
        {"record_type": "summary", "metric": "simulation_count",  "value": scenario.simulation_count,                     "source": history.source_name},
    ]
    for competitor in scenario.competitors:
        rows.append({
            "record_type": "competitor", "metric": competitor.name, "value": "",
            "source": history.source_name,
            "participation_probability": competitor.participation_probability,
            "min_bid": competitor.min_bid, "mode_bid": competitor.mode_bid,
            "max_bid": competitor.max_bid, "guaranteed": competitor.guaranteed,
        })
    for result in analysis.candidate_results:
        rows.append({
            "record_type": "candidate_bid", "metric": "bid_metrics", "value": "",
            "source": history.source_name,
            "bid_amount": result.bid_amount,
            "expected_profit": round(result.expected_profit, 2),
            "win_probability": round(result.win_probability, 4),
            "probability_of_loss": round(result.probability_of_loss, 4),
            "expected_revenue": round(result.expected_revenue, 2),
            "expected_total_cost": round(result.expected_total_cost, 2),
            "profit_p5": round(result.profit_p5, 2),
            "profit_p50": round(result.profit_p50, 2),
            "profit_p95": round(result.profit_p95, 2),
        })
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ═══════════════════════════════════════════════════════════════════

PLOT = dict(
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Outfit", color="#475569", size=12),
    margin=dict(l=10, r=10, t=30, b=40),
)
GRID  = dict(showgrid=True,  gridcolor="#f1f5f9")
NGRID = dict(showgrid=False)
TICK  = dict(tickfont=dict(family="JetBrains Mono", size=11, color="#475569"))


# ═══════════════════════════════════════════════════════════════════
# EXAMPLE DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════

def _make_example_csv() -> str:
    rng = np.random.default_rng(1)
    n = 50
    prep = rng.normal(4500, 800, n).clip(0)
    project = rng.normal(115000, 18000, n).clip(70000)
    lost = rng.random(n) < 0.25  # 25% bids lost → blank project cost
    buf = io.StringIO()
    buf.write("Prep Cost,Project Cost\n")
    for i in range(n):
        p_str = f"{project[i]:.2f}" if not lost[i] else ""
        buf.write(f"{prep[i]:.2f},{p_str}\n")
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════════

st.markdown(CSS, unsafe_allow_html=True)

st.markdown("""
<div class="header-container">
    <div style="font-size:52px;line-height:1">🏗️</div>
    <div class="header-text">
        <h1>Contract Bid Optimizer</h1>
        <p>Bootstrap-resampled historical costs · Monte Carlo simulation · Triangular competitor bids ·
        Find the bid amount that maximises expected profit across uncertain competition.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────
for k, v in {
    "history": None,
    "analysis": None,
    "scenario": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Two-column layout ────────────────────────────────────────────
left, right = st.columns([1, 2], gap="large")

with left:

    # ── Step 1: Upload data ──────────────────────────────────────
    st.markdown('<div class="panel-title">📂 Step 1 — Upload Cost Data</div>', unsafe_allow_html=True)
    st.caption(
        "CSV must have two columns: **column 1** = bid preparation cost, "
        "**column 2** = project completion cost (blank = lost bid, excluded from cost resampling)."
    )

    uploaded = st.file_uploader("Upload historical cost CSV", type=["csv"], key="csv_upload")

    with st.expander("Or load built-in example data"):
        if st.button("Load Miller Construction Example", use_container_width=True):
            example_csv = _make_example_csv()
            try:
                st.session_state.history = load_historical_costs(io.StringIO(example_csv))
                st.session_state.history = HistoricalCostData(
                    prep_costs=st.session_state.history.prep_costs,
                    project_costs=st.session_state.history.project_costs,
                    row_count=st.session_state.history.row_count,
                    source_name="example_data",
                )
                st.session_state.analysis = None
            except ValueError as exc:
                st.error(str(exc))

    if uploaded is not None:
        try:
            st.session_state.history = load_historical_costs(uploaded)
            st.session_state.analysis = None
        except ValueError as exc:
            st.error(str(exc))
            st.session_state.history = None

    if st.session_state.history is not None:
        h = st.session_state.history
        st.markdown(
            f'<div class="alert-info">'
            f'Loaded <strong>{h.row_count:,}</strong> bids · '
            f'<strong>{len(h.project_costs)}</strong> won projects · '
            f'source: <em>{h.source_name}</em>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Step 2: Cost floor ───────────────────────────────────────
    st.markdown('<div class="panel-title">🔻 Step 2 — Cost Constraints</div>', unsafe_allow_html=True)
    project_cost_floor = st.number_input(
        "Minimum project completion cost ($)",
        min_value=0.0, value=70000.0, step=1000.0,
        help="Sampled costs below this value are floored here.",
    )

    st.divider()

    # ── Step 3: Competitors ──────────────────────────────────────
    st.markdown('<div class="panel-title">⚔️ Step 3 — Competitor Setup</div>', unsafe_allow_html=True)

    optional_competitors = st.number_input(
        "Number of optional competitors", min_value=0, max_value=6, value=2, step=1,
    )

    st.markdown("**Competitor bid distribution (Triangular) — applies to all competitors**")
    cc1, cc2, cc3 = st.columns(3)
    with cc1: comp_min  = st.number_input("Min ($)",         min_value=0.0, value=90000.0,  step=1000.0)
    with cc2: comp_mode = st.number_input("Most likely ($)", min_value=0.0, value=130000.0, step=1000.0)
    with cc3: comp_max  = st.number_input("Max ($)",         min_value=0.0, value=180000.0, step=1000.0)

    optional_probabilities: list[float] = []
    for idx in range(int(optional_competitors)):
        prob = st.slider(
            f"Optional competitor {idx + 1} — participation probability",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        )
        optional_probabilities.append(prob)

    st.divider()

    # ── Step 4: Simulation settings ──────────────────────────────
    st.markdown('<div class="panel-title">⚙️ Step 4 — Simulation Settings</div>', unsafe_allow_html=True)

    simulation_count = st.number_input(
        "Monte Carlo iterations", min_value=1000, max_value=200000, value=20000, step=1000,
    )
    random_seed_input = st.text_input("Random seed (leave blank for random)", value="42")

    st.markdown("**Bid scan range**")
    bc1, bc2, bc3 = st.columns(3)
    with bc1: candidate_bid_min  = st.number_input("Min bid ($)",  min_value=0.0, value=90000.0,  step=1000.0)
    with bc2: candidate_bid_max  = st.number_input("Max bid ($)",  min_value=0.0, value=180000.0, step=1000.0)
    with bc3: candidate_bid_step = st.number_input("Step ($)",     min_value=100.0, value=2500.0, step=100.0)

    run_analysis = st.button("▶  Run Analysis", type="primary", use_container_width=True)

# ── Trigger analysis ─────────────────────────────────────────────
if run_analysis:
    if st.session_state.history is None:
        with right:
            st.error("Please upload a CSV or load the example data first.")
    else:
        try:
            seed_val = int(random_seed_input.strip()) if random_seed_input.strip() else None
        except ValueError:
            with right:
                st.error("Random seed must be blank or an integer.")
            seed_val = None

        if seed_val is not None or not random_seed_input.strip():
            competitors = [
                CompetitorConfig(
                    name="Known Competitor",
                    participation_probability=1.0,
                    min_bid=comp_min, mode_bid=comp_mode, max_bid=comp_max,
                    guaranteed=True,
                )
            ]
            for i, prob in enumerate(optional_probabilities):
                competitors.append(
                    CompetitorConfig(
                        name=f"Optional Competitor {i + 1}",
                        participation_probability=prob,
                        min_bid=comp_min, mode_bid=comp_mode, max_bid=comp_max,
                        guaranteed=False,
                    )
                )
            scenario = ScenarioConfig(
                simulation_count=int(simulation_count),
                project_cost_floor=float(project_cost_floor),
                candidate_bid_min=float(candidate_bid_min),
                candidate_bid_max=float(candidate_bid_max),
                candidate_bid_step=float(candidate_bid_step),
                competitors=competitors,
                random_seed=seed_val,
            )
            try:
                with st.spinner("Running Monte Carlo simulation…"):
                    st.session_state.analysis = analyze_scenario(st.session_state.history, scenario)
                    st.session_state.scenario = scenario
            except ValueError as exc:
                with right:
                    st.error(str(exc))

# ═══════════════════════════════════════════════════════════════════
# RIGHT PANEL — RESULTS
# ═══════════════════════════════════════════════════════════════════

with right:
    history  = st.session_state.history
    analysis = st.session_state.analysis
    scenario = st.session_state.scenario

    # ── Historical data summary ──────────────────────────────────
    if history is not None:
        summary = summarize_history(history)
        st.markdown('<div class="panel-title">📊 Historical Data</div>', unsafe_allow_html=True)
        hc1, hc2, hc3, hc4 = st.columns(4)
        hc1.metric("Historical bids",    int(summary["historical_rows"]))
        hc2.metric("Won projects",        int(summary["won_project_rows"]))
        hc3.metric("Avg bid-prep cost",  f"${summary['mean_prep_cost']:,.0f}")
        hc4.metric("Avg project cost",   f"${summary['mean_project_cost']:,.0f}")
        st.caption(
            f"Source: `{history.source_name}`. Blank completion-cost cells are treated as "
            "lost bids and excluded from project-cost resampling."
        )
        st.markdown("<br>", unsafe_allow_html=True)

    # ── No data yet ──────────────────────────────────────────────
    if history is None:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    height:420px;background:white;border:2px dashed #e2e8f0;border-radius:15px;">
            <div style="font-size:64px;margin-bottom:16px">🏗️</div>
            <div style="font-size:20px;font-weight:800;color:#1e3a8a;margin-bottom:8px;font-family:'Outfit',sans-serif">
                Ready for Analysis</div>
            <div style="font-size:14px;color:#64748b;text-align:center;max-width:360px;line-height:1.7">
                <strong>Step 1</strong>: Upload your historical cost CSV or load the example<br>
                <strong>Step 2–4</strong>: Configure constraints, competitors, and simulation<br>
                <strong>Run Analysis</strong>: Find your optimal bid
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Analysis not yet run ─────────────────────────────────────
    elif analysis is None:
        st.info("Data loaded. Adjust the inputs on the left, then click **▶ Run Analysis**.")

    # ── Analysis results ─────────────────────────────────────────
    else:
        best_bid  = analysis.best_bid
        best      = analysis.best_result
        win_rate  = best.win_probability
        mean_prof = best.expected_profit
        prob_loss = best.probability_of_loss

        # Recommendation banner
        if analysis.recommendation == "No Bid":
            banner_cls = "red"
            banner_title = "⚠️ No Bid Recommended"
            banner_body = (
                f"No candidate bid produces a positive expected profit. "
                f"Best expected profit is <strong>${mean_prof:,.0f}</strong> — "
                "consider widening the bid range or re-examining cost assumptions."
            )
        elif prob_loss > 0.30:
            banner_cls = "amber"
            banner_title = f"⚡ Bid at ${best_bid:,.0f} — Elevated Risk"
            banner_body = (
                f"Expected profit <strong>${mean_prof:,.0f}</strong> · "
                f"Win probability <strong>{win_rate*100:.1f}%</strong> · "
                f"Loss probability <strong>{prob_loss*100:.1f}%</strong> — higher than 30%, proceed with caution."
            )
        else:
            banner_cls = "green"
            banner_title = f"💡 Bid at ${best_bid:,.0f}"
            banner_body = (
                f"Expected profit <strong>${mean_prof:,.0f}</strong> · "
                f"Win probability <strong>{win_rate*100:.1f}%</strong> · "
                f"Loss probability <strong>{prob_loss*100:.1f}%</strong>."
            )

        st.markdown(f"""
        <div class="rec-banner {banner_cls}">
            <div class="rec-title">{banner_title}</div>
            <div class="rec-body">{banner_body}</div>
        </div>""", unsafe_allow_html=True)

        # Key metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Optimal Bid",      f"${best_bid:,.0f}")
        c2.metric("Expected Profit",  f"${mean_prof:,.0f}")
        c3.metric("Win Probability",  f"{win_rate*100:.1f}%")
        c4.metric("Loss Probability", f"{prob_loss*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Result tabs ────────────────────────────────────────────
        t1, t2, t3, t4, t5 = st.tabs([
            "📈 Profit Analysis",
            "🎯 Bid Optimizer",
            "📉 Risk Curves",
            "📊 Historical Costs",
            "📋 Full Scan Table",
        ])

        # ── TAB 1: Profit Analysis ─────────────────────────────────
        with t1:
            profit = np.array(best.simulated_profits)

            dc1, dc2, dc3 = st.columns(3)
            with dc1: show_hist_p = st.checkbox("Histogram",      value=True,  key="ph")
            with dc2: show_kde_p  = st.checkbox("Smoothed (KDE)", value=True,  key="pk")
            with dc3: show_norm_p = st.checkbox("Normal overlay",  value=False, key="pn")

            fig_p = go.Figure()
            if show_hist_p:
                fig_p.add_trace(go.Histogram(
                    x=profit, nbinsx=60, name="All outcomes",
                    histnorm="probability density",
                    marker=dict(color="#1e88e5", opacity=0.6, line=dict(color="white", width=0.3)),
                    hovertemplate="Profit: $%{x:,.0f}<extra></extra>",
                ))
            if show_kde_p:
                kde_x = np.linspace(profit.min(), profit.max(), 400)
                kde   = stats.gaussian_kde(profit)
                fig_p.add_trace(go.Scatter(
                    x=kde_x, y=kde(kde_x), mode="lines", name="Smoothed (KDE)",
                    line=dict(color="#1e3a8a", width=2.5),
                ))
            if show_norm_p:
                mu, sg = profit.mean(), profit.std()
                xn = np.linspace(profit.min(), profit.max(), 300)
                fig_p.add_trace(go.Scatter(
                    x=xn, y=stats.norm.pdf(xn, mu, sg), mode="lines", name="Normal",
                    line=dict(color="#ef4444", width=1.8, dash="dash"),
                ))
            fig_p.add_vline(x=0, line=dict(color="#ef4444", width=1.5, dash="dot"),
                            annotation_text=" Break-even",
                            annotation_font=dict(color="#ef4444", size=11, family="Outfit"))
            fig_p.add_vline(x=float(profit.mean()), line=dict(color="#1e3a8a", width=2, dash="dash"),
                            annotation_text=f" Mean=${profit.mean():,.0f}",
                            annotation_font=dict(color="#1e3a8a", size=11, family="Outfit"))
            fig_p.update_layout(**PLOT, height=300,
                title=dict(text=f"Profit Distribution at Optimal Bid ${best_bid:,.0f}", font=dict(size=13, color="#1e3a8a")),
                xaxis=dict(title="Profit ($)", tickprefix="$", **NGRID, **TICK),
                yaxis=dict(title="Density", **GRID, **TICK),
                legend=dict(font=dict(size=11)))
            st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False})

            # Profit breakdown rows
            st.markdown('<div class="sec-lbl">Profit Percentiles at Optimal Bid</div>', unsafe_allow_html=True)
            for lbl, val, note, fg in [
                ("Expected profit (mean)", f"${profit.mean():,.0f}", "", "#1e3a8a"),
                ("P5 profit",  f"${best.profit_p5:,.0f}",  "5% of outcomes fall below this", "#ef4444"),
                ("Median (P50)",f"${best.profit_p50:,.0f}", "",                              "#475569"),
                ("P95 profit", f"${best.profit_p95:,.0f}", "Only 5% of outcomes exceed this","#22c55e"),
            ]:
                note_html = f'<div style="font-size:11px;color:#94a3b8;margin-top:2px">{note}</div>' if note else ""
                st.markdown(f"""
                <div class="brow">
                    <div><div>{lbl}</div>{note_html}</div>
                    <span style="color:{fg};font-family:'JetBrains Mono',monospace;font-weight:700;font-size:14px">{val}</span>
                </div>""", unsafe_allow_html=True)

        # ── TAB 2: Bid Optimizer ───────────────────────────────────
        with t2:
            bid_amounts   = [r.bid_amount        for r in analysis.candidate_results]
            mean_profits  = [r.expected_profit   for r in analysis.candidate_results]
            p5_profits    = [r.profit_p5         for r in analysis.candidate_results]
            p95_profits   = [r.profit_p95        for r in analysis.candidate_results]

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=bid_amounts, y=p95_profits,
                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig2.add_trace(go.Scatter(
                x=bid_amounts, y=p5_profits,
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(30,136,229,0.07)",
                name="P5–P95 range", hoverinfo="skip"))
            fig2.add_trace(go.Scatter(
                x=bid_amounts, y=mean_profits,
                mode="lines+markers", line=dict(color="#1e88e5", width=3),
                marker=dict(size=5), name="Expected Profit",
                hovertemplate="Bid: $%{x:,.0f}<br>Profit: $%{y:,.0f}<extra></extra>"))
            fig2.add_hline(y=0, line=dict(color="#ef4444", width=1.5, dash="dot"),
                           annotation_text=" Break-even",
                           annotation_font=dict(color="#ef4444", size=11, family="Outfit"))
            fig2.add_vline(x=best_bid, line=dict(color="#22c55e", width=2),
                           annotation_text=f" Optimal ${best_bid:,.0f}",
                           annotation_font=dict(color="#22c55e", size=12, family="Outfit"))
            fig2.update_layout(**PLOT, height=320,
                title=dict(text="Expected Profit vs Bid Amount", font=dict(size=13, color="#1e3a8a")),
                xaxis=dict(title="Bid Amount ($)", tickprefix="$", **NGRID, **TICK),
                yaxis=dict(title="Expected Profit ($)", tickprefix="$", **GRID, **TICK),
                legend=dict(font=dict(size=11)))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

            # Top 8 candidates table
            st.markdown('<div class="sec-lbl">Top Candidate Bids by Expected Profit</div>', unsafe_allow_html=True)
            top_results = sorted(analysis.candidate_results, key=lambda r: r.expected_profit, reverse=True)[:8]
            for r in top_results:
                is_best = r.bid_amount == best_bid
                fg = "#22c55e" if is_best else "#1e3a8a"
                label = f"${r.bid_amount:,.0f}" + (" ← optimal" if is_best else "")
                st.markdown(f"""
                <div class="brow">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:13px;color:{fg};font-weight:700">{label}</span>
                    <span style="font-size:12px;color:#475569">
                        E[profit] <strong>${r.expected_profit:,.0f}</strong> ·
                        win {r.win_probability*100:.1f}% ·
                        loss {r.probability_of_loss*100:.1f}%
                    </span>
                </div>""", unsafe_allow_html=True)

        # ── TAB 3: Risk Curves ─────────────────────────────────────
        with t3:
            win_probs  = [r.win_probability    for r in analysis.candidate_results]
            loss_probs = [r.probability_of_loss for r in analysis.candidate_results]

            fig3 = make_subplots(rows=1, cols=2, subplot_titles=["Win Probability (%)", "Loss Probability (%)"])
            fig3.add_trace(go.Scatter(
                x=bid_amounts, y=[w * 100 for w in win_probs],
                mode="lines", line=dict(color="#22c55e", width=2.5), name="Win Rate",
                hovertemplate="Bid: $%{x:,.0f}<br>Win: %{y:.1f}%<extra></extra>"),
                row=1, col=1)
            fig3.add_trace(go.Scatter(
                x=bid_amounts, y=[l * 100 for l in loss_probs],
                mode="lines", line=dict(color="#ef4444", width=2.5), name="Loss Prob",
                hovertemplate="Bid: $%{x:,.0f}<br>Loss: %{y:.1f}%<extra></extra>"),
                row=1, col=2)
            for col in [1, 2]:
                fig3.add_vline(x=best_bid, line=dict(color="#22c55e", width=1.5, dash="dash"), row=1, col=col)
            fig3.update_layout(
                paper_bgcolor="white", plot_bgcolor="white",
                font=dict(family="Outfit", color="#475569", size=11),
                margin=dict(l=10, r=10, t=40, b=40), height=300, showlegend=False,
            )
            fig3.update_xaxes(showgrid=False, tickprefix="$",
                              tickfont=dict(family="JetBrains Mono", size=10))
            fig3.update_yaxes(showgrid=True, gridcolor="#f1f5f9",
                              tickfont=dict(family="JetBrains Mono", size=10))
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

            st.markdown('<div class="sec-lbl">Interpretation</div>', unsafe_allow_html=True)
            for lbl, val, fg in [
                ("Win probability at optimal bid",   f"{win_rate*100:.1f}%",  "#22c55e"),
                ("Loss probability at optimal bid",  f"{prob_loss*100:.1f}%", "#ef4444"),
                ("Expected revenue at optimal bid",  f"${best.expected_revenue:,.0f}",    "#1e88e5"),
                ("Expected total cost at optimal bid", f"${best.expected_total_cost:,.0f}", "#f59e0b"),
            ]:
                st.markdown(f"""
                <div class="brow">
                    <span>{lbl}</span>
                    <span style="color:{fg};font-family:'JetBrains Mono',monospace;font-weight:700;font-size:14px">{val}</span>
                </div>""", unsafe_allow_html=True)

        # ── TAB 4: Historical Costs ────────────────────────────────
        with t4:
            prep_arr    = np.array(history.prep_costs)
            project_arr = np.array(history.project_costs)

            hc1, hc2 = st.columns(2)

            with hc1:
                fig_prep = go.Figure()
                fig_prep.add_trace(go.Histogram(
                    x=prep_arr, nbinsx=30, name="Bid prep cost",
                    histnorm="probability density",
                    marker=dict(color="#8b5cf6", opacity=0.7, line=dict(color="white", width=0.3)),
                    hovertemplate="Prep cost: $%{x:,.0f}<extra></extra>",
                ))
                if len(prep_arr) > 5:
                    kd = stats.gaussian_kde(prep_arr)
                    xk = np.linspace(prep_arr.min(), prep_arr.max(), 300)
                    fig_prep.add_trace(go.Scatter(
                        x=xk, y=kd(xk), mode="lines", name="KDE",
                        line=dict(color="#4c1d95", width=2.5),
                    ))
                fig_prep.update_layout(**PLOT, height=280,
                    title=dict(text="Bid Preparation Cost", font=dict(size=12, color="#1e3a8a")),
                    xaxis=dict(title="Cost ($)", tickprefix="$", **NGRID, **TICK),
                    yaxis=dict(title="Density", **GRID, **TICK),
                    legend=dict(font=dict(size=10)))
                st.plotly_chart(fig_prep, use_container_width=True, config={"displayModeBar": False})
                s1, s2 = st.columns(2)
                s1.metric("Mean prep cost",   f"${prep_arr.mean():,.0f}")
                s2.metric("Median prep cost", f"${float(np.median(prep_arr)):,.0f}")

            with hc2:
                fig_proj = go.Figure()
                fig_proj.add_trace(go.Histogram(
                    x=project_arr, nbinsx=30, name="Project completion cost",
                    histnorm="probability density",
                    marker=dict(color="#1e88e5", opacity=0.7, line=dict(color="white", width=0.3)),
                    hovertemplate="Project cost: $%{x:,.0f}<extra></extra>",
                ))
                if len(project_arr) > 5:
                    kd2 = stats.gaussian_kde(project_arr)
                    xk2 = np.linspace(project_arr.min(), project_arr.max(), 300)
                    fig_proj.add_trace(go.Scatter(
                        x=xk2, y=kd2(xk2), mode="lines", name="KDE",
                        line=dict(color="#1e3a8a", width=2.5),
                    ))
                fig_proj.add_vline(
                    x=float(project_cost_floor),
                    line=dict(color="#ef4444", width=1.5, dash="dash"),
                    annotation_text=" Floor",
                    annotation_font=dict(color="#ef4444", size=10, family="Outfit"),
                )
                fig_proj.update_layout(**PLOT, height=280,
                    title=dict(text="Project Completion Cost (won bids only)", font=dict(size=12, color="#1e3a8a")),
                    xaxis=dict(title="Cost ($)", tickprefix="$", **NGRID, **TICK),
                    yaxis=dict(title="Density", **GRID, **TICK),
                    legend=dict(font=dict(size=10)))
                st.plotly_chart(fig_proj, use_container_width=True, config={"displayModeBar": False})
                s3, s4 = st.columns(2)
                s3.metric("Mean project cost",   f"${project_arr.mean():,.0f}")
                s4.metric("Median project cost", f"${float(np.median(project_arr)):,.0f}")

        # ── TAB 5: Full Scan Table ─────────────────────────────────
        with t5:
            import pandas as pd

            scan_rows = [
                {
                    "Bid Amount":         r.bid_amount,
                    "Expected Profit":    round(r.expected_profit, 2),
                    "Win Probability (%)": round(r.win_probability * 100, 1),
                    "Loss Probability (%)":round(r.probability_of_loss * 100, 1),
                    "P5 Profit":          round(r.profit_p5, 2),
                    "Median Profit":      round(r.profit_p50, 2),
                    "P95 Profit":         round(r.profit_p95, 2),
                    "Expected Revenue":   round(r.expected_revenue, 2),
                    "Expected Total Cost":round(r.expected_total_cost, 2),
                }
                for r in analysis.candidate_results
            ]
            scan_df = pd.DataFrame(scan_rows).sort_values("Expected Profit", ascending=False)
            st.markdown('<div class="sec-lbl">All Candidate Bids — Sorted by Expected Profit</div>', unsafe_allow_html=True)
            st.dataframe(scan_df, use_container_width=True, hide_index=True)

        # ── Download ───────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        export_csv = build_export_csv(analysis, scenario, history)
        st.download_button(
            "⬇ Download summary CSV",
            data=export_csv,
            file_name="rfp_bid_analysis_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.markdown("---")
st.caption(
    "Model: Monte Carlo simulation · Bootstrap resampling of historical costs · "
    "Triangular competitor bid distribution · Tie-split win-share accounting"
)
