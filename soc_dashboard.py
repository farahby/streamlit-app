import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, json, requests
from supabase import create_client as _sb_create
from datetime import datetime
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def _get_supabase():
    url = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL", ""))
    key = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY", ""))
    if url and key:
        return _sb_create(url, key)
    return None

@st.cache_data(ttl=60)
def _load_feedback_from_supabase():
    sb = _get_supabase()
    if sb:
        res = sb.table("analyst_feedback").select("*").order("feedback_time", desc=True).execute()
        return pd.DataFrame(res.data)
    return pd.DataFrame()

@st.cache_data(ttl=30)
def _load_queue_from_supabase():
    sb = _get_supabase()
    if sb:
        try:
            res = sb.table("review_queue").select("*").execute()
            return pd.DataFrame(res.data)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

# CONFIG
st.set_page_config(page_title="SOC AI Platform", layout="wide", page_icon="")

BASE_DIR    = os.environ.get("SOC_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
API_URL     = os.environ.get("SOC_API_URL",  "http://localhost:8000")

SCORED_CSV   = os.path.join(BASE_DIR, "normalized_alerts", "scored_findings.csv")
ALL_CSV      = os.path.join(BASE_DIR, "normalized_alerts", "all_findings.csv")
TRENDS_LOG   = os.path.join(BASE_DIR, "logs", "risk_trends.jsonl")
FEEDBACK     = os.path.join(BASE_DIR, "feedback", "analyst_labels.csv")
FP_LOG       = os.path.join(BASE_DIR, "feedback", "false_positives.csv")
REVIEW_PATH  = os.path.join(BASE_DIR, "feedback", "review_queue.csv")
AGENTS_JSON  = os.path.join(BASE_DIR, "reports", "agent_results.json")
PR_MD        = os.path.join(BASE_DIR, "remediation", "pr_templates.md")

COLORS = {
    "CRITICAL": "#d32f2f",
    "HIGH":     "#f57c00",
    "MEDIUM":   "#fbc02d",
    "LOW":      "#388e3c",
    "UNKNOWN":  "#757575",
}

PRIORITY_COLORS = {
    "P0_FIX_NOW": "#7b1fa2",
    "P1_24H":     "#d32f2f",
    "P2_72H":     "#f57c00",
    "P3_7D":      "#fbc02d",
    "P4_30D":     "#66bb6a",
    "P5_BACKLOG": "#90a4ae",
}

PRIORITY_LABELS = {
    "P0_FIX_NOW": "Fix NOW (Emergency)",
    "P1_24H":     "Within 24 hours",
    "P2_72H":     "Within 72 hours",
    "P3_7D":      "Within 7 days",
    "P4_30D":     "Within 30 days",
    "P5_BACKLOG": "Backlog",
}

ROLES = {
    "soc_analyst":    {"read": True,  "write": False, "admin": False},
    "senior_analyst": {"read": True,  "write": True,  "admin": False},
    "soc_lead":       {"read": True,  "write": True,  "admin": True},
}

# HELPER: render a dataframe as a matplotlib table (localtunnel-safe)
def render_table(df_show, title="", figsize=None, max_rows=500):
    """Render a real interactive Streamlit dataframe (sortable, scrollable, crisp)."""
    df_show = df_show.fillna("")
    if title:
        st.markdown(f"**{title}**")
    if len(df_show) == 0:
        st.info("No data to display.")
        return
    if len(df_show) > max_rows:
        st.caption(f"Showing first {max_rows} of {len(df_show)} rows")
        df_show = df_show.head(max_rows)
    st.dataframe(df_show, use_container_width=True, hide_index=True)

@st.cache_data
def load_agent_results():
    """Load LLM multi-agent results from JSON.
    Handles both formats produced by Layer 7:
      - dict: {finding_id: {severity, triage, compliance, ...}}
      - list: [{finding_id: str, severity: str, triage: {}, ...}, ...]
    Always returns a dict keyed by finding_id.
    """
    if not os.path.exists(AGENTS_JSON):
        return {}
    try:
        with open(AGENTS_JSON) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            result = {}
            for item in data:
                if not isinstance(item, dict):
                    continue
                fid = str(item.get("finding_id") or item.get("id") or item.get("cve_id") or "unknown")
                result[fid] = item
            return result
    except Exception:
        pass
    return {}

@st.cache_data
def load_scored_csv(path, mtime):
    return pd.read_csv(path)

# SIDEBAR
with st.sidebar:
    st.title("SOC AI Platform")
    st.markdown("---")
    tenant_id = "default"  # multi-tenancy not claimed; constant keeps the Supabase column valid
    role      = st.selectbox("Role",   ["soc_analyst", "senior_analyst", "soc_lead"])
    access_label = (
        "Read-Only"   if role == "soc_analyst"   else
        "Read+Write"  if role == "senior_analyst" else
        "Admin"
    )
    st.markdown(f"**Access:** {access_label}")
    if role in ("senior_analyst", "soc_lead"):
        # Password is read from Streamlit secrets / env, never hardcoded.
        # Set DASHBOARD_WRITE_PASSWORD in .streamlit/secrets.toml or the
        # environment. If unset, write roles are disabled (fail closed).
        _expected_pwd = st.secrets.get(
            "DASHBOARD_WRITE_PASSWORD",
            os.environ.get("DASHBOARD_WRITE_PASSWORD", ""),
        )
        pwd = st.text_input("Password", type="password", key="role_pwd")
        if not _expected_pwd:
            st.sidebar.error(
                "Write access not configured. Set DASHBOARD_WRITE_PASSWORD "
                "in Streamlit secrets to enable senior_analyst / soc_lead roles."
            )
            st.stop()
        if pwd != _expected_pwd:
            st.sidebar.error("Incorrect password - access denied")
            st.stop()
    st.markdown("**Logged in as:**")
    st.info(role)
    st.markdown("---")

    if os.path.exists(SCORED_CSV):
        _df_check = pd.read_csv(SCORED_CSV)
        st.success(f"{len(_df_check)} findings scored")
    else:
        st.error("Run pipeline first")
    st.caption(f"{datetime.now().strftime('%H:%M:%S')}")

# GUARD
if not os.path.exists(SCORED_CSV):
    st.error("No data found. Run the Colab pipeline first (Layers 1-9), then launch this dashboard.")
    st.markdown("""
    **Quick start:**
    ```bash
    !streamlit run /content/SOC-Audit/soc_dashboard.py &
    !npx localtunnel --port 8501
    ```
    """)
    st.stop()

df = load_scored_csv(SCORED_CSV, os.path.getmtime(SCORED_CSV))

# KPI HELPERS
total         = len(df)
n_critical    = int((df["severity"] == "CRITICAL").sum())
n_high        = int((df["severity"] == "HIGH").sum())
n_medium      = int((df["severity"] == "MEDIUM").sum())
n_low         = int((df["severity"] == "LOW").sum())
n_kev         = int(df["in_kev"].sum())          if "in_kev"          in df.columns else 0
n_exploit     = int(df["has_exploit"].sum())      if "has_exploit"     in df.columns else 0
n_inet        = int(df["internet_facing"].sum())  if "internet_facing" in df.columns else 0
n_p0          = int((df.get("priority", pd.Series([""] * total)) == "P0_FIX_NOW").sum())
n_p1          = int((df.get("priority", pd.Series([""] * total)) == "P1_24H").sum())
avg_risk      = df["risk_score"].mean() if "risk_score" in df.columns else 0.0
n_blast       = int(df["in_blast_radius"].sum()) if "in_blast_radius" in df.columns else 0
n_uncertainty = int((df["model_uncertainty"] > 20).sum()) if "model_uncertainty" in df.columns else 0

# HEADER
st.title("SOC AI Platform Enterprise")
st.caption(f"12-Layer Decision Engine | Trivy Snyk Semgrep ZAP Nikto Grype Gitleaks | Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Emergency alert banner
if n_p0 > 0:
    st.error(f"P0 EMERGENCY {n_p0} finding(s) require IMMEDIATE action!")
elif n_p1 > 0:
    st.warning(f"{n_p1} finding(s) due within 24 hours act now!")

# KPI row
c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
c1.metric("Total Findings",     total)
c2.metric("Critical",        n_critical)
c3.metric("High",            n_high)
c4.metric("CISA KEV",        n_kev)
c5.metric("Has Exploit",     n_exploit)
c6.metric("Internet Facing", n_inet)
c7.metric("P0/P1 Actions",   n_p0 + n_p1)
c8.metric("Avg Risk Score",     f"{avg_risk:.1f}")

st.divider()

# TABS
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15 = st.tabs([
    "Overview",
    "Risk Board",
    "KEV Tracker",
    "Trends and Graph",
    "LLM Agent Results",
    "Analyst Queue",
    "AI Triage Intelligence",
    "Remediation and Compliance",
    "Advanced Analytics",
    "System Health",
    "Ablation Study",
    "Architecture",
    "System Card",
    "Compliance Gaps",
    "Assistant SOC",
])

# ── shared report loader (used by analytics / health / new tabs) ──
import json as _vjson
_REPORTS = os.path.join(BASE_DIR, "reports")
def _vload(name, base=None):
    p = os.path.join(base or _REPORTS, name)
    if os.path.exists(p):
        try:
            with open(p) as _fh:
                return _vjson.load(_fh)
        except Exception:
            return None
    return None


# TAB 1 OVERVIEW
with tab1:
    st.header("Executive Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Severity Distribution")
        sev = df["severity"].value_counts().reindex(["CRITICAL","HIGH","MEDIUM","LOW"], fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.bar(sev.index, sev.values, color=[COLORS.get(s, "#757575") for s in sev.index], edgecolor="white")
        for b, v in zip(bars, sev.values):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3, str(v), ha="center", fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Risk Score Heatmap (Tool x Severity)")
        tools  = df["tool"].unique().tolist()
        sevs   = ["CRITICAL","HIGH","MEDIUM","LOW"]
        matrix = np.zeros((len(tools), len(sevs)), dtype=int)
        for ti, t in enumerate(tools):
            for si, s in enumerate(sevs):
                matrix[ti, si] = int(((df["tool"] == t) & (df["severity"] == s)).sum())
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        im = ax2.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax2.set_xticks(range(len(sevs)));  ax2.set_xticklabels(sevs, fontsize=8)
        ax2.set_yticks(range(len(tools))); ax2.set_yticklabels([t[:15] for t in tools], fontsize=7)
        for ti in range(len(tools)):
            for si in range(len(sevs)):
                if matrix[ti, si]:
                    ax2.text(si, ti, str(matrix[ti, si]), ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax2, shrink=0.8)
        st.pyplot(fig2)
        plt.close(fig2)

    col3, col4 = st.columns(2)

    with col3:
        if "priority" in df.columns:
            st.subheader("Priority Fix Queue (Decision Engine)")
            pq = df["priority"].value_counts().reindex(list(PRIORITY_LABELS.keys()), fill_value=0)
            fig3, ax3 = plt.subplots(figsize=(8, 2.5))
            p_colors = [PRIORITY_COLORS.get(p, "#90a4ae") for p in pq.index]
            bars3 = ax3.barh(
                [PRIORITY_LABELS.get(p, p) for p in pq.index],
                pq.values,
                color=p_colors,
                edgecolor="white"
            )
            for b, v in zip(bars3, pq.values):
                if v:
                    ax3.text(v + 0.1, b.get_y() + b.get_height()/2, str(v), va="center", fontweight="bold")
            ax3.spines[["top","right"]].set_visible(False)
            ax3.set_xlabel("Findings")
            st.pyplot(fig3)
            plt.close(fig3)

    with col4:
        st.subheader("Exploitability Signals")
        signal_counts = {
            "KEV":            n_kev,
            "Has Exploit":    n_exploit,
            "Internet Facing":n_inet,
            "Blast Radius":   n_blast,
        }
        if "requires_auth" in df.columns:
            signal_counts["No Auth\nRequired"] = int((df["requires_auth"] == 0).sum())
        fig_sig, ax_sig = plt.subplots(figsize=(6, 3))
        ax_sig.bar(list(signal_counts.keys()), list(signal_counts.values()),
                   color=["#d32f2f","#e65100","#f57c00","#827717","#fbc02d"], edgecolor="white")
        ax_sig.spines[["top","right"]].set_visible(False)
        ax_sig.set_ylabel("Count")
        st.pyplot(fig_sig)
        plt.close(fig_sig)

    col6, = st.columns(1)
    with col6:
        if "confidence_score" in df.columns:
            st.subheader("Confidence Scores")
            conf_data = pd.to_numeric(df["confidence_score"], errors="coerce").dropna()
            fig_conf, ax_conf = plt.subplots(figsize=(5, 2.5))
            ax_conf.hist(conf_data, bins=10, color="#00897b", edgecolor="white", alpha=0.85)
            ax_conf.axvline(0.7, color="red", linestyle="--", lw=1.5, label="High confidence (0.7)")
            ax_conf.legend(fontsize=8)
            ax_conf.set_xlabel("Confidence Score")
            ax_conf.set_ylabel("Findings")
            ax_conf.spines[["top","right"]].set_visible(False)
            st.pyplot(fig_conf)
            plt.close(fig_conf)

    if "epss_score" in df.columns:
        st.subheader("EPSS Score Distribution")
        epss_data = pd.to_numeric(df["epss_score"], errors="coerce").dropna()
        fig4, ax4 = plt.subplots(figsize=(10, 2.5))
        ax4.hist(epss_data, bins=20, color="#0288d1", edgecolor="white", alpha=0.85)
        ax4.axvline(0.5, color="red", linestyle="--", lw=1.5, label="High risk (EPSS > 0.5)")
        ax4.axvline(0.7, color="darkred", linestyle="--", lw=1.5, label="High probability (EPSS > 0.7)")
        ax4.legend(fontsize=8)
        ax4.spines[["top","right"]].set_visible(False)
        ax4.set_xlabel("EPSS Score")
        ax4.set_ylabel("Findings")
        st.pyplot(fig4)
        plt.close(fig4)


# TAB 2 RISK BOARD
with tab2:
    st.header("Risk Board")

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    with col_f1:
        sev_filter = st.radio(
            "Severity",
            ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
            index=0, horizontal=True,
        )
    with col_f2:
        tool_options = ["ALL"] + sorted(df["tool"].dropna().unique().tolist())
        tool_filter  = st.selectbox("Tool", tool_options)
    with col_f3:
        priority_options = ["ALL"] + list(PRIORITY_LABELS.keys())
        priority_filter  = st.selectbox("Priority", priority_options)
    with col_f4:
        kev_filter  = st.radio("KEV Only",           ["No", "Yes"], horizontal=True)
        inet_filter = st.radio("Internet-Facing Only",["No", "Yes"], horizontal=True)

    if "asset_env" in df.columns:
        env_options = ["ALL"] + sorted(df["asset_env"].dropna().unique().tolist())
        env_filter  = st.selectbox("Asset Environment", env_options)
    else:
        env_filter = "ALL"

    filtered = df.copy()
    if sev_filter != "ALL":
        filtered = filtered[filtered["severity"] == sev_filter]
    if tool_filter != "ALL":
        filtered = filtered[filtered["tool"] == tool_filter]
    if priority_filter != "ALL" and "priority" in filtered.columns:
        filtered = filtered[filtered["priority"] == priority_filter]
    if kev_filter == "Yes" and "in_kev" in filtered.columns:
        filtered = filtered[filtered["in_kev"] == 1]
    if inet_filter == "Yes" and "internet_facing" in filtered.columns:
        filtered = filtered[filtered["internet_facing"] == 1]
    if env_filter != "ALL" and "asset_env" in filtered.columns:
        filtered = filtered[filtered["asset_env"] == env_filter]

    # NO CAP: show ALL findings
    st.caption(f"Showing all {len(filtered)} findings")

    show_cols = [c for c in [
        "id", "tool", "severity", "risk_score", "priority",
        "epss_score", "asset_criticality", "asset_env", "asset_owner",
        "in_kev", "has_exploit", "internet_facing", "in_blast_radius",
        "confidence_score", "tool_count", "fix_version", "title"
    ] if c in filtered.columns]

    render_table(
        filtered[show_cols].sort_values("risk_score", ascending=False),
        title=f"All Findings ({len(filtered)}) sorted by risk score",
    )

    # Top 30 by risk score as a compact, readable chart
    st.subheader("Top 30 Findings by Risk Score")
    top_all = filtered.sort_values("risk_score", ascending=False) if "risk_score" in filtered.columns else filtered
    top30 = top_all.head(30)
    if not top30.empty:
        fig5, ax5 = plt.subplots(figsize=(11, max(4, len(top30) * 0.32)))
        bar_colors = [COLORS.get(s, "#757575") for s in top30["severity"]]
        ax5.barh(range(len(top30)), top30["risk_score"], color=bar_colors, edgecolor="white")
        labels = [f"{str(row['id'])[:26]} [{row.get('priority','?')}]" for _, row in top30.iterrows()]
        ax5.set_yticks(range(len(top30)))
        ax5.set_yticklabels(labels, fontsize=7)
        ax5.set_xlabel("Risk Score")
        ax5.invert_yaxis()
        ax5.axvline(70, color="red", linestyle="--", lw=1, label="Critical (70)")
        ax5.axvline(50, color="orange", linestyle="--", lw=1, label="High (50)")
        ax5.legend(fontsize=8)
        ax5.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

# TAB 3 KEV TRACKER
with tab3:
    st.header("CISA KEV Tracker")

    if "in_kev" not in df.columns or df["in_kev"].sum() == 0:
        st.info("No KEV matches in current scan. Run Layer 3 (Threat Intel) to fetch live CISA KEV feed.")
        st.markdown("""
        **What is CISA KEV?**
        The CISA Known Exploited Vulnerabilities catalog lists CVEs confirmed to be actively exploited in the wild.
        All KEV findings are auto-escalated to CRITICAL in Layer 3.
        """)
    else:
        kev_df = df[df["in_kev"] == 1].copy()
        st.error(f"{len(kev_df)} findings are ACTIVELY EXPLOITED in the wild (CISA KEV)")

        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric("KEV Findings",       len(kev_df))
        kc2.metric("Internet Facing",    int(kev_df.get("internet_facing", pd.Series([0]*len(kev_df))).sum()))
        kc3.metric("Has Public Exploit", int(kev_df.get("has_exploit",     pd.Series([0]*len(kev_df))).sum()))
        kc4.metric("P1/P2 Priority",     int(kev_df.get("priority", pd.Series([""]*len(kev_df))).isin(["P1_24H","P2_72H"]).sum()))

        kev_cols = [c for c in [
            "id", "tool", "severity", "risk_score", "priority",
            "epss_score", "kev_ransomware", "kev_due_date",
            "internet_facing", "has_exploit", "asset_criticality", "title"
        ] if c in kev_df.columns]

        # ALL KEV findings no cap
        render_table(kev_df[kev_cols].sort_values("risk_score", ascending=False),
                     title=f"All KEV Matches ({len(kev_df)}) Act Immediately")


# TAB 4 TRENDS AND KNOWLEDGE GRAPH
with tab4:
    st.header("Risk Trends and Asset Intelligence")

    if os.path.exists(TRENDS_LOG):
        history = []
        with open(TRENDS_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        history.append(json.loads(line))
                    except Exception:
                        pass

        if len(history) >= 2:
            st.subheader("Risk Trends Over Pipeline Runs")
            th = pd.DataFrame(history)
            th["run"] = range(len(th))
            th["timestamp"] = pd.to_datetime(th["timestamp"])

            fig_t, axes = plt.subplots(2, 2, figsize=(14, 6))

            axes[0,0].plot(th["run"], th["critical_count"], "r-o", markersize=5, label="Critical")
            axes[0,0].plot(th["run"], th["high_count"] if "high_count" in th.columns else [0]*len(th), "o--", color="#f57c00", markersize=4, label="High")
            axes[0,0].set_title("Critical + High Count Over Runs", fontweight="bold")
            axes[0,0].legend(fontsize=8)
            axes[0,0].spines[["top","right"]].set_visible(False)

            axes[0,1].plot(th["run"], th["kev_count"], "k--s", markersize=5, label="KEV")
            axes[0,1].plot(th["run"], th["has_exploit"] if "has_exploit" in th.columns else [0]*len(th), "r-^", markersize=4, label="Has Exploit")
            axes[0,1].set_title("KEV and Exploit Exposure", fontweight="bold")
            axes[0,1].legend(fontsize=8)
            axes[0,1].spines[["top","right"]].set_visible(False)

            axes[1,0].plot(th["run"], th["avg_risk_score"], "b-o", markersize=5)
            axes[1,0].axhline(70, color="red", linestyle="--", lw=1, label="Critical threshold")
            axes[1,0].axhline(50, color="orange", linestyle="--", lw=1, label="High threshold")
            axes[1,0].set_title("Avg Risk Score Over Runs", fontweight="bold")
            axes[1,0].legend(fontsize=8)
            axes[1,0].spines[["top","right"]].set_visible(False)

            if "p1_24h" in th.columns and "p0_fix_now" in th.columns:
                axes[1,1].bar(th["run"], th["p1_24h"], color="#d32f2f", label="P1 (24h)")
                axes[1,1].bar(th["run"], th["p0_fix_now"] if "p0_fix_now" in th.columns else [0]*len(th),
                              bottom=th["p1_24h"], color="#7b1fa2", label="P0 (NOW)")
                axes[1,1].set_title("P0/P1 Action Items", fontweight="bold")
                axes[1,1].legend(fontsize=8)
                axes[1,1].spines[["top","right"]].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig_t)
            plt.close(fig_t)

            if len(history) >= 2:
                curr = history[-1]
                prev = history[-2]
                delta_crit = curr.get("critical_count", 0) - prev.get("critical_count", 0)
                delta_kev  = curr.get("kev_count", 0) - prev.get("kev_count", 0)
                trend = "WORSE" if delta_crit > 0 else ("IMPROVING" if delta_crit < 0 else "STABLE")
                tc1, tc2, tc3 = st.columns(3)
                tc1.metric("Critical delta vs last run", f"{delta_crit:+d}")
                tc2.metric("KEV delta vs last run",      f"{delta_kev:+d}")
                tc3.metric("Trend", trend)
        else:
            st.info("Run the pipeline multiple times to see trend data. Only 1 snapshot so far.")
    else:
        st.info("Run Layer 9 (Risk Trends) in the notebook to generate trend data.")

    st.divider()


    if "asset_owner" in df.columns:
        st.subheader("Findings by Asset Owner All Owners")
        owner_df = df.groupby("asset_owner").agg(
            total=("id","count"),
            critical=("severity", lambda x: int((x == "CRITICAL").sum())),
            avg_risk=("risk_score","mean")
        )
        owner_df["avg_risk"] = owner_df["avg_risk"].round(1)
        owner_df = owner_df.sort_values("critical", ascending=False)
        # ALL owners no .head(10) cap
        render_table(owner_df.reset_index(), title="All Asset Owners by Critical Findings")

# TAB 5 LLM AGENT RESULTS
with tab5:
    st.header("Multi-Agent LLM Analysis (Tiered Triage, Compliance, Remediation, Risk Mgmt)")
    st.caption("CRITICAL/HIGH 4 agents full analysis | MEDIUM template | LOW minimal")

    agent_results = load_agent_results()

    if not agent_results:
        st.info("Run Layer 7 (Multi-Agent LLM Triage) in the notebook to generate AI analysis.")
        st.markdown("""
        **Layer 7 Strategy:**
        - CRITICAL / HIGH 4 separate agent calls (Triage, Compliance, Remediation, Risk Management)
        - MEDIUM template-based analysis (fast)
        - LOW minimal template
        - Drive cache reuse previous agent state on later runs
        """)
    else:
        total_analyzed = len(agent_results)
        urgency_immediate = sum(
            1 for r in agent_results.values()
            if isinstance(r, dict) and (r.get("triage") or {}).get("urgency") == "IMMEDIATE"
        )
        st.success(f"{total_analyzed} findings analyzed by LLM agents")

        ac1, ac2, ac3 = st.columns(3)
        ac1.metric("Findings Analyzed",   total_analyzed)
        ac2.metric("Urgent (IMMEDIATE)", urgency_immediate)
        ac3.metric("Risk Avoidance Strategy",
                   sum(1 for r in agent_results.values()
                       if isinstance(r, dict) and
                       (r.get("risk_management") or r.get("risk_mgmt") or {}).get("risk_strategy") == "AVOIDANCE"))

        sev_llm = st.radio(
            "Show findings",
            ["CRITICAL", "HIGH", "MEDIUM", "LOW", "ALL"],
            horizontal=True,
            index=0,
        )

        # ALL matching findings no shown >= 10 cap
        shown = 0
        for finding_id, result in agent_results.items():
            if not isinstance(result, dict):
                continue

            sev = str(result.get("severity", "?"))
            if sev_llm != "ALL" and sev != sev_llm:
                continue
            shown += 1

            triage      = result.get("triage", {})
            compliance  = result.get("compliance", {})
            remediation = result.get("remediation", {})
            risk_mgmt   = result.get("risk_management") or result.get("risk_mgmt") or {}
            if not isinstance(risk_mgmt, dict): risk_mgmt = {}
            border = COLORS.get(sev, "#555")

            st.markdown(f"""
<div style="border-left: 5px solid {border}; padding: 14px 18px; margin: 10px 0;
            background: #111827; border-radius: 6px; color: #e5e7eb;">
  <div style="font-size:14px; font-weight:700; margin-bottom:6px;">
    <span style="color:{border}">[{sev}]</span> &nbsp;{finding_id}
  </div>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; font-size:12px; margin-top:8px;">
    <div style="background:#1f2937; padding:10px; border-radius:4px;">
      <div style="color:#60a5fa; font-weight:700; margin-bottom:4px;">TRIAGE AGENT</div>
      <div>Urgency: <b style="color:#fca5a5">{triage.get('urgency','N/A')}</b></div>
      <div>Exploitability: {triage.get('exploitability','N/A')}/10</div>
      <div>Attack Vector: {triage.get('attack_vector','N/A')}</div>
      <div>Confidence: {triage.get('confidence','N/A')}</div>
      <div style="margin-top:4px; color:#9ca3af; font-size:11px;">{str(triage.get('rationale',''))[:200]}</div>
    </div>
    <div style="background:#1f2937; padding:10px; border-radius:4px;">
      <div style="color:#34d399; font-weight:700; margin-bottom:4px;">REMEDIATION AGENT</div>
      <div>{str(remediation.get('immediate_action','N/A'))[:200]}</div>
      <div style="margin-top:4px; color:#fbbf24; font-family:monospace; font-size:11px;">{str(remediation.get('patch_command',''))[:150]}</div>
      <div style="margin-top:2px; color:#9ca3af; font-size:11px;">Effort: {remediation.get('effort_hours','?')}h</div>
    </div>
    <div style="background:#1f2937; padding:10px; border-radius:4px;">
      <div style="color:#a78bfa; font-weight:700; margin-bottom:4px;">COMPLIANCE AGENT</div>
      <div>OWASP: {str(compliance.get('owasp_category','N/A'))[:80]}</div>
      <div>PCI: {compliance.get('pci_dss','N/A')}</div>
      <div>GDPR: {compliance.get('gdpr_impact','N/A')}</div>
      <div style="margin-top:4px; color:#9ca3af; font-size:11px;">{str(compliance.get('compliance_summary',''))[:200]}</div>
    </div>
    <div style="background:#1f2937; padding:10px; border-radius:4px;">
      <div style="color:#fb923c; font-weight:700; margin-bottom:4px;">RISK MGMT AGENT</div>
      <div>Strategy: <b>{risk_mgmt.get('risk_strategy','N/A')}</b></div>
      <div>Business Impact: {risk_mgmt.get('business_impact','N/A')}</div>
      <div>Owner: {risk_mgmt.get('owner','N/A')}</div>
      <div>Insurance: {risk_mgmt.get('insurance_applicable','N/A')}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        if shown == 0:
            st.info(f"No findings with severity '{sev_llm}' in agent results.")

# TAB 6 ANALYST QUEUE
with tab6:
    st.header("Analyst Feedback Queue (Layer 9 RL Feedback Loop)")
    can_write = role in ("senior_analyst", "soc_lead")

    queue = _load_queue_from_supabase()
    _qsrc = "Supabase live"
    if queue.empty and os.path.exists(REVIEW_PATH):
        queue = pd.read_csv(REVIEW_PATH)
        _qsrc = "repo CSV fallback"
    if not queue.empty:
        st.info(f"{len(queue)} findings queued for review — {_qsrc} (highest model uncertainty active learning)")

        q_cols = [c for c in [
            "id", "tool", "severity", "risk_score", "ml_risk_score",
            "model_uncertainty", "in_kev", "confidence_score", "priority", "title"
        ] if c in queue.columns]
        # ALL queue entries no cap
        render_table(queue[q_cols], title=f"Full Review Queue ({len(queue)}) Highest Uncertainty First (Active Learning)")

        if "model_uncertainty" in queue.columns:
            n_unc = len(queue)
            fig_h_unc = max(3, n_unc * 0.28)
            fig_unc, ax_unc = plt.subplots(figsize=(10, fig_h_unc))
            ax_unc.barh(
                range(n_unc),
                queue["model_uncertainty"],
                color="#e65100", edgecolor="white"
            )
            ax_unc.set_yticks(range(n_unc))
            ax_unc.set_yticklabels([str(i)[:30] for i in queue["id"]], fontsize=7)
            ax_unc.set_xlabel("XGB LGB Score Gap (uncertainty)")
            ax_unc.set_title(f"All {n_unc} Uncertain Findings (XGBoost vs LightGBM)", fontweight="bold")
            ax_unc.invert_yaxis()
            ax_unc.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_unc)
            plt.close(fig_unc)

        if can_write:
            st.subheader("Submit Analyst Label Triggers RL Retraining")
            with st.form("feedback_form"):
                finding_id    = st.selectbox("Finding ID", queue["id"].tolist())
                analyst_score = st.slider("Your Risk Score (0-100)", 0, 100, 50)
                new_severity  = st.selectbox("Override Severity", ["(Keep original)", "CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL"])
                fp_flag       = st.selectbox("False Positive?", ["No", "Yes test_environment", "Yes compensating_control", "Yes not_applicable", "Yes accepted_risk", "Yes scanner_noise"])
                notes         = st.text_input("Notes / reason (optional)")
                submitted     = st.form_submit_button("Submit Label")

            if submitted:
                row = {
                    "id":            finding_id,
                    "analyst_id":    f"analyst_{role}",
                    "analyst_score": analyst_score,
                    "new_severity":  new_severity,
                    "fp_flag":       fp_flag,
                    "notes":         notes,
                    "tenant_id":     tenant_id,
                    "feedback_time": datetime.utcnow().isoformat(),
                }
                sb = _get_supabase()
                if sb:
                    sb.table("analyst_feedback").upsert(row, on_conflict="id,analyst_id,tenant_id").execute()
                    st.success(f"Saved to Supabase for {finding_id} (updated if already labeled)")
                else:
                    st.error("Supabase not configured — check Streamlit secrets (SUPABASE_URL / SUPABASE_KEY)")

        fb = _load_feedback_from_supabase()
        if not fb.empty:
            st.subheader(f"Submitted Labels ({len(fb)}) — live from Supabase")
            fb_cols = [c for c in ["id","analyst_id","analyst_score","new_severity","fp_flag","notes","tenant_id","feedback_time"] if c in fb.columns]
            render_table(fb[fb_cols], title=f"All Analyst Labels ({len(fb)})")

            if "analyst_score" in fb.columns and "ml_risk_score" in fb.columns:
                drift = (pd.to_numeric(fb["analyst_score"], errors="coerce") - pd.to_numeric(fb["ml_risk_score"], errors="coerce")).dropna()
                if not drift.empty:
                    fig7, ax7 = plt.subplots(figsize=(8, 2.5))
                    ax7.hist(drift, bins=10, color="#e65100", edgecolor="white", alpha=0.85)
                    ax7.axvline(0, color="black", linestyle="--", lw=1, label="No drift")
                    ax7.legend(fontsize=8)
                    ax7.set_xlabel("Analyst Score minus ML Score")
                    ax7.set_ylabel("Count")
                    ax7.spines[["top","right"]].set_visible(False)
                    st.pyplot(fig7)
                    plt.close(fig7)
                    avg_drift    = drift.mean()
                    confirm_rate = (drift.abs() < 15).mean()
                    st.caption(f"Avg drift: {avg_drift:.1f} pts | ML confirmation rate: {confirm_rate:.0%} | "
                               f"{'High drift — retrain recommended' if abs(avg_drift) > 15 else 'Model well-calibrated'}")

        if os.path.exists(FP_LOG):
            fp_df = pd.read_csv(FP_LOG)
            st.subheader(f"False Positives ({len(fp_df)})")
            if "fp_category" in fp_df.columns:
                fp_cats = fp_df["fp_category"].value_counts()
                fig_fp, ax_fp = plt.subplots(figsize=(7, 2))
                ax_fp.barh(fp_cats.index, fp_cats.values, color="#9e9e9e", edgecolor="white")
                ax_fp.set_title("FP Categories", fontweight="bold")
                ax_fp.spines[["top","right"]].set_visible(False)
                st.pyplot(fig_fp)
                plt.close(fig_fp)
    else:
        st.info("Run Layer 9 in the notebook to generate the analyst review queue.")

# TAB 7 AI TRIAGE INTELLIGENCE
with tab7:
    st.header("AI Triage Intelligence (Layer 7 Merged View)")
    st.caption("Combines scored_findings.csv columns with multi-agent LLM output for a unified triage view.")

    agent_results_t8 = load_agent_results()

    # KPI row
    has_llm_cols = "llm_urgency" in df.columns

    urgency_counts_t8 = {}
    if has_llm_cols:
        urgency_counts_t8 = df["llm_urgency"].fillna("UNKNOWN").value_counts().to_dict()
    elif agent_results_t8:
        for r in agent_results_t8.values():
            if isinstance(r, dict):
                u = (r.get("triage") or {}).get("urgency", "UNKNOWN")
                urgency_counts_t8[u] = urgency_counts_t8.get(u, 0) + 1

    n_immediate = urgency_counts_t8.get("IMMEDIATE", 0)
    n_high_u    = urgency_counts_t8.get("HIGH", 0)
    n_medium_u  = urgency_counts_t8.get("MEDIUM", 0)
    n_low_u     = urgency_counts_t8.get("LOW", 0)
    n_analyzed  = len(agent_results_t8) if agent_results_t8 else (len(df) if has_llm_cols else 0)

    tk1, tk2, tk3, tk4, tk5 = st.columns(5)
    tk1.metric("Findings Analyzed", n_analyzed)
    tk2.metric("IMMEDIATE",       n_immediate)
    tk3.metric("HIGH Urgency",    n_high_u)
    tk4.metric("MEDIUM Urgency",  n_medium_u)
    tk5.metric("LOW Urgency",     n_low_u)

    st.divider()

    # Path A: llm_* columns exist in the CSV
    if has_llm_cols:
        ai_cols = [c for c in [
            "id", "severity", "risk_score", "priority",
            "llm_urgency", "llm_confidence", "llm_effort_hours",
            "llm_rationale", "llm_action",
        ] if c in df.columns]

        sev_t8 = st.radio(
            "Filter by severity",
            ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
            horizontal=True,
            index=0,
            key="t7_sev_radio",
        )
        t8_df = df.copy()
        if sev_t8 != "ALL":
            t8_df = t8_df[t8_df["severity"] == sev_t8]
        t8_df = t8_df.sort_values("risk_score", ascending=False)

        # ALL rows no cap
        render_table(t8_df[ai_cols], f"Layer 7 AI Agent Output All {len(t8_df)} findings (from scored_findings.csv)")

        # Urgency breakdown bar chart
        st.subheader("Urgency Breakdown")
        urgency_series = t8_df["llm_urgency"].fillna("UNKNOWN").value_counts()
        fig_urg, ax_urg = plt.subplots(figsize=(8, 2.8))
        urg_colors = {"IMMEDIATE": "#d32f2f", "HIGH": "#f57c00", "MEDIUM": "#fbc02d",
                      "LOW": "#388e3c", "UNKNOWN": "#757575"}
        ax_urg.bar(
            urgency_series.index.astype(str),
            urgency_series.values,
            color=[urg_colors.get(u, "#607d8b") for u in urgency_series.index],
            edgecolor="white",
        )
        for i, v in enumerate(urgency_series.values):
            ax_urg.text(i, v + 0.2, str(v), ha="center", fontweight="bold", fontsize=10)
        ax_urg.spines[["top", "right"]].set_visible(False)
        ax_urg.set_ylabel("Findings")
        ax_urg.set_title("AI-Assessed Urgency Distribution", fontweight="bold")
        st.pyplot(fig_urg)
        plt.close(fig_urg)

        # Effort estimate scatter
        if "llm_effort_hours" in t8_df.columns and "risk_score" in t8_df.columns:
            st.subheader("Effort vs Risk Score")
            effort_data_t8 = pd.to_numeric(t8_df["llm_effort_hours"], errors="coerce")
            risk_data_t8   = pd.to_numeric(t8_df["risk_score"], errors="coerce")
            valid_mask     = effort_data_t8.notna() & risk_data_t8.notna()
            if valid_mask.sum() > 0:
                fig_ef, ax_ef = plt.subplots(figsize=(9, 3))
                sc_colors = [COLORS.get(s, "#757575") for s in t8_df.loc[valid_mask, "severity"]]
                ax_ef.scatter(effort_data_t8[valid_mask], risk_data_t8[valid_mask],
                              c=sc_colors, alpha=0.65, s=40)
                ax_ef.set_xlabel("Estimated Effort (hours)")
                ax_ef.set_ylabel("Risk Score")
                ax_ef.set_title("Remediation Effort vs Risk Score (colored by severity)", fontweight="bold")
                ax_ef.spines[["top", "right"]].set_visible(False)
                from matplotlib.patches import Patch
                legend_els_t8 = [Patch(facecolor=COLORS[s], label=s) for s in ["CRITICAL","HIGH","MEDIUM","LOW"]]
                ax_ef.legend(handles=legend_els_t8, fontsize=8, loc="upper right")
                st.pyplot(fig_ef)
                plt.close(fig_ef)

        # Rationale explorer
        st.subheader("Rationale Explorer")
        st.caption("Select a finding to read the full AI rationale and recommended action.")
        if "id" in t8_df.columns and "llm_rationale" in t8_df.columns:
            sel_id = st.selectbox(
                "Select finding ID",
                t8_df["id"].astype(str).tolist(),
                key="t7_rationale_select",
            )
            sel_row = t8_df[t8_df["id"].astype(str) == sel_id]
            if not sel_row.empty:
                r = sel_row.iloc[0]
                sev_r  = str(r.get("severity", "?"))
                border_r = COLORS.get(sev_r, "#555")
                st.markdown(f"""
<div style="border-left:5px solid {border_r}; padding:14px 18px; background:#111827;
            border-radius:6px; color:#e5e7eb; font-size:13px;">
  <b style="color:{border_r}">[{sev_r}]</b> &nbsp; {sel_id} &nbsp;
  <span style="color:#9ca3af;">Risk: {r.get('risk_score','?')} | Priority: {r.get('priority','?')}</span><br/><br/>
  <b style="color:#60a5fa;">AI Rationale:</b><br/>
  <span style="color:#d1d5db;">{str(r.get('llm_rationale','N/A'))}</span><br/><br/>
  <b style="color:#34d399;">Recommended Action:</b><br/>
  <span style="color:#d1d5db;">{str(r.get('llm_action','N/A'))}</span>
</div>
""", unsafe_allow_html=True)

    # Path B: no llm_* CSV columns pull from agent_results dict
    elif agent_results_t8:

        sev_t8b = st.radio(
            "Filter by severity",
            ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
            horizontal=True,
            index=0,
            key="t7b_sev_radio",
        )

        rows_t8 = []
        for fid, res in agent_results_t8.items():
            if not isinstance(res, dict):
                continue
            triage_t8 = res.get("triage") or {}
            rem_t8    = res.get("remediation") or {}
            rows_t8.append({
                "id":           fid,
                "severity":     res.get("severity", "?"),
                "llm_urgency":  triage_t8.get("urgency", "N/A"),
                "exploitability": triage_t8.get("exploitability", "N/A"),
                "attack_vector":  triage_t8.get("attack_vector", "N/A"),
                "confidence":     triage_t8.get("confidence", "N/A"),
                "effort_hours":   rem_t8.get("effort_hours", "N/A"),
                "rationale":      str(triage_t8.get("rationale", ""))[:120],
                "action":         str(rem_t8.get("immediate_action", ""))[:120],
            })

        agent_display_df = pd.DataFrame(rows_t8)
        if sev_t8b != "ALL":
            agent_display_df = agent_display_df[agent_display_df["severity"] == sev_t8b]

        # ALL rows no cap
        render_table(agent_display_df, f"Layer 7 Agent Triage Summary All {len(agent_display_df)} findings (from agent_results.json)")

        if not agent_display_df.empty and "llm_urgency" in agent_display_df.columns:
            st.subheader("Urgency Breakdown")
            urg_s = agent_display_df["llm_urgency"].fillna("UNKNOWN").value_counts()
            fig_urg2, ax_urg2 = plt.subplots(figsize=(8, 2.8))
            urg_colors2 = {"IMMEDIATE": "#d32f2f", "HIGH": "#f57c00", "MEDIUM": "#fbc02d",
                           "LOW": "#388e3c", "UNKNOWN": "#757575"}
            ax_urg2.bar(
                urg_s.index.astype(str),
                urg_s.values,
                color=[urg_colors2.get(u, "#607d8b") for u in urg_s.index],
                edgecolor="white",
            )
            for i, v in enumerate(urg_s.values):
                ax_urg2.text(i, v + 0.2, str(v), ha="center", fontweight="bold", fontsize=10)
            ax_urg2.spines[["top", "right"]].set_visible(False)
            ax_urg2.set_ylabel("Findings")
            ax_urg2.set_title("AI-Assessed Urgency Distribution", fontweight="bold")
            st.pyplot(fig_urg2)
            plt.close(fig_urg2)

        st.subheader("Rationale Explorer")
        if not agent_display_df.empty:
            sel_id_b = st.selectbox(
                "Select finding ID",
                agent_display_df["id"].astype(str).tolist(),
                key="t7b_rationale_select",
            )
            full_res = agent_results_t8.get(sel_id_b, {})
            triage_full = full_res.get("triage") or {}
            rem_full    = full_res.get("remediation") or {}
            sev_b = str(full_res.get("severity", "?"))
            border_b = COLORS.get(sev_b, "#555")
            st.markdown(f"""
<div style="border-left:5px solid {border_b}; padding:14px 18px; background:#111827;
            border-radius:6px; color:#e5e7eb; font-size:13px;">
  <b style="color:{border_b}">[{sev_b}]</b> &nbsp; {sel_id_b}<br/><br/>
  <b style="color:#60a5fa;">AI Rationale:</b><br/>
  <span style="color:#d1d5db;">{str(triage_full.get('rationale','N/A'))}</span><br/><br/>
  <b style="color:#34d399;">Recommended Action:</b><br/>
  <span style="color:#d1d5db;">{str(rem_full.get('immediate_action','N/A'))}</span>
</div>
""", unsafe_allow_html=True)

    # Path C: no data at all
    else:
        st.warning("No AI triage data found. Run Layer 7 (Multi-Agent LLM Triage) in the notebook first.")
        st.markdown("""
        **What Layer 7 produces:**
        - `llm_urgency` IMMEDIATE / HIGH / MEDIUM / LOW
        - `llm_rationale` AI reasoning for the triage decision
        - `llm_action` Recommended immediate action
        - `llm_confidence` Agent confidence score
        - `llm_effort_hours` Estimated remediation effort

        These columns are merged into `scored_findings.csv` and also saved to `agent_results.json`.
        This tab reads from both sources, preferring the CSV.
        """)

# TAB 8 REMEDIATION AND COMPLIANCE
with tab8:
    st.header("Remediation and Compliance Center")
    st.caption("ISO 27001 NIST SP 800-53 OWASP Top 10 PCI DSS GDPR per finding")

    agent_results_compliance = load_agent_results()

    COMPLIANCE_DB = {
        "CVE-2022-22965": {
            "iso":    "A.12.6.1 Technical Vulnerability Management",
            "nist":   "SI-2 Flaw Remediation RA-5 Vulnerability Scanning",
            "owasp":  "A06:2021 Vulnerable and Outdated Components",
            "pci":    "PCI DSS 6.3 Address vulnerabilities",
            "gdpr":   "Art. 32 Technical security measures",
            "action": "Upgrade Spring Framework to 5.3.18+ or 5.2.20.RELEASE",
            "patch":  "mvn versions:use-dep-version -Dincludes=org.springframework:spring-webmvc -DdepVersion=5.3.18",
            "verify": "mvn dependency:tree | grep spring re-scan with Trivy",
            "rollback":"Revert pom.xml to previous Spring version, rebuild",
            "effort": 2,
        },
        "CVE-2022-22950": {
            "iso":    "A.12.6.1 Technical Vulnerability Management",
            "nist":   "SI-2 Flaw Remediation SC-5 Denial of Service Protection",
            "owasp":  "A06:2021 Vulnerable and Outdated Components",
            "pci":    "PCI DSS 6.3",
            "gdpr":   "Art. 32",
            "action": "Upgrade Spring Framework to 5.3.16+ or 5.2.19.RELEASE",
            "patch":  "mvn versions:use-dep-version -Dincludes=org.springframework -DdepVersion=5.3.16",
            "verify": "Re-scan with Trivy/Grype after upgrade",
            "rollback":"Revert to previous Spring version in pom.xml",
            "effort": 2,
        },
        "CVE-2016-1000027": {
            "iso":    "A.14.2.5 Secure System Engineering A.12.6.1",
            "nist":   "SI-2 SA-11 Developer Security Testing",
            "owasp":  "A08:2021 Software and Data Integrity Failures",
            "pci":    "PCI DSS 6.3 6.5",
            "gdpr":   "Art. 25 Data Protection by Design",
            "action": "Upgrade spring-web to 6.0.0+ or replace HttpInvoker usage",
            "patch":  "mvn versions:use-dep-version -Dincludes=org.springframework:spring-web -DdepVersion=6.0.0",
            "verify": "Confirm HttpInvokerServiceExporter not used; re-scan",
            "rollback":"Revert spring-web version, test endpoints",
            "effort": 4,
        },
        "DEFAULT": {
            "iso":    "A.12.6.1 Technical Vulnerability Management",
            "nist":   "SI-2 Flaw Remediation RA-5 Vulnerability Scanning",
            "owasp":  "A06:2021 Vulnerable and Outdated Components",
            "pci":    "PCI DSS 6.3 Address vulnerabilities",
            "gdpr":   "Art. 32 Technical security measures",
            "action": "Apply vendor patch or upgrade to fixed version",
            "patch":  "See fix_version field apply via package manager",
            "verify": "Re-scan with originating tool after patch",
            "rollback":"Revert to previous package version, run regression tests",
            "effort": 2,
        },
    }

    def _as_control_list(value):
        """Normalize a controls field that may be a real list OR a stringified
        list like \"['A.8.8', 'A.8.2']\" (a known upstream serialization quirk).
        Always returns a clean list of control-id strings."""
        import ast as _ast
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return []
            # try to parse a python/json list embedded in the string
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = _ast.literal_eval(s)
                    if isinstance(parsed, list):
                        return [str(v).strip() for v in parsed if str(v).strip()]
                except Exception:
                    pass
            # fall back: comma-separated ids
            return [p.strip() for p in s.split(",") if p.strip()]
        return [str(value).strip()]

    def get_compliance(cve_id):
        agent_data = agent_results_compliance.get(str(cve_id), {})
        if agent_data and isinstance(agent_data, dict):
            comp = agent_data.get("compliance") or {}
            rem  = agent_data.get("remediation") or {}
            rm   = agent_data.get("risk_management") or agent_data.get("risk_mgmt") or {}
            base = COMPLIANCE_DB.get(str(cve_id), COMPLIANCE_DB["DEFAULT"])
            return {
                "iso":      ", ".join(_as_control_list(comp.get("iso_controls"))) or base["iso"],
                "nist":     ", ".join(_as_control_list(comp.get("nist_controls"))) or base["nist"],
                "owasp":    comp.get("owasp_category", base["owasp"]),
                "pci":      comp.get("pci_dss", base["pci"]),
                "gdpr":     comp.get("gdpr_impact", base["gdpr"]),
                "action":   rem.get("immediate_action", base["action"]),
                "patch":    rem.get("patch_command") or base["patch"],
                "verify":   rem.get("verification_step", base["verify"]),
                "rollback": rem.get("rollback_plan", base["rollback"]),
                "effort":   rem.get("effort_hours", base["effort"]),
                "source":   "LLM Agent",
            }
        return {**COMPLIANCE_DB.get(str(cve_id), COMPLIANCE_DB["DEFAULT"]), "source": "Static DB"}

    st.subheader("Regulatory Impact Summary")

    reg_data = {
        "ISO 27001\nA.12.6.1": int((df["severity"].isin(["CRITICAL","HIGH"])).sum()),
        "NIST SP 800-53\nSI-2": int((df["severity"].isin(["CRITICAL","HIGH","MEDIUM"])).sum()),
        "OWASP Top 10\nA06:2021": int((df["tool"].isin(["Trivy","Grype","Snyk"])).sum()),
        "PCI DSS\n6.3": int(df.get("pci_scope", pd.Series([0]*len(df))).sum()) or int((df["severity"]=="CRITICAL").sum()),
        "GDPR\nArt. 32": int(df.get("gdpr_scope", pd.Series([0]*len(df))).sum()) or int((df["severity"].isin(["CRITICAL","HIGH"])).sum()),
    }

    fig_reg, ax_reg = plt.subplots(figsize=(12, 3))
    reg_colors = ["#1565c0","#283593","#c62828","#4a148c","#1b5e20"]
    bars_reg = ax_reg.bar(list(reg_data.keys()), list(reg_data.values()), color=reg_colors, edgecolor="white", width=0.5)
    for b, v in zip(bars_reg, reg_data.values()):
        ax_reg.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, str(v),
                    ha="center", fontweight="bold", fontsize=11)
    ax_reg.set_ylabel("Findings Affected")
    ax_reg.set_title("Findings per Regulatory Framework", fontweight="bold")
    ax_reg.spines[["top","right"]].set_visible(False)
    st.pyplot(fig_reg)
    plt.close(fig_reg)

    st.divider()

    st.subheader("Per-Finding Remediation + Compliance Mapping")
    st.caption("LLM Agent results used when available (Layer 7), otherwise static DB")

    sev_rem = st.radio(
        "Show findings by severity",
        ["CRITICAL", "HIGH", "MEDIUM", "LOW", "ALL"],
        horizontal=True,
        index=0,
    )

    rem_df = df.copy()
    if sev_rem != "ALL":
        rem_df = rem_df[rem_df["severity"] == sev_rem]

    # ALL findings for the selected severity no .head(20) cap
    rem_df = rem_df.sort_values("risk_score", ascending=False)

    st.caption(f"Showing all {len(rem_df)} findings for severity: {sev_rem}")

    if rem_df.empty:
        st.info("No findings for selected severity.")
    else:
        for _, row in rem_df.iterrows():
            cve_id      = str(row.get("id", "N/A"))
            title       = str(row.get("title", "N/A"))[:80]
            sev         = str(row.get("severity", "?"))
            risk        = row.get("risk_score", "N/A")
            tool        = str(row.get("tool", "N/A"))
            fix_ver     = str(row.get("fix_version", "N/A"))
            target      = str(row.get("target", "N/A"))
            epss        = float(row.get("epss_score", 0))
            in_kev      = int(row.get("in_kev", 0))
            priority    = str(row.get("priority", "N/A"))
            conf        = float(row.get("confidence_score", 0.5))
            asset_env   = str(row.get("asset_env", "?"))
            asset_owner = str(row.get("asset_owner", "?"))
            blast       = int(row.get("in_blast_radius", 0))

            comp        = get_compliance(cve_id)
            border      = {"CRITICAL":"#d32f2f","HIGH":"#f57c00","MEDIUM":"#fbc02d","LOW":"#388e3c"}.get(sev,"#555")
            kev_badge   = "CISA KEV ACTIVELY EXPLOITED &nbsp;" if in_kev else ""
            blast_badge = "BLAST RADIUS &nbsp;" if blast else ""
            llm_badge   = "LLM" if comp.get("source") == "LLM Agent" else "DB"

            st.markdown(f"""
<div style="border-left: 5px solid {border}; padding: 16px 20px; margin: 12px 0;
            background: #111827; border-radius: 6px; color: #e5e7eb;">

  <div style="font-size:15px; font-weight:700; margin-bottom:6px;">
    {kev_badge}{blast_badge}<span style="color:{border}">[{sev}]</span> &nbsp;{title}
    <span style="float:right; font-size:11px; background:#374151; padding:2px 6px; border-radius:4px;">{llm_badge}</span>
  </div>
  <div style="font-size:12px; color:#9ca3af; margin-bottom:12px;">
    ID: {cve_id} &nbsp;|&nbsp; Tool: {tool} &nbsp;|&nbsp; Target: {target}
    &nbsp;|&nbsp; Env: {asset_env} | Owner: {asset_owner}
    &nbsp;|&nbsp; Risk: <b style="color:{border}">{risk}</b>
    &nbsp;|&nbsp; EPSS: {epss:.3f} &nbsp;|&nbsp; Priority: {priority}
    &nbsp;|&nbsp; Confidence: {conf:.0%}
  </div>

  <table style="width:100%; border-collapse:collapse; font-size:13px;">
    <tr style="background:#1f2937;">
      <td style="padding:8px 12px; color:#60a5fa; font-weight:600; width:22%">Fix Version</td>
      <td style="padding:8px 12px; color:#34d399; font-family:monospace">{fix_ver}</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; color:#60a5fa; font-weight:600">Immediate Action</td>
      <td style="padding:8px 12px">{comp["action"]}</td>
    </tr>
    <tr style="background:#1f2937;">
      <td style="padding:8px 12px; color:#60a5fa; font-weight:600">Patch Command</td>
      <td style="padding:8px 12px; font-family:monospace; color:#fbbf24">{comp["patch"] or "See vendor advisory"}</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; color:#60a5fa; font-weight:600">Verification</td>
      <td style="padding:8px 12px">{comp["verify"]}</td>
    </tr>
    <tr style="background:#1f2937;">
      <td style="padding:8px 12px; color:#60a5fa; font-weight:600">Rollback Plan</td>
      <td style="padding:8px 12px">{comp["rollback"]}</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; color:#60a5fa; font-weight:600">Effort</td>
      <td style="padding:8px 12px">{comp["effort"]}h estimated</td>
    </tr>
  </table>

  <div style="margin-top:12px; padding:10px 12px; background:#1f2937; border-radius:4px;">
    <div style="font-size:12px; font-weight:700; color:#a78bfa; margin-bottom:6px;">
      REGULATORY MAPPING
    </div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; font-size:12px;">
      <div><b>ISO 27001:</b> {comp["iso"]}</div>
      <div><b>NIST SP 800-53:</b> {comp["nist"]}</div>
      <div><b>OWASP Top 10:</b> {comp["owasp"]}</div>
      <div><b>PCI DSS:</b> {comp["pci"]}</div>
      <div><b>GDPR:</b> {comp["gdpr"]}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    st.subheader("Compliance Matrix Severity x Framework")

    frameworks = ["ISO 27001", "NIST 800-53", "OWASP Top 10", "PCI DSS", "GDPR"]
    severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    coverage   = [1.0, 1.0, 0.85, 0.70, 0.65]

    matrix_comp = np.zeros((len(severities), len(frameworks)), dtype=int)
    for si, sev in enumerate(severities):
        count = int((df["severity"] == sev).sum())
        for fi, factor in enumerate(coverage):
            matrix_comp[si, fi] = int(count * factor)

    fig_mat, ax_mat = plt.subplots(figsize=(12, 3.5))
    im_mat = ax_mat.imshow(matrix_comp, cmap="Blues", aspect="auto")
    ax_mat.set_xticks(range(len(frameworks)))
    ax_mat.set_xticklabels(frameworks, fontsize=9, fontweight="bold")
    ax_mat.set_yticks(range(len(severities)))
    ax_mat.set_yticklabels(severities, fontsize=9)
    for si in range(len(severities)):
        for fi in range(len(frameworks)):
            v = matrix_comp[si, fi]
            ax_mat.text(fi, si, str(v), ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if v > matrix_comp.max()*0.5 else "black")
    plt.colorbar(im_mat, ax=ax_mat, shrink=0.8, label="Findings affected")
    ax_mat.set_title("Compliance Violation Matrix (findings count per framework)", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_mat)
    plt.close(fig_mat)

    st.subheader("Remediation Effort Estimate")

    effort_data = {
        "CRITICAL\n(patch now)":   n_critical * 2,
        "HIGH\n(this week)":       n_high * 2,
        "MEDIUM\n(this month)":    n_medium * 1,
        "LOW\n(backlog)":          n_low * 1,
    }
    total_effort = sum(effort_data.values())

    col_e1, col_e2 = st.columns([2, 1])
    with col_e1:
        fig_eff, ax_eff = plt.subplots(figsize=(7, 2.5))
        eff_colors = ["#d32f2f","#f57c00","#fbc02d","#388e3c"]
        ax_eff.barh(list(effort_data.keys()), list(effort_data.values()), color=eff_colors, edgecolor="white")
        for i, v in enumerate(effort_data.values()):
            ax_eff.text(v + 0.5, i, f"{v}h", va="center", fontweight="bold")
        ax_eff.set_xlabel("Estimated Hours")
        ax_eff.set_title("Remediation Effort by Severity", fontweight="bold")
        ax_eff.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_eff)
        plt.close(fig_eff)
    with col_e2:
        st.metric("Total Estimated Effort", f"{total_effort}h")
        st.metric("In person-days (8h)",    f"{total_effort//8}d {total_effort%8}h")
        st.metric("CRITICAL findings",      n_critical)
        st.caption("Estimate: 2h/Critical High, 1h/Medium Low")

    if os.path.exists(PR_MD):
        st.subheader("Auto-Generated PR Templates (Layer 8)")
        st.success(f"PR templates generated: {PR_MD}")
        st.caption("Run Layer 8 in notebook to generate patch PR templates. Download from Colab.")


# TAB 9 ADVANCED ANALYTICS
with tab9:
    st.header("Advanced analytics")
    _c1, _c2, _c3 = st.columns(3)

    _cal = _vload("risk_explanations.json")
    with _c1:
        st.subheader("Calibration (5B)")
        if _cal and _cal.get("calibration"):
            _c = _cal["calibration"]
            st.metric("Brier score", _c.get("brier_score", "n/a"))
            st.caption(f"Threshold {_c.get('positive_threshold')} | positive rate {_c.get('positive_rate')}")
        else:
            st.caption("Run Layer 5B to populate.")

    _ev = _vload("evaluation_harness_report.json")
    with _c2:
        st.subheader("Evaluation (12)")
        if _ev and _ev.get("risk_model_metrics"):
            _m = _ev["risk_model_metrics"]
            for _k in ("precision", "recall", "f1", "accuracy"):
                if _k in _m:
                    try:
                        st.metric(_k.capitalize(), round(float(_m[_k]), 3))
                    except Exception:
                        st.metric(_k.capitalize(), str(_m[_k]))
            if "label_source" in _m:
                st.caption(f"Labels: {_m['label_source']}")
        else:
            st.caption("Run Layer 12 to populate.")

    _dr = _vload("scanner_drift_report.json")
    with _c3:
        st.subheader("Scanner drift (9B)")
        if _dr:
            st.metric("Status", str(_dr.get("status", "n/a")))
            st.caption(f"History {_dr.get('history_size', 0)} | alerts {len(_dr.get('alerts') or [])}")
            for _a in (_dr.get("alerts") or [])[:5]:
                st.warning(str(_a)[:160])
        else:
            st.caption("Run Layer 9B to populate.")

    try:
        _sdf = pd.read_csv(SCORED_CSV)
    except Exception:
        _sdf = pd.DataFrame()

    if not _sdf.empty and "risk_probability_calibrated" in _sdf.columns:
        st.subheader("Calibrated risk probabilities (5B)")
        _cols = [c for c in ["id", "severity", "risk_score", "risk_probability_calibrated",
                             "calibration_bucket", "blast_radius_score", "crown_jewel_reachable",
                             "fp_probability"] if c in _sdf.columns]
        st.dataframe(_sdf.sort_values("risk_probability_calibrated", ascending=False)[_cols].head(15),
                     use_container_width=True)

    if not _sdf.empty and "top_shap_drivers" in _sdf.columns:
        st.subheader("Top SHAP drivers — highest-risk finding (5B)")
        try:
            _row = _sdf.sort_values("risk_score", ascending=False).iloc[0]
            _drv = _vjson.loads(_row.get("top_shap_drivers") or "[]")
            if _drv:
                st.caption(f"Finding: {_row.get('id', 'unknown')}")
                render_table(pd.DataFrame(_drv), title="Top SHAP drivers")
            else:
                st.caption("No drivers stored for this finding.")
        except Exception as _e:
            st.caption(f"Drivers unavailable: {_e}")

    _ag = _vload("agent_results.json")
    if _ag:
        _items = _ag if isinstance(_ag, list) else _ag.get("results", [])
        _verd = {}
        for _it in _items:
            _v = ((_it or {}).get("red_team_challenge") or {}).get("final_verdict")
            if _v:
                _verd[_v] = _verd.get(_v, 0) + 1
        if _verd:
            st.subheader("Red-team verdicts (7B)")
            st.bar_chart(pd.Series(_verd))

    _br = _vload("blast_radius_simulation.json", base=os.path.join(BASE_DIR, "knowledge_graph"))
    if _br:
        st.subheader("Weighted blast radius (4B)")
        _summary = {k: _br[k] for k in list(_br)[:8] if not isinstance(_br[k], (list, dict))}
        if _summary:
            st.json(_summary)
        _paths = _br.get("simulated_paths") or _br.get("paths") or []
        if isinstance(_paths, list) and _paths:
            st.caption("Top simulated attack paths")
            st.dataframe(pd.DataFrame(_paths[:10]), use_container_width=True)


# TAB 10 SYSTEM HEALTH
with tab10:
    st.header("System health & monitoring")

    _hc = _vload("system_health_report.json")
    _dr2 = _vload("distribution_drift_report.json")

    _hcol1, _hcol2, _hcol3 = st.columns(3)

    with _hcol1:
        st.subheader("Run health")
        if _hc:
            st.metric("Checks passed", _hc.get("ok_count", 0))
            _iss = _hc.get("issue_count", 0)
            st.metric("Issues", _iss)
            st.caption(f"Approved training examples: {_hc.get('approved_examples', 0)}")
            for _i in (_hc.get("issues") or [])[:5]:
                st.warning(str(_i)[:150])
        else:
            st.caption("Run the health-check cell to populate.")

    with _hcol2:
        st.subheader("Cache freshness")
        if _hc:
            st.metric("Fresh entries", _hc.get("cache_fresh", 0))
            st.metric("Stale entries", _hc.get("cache_stale", 0))
            st.caption(f"Config hash: {_hc.get('cache_config_hash', 'n/a')}")
            if _hc.get("cache_stale", 0) > 0:
                st.info("Stale cache: bump rag_library_version to regenerate.")
        else:
            st.caption("Run the staleness monitor to populate.")

    with _hcol3:
        st.subheader("Output drift")
        if _dr2:
            _status = _dr2.get("status", "n/a")
            st.metric("Status", _status)
            _alerts = _dr2.get("alerts") or []
            st.metric("Drift alerts", len(_alerts))
            for _a in _alerts[:5]:
                st.warning(str(_a)[:150])
            if _status == "stable":
                st.success("No post-retraining distribution drift detected.")
        else:
            st.caption("Run the drift-check cell after a retraining cycle.")

    if _dr2 and _dr2.get("current"):
        _bc = _dr2.get("baseline", {})
        _cc = _dr2.get("current", {})
        _rows = []
        for _k in sorted(set(list(_bc) + list(_cc))):
            _bv, _cv = _bc.get(_k), _cc.get(_k)
            if not isinstance(_bv, dict) and not isinstance(_cv, dict):
                _rows.append({"metric": _k, "baseline": _bv, "current": _cv})
        if _rows:
            st.caption("Distribution baseline vs current")
            st.dataframe(pd.DataFrame(_rows), use_container_width=True)
    # rollback decision (from Layer 9 MAE guard)
    _rb = _vload("rollback_decision.json", base=os.path.join(BASE_DIR, "feedback"))
    if _rb:
        st.markdown("---")
        st.subheader("Last retraining decision (MAE rollback guard)")
        _r1, _r2, _r3 = st.columns(3)
        _r1.metric("MAE before", _rb.get("mae_before", "n/a"))
        _r2.metric("MAE after", _rb.get("mae_after", "n/a"))
        _r3.metric("Improvement %", _rb.get("improvement_pct", "n/a"))
        if _rb.get("deploy"):
            st.success("Decision: DEPLOY new checkpoint")
        elif _rb.get("rollback"):
            st.error("Decision: ROLLBACK to previous checkpoint")

# TAB 11 ABLATION STUDY
with tab11:
    st.header("Ablation study — layer contribution")
    st.caption("Cumulative contribution of each pipeline capability, with the asymmetric security cost model (FN x10, FP x2).")
    _ab = _vload("ablation_study.json")
    if _ab and _ab.get("rows"):
        st.caption(f"Label source: {_ab.get('label_source','n/a')} | findings: {_ab.get('n_findings','n/a')} | "
                   f"cost FN={_ab.get('cost_false_negative')} FP={_ab.get('cost_false_positive')}")
        _abdf = pd.DataFrame(_ab["rows"])
        _show = [c for c in ["version","precision","recall","f1","fp_rate","cost","cost_savings_vs_baseline","notes"] if c in _abdf.columns]
        st.dataframe(_abdf[_show], use_container_width=True, hide_index=True)
        if "f1" in _abdf.columns and "version" in _abdf.columns:
            st.subheader("F1 by pipeline stage")
            _f = _abdf.set_index("version")["f1"]
            st.bar_chart(_f)
        if _ab.get("label_source") == "pseudo_ground_truth":
            st.warning("Pseudo-labels are partially circular with severity-based stages. Use analyst labels for defense numbers.")
    else:
        st.info("No ablation_study.json found. Run the ablation cell (Layer 13).")

# TAB 12 ARCHITECTURE
with tab12:
    st.header("System architecture")
    _arch = os.path.join(BASE_DIR, "reports", "architecture_diagram.png")
    if os.path.exists(_arch):
        st.image(_arch, use_container_width=True)
    else:
        st.info("No architecture_diagram.png found. Run the architecture diagram cell.")

# TAB 13 SYSTEM CARD
with tab13:
    st.header("System card")
    _card = os.path.join(BASE_DIR, "reports", "system_card.md")
    if os.path.exists(_card):
        with open(_card) as _fh:
            st.markdown(_fh.read())
    else:
        st.info("No system_card.md found. Run the system card cell.")

# TAB 14 COMPLIANCE GAPS
with tab14:
    st.header("Compliance Control Coverage Map")
    st.caption("Framework controls implicated by current findings, ranked by worst severity. 'Shared' controls are hit by 2+ findings.")
    _gap = _vload("compliance_gap_map.json")
    # Resolve to the list of per-control entries regardless of the saved shape.
    _entries = []
    if isinstance(_gap, dict) and isinstance(_gap.get("gap_map"), list):
        _entries = _gap["gap_map"]
    elif isinstance(_gap, list):
        _entries = _gap
    elif isinstance(_gap, dict):
        for _fw, _v in _gap.items():
            if isinstance(_v, list):
                for _ctrl in _v:
                    _entries.append({"control_id": str(_ctrl), "framework": _fw})
    if _entries:
        _gdf = pd.DataFrame(_entries)
        if "finding_ids" in _gdf.columns:
            _gdf["finding_ids"] = _gdf["finding_ids"].apply(
                lambda v: ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v))
        _order = ["control_id", "framework", "worst_severity", "finding_count", "is_shared_gap", "finding_ids"]
        _cols = [c for c in _order if c in _gdf.columns]
        _gdf = _gdf[_cols + [c for c in _gdf.columns if c not in _cols]]
        _fws = ["All"] + sorted(str(x) for x in _gdf.get("framework", pd.Series(dtype=str)).dropna().unique())
        _pick = st.selectbox("Framework", _fws, key="gapmap_fw")
        _view = _gdf if _pick == "All" else _gdf[_gdf["framework"].astype(str) == _pick]
        _c1, _c2, _c3 = st.columns(3)
        _c1.metric("Controls implicated", len(_view))
        if "is_shared_gap" in _view.columns:
            _c2.metric("Shared (2+ findings)", int(_view["is_shared_gap"].fillna(False).astype(bool).sum()))
        if "worst_severity" in _view.columns:
            _c3.metric("Critical / High", int(_view["worst_severity"].astype(str).isin(["CRITICAL", "HIGH"]).sum()))
        st.dataframe(_view, use_container_width=True, hide_index=True)
        st.download_button("Download coverage map (CSV)",
                           _view.to_csv(index=False).encode("utf-8"),
                           "compliance_coverage_map.csv", "text/csv", key="gapmap_dl")
    elif _gap:
        st.json(_gap)
    else:
        st.info("No compliance_gap_map.json found. Run the compliance gap cell (Layer 6).")


with tab15:
    # ══ CHATBOT TAB — Assistant SOC (call-to-action onglet) ══
    # NV61: onglet chatbot isole. try/except sur TOUT le contenu -> une panne
    # (cle API absente, artefact manquant, import rate) n affecte QUE cet onglet.
    try:
        import importlib, soc_chatbot
        importlib.reload(soc_chatbot)
        soc_chatbot.render_chatbot_tab()
    except Exception as _chat_e:
        st.error(f"Assistant SOC indisponible: {_chat_e}")
        st.caption("Les autres onglets ne sont pas affectes.")
