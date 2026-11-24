"""
ui_components.py
----------------
Reusable UI helpers, custom styling, and Plotly visualization builders
for the AI Internship Recommendation & Skill Gap Analysis Streamlit dashboard.
"""

import re
from collections import Counter
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# -----------------------------------------------------------------------------
# CUSTOM CSS THEME (DARK BLUE & PURPLE EXPO-READY PALETTE)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Global App Background and Font */
.stApp {
    background-color: #0B0F19;
    color: #F1F5F9;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #111827 50%, #1E1B4B 100%) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.2);
}

/* Top App Header / Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(30, 27, 75, 0.85) 0%, rgba(15, 23, 42, 0.95) 50%, rgba(88, 28, 135, 0.4) 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 0 15px rgba(139, 92, 246, 0.15);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #A78BFA 0%, #60A5FA 50%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94A3B8;
    margin-bottom: 0;
    line-height: 1.5;
}

/* Metric / Stat Cards */
.metric-card {
    background: linear-gradient(145deg, #151D2F 0%, #1A2238 100%);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(139, 92, 246, 0.5);
}

.metric-label {
    font-size: 0.85rem;
    color: #94A3B8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 2px;
}

.metric-delta {
    font-size: 0.8rem;
    color: #A78BFA;
    font-weight: 500;
}

/* Recommendation Card */
.rec-card {
    background: linear-gradient(145deg, #131B2E 0%, #18223B 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 18px;
    transition: all 0.25s ease;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
}

.rec-card:hover {
    border-color: #8B5CF6;
    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.2);
    transform: translateY(-3px);
}

/* Badges */
.badge-match-high {
    background: linear-gradient(135deg, #059669 0%, #10B981 100%);
    color: #FFFFFF;
    font-weight: 700;
    font-size: 0.9rem;
    padding: 6px 14px;
    border-radius: 20px;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.4);
    display: inline-block;
}

.badge-match-med {
    background: linear-gradient(135deg, #D97706 0%, #F59E0B 100%);
    color: #FFFFFF;
    font-weight: 700;
    font-size: 0.9rem;
    padding: 6px 14px;
    border-radius: 20px;
    box-shadow: 0 0 10px rgba(245, 158, 11, 0.4);
    display: inline-block;
}

.badge-match-low {
    background: linear-gradient(135deg, #475569 0%, #64748B 100%);
    color: #FFFFFF;
    font-weight: 700;
    font-size: 0.9rem;
    padding: 6px 14px;
    border-radius: 20px;
    display: inline-block;
}

.skill-tag-matched {
    background-color: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.4);
    border-radius: 6px;
    padding: 2px 9px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 5px;
    margin-bottom: 5px;
    display: inline-block;
}

.skill-tag-missing {
    background-color: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 6px;
    padding: 2px 9px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 5px;
    margin-bottom: 5px;
    display: inline-block;
}

.domain-pill {
    background: rgba(139, 92, 246, 0.18);
    color: #C084FC;
    border: 1px solid rgba(139, 92, 246, 0.35);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    display: inline-block;
}

/* Button & Form Input overrides */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 100%);
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.35);
}

div.stButton > button:first-child:hover {
    background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%);
    box-shadow: 0 6px 18px rgba(139, 92, 246, 0.5);
    transform: translateY(-1px);
}

/* Dataframe & Tables */
[data-testid="stDataFrame"] {
    background-color: #111827;
    border-radius: 10px;
    border: 1px solid rgba(139, 92, 246, 0.2);
}

/* Streamlit Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #0F172A;
    padding: 6px;
    border-radius: 10px;
    border: 1px solid rgba(139, 92, 246, 0.2);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94A3B8;
    font-weight: 600;
    padding: 8px 16px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6D28D9 0%, #4338CA 100%) !important;
    color: #FFFFFF !important;
}
</style>
"""


# -----------------------------------------------------------------------------
# PLOTLY THEME DEFAULTS
# -----------------------------------------------------------------------------
PLOTLY_DARK_LAYOUT = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(color="#E2E8F0", family="Inter, sans-serif"),
    margin=dict(l=30, r=30, t=45, b=30),
    legend=dict(
        font=dict(color="#CBD5E1"),
        bgcolor="rgba(17, 24, 39, 0.8)",
        bordercolor="rgba(139, 92, 246, 0.2)",
        borderwidth=1
    )
)


def get_match_badge_html(match_pct: float) -> str:
    """Returns styled HTML badge for match percentage."""
    if match_pct >= 70.0:
        cls = "badge-match-high"
    elif match_pct >= 45.0:
        cls = "badge-match-med"
    else:
        cls = "badge-match-low"
    return f'<span class="{cls}">{match_pct:.1f}% Match</span>'


def get_skills_badges_html(matched: List[str], missing: List[str]) -> str:
    """Returns HTML snippet with green matched tags and red missing tags."""
    matched_html = "".join([f'<span class="skill-tag-matched">✓ {s}</span>' for s in matched])
    missing_html = "".join([f'<span class="skill-tag-missing">✗ {s}</span>' for s in missing])
    
    html = f"""
    <div style="margin-top: 8px;">
        <div style="margin-bottom: 6px;">
            <strong style="color: #34D399; font-size: 0.82rem;">Matched Skills ({len(matched)}):</strong><br>
            {matched_html if matched else '<span style="color: #64748B; font-size: 0.8rem;">None</span>'}
        </div>
        <div>
            <strong style="color: #F87171; font-size: 0.82rem;">Skill Gap ({len(missing)}):</strong><br>
            {missing_html if missing else '<span style="color: #34D399; font-size: 0.8rem;">Full Match! No missing skills.</span>'}
        </div>
    </div>
    """
    return html


# -----------------------------------------------------------------------------
# PLOTLY VISUALIZATION GENERATORS
# -----------------------------------------------------------------------------
def create_suggested_skills_bar_chart(suggestions: List[Dict[str, Any]]) -> go.Figure:
    """
    Generates a horizontal bar chart showing the highest-impact skills
    needed across the student's top matching internships.
    """
    if not suggestions:
        fig = go.Figure()
        fig.add_annotation(
            text="No missing skills identified! Great match coverage.",
            showarrow=False,
            font=dict(color="#10B981", size=14)
        )
        fig.update_layout(**PLOTLY_DARK_LAYOUT)
        return fig

    df_sug = pd.DataFrame(suggestions)
    # Reverse order so highest percentage is at the top
    df_sug = df_sug.iloc[::-1]

    fig = go.Figure(go.Bar(
        x=df_sug["percentage_of_recommendations"],
        y=df_sug["skill"],
        orientation="h",
        text=[f"{pct:.1f}% ({cnt} jobs)" for pct, cnt in zip(df_sug["percentage_of_recommendations"], df_sug["frequency"])],
        textposition="inside",
        marker=dict(
            color=df_sug["percentage_of_recommendations"],
            colorscale=[[0, "#4C1D95"], [0.5, "#7C3AED"], [1.0, "#A78BFA"]],
            line=dict(color="#C084FC", width=1)
        )
    ))

    fig.update_layout(
        title="<b>Top High-Impact Skills to Learn</b> (Frequency across Top Matches)",
        xaxis_title="Percentage of Top Matches Requiring Skill (%)",
        yaxis_title="",
        height=320,
        **PLOTLY_DARK_LAYOUT
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.1)", range=[0, 105])
    fig.update_yaxes(showgrid=False)
    return fig


def create_skill_match_gauge(avg_match_rate: float) -> go.Figure:
    """
    Creates an interactive gauge meter showing the student's average
    skill compatibility across their top recommendations.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_match_rate,
        number=dict(suffix="%", font=dict(color="#F8FAFC", size=32)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#94A3B8"),
            bar=dict(color="#8B5CF6", thickness=0.25),
            bgcolor="#1E293B",
            borderwidth=1,
            bordercolor="rgba(139, 92, 246, 0.3)",
            steps=[
                dict(range=[0, 40], color="rgba(239, 68, 68, 0.25)"),
                dict(range=[40, 70], color="rgba(245, 158, 11, 0.25)"),
                dict(range=[70, 100], color="rgba(16, 185, 129, 0.25)")
            ],
            threshold=dict(
                line=dict(color="#10B981", width=3),
                thickness=0.75,
                value=75
            )
        )
    ))

    fig.update_layout(
        title="<b>Average Skill Match Rate</b>",
        height=260,
        **PLOTLY_DARK_LAYOUT
    )
    return fig


def create_domain_donut_chart(df: pd.DataFrame) -> go.Figure:
    """
    Creates a donut chart displaying the distribution of internships across domains.
    """
    domain_counts = df["Domain"].value_counts().reset_index()
    domain_counts.columns = ["Domain", "Count"]

    fig = px.pie(
        domain_counts,
        values="Count",
        names="Domain",
        hole=0.55,
        color_discrete_sequence=px.colors.sequential.Purp_r
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker=dict(line=dict(color="#0F172A", width=2))
    )

    fig.update_layout(
        title="<b>Internship Distribution by Technical Domain</b>",
        showlegend=False,
        height=380,
        **PLOTLY_DARK_LAYOUT
    )
    return fig


def create_top_skills_chart(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Extracts and graphs the top N most in-demand skills across the entire dataset.
    """
    skill_counter = Counter()
    for skills_str in df["Required Skills"].dropna():
        for s in skills_str.split(","):
            s_clean = s.strip()
            if s_clean:
                skill_counter[s_clean] += 1

    top_skills = skill_counter.most_common(top_n)
    df_skills = pd.DataFrame(top_skills, columns=["Skill", "Count"])
    df_skills = df_skills.iloc[::-1]

    fig = go.Figure(go.Bar(
        x=df_skills["Count"],
        y=df_skills["Skill"],
        orientation="h",
        marker=dict(
            color=df_skills["Count"],
            colorscale=[[0, "#3B82F6"], [0.5, "#6366F1"], [1.0, "#A855F7"]],
            line=dict(color="#C084FC", width=1)
        ),
        text=df_skills["Count"],
        textposition="outside"
    ))

    fig.update_layout(
        title=f"<b>Top {top_n} In-Demand Technical Skills</b>",
        xaxis_title="Number of Internship Postings",
        yaxis_title="",
        height=450,
        **PLOTLY_DARK_LAYOUT
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.1)")
    fig.update_yaxes(showgrid=False)
    return fig


def _parse_stipend_numeric(stipend_str: str) -> float:
    """Helper to convert stipend string (e.g. ₹35,000 / month) into numeric amount."""
    if not isinstance(stipend_str, str):
        return 0.0
    # Search for INR values
    inr_match = re.search(r"₹\s*([\d,]+)", stipend_str)
    if inr_match:
        return float(inr_match.group(1).replace(",", ""))
    # USD values (assume $1 = ₹85)
    usd_match = re.search(r"\$\s*([\d,]+)", stipend_str)
    if usd_match:
        return float(usd_match.group(1).replace(",", "")) * 85.0
    return 0.0


def create_stipend_by_domain_chart(df: pd.DataFrame) -> go.Figure:
    """
    Calculates and visualizes the average monthly stipend across domains.
    """
    df_temp = df.copy()
    df_temp["Numeric_Stipend"] = df_temp["Stipend"].apply(_parse_stipend_numeric)
    
    # Filter out unpaid/zero values for average calculation
    paid_df = df_temp[df_temp["Numeric_Stipend"] > 0]
    avg_stipends = paid_df.groupby("Domain")["Numeric_Stipend"].mean().reset_index()
    avg_stipends = avg_stipends.sort_values(by="Numeric_Stipend", ascending=True)

    fig = go.Figure(go.Bar(
        x=avg_stipends["Numeric_Stipend"],
        y=avg_stipends["Domain"],
        orientation="h",
        text=[f"₹{val:,.0f}" for val in avg_stipends["Numeric_Stipend"]],
        textposition="inside",
        marker=dict(
            color=avg_stipends["Numeric_Stipend"],
            colorscale=[[0, "#1E3A8A"], [0.5, "#4338CA"], [1.0, "#7E22CE"]],
            line=dict(color="#A855F7", width=1)
        )
    ))

    fig.update_layout(
        title="<b>Average Monthly Stipend by Technical Domain (₹)</b>",
        xaxis_title="Average Stipend (INR / Month)",
        yaxis_title="",
        height=450,
        **PLOTLY_DARK_LAYOUT
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.1)")
    fig.update_yaxes(showgrid=False)
    return fig


def create_work_model_chart(df: pd.DataFrame) -> go.Figure:
    """
    Categorizes internships into Remote, Hybrid, and On-site.
    """
    def categorize_location(loc: str) -> str:
        loc_l = str(loc).lower()
        if "remote" in loc_l:
            return "Remote"
        elif "hybrid" in loc_l:
            return "Hybrid"
        else:
            return "On-Site"

    models = df["Location"].apply(categorize_location).value_counts().reset_index()
    models.columns = ["Work Model", "Count"]

    fig = px.pie(
        models,
        values="Count",
        names="Work Model",
        color="Work Model",
        color_discrete_map={
            "Remote": "#10B981",
            "Hybrid": "#8B5CF6",
            "On-Site": "#3B82F6"
        },
        hole=0.45
    )

    fig.update_layout(
        title="<b>Workplace Model Breakdown</b>",
        height=280,
        **PLOTLY_DARK_LAYOUT
    )
    return fig
