"""
app.py
------
AI-Based Internship Recommendation and Skill Gap Analysis System
Interactive Streamlit Dashboard (Expo-Ready Edition)

Connects directly to the Scikit-learn TF-IDF + Cosine Similarity
recommendation engine in src/recommendation_engine.py.
"""

import datetime
from typing import List
import pandas as pd
import streamlit as st

# Import backend ML recommender and UI components
from src.recommendation_engine import InternshipRecommender
from src.ui_components import (
    CUSTOM_CSS,
    get_match_badge_html,
    get_skills_badges_html,
    create_suggested_skills_bar_chart,
    create_skill_match_gauge,
    create_domain_donut_chart,
    create_top_skills_chart,
    create_stipend_by_domain_chart,
    create_work_model_chart
)


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Internship Recommender & Skill Gap Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom Dark Blue & Purple CSS styling
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CACHED ENGINE & DATA LOADER
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing AI Recommendation Engine...")
def load_recommender():
    """Initializes and caches the TF-IDF recommendation engine."""
    return InternshipRecommender(dataset_path="datasets/internships.csv")


recommender = load_recommender()
df_all = recommender.df

# Extract unique skills, domains, and locations for UI pickers
@st.cache_data
def get_dataset_metadata(df: pd.DataFrame):
    all_skills = set()
    for s_str in df["Required Skills"].dropna():
        for s in s_str.split(","):
            s_clean = s.strip()
            if s_clean:
                all_skills.add(s_clean)
    sorted_skills = sorted(list(all_skills))
    sorted_domains = sorted(df["Domain"].unique().tolist())
    sorted_locations = sorted(df["Location"].unique().tolist())
    return sorted_skills, sorted_domains, sorted_locations


ALL_SKILLS, ALL_DOMAINS, ALL_LOCATIONS = get_dataset_metadata(df_all)


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "student_name" not in st.session_state:
    st.session_state.student_name = "Alex Chen"

if "student_dept" not in st.session_state:
    st.session_state.student_dept = "Computer Science & Engineering"

if "student_batch" not in st.session_state:
    st.session_state.student_batch = "2026"

if "student_cgpa" not in st.session_state:
    st.session_state.student_cgpa = 8.6

if "student_skills" not in st.session_state:
    st.session_state.student_skills = ["Python", "Machine Learning", "Scikit-learn", "Pandas", "SQL"]

if "student_interests" not in st.session_state:
    st.session_state.student_interests = ["Artificial Intelligence & Machine Learning", "Data Science & Analytics"]

if "preferred_location" not in st.session_state:
    st.session_state.preferred_location = "Bangalore, India"

if "tracked_applications" not in st.session_state:
    # Initialize with 2 sample demonstration tracked items
    st.session_state.tracked_applications = [
        {
            "id": "INT-0581",
            "company": "BrowserStack",
            "role": "AI Research Intern",
            "location": "San Francisco, CA (Remote)",
            "stipend": "₹80,000 / month",
            "status": "Applied",
            "date": "2026-11-20",
            "notes": "Submitted GitHub portfolio and resume."
        },
        {
            "id": "INT-0190",
            "company": "Oracle",
            "role": "Deep Learning Intern",
            "location": "Hybrid - Bangalore",
            "stipend": "₹35,000 / month",
            "status": "Interview Scheduled",
            "date": "2026-11-22",
            "notes": "Technical round scheduled for next Tuesday."
        }
    ]


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & PROFILE SUMMARY
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 12px 0 16px 0;">
        <div style="font-size: 2.2rem; margin-bottom: 4px;">🎓⚡</div>
        <h2 style="margin: 0; font-size: 1.35rem; font-weight: 800; background: linear-gradient(90deg, #A78BFA, #60A5FA); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            InternAI Navigator
        </h2>
        <span style="color: #94A3B8; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">
            Recommendation & Skill Gap Engine
        </span>
    </div>
    """, unsafe_allow_html=True)

    nav_selection = st.radio(
        "Navigation",
        [
            "🎯 AI Recommendations",
            "📊 Skill Gap Studio",
            "🔍 Explore Internships",
            "📌 Application Tracker",
            "📈 Market Analytics",
            "👤 Student Profile"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Persistent Student Profile Card in Sidebar
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 10px; padding: 14px; margin-bottom: 12px;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
            <strong style="color: #F8FAFC; font-size: 0.95rem;">👤 {st.session_state.student_name}</strong>
            <span style="background: #4C1D95; color: #C084FC; font-size: 0.72rem; padding: 2px 7px; border-radius: 10px; font-weight: 700;">Batch {st.session_state.student_batch}</span>
        </div>
        <div style="color: #94A3B8; font-size: 0.8rem; line-height: 1.4;">
            🏛️ {st.session_state.student_dept}<br>
            📍 {st.session_state.preferred_location}<br>
            ⭐ CGPA: {st.session_state.student_cgpa} / 10.0<br>
            🛠️ <strong>{len(st.session_state.student_skills)} Skills:</strong> {', '.join(st.session_state.student_skills[:3])}{'...' if len(st.session_state.student_skills) > 3 else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_side1, col_side2 = st.columns(2)
    with col_side1:
        st.metric("Tracked Jobs", len(st.session_state.tracked_applications))
    with col_side2:
        st.metric("Catalog Size", f"{len(df_all):,}")

    st.markdown("""
    <div style="font-size: 0.75rem; color: #64748B; text-align: center; margin-top: 20px;">
        Powered by Scikit-learn TF-IDF & Cosine Similarity<br>
        <strong>Expo Ready • 2026 Edition</strong>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# VIEW 1: STUDENT PROFILE
# =============================================================================
if nav_selection == "👤 Student Profile":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">👤 Student Profile Setup</h1>
        <p class="hero-subtitle">
            Configure your technical skills, academic background, and career interests to generate tailored AI internship matches.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Preset Career Track Buttons
    st.markdown("#### ⚡ Quick Preset Tracks (One-Click Setup)")
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        if st.button("🤖 AI / Machine Learning", use_container_width=True):
            st.session_state.student_skills = ["Python", "Machine Learning", "Scikit-learn", "Pandas", "PyTorch", "SQL"]
            st.session_state.student_interests = ["Artificial Intelligence & Machine Learning", "Data Science & Analytics"]
            st.rerun()
    with pcol2:
        if st.button("🌐 Full Stack Developer", use_container_width=True):
            st.session_state.student_skills = ["JavaScript", "React.js", "Node.js", "HTML5", "CSS3", "Git", "MongoDB"]
            st.session_state.student_interests = ["Full Stack Web Development", "Frontend Engineering"]
            st.rerun()
    with pcol3:
        if st.button("📊 Data Science & BI", use_container_width=True):
            st.session_state.student_skills = ["Python", "SQL", "Data Analysis", "Pandas", "Tableau", "Power BI", "Statistics"]
            st.session_state.student_interests = ["Data Science & Analytics"]
            st.rerun()
    with pcol4:
        if st.button("☁️ Cloud & DevOps", use_container_width=True):
            st.session_state.student_skills = ["Linux", "Docker", "Git", "Kubernetes", "AWS", "CI/CD", "Python"]
            st.session_state.student_interests = ["Cloud Computing & DevOps"]
            st.rerun()

    st.markdown("---")

    with st.form("student_profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            name_input = st.text_input("Full Name", value=st.session_state.student_name)
            dept_options = [
                "Computer Science & Engineering",
                "Information Technology",
                "Artificial Intelligence & Data Science",
                "Electronics & Communication Engineering",
                "Electrical & Electronics Engineering",
                "Mechanical Engineering",
                "Master of Computer Applications (MCA)",
                "B.Sc / M.Sc Computer Science",
                "Other / Interdisciplinary"
            ]
            current_dept_idx = dept_options.index(st.session_state.student_dept) if st.session_state.student_dept in dept_options else 0
            dept_input = st.selectbox("Academic Department / Degree", dept_options, index=current_dept_idx)
            batch_input = st.selectbox("Graduation Batch", ["2025", "2026", "2027", "2028"], index=["2025", "2026", "2027", "2028"].index(st.session_state.student_batch))

        with col2:
            cgpa_input = st.slider("Current CGPA (out of 10.0)", min_value=5.0, max_value=10.0, value=float(st.session_state.student_cgpa), step=0.1)
            loc_options = ["Any / Remote"] + ALL_LOCATIONS
            current_loc_idx = loc_options.index(st.session_state.preferred_location) if st.session_state.preferred_location in loc_options else 0
            loc_input = st.selectbox("Preferred Location", loc_options, index=current_loc_idx)

        st.markdown("#### 🛠️ Skills & Competencies")
        # Ensure default skills are present in options
        valid_default_skills = [s for s in st.session_state.student_skills if s in ALL_SKILLS]
        skills_input = st.multiselect(
            "Select your technical skills (choose from catalog or type custom)",
            options=ALL_SKILLS,
            default=valid_default_skills,
            help="Select all technical skills, frameworks, languages, and tools you have experience with."
        )

        custom_skill_str = st.text_input(
            "Add Custom Skills (comma-separated, if not in the list above)",
            placeholder="e.g. LangGraph, Ray, vLLM, Neo4j"
        )

        st.markdown("#### 💡 Technical Domains of Interest")
        valid_default_interests = [i for i in st.session_state.student_interests if i in ALL_DOMAINS]
        interests_input = st.multiselect(
            "Target domains / career tracks",
            options=ALL_DOMAINS,
            default=valid_default_interests
        )

        submitted = st.form_submit_button("💾 Save Profile & Update Recommendations")
        if submitted:
            st.session_state.student_name = name_input.strip() or "Student"
            st.session_state.student_dept = dept_input
            st.session_state.student_batch = batch_input
            st.session_state.student_cgpa = cgpa_input
            st.session_state.preferred_location = loc_input
            
            # Combine multiselect skills with custom skills
            combined_skills = list(skills_input)
            if custom_skill_str.strip():
                for cs in custom_skill_str.split(","):
                    c_clean = cs.strip()
                    if c_clean and c_clean not in combined_skills:
                        combined_skills.append(c_clean)
            
            st.session_state.student_skills = combined_skills
            st.session_state.student_interests = interests_input
            st.success("✅ Profile updated successfully! Navigate to 'AI Recommendations' to view updated matches.")


# =============================================================================
# VIEW 2: AI RECOMMENDATIONS
# =============================================================================
elif nav_selection == "🎯 AI Recommendations":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">🎯 AI Internship Recommendations</h1>
        <p class="hero-subtitle">
            Personalized matches generated by Scikit-learn TF-IDF Vectorizer and Cosine Similarity, featuring real-time skill gap analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Active profile reminder bar
    p_skills_str = ", ".join(st.session_state.student_skills) if st.session_state.student_skills else "None specified"
    p_interests_str = ", ".join(st.session_state.student_interests) if st.session_state.student_interests else "All Domains"
    
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 12px; padding: 14px 18px; margin-bottom: 20px;">
        <span style="color: #A78BFA; font-weight: 700; font-size: 0.85rem; text-transform: uppercase;">Active Profile Filters</span><br>
        <span style="color: #94A3B8; font-size: 0.85rem;">
            🛠️ <strong>Skills:</strong> {p_skills_str} | 💡 <strong>Interests:</strong> {p_interests_str} | 📍 <strong>Location:</strong> {st.session_state.preferred_location}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Control filters for recommendation
    f_col1, f_col2, f_col3 = st.columns([1, 1, 1])
    with f_col1:
        top_n = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10, step=1)
    with f_col2:
        min_match_pct = st.slider("Minimum Match Threshold (%)", min_value=10, max_value=90, value=30, step=5)
    with f_col3:
        filter_by_loc = st.checkbox("Prioritize Preferred Location / Remote", value=True)

    # Call ML Recommendation Engine
    if not st.session_state.student_skills and not st.session_state.student_interests:
        st.warning("⚠️ Your profile skills and interests are empty! Please visit 'Student Profile' to add your skills.")
    else:
        with st.spinner("Computing TF-IDF feature matrices & Cosine Similarities..."):
            recommendation_result = recommender.recommend(
                student_skills=st.session_state.student_skills,
                department=st.session_state.student_dept,
                interests=", ".join(st.session_state.student_interests),
                preferred_location=st.session_state.preferred_location,
                top_n=top_n,
                filter_location=filter_by_loc
            )

        recommendations = recommendation_result["recommendations"]
        # Filter by minimum match threshold
        filtered_recs = [r for r in recommendations if r["match_percentage"] >= min_match_pct]

        # Top summary stats
        stat1, stat2, stat3, stat4 = st.columns(4)
        with stat1:
            best_match = filtered_recs[0]["match_percentage"] if filtered_recs else 0.0
            st.metric("Highest Match Score", f"{best_match:.1f}%")
        with stat2:
            avg_match = sum(r["match_percentage"] for r in filtered_recs) / len(filtered_recs) if filtered_recs else 0.0
            st.metric("Average Match Score", f"{avg_match:.1f}%")
        with stat3:
            st.metric("Total Matches Returned", len(filtered_recs))
        with stat4:
            avg_skill_cov = sum(r["skill_match_rate"] for r in filtered_recs) / len(filtered_recs) if filtered_recs else 0.0
            st.metric("Avg Skill Coverage", f"{avg_skill_cov:.1f}%")

        st.markdown("---")

        if not filtered_recs:
            st.info(f"No internships met the {min_match_pct}% match threshold. Try lowering the threshold or adding more skills in your profile.")
        else:
            st.markdown(f"### 🏆 Top {len(filtered_recs)} Recommended Opportunities")

            for item in filtered_recs:
                with st.container():
                    st.markdown(f"""
                    <div class="rec-card">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                            <div>
                                <h3 style="margin: 0 0 4px 0; font-size: 1.25rem; font-weight: 700; color: #F8FAFC;">
                                    {item['company_name']} — {item['internship_role']}
                                </h3>
                                <span class="domain-pill">{item['domain']}</span>
                                <span style="color: #64748B; font-size: 0.8rem; margin-left: 8px;">ID: {item['internship_id']}</span>
                            </div>
                            <div>
                                {get_match_badge_html(item['match_percentage'])}
                            </div>
                        </div>
                        <div style="color: #CBD5E1; font-size: 0.88rem; margin-bottom: 10px;">
                            📍 <strong>Location:</strong> {item['location']} &nbsp;|&nbsp;
                            ⏱️ <strong>Duration:</strong> {item['duration']} &nbsp;|&nbsp;
                            💰 <strong>Stipend:</strong> <span style="color: #34D399; font-weight: 600;">{item['stipend']}</span>
                        </div>
                        <div style="color: #94A3B8; font-size: 0.82rem; margin-bottom: 8px;">
                            🎓 <strong>Eligibility:</strong> {item['eligibility']}
                        </div>
                        {get_skills_badges_html(item['matched_skills'], item['missing_skills'])}
                    </div>
                    """, unsafe_allow_html=True)

                    # Interactive Application Tracking Action
                    btn_col1, btn_col2 = st.columns([1, 3])
                    with btn_col1:
                        # Check if already tracked
                        is_tracked = any(t["id"] == item["internship_id"] for t in st.session_state.tracked_applications)
                        if is_tracked:
                            st.button(f"✓ In Tracker ({item['internship_id']})", key=f"track_{item['internship_id']}", disabled=True)
                        else:
                            if st.button(f"📌 Add to Tracker", key=f"track_{item['internship_id']}"):
                                st.session_state.tracked_applications.append({
                                    "id": item["internship_id"],
                                    "company": item["company_name"],
                                    "role": item["internship_role"],
                                    "location": item["location"],
                                    "stipend": item["stipend"],
                                    "status": "Saved",
                                    "date": str(datetime.date.today()),
                                    "notes": f"Match Score: {item['match_percentage']}%. Skill Coverage: {item['skill_match_rate']}%."
                                })
                                st.toast(f"Added {item['company_name']} ({item['internship_role']}) to tracker!", icon="📌")
                                st.rerun()

                    with btn_col2:
                        with st.expander(f"🔍 Detailed Analysis & Why This Matches ({item['internship_id']})"):
                            st.write(f"**Required Competencies ({len(item['required_skills'])}):** {', '.join(item['required_skills'])}")
                            st.write(f"**Your Matched Competencies:** {', '.join(item['matched_skills']) if item['matched_skills'] else 'None'}")
                            st.write(f"**Exact Skill Gap:** {', '.join(item['missing_skills']) if item['missing_skills'] else 'None (100% matched!)'}")
                            st.write(f"**Skill Match Ratio:** {len(item['matched_skills'])} / {len(item['required_skills'])} required skills ({item['skill_match_rate']}%)")


# =============================================================================
# VIEW 3: SKILL GAP STUDIO
# =============================================================================
elif nav_selection == "📊 Skill Gap Studio":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">📊 Skill Gap Analysis Studio</h1>
        <p class="hero-subtitle">
            Visual breakdown of your missing competencies and intelligent upskilling roadmaps to maximize your internship match score.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Analyzing skill distributions and recommendations..."):
        recommendation_result = recommender.recommend(
            student_skills=st.session_state.student_skills,
            department=st.session_state.student_dept,
            interests=", ".join(st.session_state.student_interests),
            preferred_location=st.session_state.preferred_location,
            top_n=10
        )

    recommendations = recommendation_result["recommendations"]
    suggested_skills = recommendation_result["suggested_skills_to_learn"]

    # Visual Gauges & Top Upskilling Bar Chart
    v_col1, v_col2 = st.columns([1, 2])
    with v_col1:
        avg_rate = sum(r["skill_match_rate"] for r in recommendations) / len(recommendations) if recommendations else 0.0
        st.plotly_chart(create_skill_match_gauge(avg_rate), use_container_width=True)

        st.markdown(f"""
        <div style="background: #111827; border: 1px solid rgba(139, 92, 246, 0.2); border-radius: 10px; padding: 14px; margin-top: 10px;">
            <strong style="color: #A78BFA; font-size: 0.9rem;">💡 Readiness Insight</strong>
            <p style="color: #94A3B8; font-size: 0.82rem; margin: 4px 0 0 0;">
                You currently possess <strong>{avg_rate:.1f}%</strong> of all core competencies required across your top 10 recommended internships.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with v_col2:
        st.plotly_chart(create_suggested_skills_bar_chart(suggested_skills), use_container_width=True)

    st.markdown("---")

    # Actionable Upskilling Roadmap
    st.markdown("### 🚀 High-Impact Skills Learning Pathway")
    st.write("Acquiring these specific competencies will yield the largest immediate increase in your internship eligibility:")

    sug_cols = st.columns(min(len(suggested_skills), 3) if suggested_skills else 1)
    if suggested_skills:
        for idx, sug in enumerate(suggested_skills[:3]):
            with sug_cols[idx]:
                st.markdown(f"""
                <div style="background: linear-gradient(145deg, #161F37 0%, #1A2442 100%); border: 1px solid rgba(139, 92, 246, 0.35); border-radius: 12px; padding: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 1.15rem; font-weight: 700; color: #F1F5F9;">{sug['skill']}</span>
                        <span style="background: #4C1D95; color: #C084FC; font-size: 0.75rem; padding: 3px 8px; border-radius: 12px; font-weight: 700;">#{idx+1} Priority</span>
                    </div>
                    <div style="color: #34D399; font-weight: 600; font-size: 0.85rem; margin-bottom: 6px;">
                        +{sug['percentage_of_recommendations']}% Match Reach
                    </div>
                    <p style="color: #94A3B8; font-size: 0.8rem; margin: 0;">
                        {sug['insight']}. Learning this will directly unlock candidate qualification for these postings.
                    </p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("🎉 You have 100% skill coverage across your top recommendations!")

    st.markdown("---")

    # Detailed Skill Gap Matrix Table
    st.markdown("### 📋 Opportunity-by-Opportunity Skill Gap Matrix")
    gap_table_data = []
    for r in recommendations:
        gap_table_data.append({
            "Rank": f"#{r['rank']}",
            "Match %": f"{r['match_percentage']:.1f}%",
            "Company": r["company_name"],
            "Role": r["internship_role"],
            "Skill Coverage": f"{r['skill_match_rate']:.1f}%",
            "Matched Skills": ", ".join(r["matched_skills"]) if r["matched_skills"] else "None",
            "Missing Skills (Gap)": ", ".join(r["missing_skills"]) if r["missing_skills"] else "None (Full Match)"
        })

    st.dataframe(pd.DataFrame(gap_table_data), use_container_width=True, hide_index=True)


# =============================================================================
# VIEW 4: EXPLORE INTERNSHIPS (SEARCH & FILTER CATALOG)
# =============================================================================
elif nav_selection == "🔍 Explore Internships":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">🔍 Explore All Internships</h1>
        <p class="hero-subtitle">
            Search, filter, and inspect the complete catalog of 1,000 synthetic technical internship records.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Search & Filter Controls
    c_search, c_domain, c_loc = st.columns([2, 1.5, 1.5])
    with c_search:
        search_query = st.text_input("🔎 Search by Role, Company, or Skill keyword", placeholder="e.g. PyTorch, Google, Deep Learning, Bangalore")
    with c_domain:
        domain_filter = st.selectbox("Filter by Domain", ["All Domains"] + ALL_DOMAINS)
    with c_loc:
        location_filter = st.selectbox("Filter by Location", ["All Locations", "Remote Only", "Hybrid Only"] + ALL_LOCATIONS)

    dur_col, stip_col = st.columns(2)
    with dur_col:
        duration_filter = st.multiselect("Filter by Duration", ["2 Months", "3 Months", "4 Months", "6 Months"], default=[])
    with stip_col:
        stipend_type = st.radio("Stipend Type", ["All Stipends", "Paid Only", "Unpaid / Credit"], horizontal=True)

    # Apply Filters
    df_filtered = df_all.copy()

    if search_query.strip():
        q = search_query.strip().lower()
        df_filtered = df_filtered[
            df_filtered["Company Name"].str.lower().str.contains(q, na=False) |
            df_filtered["Internship Role"].str.lower().str.contains(q, na=False) |
            df_filtered["Required Skills"].str.lower().str.contains(q, na=False) |
            df_filtered["Domain"].str.lower().str.contains(q, na=False)
        ]

    if domain_filter != "All Domains":
        df_filtered = df_filtered[df_filtered["Domain"] == domain_filter]

    if location_filter == "Remote Only":
        df_filtered = df_filtered[df_filtered["Location"].str.contains("Remote", case=False, na=False)]
    elif location_filter == "Hybrid Only":
        df_filtered = df_filtered[df_filtered["Location"].str.contains("Hybrid", case=False, na=False)]
    elif location_filter != "All Locations":
        df_filtered = df_filtered[df_filtered["Location"] == location_filter]

    if duration_filter:
        df_filtered = df_filtered[df_filtered["Duration"].isin(duration_filter)]

    if stipend_type == "Paid Only":
        df_filtered = df_filtered[~df_filtered["Stipend"].str.contains("Unpaid", case=False, na=False)]
    elif stipend_type == "Unpaid / Credit":
        df_filtered = df_filtered[df_filtered["Stipend"].str.contains("Unpaid", case=False, na=False)]

    st.markdown(f"**Showing {len(df_filtered):,} of {len(df_all):,} internships**")

    # Tabbed View: Table vs Card View
    v_tab1, v_tab2 = st.tabs(["📋 Data Table View", "🗂️ Interactive Cards View"])

    with v_tab1:
        st.dataframe(
            df_filtered,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Stipend": st.column_config.TextColumn("Monthly Stipend"),
                "Required Skills": st.column_config.TextColumn("Skills"),
            }
        )

        # CSV Download Button
        csv_data = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Results as CSV",
            data=csv_data,
            file_name="filtered_internships.csv",
            mime="text/csv"
        )

    with v_tab2:
        if len(df_filtered) > 50:
            st.caption("Showing the first 50 results in Card View for optimal rendering speed.")
            card_subset = df_filtered.head(50)
        else:
            card_subset = df_filtered

        for _, row in card_subset.iterrows():
            skills_list = [s.strip() for s in row["Required Skills"].split(",")]
            skills_tags = "".join([f'<span class="skill-tag-matched">{s}</span>' for s in skills_list])
            st.markdown(f"""
            <div class="rec-card" style="margin-bottom: 12px; padding: 14px 18px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <h4 style="margin: 0; color: #F8FAFC;">{row['Company Name']} — {row['Internship Role']}</h4>
                    <span class="domain-pill">{row['Domain']}</span>
                </div>
                <div style="color: #94A3B8; font-size: 0.82rem; margin-bottom: 6px;">
                    📍 {row['Location']} &nbsp;|&nbsp; ⏱️ {row['Duration']} &nbsp;|&nbsp; 💰 <span style="color: #34D399;">{row['Stipend']}</span> &nbsp;|&nbsp; ID: {row['Internship ID']}
                </div>
                <div>{skills_tags}</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# VIEW 5: APPLICATION TRACKER
# =============================================================================
elif nav_selection == "📌 Application Tracker":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">📌 Internship Application Tracker</h1>
        <p class="hero-subtitle">
            Manage your recruitment pipeline, track application statuses, and add personal interview notes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tracked = st.session_state.tracked_applications

    # Metrics Bar
    cnt_saved = sum(1 for t in tracked if t["status"] == "Saved")
    cnt_applied = sum(1 for t in tracked if t["status"] == "Applied")
    cnt_interview = sum(1 for t in tracked if t["status"] == "Interview Scheduled")
    cnt_offered = sum(1 for t in tracked if t["status"] == "Offer Received")

    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    with t_col1:
        st.metric("Saved / Bookmarked", cnt_saved)
    with t_col2:
        st.metric("Applied", cnt_applied)
    with t_col3:
        st.metric("Interviewing", cnt_interview)
    with t_col4:
        st.metric("Offers Received", cnt_offered)

    st.markdown("---")

    if not tracked:
        st.info("You haven't saved any applications yet. Go to 'AI Recommendations' to add matching internships!")
    else:
        st.markdown(f"### 📋 Current Applications ({len(tracked)})")

        for idx, app in enumerate(tracked):
            with st.container():
                st.markdown(f"""
                <div style="background: #151D2F; border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 12px; padding: 14px 18px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <h4 style="margin: 0; color: #F8FAFC;">{app['company']} — {app['role']}</h4>
                        <span style="background: #3B82F6; color: #FFFFFF; font-size: 0.75rem; padding: 3px 10px; border-radius: 12px; font-weight: 700;">{app['status']}</span>
                    </div>
                    <div style="color: #94A3B8; font-size: 0.82rem;">
                        📍 {app['location']} &nbsp;|&nbsp; 💰 {app['stipend']} &nbsp;|&nbsp; 📅 Added: {app.get('date', 'N/A')} &nbsp;|&nbsp; ID: {app['id']}
                    </div>
                    <div style="color: #CBD5E1; font-size: 0.82rem; margin-top: 6px; font-style: italic;">
                        Notes: {app.get('notes', 'No notes added')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                act_col1, act_col2, act_col3 = st.columns([1.5, 2.5, 1])
                with act_col1:
                    new_status = st.selectbox(
                        "Update Status",
                        ["Saved", "Applied", "Interview Scheduled", "Offer Received", "Rejected"],
                        index=["Saved", "Applied", "Interview Scheduled", "Offer Received", "Rejected"].index(app["status"]),
                        key=f"status_sel_{idx}"
                    )
                    if new_status != app["status"]:
                        app["status"] = new_status
                        st.toast(f"Status updated to '{new_status}'!", icon="✅")
                        st.rerun()

                with act_col2:
                    note_edit = st.text_input("Edit Notes", value=app.get("notes", ""), key=f"note_edit_{idx}")
                    if note_edit != app.get("notes", ""):
                        app["notes"] = note_edit

                with act_col3:
                    st.write("")
                    st.write("")
                    if st.button("🗑️ Remove", key=f"del_{idx}"):
                        st.session_state.tracked_applications.pop(idx)
                        st.toast("Application removed from tracker.", icon="🗑️")
                        st.rerun()

        # Export tracker as CSV
        st.markdown("---")
        df_track = pd.DataFrame(st.session_state.tracked_applications)
        st.download_button(
            label="📥 Export Application Tracker (CSV)",
            data=df_track.to_csv(index=False).encode("utf-8"),
            file_name="my_tracked_applications.csv",
            mime="text/csv"
        )


# =============================================================================
# VIEW 6: MARKET ANALYTICS
# =============================================================================
elif nav_selection == "📈 Market Analytics":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">📈 Ecosystem Market Analytics</h1>
        <p class="hero-subtitle">
            Macro-level trends, in-demand technical competencies, and compensation insights across the synthetic internship market.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # High-level overview metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Internship Postings", f"{len(df_all):,}")
    with m2:
        st.metric("Hiring Companies", df_all["Company Name"].nunique())
    with m3:
        st.metric("Technical Domains", df_all["Domain"].nunique())
    with m4:
        st.metric("Unique Technical Skills", len(ALL_SKILLS))

    st.markdown("---")

    # Charts Row 1: Domain Distribution & Work Model
    ch_col1, ch_col2 = st.columns([1.5, 1])
    with ch_col1:
        st.plotly_chart(create_domain_donut_chart(df_all), use_container_width=True)
    with ch_col2:
        st.plotly_chart(create_work_model_chart(df_all), use_container_width=True)

    st.markdown("---")

    # Charts Row 2: Top Skills & Stipend by Domain
    st.plotly_chart(create_top_skills_chart(df_all, top_n=15), use_container_width=True)
    st.markdown("---")
    st.plotly_chart(create_stipend_by_domain_chart(df_all), use_container_width=True)
