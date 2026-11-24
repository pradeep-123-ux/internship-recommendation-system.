"""
recommendation_engine.py
-------------------------
AI-Based Internship Recommendation and Skill Gap Analysis Engine

This module provides content-based recommendation using Scikit-learn's
TF-IDF Vectorizer and Cosine Similarity to match student profiles
(skills, department, interests, and preferred location) against
internship opportunities.

Key Capabilities:
1. Computes Cosine Similarity between student profile and internship features.
2. Formats similarity as a matching percentage (e.g., 85.4%).
3. Identifies exact missing skills (Skill Gap Analysis) for every recommendation.
4. Generates high-impact skill recommendations to help students improve match scores.
5. Supports location filtering (in-city, hybrid, or remote).

Usage:
    python src/recommendation_engine.py
"""

import os
from collections import Counter
from typing import List, Dict, Any, Union
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class InternshipRecommender:
    """
    Recommendation Engine for matching student profiles to internships
    and analyzing skill gaps using TF-IDF and Cosine Similarity.
    """

    def __init__(self, dataset_path: str = "datasets/internships.csv"):
        """
        Initializes the recommendation engine by loading data and
        pre-computing TF-IDF vectors for all internships.

        Parameters:
            dataset_path (str): Path to the internships CSV dataset.
        """
        self.dataset_path = dataset_path
        self.df = None
        self.vectorizer = None
        self.internship_tfidf_matrix = None

        # Load and prepare the dataset
        self.load_data()
        self.prepare_features()

    def load_data(self):
        """Loads the internships dataset from the specified CSV path."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset not found at '{self.dataset_path}'. "
                f"Please run 'python src/generate_dataset.py' first."
            )

        self.df = pd.read_csv(self.dataset_path)

        # Ensure all required columns exist
        required_cols = [
            "Internship ID", "Company Name", "Internship Role", "Domain",
            "Required Skills", "Location", "Duration", "Stipend", "Eligibility"
        ]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Dataset is missing required columns: {missing_cols}")

        # Fill any unexpected missing values with empty strings
        self.df = self.df.fillna("")

    def _create_internship_text_representation(self, row: pd.Series) -> str:
        """
        Constructs a combined textual representation for an internship.
        Required Skills are weighted heavily (repeated 3x) to ensure skill
        relevance drives the recommendation score.
        """
        skills_weighted = f"{row['Required Skills']} {row['Required Skills']} {row['Required Skills']}"
        role = row["Internship Role"]
        domain = row["Domain"]
        eligibility = row["Eligibility"]

        # Combined representation
        return f"{skills_weighted} {role} {domain} {eligibility}"

    def prepare_features(self):
        """
        Builds the TF-IDF feature matrix across all internship descriptions.
        Uses a custom token pattern to preserve technical tokens such as
        'react.js', 'node.js', 'scikit-learn', 'c++', etc.
        """
        # Create combined text for each internship record
        combined_texts = self.df.apply(self._create_internship_text_representation, axis=1)

        # Initialize TF-IDF Vectorizer
        # token_pattern preserves hyphens, dots, and plus signs (e.g. C++, .NET, React.js)
        self.vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b[\w\.\#\+\-]+\b",
            stop_words="english",
            lowercase=True
        )

        # Compute TF-IDF matrix for all internships
        self.internship_tfidf_matrix = self.vectorizer.fit_transform(combined_texts)

    @staticmethod
    def _parse_skills_list(skills_input: Union[str, List[str]]) -> List[str]:
        """
        Normalizes a skills input (comma-separated string or list) into
        a clean, whitespace-stripped list of skill names.
        """
        if isinstance(skills_input, str):
            skills = [s.strip() for s in skills_input.split(",") if s.strip()]
        elif isinstance(skills_input, list):
            skills = [str(s).strip() for s in skills_input if str(s).strip()]
        else:
            skills = []
        return skills

    @staticmethod
    def identify_skill_gap(required_skills_str: str,
                            student_skills: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Compares the internship's required skills with the student's skills
        to identify matched skills and missing skills (Skill Gap).

        Parameters:
            required_skills_str (str): Comma-separated required skills from dataset.
            student_skills (str or list): Skills currently possessed by the student.

        Returns:
            dict containing:
                - matched_skills (list of str): Skills the student possesses.
                - missing_skills (list of str): Skills the student lacks.
                - match_rate (float): Percentage of required skills the student has.
        """
        internship_skills = InternshipRecommender._parse_skills_list(required_skills_str)
        student_skills_list = InternshipRecommender._parse_skills_list(student_skills)

        # Case-insensitive comparison sets
        student_skills_lower = {s.lower() for s in student_skills_list}

        matched = []
        missing = []

        for skill in internship_skills:
            if skill.lower() in student_skills_lower:
                matched.append(skill)
            else:
                missing.append(skill)

        total_req = len(internship_skills)
        match_rate = round((len(matched) / total_req * 100), 1) if total_req > 0 else 0.0

        return {
            "matched_skills": matched,
            "missing_skills": missing,
            "match_rate": match_rate
        }

    @staticmethod
    def suggest_skills_to_learn(recommended_internships: List[Dict[str, Any]],
                                 top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Aggregates missing skills across all top recommendations and ranks
        the most frequently required skills to guide student upskilling.

        Parameters:
            recommended_internships (list of dict): Top recommended internship records.
            top_k (int): Number of top skills to suggest (default: 5).

        Returns:
            list of dicts containing suggested skill name, frequency, and percentage.
        """
        missing_skills_counter = Counter()

        for item in recommended_internships:
            for skill in item.get("missing_skills", []):
                missing_skills_counter[skill] += 1

        total_recommendations = len(recommended_internships)
        suggestions = []

        for skill, count in missing_skills_counter.most_common(top_k):
            pct = round((count / total_recommendations * 100), 1) if total_recommendations > 0 else 0.0
            suggestions.append({
                "skill": skill,
                "frequency": count,
                "percentage_of_recommendations": pct,
                "insight": f"Required by {count}/{total_recommendations} of your top matches ({pct}%)"
            })

        return suggestions

    def recommend(self,
                  student_skills: Union[str, List[str]],
                  department: str = "",
                  interests: str = "",
                  preferred_location: str = "Any",
                  top_n: int = 10,
                  filter_location: bool = True) -> Dict[str, Any]:
        """
        Generates personalized internship recommendations based on student profile.

        Parameters:
            student_skills (str or list): Skills possessed by the student.
            department (str): Student's academic department (e.g., 'Computer Science').
            interests (str): Areas of interest (e.g., 'Machine Learning, Computer Vision').
            preferred_location (str): Preferred city or 'Remote' or 'Any'.
            top_n (int): Number of top recommendations to return (default: 10).
            filter_location (bool): If True, filters or prioritizes preferred location.

        Returns:
            dict containing:
                - student_profile: Echo of user input
                - recommendations: List of top matching internships with match % and skill gap
                - suggested_skills: Top skills to learn to boost match rate
        """
        # Parse and format student skills
        skills_list = self._parse_skills_list(student_skills)
        skills_str = ", ".join(skills_list)

        # Construct weighted student profile query
        # Skills are weighted 3x to match the internship representation
        query_text = (
            f"{skills_str} {skills_str} {skills_str} "
            f"{interests} {interests} "
            f"{department}"
        ).strip()

        if not query_text:
            raise ValueError("Student profile cannot be empty. Please provide skills or interests.")

        # Transform student query using the fitted TF-IDF Vectorizer
        student_vector = self.vectorizer.transform([query_text])

        # Compute Cosine Similarity between student vector and all internships
        # similarity_scores is an array of shape (1, num_internships)
        similarity_scores = cosine_similarity(student_vector, self.internship_tfidf_matrix).flatten()

        # Create a working copy of the dataframe with similarity scores
        results_df = self.df.copy()
        results_df["Similarity_Score"] = similarity_scores
        results_df["Match_Percentage"] = (results_df["Similarity_Score"] * 100).round(1)

        # Apply location filtering if requested and not "Any"
        loc_pref = preferred_location.strip().lower()
        if filter_location and loc_pref and loc_pref not in ["any", "all", "none"]:
            # Match preferred location or remote options (as remote is accessible from anywhere)
            loc_mask = (
                results_df["Location"].str.lower().str.contains(loc_pref, na=False) |
                results_df["Location"].str.lower().str.contains("remote", na=False)
            )
            filtered_df = results_df[loc_mask]

            # If enough matches exist in preferred location, use filtered subset
            # Otherwise keep all but preserve matching ones at the top
            if len(filtered_df) >= top_n:
                results_df = filtered_df
            elif len(filtered_df) > 0:
                # Add a location relevance boost to matching locations
                results_df.loc[loc_mask, "Similarity_Score"] += 0.05
                results_df["Match_Percentage"] = (results_df["Similarity_Score"] * 100).round(1)

        # Sort internships by similarity score in descending order
        top_matches_df = results_df.sort_values(by="Similarity_Score", ascending=False).head(top_n)

        # Build detailed recommendation list with Skill Gap Analysis
        recommendations = []
        for rank, (_, row) in enumerate(top_matches_df.iterrows(), start=1):
            gap_info = self.identify_skill_gap(row["Required Skills"], skills_list)

            # Cap display percentage at 100%
            match_pct = min(100.0, float(row["Match_Percentage"]))

            recommendation_item = {
                "rank": rank,
                "internship_id": row["Internship ID"],
                "company_name": row["Company Name"],
                "internship_role": row["Internship Role"],
                "domain": row["Domain"],
                "location": row["Location"],
                "duration": row["Duration"],
                "stipend": row["Stipend"],
                "eligibility": row["Eligibility"],
                "required_skills": [s.strip() for s in row["Required Skills"].split(",")],
                "match_percentage": match_pct,
                "matched_skills": gap_info["matched_skills"],
                "missing_skills": gap_info["missing_skills"],
                "skill_match_rate": gap_info["match_rate"]
            }
            recommendations.append(recommendation_item)

        # Identify top skills to learn across all recommended opportunities
        suggested_skills = self.suggest_skills_to_learn(recommendations, top_k=5)

        return {
            "student_profile": {
                "skills": skills_list,
                "department": department,
                "interests": interests,
                "preferred_location": preferred_location
            },
            "recommendations": recommendations,
            "suggested_skills_to_learn": suggested_skills
        }


def display_recommendations_cli(result: Dict[str, Any]):
    """
    Neatly formats and prints recommendation results and skill gap analysis
    to the command-line console.
    """
    profile = result["student_profile"]
    recommendations = result["recommendations"]
    suggestions = result["suggested_skills_to_learn"]

    print("=" * 85)
    print("           AI INTERNSHIP RECOMMENDATION & SKILL GAP REPORT")
    print("=" * 85)
    print(f" Department         : {profile['department'] or 'Not specified'}")
    print(f" Key Interests      : {profile['interests'] or 'Not specified'}")
    print(f" Preferred Location : {profile['preferred_location']}")
    print(f" Student Skills ({len(profile['skills'])}): {', '.join(profile['skills'])}")
    print("=" * 85)

    print(f"\n>>> TOP {len(recommendations)} MATCHING INTERNSHIPS:\n")

    for item in recommendations:
        print(f"#{item['rank']} | Match: {item['match_percentage']:.1f}% | {item['company_name']} - {item['internship_role']}")
        print(f"     ID         : {item['internship_id']} | Domain: {item['domain']}")
        print(f"     Location   : {item['location']} | Duration: {item['duration']} | Stipend: {item['stipend']}")
        print(f"     Eligibility: {item['eligibility']}")
        print(f"     Required   : {', '.join(item['required_skills'])}")
        
        # Skill gap breakdown
        matched_str = ", ".join(item['matched_skills']) if item['matched_skills'] else "None"
        missing_str = ", ".join(item['missing_skills']) if item['missing_skills'] else "None (Full match!)"
        print(f"     [+] Matched Skills ({len(item['matched_skills'])}): {matched_str}")
        print(f"     [-] Missing Skills ({len(item['missing_skills'])}): {missing_str}")
        print("-" * 85)

    print("\n" + "=" * 85)
    print(">>> RECOMMENDED SKILLS TO LEARN (To maximize your internship match score)")
    print("=" * 85)
    if suggestions:
        for idx, sug in enumerate(suggestions, 1):
            print(f"  {idx}. {sug['skill']} -> {sug['insight']}")
    else:
        print("  You already possess all required skills for your top recommendations! Great job!")
    print("=" * 85 + "\n")


def test_recommendation_engine():
    """
    Self-test function demonstrating recommendations on sample student profiles.
    """
    print("[*] Initializing Recommendation Engine...")
    recommender = InternshipRecommender(dataset_path="datasets/internships.csv")
    print(f"[+] Loaded {len(recommender.df)} internships from dataset.\n")

    # Sample Test Student 1: AI / Machine Learning Enthusiast
    print("\n" + "#" * 85)
    print("# TEST CASE 1: AI & Machine Learning Student")
    print("#" * 85)
    result_ai = recommender.recommend(
        student_skills="Python, Machine Learning, Scikit-learn, Pandas, SQL",
        department="Computer Science & Engineering",
        interests="Artificial Intelligence, Machine Learning, Deep Learning",
        preferred_location="Bangalore",
        top_n=10
    )
    display_recommendations_cli(result_ai)

    # Sample Test Case 2: Full Stack / Web Developer Student
    print("\n" + "#" * 85)
    print("# TEST CASE 2: Full Stack Web Development Student")
    print("#" * 85)
    result_web = recommender.recommend(
        student_skills=["JavaScript", "React.js", "HTML5", "CSS3", "Git"],
        department="Information Technology",
        interests="Full Stack Web Development, Frontend, React",
        preferred_location="Remote",
        top_n=10
    )
    display_recommendations_cli(result_web)


def run_interactive_session():
    """
    Runs an interactive terminal session where the user can enter
    their custom profile and receive immediate recommendations.
    """
    print("\n" + "=" * 85)
    print("      INTERACTIVE AI INTERNSHIP RECOMMENDATION & SKILL GAP TESTER")
    print("=" * 85)

    recommender = InternshipRecommender(dataset_path="datasets/internships.csv")

    print("\nPlease enter the student profile details below:")
    skills_in = input("1. Skills (comma-separated, e.g. Python, SQL, Machine Learning): ").strip()
    dept_in = input("2. Academic Department (e.g. Computer Science): ").strip()
    interests_in = input("3. Interests (e.g. Data Science, Machine Learning): ").strip()
    location_in = input("4. Preferred Location (e.g. Bangalore, Remote, or Any) [Any]: ").strip() or "Any"
    top_n_str = input("5. Number of recommendations (default: 10): ").strip()
    top_n = int(top_n_str) if top_n_str.isdigit() else 10

    print("\n[*] Processing profile and computing similarity scores...")
    result = recommender.recommend(
        student_skills=skills_in,
        department=dept_in,
        interests=interests_in,
        preferred_location=location_in,
        top_n=top_n
    )
    display_recommendations_cli(result)


def main():
    """
    Main entry point for testing the recommendation engine.
    Supports both automated test cases and an interactive terminal mode.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="AI Internship Recommendation and Skill Gap Analysis Engine"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode to test with custom student inputs."
    )
    args = parser.parse_args()

    if args.interactive:
        run_interactive_session()
    else:
        test_recommendation_engine()


if __name__ == "__main__":
    main()

