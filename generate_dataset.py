"""
generate_dataset.py
-------------------
Synthetic Dataset Generator for AI-Based Internship Recommendation
and Skill Gap Analysis System.

DISCLAIMER / DATASET LABEL:
This script generates a SYNTHETIC demonstration dataset. All company postings,
stipends, roles, requirements, and IDs are simulated for educational, prototyping,
and algorithm testing purposes.

Target Output: datasets/internships.csv (1,000 records)
Columns:
    1. Internship ID
    2. Company Name
    3. Internship Role
    4. Domain
    5. Required Skills
    6. Location
    7. Duration
    8. Stipend
    9. Eligibility
"""

import os
import random
import argparse
import pandas as pd


# -----------------------------------------------------------------------------
# DOMAIN, ROLE, AND SKILL CONFIGURATION
# -----------------------------------------------------------------------------
DOMAINS_CONFIG = {
    "Artificial Intelligence & Machine Learning": {
        "roles": [
            "Machine Learning Engineer Intern",
            "AI Research Intern",
            "Deep Learning Intern",
            "Applied AI Intern",
            "MLOps Intern"
        ],
        "core_skills": ["Python", "Machine Learning", "Scikit-learn"],
        "extended_skills": [
            "PyTorch", "TensorFlow", "Deep Learning", "Keras", "NumPy",
            "Pandas", "Model Evaluation", "Docker", "Git", "MLflow",
            "Data Preprocessing", "Hugging Face"
        ]
    },
    "Data Science & Analytics": {
        "roles": [
            "Data Science Intern",
            "Data Analyst Intern",
            "Business Intelligence Intern",
            "Data Engineering Intern",
            "Quantitative Research Intern"
        ],
        "core_skills": ["Python", "SQL", "Data Analysis"],
        "extended_skills": [
            "Pandas", "NumPy", "Tableau", "Power BI", "Scikit-learn",
            "Matplotlib", "Seaborn", "Statistics", "Excel", "ETL Pipelines",
            "BigQuery", "Snowflake"
        ]
    },
    "Natural Language Processing & Generative AI": {
        "roles": [
            "Generative AI Intern",
            "NLP Engineer Intern",
            "Prompt Engineering Intern",
            "LLM Solutions Intern"
        ],
        "core_skills": ["Python", "Natural Language Processing", "Transformers"],
        "extended_skills": [
            "PyTorch", "Hugging Face", "LangChain", "LlamaIndex",
            "OpenAI API", "Vector Databases", "BERT", "Prompt Engineering",
            "RAG Architecture", "Pinecone", "Fine-Tuning"
        ]
    },
    "Computer Vision & Robotics": {
        "roles": [
            "Computer Vision Intern",
            "Robotics Software Intern",
            "Perception Systems Intern",
            "Autonomous Systems Intern"
        ],
        "core_skills": ["Python", "Computer Vision", "OpenCV"],
        "extended_skills": [
            "PyTorch", "TensorFlow", "YOLO", "Image Processing",
            "Object Detection", "ROS", "C++", "3D Reconstruction",
            "Point Cloud Library", "CUDA"
        ]
    },
    "Full Stack Web Development": {
        "roles": [
            "Full Stack Developer Intern",
            "MERN Stack Intern",
            "Web Development Intern",
            "Software Development Engineer (SDE) Intern"
        ],
        "core_skills": ["JavaScript", "React.js", "Node.js"],
        "extended_skills": [
            "TypeScript", "Express.js", "MongoDB", "PostgreSQL",
            "REST APIs", "HTML5", "CSS3", "Tailwind CSS", "Next.js",
            "GraphQL", "Git", "Redux"
        ]
    },
    "Backend Engineering": {
        "roles": [
            "Backend Developer Intern",
            "Python Backend Intern",
            "Java Software Intern",
            "Go Backend Intern",
            "API Developer Intern"
        ],
        "core_skills": ["REST APIs", "Git", "Databases"],
        "extended_skills": [
            "Python", "Django", "FastAPI", "Java", "Spring Boot",
            "Go", "PostgreSQL", "MySQL", "Redis", "Docker",
            "Microservices", "Kafka"
        ]
    },
    "Frontend Engineering": {
        "roles": [
            "Frontend Developer Intern",
            "React Developer Intern",
            "UI Developer Intern",
            "Web Design & Frontend Intern"
        ],
        "core_skills": ["JavaScript", "HTML5", "CSS3"],
        "extended_skills": [
            "React.js", "TypeScript", "Next.js", "Tailwind CSS",
            "Redux Toolkit", "Vue.js", "Responsive Design",
            "Webpack", "Bootstrap", "Figma to Code"
        ]
    },
    "Cloud Computing & DevOps": {
        "roles": [
            "DevOps Engineer Intern",
            "Cloud Computing Intern",
            "Site Reliability Engineering (SRE) Intern",
            "Platform Engineering Intern"
        ],
        "core_skills": ["Linux", "Docker", "Git"],
        "extended_skills": [
            "Kubernetes", "AWS", "CI/CD", "Terraform", "GitHub Actions",
            "Azure", "Google Cloud Platform", "Bash Scripting",
            "Prometheus", "Grafana", "Ansible"
        ]
    },
    "Cybersecurity & InfoSec": {
        "roles": [
            "Cybersecurity Analyst Intern",
            "Information Security Intern",
            "SOC Analyst Intern",
            "Penetration Testing Intern"
        ],
        "core_skills": ["Network Security", "Linux", "Cybersecurity Fundamentals"],
        "extended_skills": [
            "Wireshark", "Python", "Vulnerability Assessment",
            "Penetration Testing", "SIEM", "Cryptography",
            "OWASP Top 10", "Metasploit", "Incident Response", "Nmap"
        ]
    },
    "Mobile Application Development": {
        "roles": [
            "Mobile App Developer Intern",
            "Flutter Developer Intern",
            "Android Developer Intern",
            "iOS Developer Intern",
            "React Native Intern"
        ],
        "core_skills": ["Mobile App Development", "Git", "REST APIs"],
        "extended_skills": [
            "Flutter", "Dart", "Kotlin", "Swift", "React Native",
            "Android Studio", "Xcode", "Firebase", "State Management",
            "SQLite", "Push Notifications"
        ]
    },
    "UI/UX Design & Product Design": {
        "roles": [
            "UI/UX Design Intern",
            "Product Design Intern",
            "User Researcher Intern",
            "Interaction Design Intern"
        ],
        "core_skills": ["Figma", "UI/UX Design", "Wireframing"],
        "extended_skills": [
            "Prototyping", "User Research", "Usability Testing",
            "Adobe XD", "Design Systems", "Information Architecture",
            "Design Thinking", "Interaction Design", "Responsive Layouts"
        ]
    },
    "Blockchain & Web3": {
        "roles": [
            "Blockchain Developer Intern",
            "Smart Contract Intern",
            "Web3 Research Intern"
        ],
        "core_skills": ["Blockchain Fundamentals", "Solidity", "Git"],
        "extended_skills": [
            "Ethereum", "Web3.js", "Ethers.js", "Smart Contracts",
            "Hardhat", "Cryptography", "Rust", "DeFi Protocols", "IPFS"
        ]
    }
}

# -----------------------------------------------------------------------------
# COMPANY NAMES (Diverse mix of global giants, Indian unicorns & AI startups)
# -----------------------------------------------------------------------------
COMPANIES = [
    # Global Tech Giants
    "Google", "Microsoft", "Amazon", "Meta", "Apple", "NVIDIA", "Adobe",
    "IBM", "Oracle", "Cisco", "Intel", "Salesforce", "Atlassian",
    # Indian Unicorns & Product Leaders
    "Flipkart", "Swiggy", "Zomato", "Razorpay", "CRED", "PhonePe",
    "Zerodha", "Meesho", "Infosys", "Tata Consultancy Services (TCS)",
    "Wipro", "Jio Platforms", "Zoho Corporation", "Freshworks", "InMobi",
    # Fast-growing Startups & AI Labs
    "Anthropic Labs", "OpenAI Research", "Cohere", "Scale AI",
    "Databricks", "Snowflake", "Hugging Face Hub", "DataRobot",
    "Synthetix AI", "ByteWave Technologies", "NexaCore Analytics",
    "QuantumLeap AI", "CloudMatrix Systems", "Innovaccer Health",
    "Urban Company", "Postman", "Hasura", "BrowserStack"
]

# -----------------------------------------------------------------------------
# LOCATIONS (Tier-1 Tech Hubs, Hybrid, and Remote Opportunities)
# -----------------------------------------------------------------------------
LOCATIONS = [
    "Bangalore, India",
    "Hyderabad, India",
    "Pune, India",
    "Mumbai, India",
    "Delhi NCR (Gurgaon / Noida), India",
    "Chennai, India",
    "Remote (India)",
    "Remote (Global)",
    "Hybrid - Bangalore",
    "Hybrid - Hyderabad",
    "Hybrid - Pune",
    "Hybrid - Delhi NCR",
    "Hybrid - Mumbai",
    "San Francisco, CA (Remote)",
    "New York, NY (Hybrid)",
    "London, UK (Remote)",
    "Singapore (Hybrid)"
]

# -----------------------------------------------------------------------------
# DURATIONS
# -----------------------------------------------------------------------------
DURATIONS = [
    "2 Months",
    "3 Months",
    "4 Months",
    "6 Months"
]

# -----------------------------------------------------------------------------
# STIPENDS (Realistic competitive monthly stipends)
# -----------------------------------------------------------------------------
STIPENDS = [
    "₹15,000 / month",
    "₹20,000 / month",
    "₹25,000 / month",
    "₹30,000 / month",
    "₹35,000 / month",
    "₹40,000 / month",
    "₹45,000 / month",
    "₹50,000 / month",
    "₹60,000 / month",
    "₹75,000 / month",
    "₹80,000 / month",
    "₹1,00,000 / month",
    "Performance-based (₹15,000 - ₹30,000 / month)",
    "Unpaid (Academic Research Credit)"
]

# -----------------------------------------------------------------------------
# ELIGIBILITY CRITERIA
# -----------------------------------------------------------------------------
ELIGIBILITIES = [
    "Pre-final or Final year B.Tech/B.E (CS/IT/ECE/AI/DS) or MCA with 7.0+ CGPA",
    "B.Tech/B.E/BCA/B.Sc in Computer Science, IT, or related quantitative discipline",
    "Undergraduate or Master's students in CS, Data Science, Statistics, or Math",
    "3rd or 4th-year Engineering students with demonstrable coding portfolio/projects",
    "B.Tech/Dual Degree students available for full-time 6 months internship",
    "Open to all engineering students with strong algorithmic and problem-solving skills",
    "Graduating Batch 2025/2026/2027 in CS/IT/Data Science with no active backlogs",
    "Bachelor's or Master's degree candidates in Design, HCI, or Computer Science",
    "Postgraduate students (M.Tech/MS/Ph.D.) with specialized research experience",
    "Self-taught developers or college students with active GitHub projects"
]


def generate_synthetic_record(rng: random.Random, record_index: int) -> dict:
    """
    Generates a single synthetic internship record with realistic associations
    between Domain, Role, and Required Skills.
    """
    domain = rng.choice(list(DOMAINS_CONFIG.keys()))
    config = DOMAINS_CONFIG[domain]
    
    role = rng.choice(config["roles"])
    
    # Select 2 core skills + 2 to 4 complementary extended skills
    num_core = rng.randint(2, min(3, len(config["core_skills"])))
    selected_core = rng.sample(config["core_skills"], num_core)
    
    num_extended = rng.randint(2, 4)
    selected_extended = rng.sample(config["extended_skills"], num_extended)
    
    # Combined unique skills
    skills_list = selected_core + [s for s in selected_extended if s not in selected_core]
    skills_str = ", ".join(skills_list)
    
    company = rng.choice(COMPANIES)
    location = rng.choice(LOCATIONS)
    duration = rng.choice(DURATIONS)
    stipend = rng.choice(STIPENDS)
    eligibility = rng.choice(ELIGIBILITIES)
    internship_id = f"INT-{record_index:04d}"

    return {
        "Internship ID": internship_id,
        "Company Name": company,
        "Internship Role": role,
        "Domain": domain,
        "Required Skills": skills_str,
        "Location": location,
        "Duration": duration,
        "Stipend": stipend,
        "Eligibility": eligibility
    }


def generate_dataset(num_records: int = 1000,
                     output_path: str = "datasets/internships.csv",
                     seed: int = 42) -> pd.DataFrame:
    """
    Generates `num_records` synthetic internship entries and writes them to CSV.
    """
    rng = random.Random(seed)
    
    records = []
    for i in range(1, num_records + 1):
        record = generate_synthetic_record(rng, i)
        records.append(record)
        
    df = pd.DataFrame(records)
    
    # Ensure destination directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    return df


def print_synthetic_banner():
    banner = """
================================================================================
  AI-BASED INTERNSHIP RECOMMENDATION & SKILL GAP ANALYSIS SYSTEM
  [NOTICE: SYNTHETIC DATASET GENERATION - NOT ACTUAL COMPANY HIRING POSTINGS]
================================================================================
"""
    print(banner)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic internship records for recommendation and skill gap analysis."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of synthetic records to generate (default: 1000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/internships.csv",
        help="Output CSV file path (default: datasets/internships.csv)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic dataset reproducibility (default: 42)"
    )
    args = parser.parse_args()

    print_synthetic_banner()
    print(f"[*] Generating {args.count:,} synthetic internship records...")
    print(f"[*] Random Seed: {args.seed}")
    print(f"[*] Target Output: {args.output}")

    df = generate_dataset(
        num_records=args.count,
        output_path=args.output,
        seed=args.seed
    )

    print("\n[+] Dataset successfully generated and saved!")
    print(f"[+] Total Records: {len(df)}")
    print(f"[+] Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"[+] File Size: {os.path.getsize(args.output):,} bytes")
    print(f"[+] Unique Domains ({df['Domain'].nunique()}):")
    for domain, count in df['Domain'].value_counts().items():
        print(f"    - {domain}: {count} records")

    print("\n--- SAMPLE SYNTHETIC RECORDS (First 3) ---")
    for idx, row in df.head(3).iterrows():
        print(f"\nRecord #{idx + 1}:")
        for col in df.columns:
            print(f"  {col}: {row[col]}")
    print("=" * 80)
    print("DATASET CLASSIFICATION: SYNTHETIC / DEMONSTRATION")
    print("=" * 80)


if __name__ == "__main__":
    main()
