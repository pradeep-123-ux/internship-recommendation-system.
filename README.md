# Synthetic Internship Dataset

> **DATASET CLASSIFICATION: SYNTHETIC / DEMONSTRATION DATA**  
> **Notice:** The records in `internships.csv` are artificially generated synthetic records designed exclusively for testing, benchmarking, and developing the **AI-Based Internship Recommendation and Skill Gap Analysis System**. None of these entries represent actual live hiring postings or official endorsements from the listed companies.

## Dataset Specifications

- **File Name:** `internships.csv`
- **Total Records:** 1,000 synthetic internship entries
- **Generator Script:** `src/generate_dataset.py`
- **Format:** Comma-Separated Values (CSV, UTF-8 encoded)

## Columns Schema

| # | Column Name | Data Type | Description | Example |
|---|-------------|-----------|-------------|---------|
| 1 | `Internship ID` | String | Unique identifier for each synthetic record | `INT-0001` |
| 2 | `Company Name` | String | Name of the hiring company (synthetic listing) | `Google`, `Flipkart`, `Cohere` |
| 3 | `Internship Role` | String | Job title for the internship | `Machine Learning Engineer Intern` |
| 4 | `Domain` | String | Technical domain category | `Artificial Intelligence & Machine Learning` |
| 5 | `Required Skills` | String | Comma-separated list of required technical skills | `Python, Machine Learning, Scikit-learn, PyTorch` |
| 6 | `Location` | String | Office location, hybrid mode, or remote designation | `Bangalore, India`, `Remote (India)` |
| 7 | `Duration` | String | Expected internship duration | `3 Months`, `6 Months` |
| 8 | `Stipend` | String | Monthly compensation or academic research credit | `₹35,000 / month` |
| 9 | `Eligibility` | String | Academic, CGPA, graduation, or branch criteria | `Pre-final or Final year B.Tech/B.E (CS/IT/ECE/AI/DS)` |

## Reproduction

To regenerate or customize this dataset:

```bash
python src/generate_dataset.py
```

Optional parameters:
- `--count 1000`: Number of records to generate
- `--seed 42`: Random seed for deterministic reproducibility
- `--output datasets/internships.csv`: Custom output path
