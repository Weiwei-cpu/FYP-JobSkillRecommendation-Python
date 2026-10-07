# Job Recommendation and Skill Matching System (Final Year Project)

An end-to-end Machine Learning and Natural Language Processing (NLP) study evaluating and comparing multiple recommendation and information retrieval techniques for job vacancies and skill matching.

---

## 📌 Project Overview

In the modern recruitment ecosystem, matching candidates with suitable job vacancies—and aligning job descriptions with standardized skill taxonomies—is a critical challenge. 

This project explores, implements, and benchmarks diverse modeling paradigms:
1. **Lexical / Frequency-based IR**: BM25 Okapi for keyword-based search and relevance.
2. **Dense Semantic Embeddings**: Sentence-BERT (SBERT) for contextual representation and semantic similarity.
3. **Gradient Boosting & Learning to Rank**: LightGBM (Ranker / Classifier) and XGBoost.
4. **Collaborative Filtering**: Bayesian Personalized Ranking (BPR) using the `implicit` library.
5. **Hybrid & Ensemble Architectures**: Combining semantic embeddings with ensemble tree models (Random Forest, XGBoost).
6. **Efficiency & Resource Benchmarking**: Monitoring inference runtime and RAM usage (`psutil`) alongside accuracy metrics.

---

## 📊 Datasets

The project utilizes two distinct datasets across the experiments:

1. **Dataset 1: MyCareersFuture Job Postings (`mycareersfuture (1).json`)**
   - Scraped / collected real-world job posting data from Singapore's MyCareersFuture portal.
   - Contains job titles, descriptions, requirements, company data, and skill requirements.

2. **Dataset 2: TechWolf Vacancy-Job-to-Skill Benchmark (`TechWolf/vacancy-job-to-skill`)**
   - Sourced via Hugging Face (`load_dataset("TechWolf/vacancy-job-to-skill")`).
   - Standardized job postings mapped to structured skill annotations for validation and benchmark testing.

---

## 🛠️ Models & Methodologies

| Category | Model / Algorithm | Purpose |
| :--- | :--- | :--- |
| **Lexical Search** | **BM25 Okapi** (`rank_bm25`) | Baseline lexical matching based on token frequencies and inverse document frequencies. |
| **Semantic Matching** | **Sentence-BERT** (`sentence-transformers`) | Dense vector embeddings capturing semantic context beyond exact keyword matches. |
| **Recommender Systems** | **Bayesian Personalized Ranking (BPR)** (`implicit`) | Matrix factorization optimized for implicit feedback in job-skill interactions. |
| **Tree-based / GBDT** | **LightGBM & XGBoost** | Learning-to-rank and classification on tabularized feature representations. |
| **Ensemble Models** | **SBERT + XGBoost** | Multi-stage stacking and ensemble approaches combining semantic features with decision forests. |

---

## 📈 Evaluation Metrics

- **Jaccard Similarity / MultiLabel Score**: Measuring overlap between predicted and ground-truth skill sets.
- **Cosine Similarity**: Assessing semantic alignment between job queries and candidates/skills.
- **Ranking & Classification Metrics**: Precision, Mean Squared Error, and ranking consistency.
- **Resource Profiling**: Execution time profiling and RAM usage monitoring during training and inference.

---

## 📁 Repository Structure

```text
├── 1st_Dataset_Model_Implementation.ipynb   # Experimentation, preprocessing, & models on MyCareersFuture data
├── 2nd Dataset Model Implementation.ipynb   # Benchmark models on TechWolf dataset
├── mycareersfuture (1).json                 # Raw dataset for Dataset 1
├── .gitignore                               # Git ignore configuration
└── README.md                                # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed (recommended Python 3.9 - 3.11).

### 2. Install Dependencies
```bash
pip install numpy pandas scikit-learn nltk lightgbm xgboost rank-bm25 sentence-transformers implicit wordcloud seaborn matplotlib psutil datasets tqdm
```

### 3. NLTK Resource Setup
When running the notebooks for the first time, required NLTK resources (`stopwords`, `wordnet`, `punkt`) are downloaded automatically in the setup cell.

### 4. Running the Notebooks
Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook
```
- Open `1st_Dataset_Model_Implementation.ipynb` to inspect pipeline on the MyCareersFuture dataset.
- Open `2nd Dataset Model Implementation.ipynb` to inspect the TechWolf benchmark pipeline.

---

## 👤 Author
- **Lee Wei Jin**
- Final Year Project (FYP)
