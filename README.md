# 🔍 Intelligent Code Clone and Plagiarism Detection System

**CS-351 Artificial Intelligence - Semester Project (Fall 2024)**

A comprehensive AI-powered system for detecting code plagiarism using AST analysis, machine learning, reinforcement learning, and explainable AI.

---

## ✨ Features

| Feature | Description | Status |
|---------|-------------|--------|
| **AST Analysis** | Abstract Syntax Tree parsing for structural comparison | ✅ |
| **A* Search** | Efficient pairwise comparison with heuristics | ✅ |
| **Best-First Feature Selection** | Information gain-based feature selection | ✅ |
| **CSP Source Attribution** | Backtracking with AC-3 for plagiarism source identification | ✅ |
| **ML Classification** | Random Forest, SVM, Logistic Regression, DNN | ✅ |
| **Q-Learning** | Adaptive threshold optimization | ✅ |
| **DBSCAN Clustering** | Coding style analysis with t-SNE visualization | ✅ |
| **SHAP Explainability** | Transparent, evidence-based detection reports | ✅ |
| **Streamlit Demo** | Interactive web interface | ✅ |

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
pip install tensorflow shap streamlit
pip install radon networkx
```

### Run the Web Demo

```bash
cd AI_Project
python scripts/run_pipeline.py --mode demo
```

Then open http://localhost:8501 in your browser.

---

## 🔧 Components

| Component | Location | Description |
|-----------|----------|-------------|
| Preprocessing | `src/preprocess/` | AST parsing, feature extraction |
| A* Search | `src/prefilter/` | Efficient pairwise comparison |
| Feature Selection | `src/feature_selection/` | BFS with Information Gain |
| CSP Attribution | `src/attribution/` | Source attribution with constraints |
| ML Models | `src/ml_models/` | RF, SVM, LR, DNN training |
| Q-Learning | `src/rl_threshold/` | Adaptive threshold learning |
| Clustering | `src/clustering/` | DBSCAN + t-SNE visualization |
| Explainability | `src/explainability/` | SHAP-based explanations |
| Web Demo | `src/demo/` | Streamlit interface |

---

## 📈 Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Random Forest | 100% | 100% |
| SVM | 100% | 100% |
| Logistic Regression | 100% | 100% |
| Deep Neural Network | 100% | 100% |

**Clustering**: 8 clusters found, Silhouette Score: 0.79

---

## 👥 Team

| Name | Roll Number |
|------|-------------|
| Muhammad Ibrahim | 2023446 |
| Tughral | 2023532 |
| Jagtar Singh | 2023266 |
| Muhammad Mehdi Raza | 2023466 |

**Instructor**: Ahmed Nawaz | **Course**: CS-351 AI | **Institution**: GIK Institute
