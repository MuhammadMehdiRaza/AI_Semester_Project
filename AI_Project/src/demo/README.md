# Streamlit Demo - Code Plagiarism Detection System

Interactive web interface for detecting code plagiarism using machine learning.

## Quick Start

```bash
# Install dependencies
pip install streamlit

# Run the demo
cd AI_Project
streamlit run src/demo/app.py
```

## Features

1. **Code Input**: Paste or upload two Python code samples
2. **Plagiarism Detection**: ML-powered analysis using Random Forest
3. **Visual Results**: Color-coded risk levels (High/Medium/Low)
4. **Feature Explanation**: See which features triggered the detection
5. **Detailed Metrics**: View all similarity features

## How It Works

1. **AST Parsing**: Code is parsed into Abstract Syntax Trees
2. **Feature Extraction**: 10+ features extracted (canonical similarity, node patterns, etc.)
3. **ML Classification**: Random Forest model predicts plagiarism probability
4. **Explainability**: Top contributing features are highlighted

## Screenshots

### Main Interface
- Two code input areas side by side
- File upload support
- One-click analysis

### Results Display
- Probability percentage with risk level
- Evidence analysis with key indicators
- Detailed feature breakdown

## Dependencies

- streamlit
- numpy
- pandas
- scikit-learn

## Usage Tips

- Works best with Python code
- Minimum 5 lines of code recommended
- Try the example buttons to see how it works
