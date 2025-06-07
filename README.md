# 📊 Financial Health Assessment Using Linear Discriminant Analysis (LDA)

This project applies **Linear Discriminant Analysis (LDA)** to classify companies as **bankrupt** or **non-bankrupt** based on financial indicators. It includes preprocessing steps and visualizations, all presented in a Dash-based interactive dashboard.

## 🚀 Features

- Preprocessing (normality test, Box-Cox transformation, outlier removal)
- LDA model and decision boundary
- Visual insights: histograms, boxplots, LDA projection, coefficient importance
- Dashboard built with Dash and Plotly


How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
Run the app:

bash
Kopiuj
Edytuj
python LDA.py
Open in your browser: http://127.0.0.1:8050

Interpretation
The LDA projection shows clear class separation. Based on the coefficients plot, variables like V₅, V₆, and V₂₀ have the strongest influence on classification.

Requirements
Python 3.9+

Libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn

scipy

dash
