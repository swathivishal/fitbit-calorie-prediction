# 🏋️ Fitbit: Calorie Burn Prediction & Workout Pattern Clustering

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> A Machine Learning project combining **Supervised Regression** and **Unsupervised Clustering** on real Fitbit fitness data to predict calorie burn and discover hidden workout patterns.

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Pipeline](#-project-pipeline)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Task 1: Calorie Burn Prediction](#-task-1-calorie-burn-prediction)
- [Task 2: Workout Pattern Clustering](#-task-2-workout-pattern-clustering)
- [Results Summary](#-results-summary)
- [Key Insights](#-key-insights)
- [How to Run](#-how-to-run)
- [Author](#-author)

---

## 🎯 Problem Statement

Fitness tracking devices like Fitbit generate massive amounts of personal health data. This project aims to:

1. **Task 1 (Supervised)** — Predict the number of calories burned during a workout session using regression models
2. **Task 2 (Unsupervised)** — Discover hidden workout patterns by clustering users into meaningful fitness profiles

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | Fitbit Fitness Tracker Data |
| **Total Records** | 14,102 rows |
| **Features** | 15 (after preprocessing) |
| **Target Variable** | `Calories_Burned (kcal)` |
| **Missing Values** | None |
| **Categorical Features** | Gender, Workout_Type |

### Key Features Used

| Feature | Type | Description |
|---|---|---|
| `Age` | Numeric | User age in years |
| `Weight (kg)` | Numeric | Body weight |
| `Height (m)` | Numeric | Body height |
| `Max_BPM` | Numeric | Maximum heart rate during workout |
| `Avg_BPM` | Numeric | Average heart rate during workout |
| `Resting_BPM` | Numeric | Resting heart rate |
| `Session_Duration (hours)` | Numeric | Duration of workout session |
| `Workout_Type` | Categorical (OHE) | HIIT, Yoga, Cardio, Strength, Mixed |
| `Gender` | Categorical (OHE) | Male / Female |
| `BMI` | Numeric | Body Mass Index |
| `Fat_Percentage` | Numeric | Body fat percentage |
| `Water_Intake (liters)` | Numeric | Daily water intake |
| `Workout_Frequency` | Numeric | Days per week |
| `Experience_Level` | Numeric | 0=Beginner, 1=Intermediate, 2=Advanced |

> ⚠️ **Note:** `Base_MET`, `HR_Intensity`, and `Effective_MET` columns were **dropped** to avoid data leakage — these are mathematically derived from the target variable and would inflate accuracy to 99%+.

---

## 🔄 Project Pipeline

```
1. Data Loading
      ↓
2. Basic Cleaning
   • Check missing values & duplicates
   • Drop leaky engineered columns
   • Cap outliers using IQR method
      ↓
3. One-Hot Encoding
   • Gender → [Gender_Female, Gender_Male]
   • Workout_Type → [Workout_Type_HIIT, _Yoga, _Cardio, _Strength, _Mixed]
      ↓
4. Correlation Analysis
   • Compute |correlation| with target
   • Select features with |corr| > 0.05
      ↓
5. Standard Scaling
   • StandardScaler → mean=0, std=1
      ↓
6. Train-Test Split
   • 80% Training | 20% Testing
   • random_state=42
      ↓
7. Model Training & Evaluation (Task 1)
   + Clustering with KMeans + PCA (Task 2)
```

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| `Python 3.12` | Core programming language |
| `Pandas` | Data loading and manipulation |
| `NumPy` | Numerical operations |
| `Matplotlib / Seaborn` | Data visualization |
| `Scikit-learn` | ML models, preprocessing, evaluation |
| `XGBoost` | Gradient boosting regression |
| `KMeans` | Unsupervised clustering |
| `PCA` | Dimensionality reduction |

---

## 📁 Project Structure

```
Fitbit_Project/
│
├── 📓 Fitbit_Calorie_Prediction_Clustering.ipynb   # Main notebook
├── 📊 Fitbit_dataset.csv                            # Dataset
├── 📄 README.md                                     # Project documentation
├── 📊 Fitbit_Presentation.pptx                     # Project presentation
│
├── 📈 Charts (auto-saved when notebook runs)/
│   ├── target_distribution.png
│   ├── categorical_distribution.png
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── feature_correlation.png
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   ├── residual_analysis.png
│   ├── feature_importance.png
│   ├── elbow_method.png
│   ├── silhouette_scores.png
│   ├── cluster_visualization.png
│   └── cluster_profiles.png
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or above
- pip package manager

### Step 1 — Clone the Repository
```bash
git clone https://github.com/your-username/fitbit-calorie-prediction.git
cd fitbit-calorie-prediction
```

### Step 2 — Install Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter ipykernel
```

### Step 3 — Launch Jupyter Notebook
```bash
jupyter notebook Fitbit_Calorie_Prediction_Clustering.ipynb
```

Or open in **VS Code** with the Jupyter extension installed.

---

## 🤖 Task 1: Calorie Burn Prediction

### Models Trained

| # | Model | MAE | RMSE | R² Score | Accuracy % |
|---|---|---|---|---|---|
| 🏆 | **KNN Regressor** | **22.41** | **31.59** | **0.9648** | **96.48%** |
| 2 | SVR | 17.72 | 32.56 | 0.9626 | 96.26% |
| 3 | XGBoost | 25.22 | 34.06 | 0.9591 | 95.91% |
| 4 | Linear Regression | 32.10 | 46.35 | 0.9242 | 92.42% |
| 5 | Ridge Regression | 32.10 | 46.35 | 0.9242 | 92.42% |
| 6 | Lasso Regression | 32.17 | 46.45 | 0.9239 | 92.39% |
| 7 | Random Forest | 36.23 | 51.69 | 0.9058 | 90.58% |
| 8 | Decision Tree | 41.38 | 57.68 | 0.8826 | 88.26% |

### Best Model: KNN Regressor
- ✅ **Accuracy: 96.48%**
- ✅ **R² Score: 0.9648** — explains 96.48% of variance in calorie burn
- ✅ **MAE: 22.41 kcal** — predictions within ±22 kcal of actual
- ✅ **5-Fold CV Mean R²: ~0.964** — consistent and stable performance

### Why KNN Won
KNN captures **non-linear relationships** in fitness data. Users with similar BPM, session duration, and workout type naturally form neighbors — making distance-based prediction highly effective.

---

## 🔵 Task 2: Workout Pattern Clustering

### Algorithm: KMeans + PCA

| Parameter | Value |
|---|---|
| Algorithm | KMeans |
| Dimensionality Reduction | PCA (2 components) |
| Optimal K | **4 clusters** |
| Silhouette Score | **0.3556** |
| Threshold | ≥ 0.15 ✅ |
| PCA Variance Explained | ~34% |

### Cluster Profiles

| Cluster | Profile | Characteristics |
|---|---|---|
| 🔴 Cluster 0 | High-Intensity Athletes | High BPM, long sessions, HIIT/Cardio |
| 🟡 Cluster 1 | Moderate Exercisers | Mixed workout types, medium BPM |
| 🟢 Cluster 2 | Low-Intensity Beginners | Short sessions, Yoga focus, low experience |
| 🔵 Cluster 3 | Experienced Trainers | High frequency, Strength training |

### How K=4 Was Selected
- **Elbow Method** used to find inertia bend point
- **Silhouette Score sweep** from K=2 to K=10
- K=4 gave best balance of **score (0.3556) and interpretability**

---

## 📋 Results Summary

### Task 1 — Regression
```
Best Model  : KNN Regressor
Accuracy    : 96.48%
R² Score    : 0.9648
MAE         : 22.41 kcal
RMSE        : 31.59 kcal
Encoding    : One-Hot Encoding
Scaling     : StandardScaler
Split       : 80% Train / 20% Test
```

### Task 2 — Clustering
```
Algorithm        : KMeans + PCA
Optimal K        : 4
Silhouette Score : 0.3556  (threshold ≥ 0.15 ✅)
PCA Components   : 2
Cluster Sizes    : ~2800–4400 per cluster
```

---

## 💡 Key Insights

1. **Session Duration** is the strongest predictor of calorie burn (correlation = 0.573)
2. **HIIT and Yoga** encoded features show the highest correlation among workout types
3. **KNN outperforms** tree-based models because fitness data has natural neighbor clusters
4. **4 distinct user profiles** exist in the data — useful for personalized fitness recommendations
5. **Removing leaky MET columns** is critical — keeping them inflates accuracy to 99%+ unrealistically

---

## ▶️ How to Run

```bash
# 1. Open VS Code
# 2. Open the Fitbit_Project folder
# 3. Open the .ipynb file
# 4. Select Python 3.12 kernel
# 5. Click "Run All"
```

All charts will be automatically saved as `.png` files in the project folder.

---


---

## 📄 License

This project is licensed under the MIT License.

---

*Built with ❤️ using Python, Scikit-learn, XGBoost, KMeans, and PCA*
