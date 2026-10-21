# 🚗 Used Car Price Prediction

**NMIMS B.Tech Computer Science – Machine Learning Mini Project**

---

## 📌 Project Objective

Build a web application that **predicts the estimated resale price of a used car**
based on its details (brand, age, kilometres driven, fuel type, etc.) using
three Machine Learning regression algorithms.

---

## 📋 Problem Statement

Used cars have highly variable resale prices that depend on many factors.
A buyer or seller often struggles to estimate a fair market price.
This project uses historical data of used car listings to train ML models
that can predict the resale price of a car given its specifications.

---

## 📂 Dataset

| Item | Detail |
|------|--------|
| **Source** | Kaggle – Car Details v3 (CarDekho) |
| **File** | `dataset.csv` |
| **Records** | ~8,128 rows |
| **Target Column** | `selling_price` (Indian Rupees, ₹) |

### Columns Used

| Column | Description | Type |
|--------|-------------|------|
| `name` → `brand` | Car manufacturer (first word extracted) | Categorical |
| `year` → `car_age` | Age of the car (2024 − year) | Numerical |
| `km_driven` | Total kilometres driven | Numerical |
| `fuel` | Fuel type (Petrol, Diesel, CNG, etc.) | Categorical |
| `seller_type` | Individual / Dealer / Trustmark Dealer | Categorical |
| `transmission` | Manual / Automatic | Categorical |
| `owner` | First Owner / Second Owner / etc. | Categorical |
| `mileage` | Fuel efficiency (kmpl) | Numerical |
| `engine` | Engine capacity (CC) | Numerical |
| `max_power` | Maximum power output (bhp) | Numerical |
| `seats` | Number of seats | Numerical |

---

## 🗂️ Project File Structure

```
used-car-price-prediction/
    app.py            →  Streamlit dashboard (UI, charts, prediction form)
    data_utils.py     →  Load, clean, and prepare the dataset
    model_utils.py    →  Train models, evaluate, and make predictions
    requirements.txt  →  Python library dependencies
    README.md         →  This file
    dataset.csv       →  The Kaggle dataset (do not delete or rename)
```

---

## ⚙️ Installation & Execution

### Step 1 – Create a virtual environment (optional but recommended)

```bash
python -m venv venv
```

**On macOS / Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### Step 2 – Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 – Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

> ⚠️ Make sure `dataset.csv` is in the **same folder** as `app.py`.

---

## 🤖 Machine Learning Algorithms

### A. Linear Regression

**What it is:** The simplest regression algorithm. It fits a straight line
(or a hyperplane in multiple dimensions) through the data.

**How it works:** It finds the best-fit equation:
`Price = w₁×feature₁ + w₂×feature₂ + … + b`

It adjusts the weights (`w`) and bias (`b`) to minimise the total prediction error.

**Advantages:** Fast, easy to understand, good baseline.  
**Disadvantages:** Assumes a linear relationship; struggles with complex, non-linear data.

---

### B. Decision Tree Regression

**What it is:** A tree-like model that makes decisions at each node based on
feature values.

**How it works:** It splits the data repeatedly on the most informative features
until it reaches leaf nodes, each of which gives a price prediction.

**Advantages:** Handles non-linear relationships, easy to visualise.  
**Disadvantages:** Prone to overfitting without depth restrictions. We use
`max_depth=8` to keep it generalised.

---

### C. Random Forest Regression

**What it is:** An ensemble of many Decision Trees. Each tree is trained on a
random subset of the data and features.

**How it works:** Each tree predicts a price independently. The final prediction
is the **average** of all trees' predictions.

**Advantages:** More accurate and stable than a single tree; less prone to
overfitting; provides feature importance scores.  
**Disadvantages:** Slower to train; harder to interpret than a single tree.

---

## 🔧 Data Preprocessing

1. **Remove duplicates** – Drop identical rows.
2. **Extract brand** – Take the first word of the car's full name as the brand.
3. **Create car_age** – Calculate age from the manufacturing year.
4. **Parse text fields** – Extract numbers from strings like "23.4 kmpl" and "1248 CC".
5. **Drop torque** – Too inconsistently formatted across rows to parse reliably.
6. **Handle missing values** – Fill numerical columns with medians; categorical columns with the mode.
7. **Train-test split** – 80% training, 20% testing with `random_state=42` for reproducibility.
8. **One-Hot Encoding** – Converts categorical columns (brand, fuel, etc.) into binary columns.
9. **Pipeline** – All preprocessing is wrapped inside a `sklearn.Pipeline` so the same transformations are applied consistently during training and prediction.

---

## 📐 Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **MAE** | Mean of \|actual − predicted\| | Average ₹ error. Lower is better. |
| **RMSE** | √(Mean of (actual − predicted)²) | Penalises large errors more. Lower is better. |
| **R² Score** | 1 − SS_res / SS_tot | Proportion of price variance explained. Closer to 1 is better. |

---

## 🏆 Model Comparison (Actual Measured Results)

> Evaluated on the held-out test set (20% of the dataset, **1,386 cars** never seen during training).
> Random seed: `random_state=42`. Target: `selling_price` (in ₹).

| Model | MAE (₹) | RMSE (₹) | R² Score | Performance Rank |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest Regressor** | **₹75,861** | **₹1,33,472** | **0.9188 (91.9%)** | 🥇 **1st (Best)** |
| **Decision Tree Regressor** | ₹93,403 | ₹1,58,249 | 0.8858 (88.6%) | 🥈 2nd |
| **Linear Regression** | ₹1,32,973 | ₹2,61,704 | 0.6877 (68.8%) | 🥉 3rd |

### Key Observations & Viva Insights
1. **Random Forest achieved the highest R² (0.9188) and lowest MAE (₹75,861)** because ensembling 100 decision trees significantly reduces variance and prevents overfitting.
2. **Decision Tree (max_depth=8) performed reasonably well (R² 0.8858)**, capturing non-linear relationships such as the exponential depreciation of luxury cars and high-mileage vehicles.
3. **Linear Regression is a useful baseline (R² 0.6877)**, but suffers because car depreciation and pricing dynamics are inherently non-linear (e.g., doubling engine power or brand prestige does not simply increase price linearly).
4. **Top 3 most important features (from Random Forest):**
   - **`max_power`** (~61.1% importance): Strongest predictor of vehicle tier and price.
   - **`car_age`** (~23.3% importance): Reflects annual vehicle depreciation.
   - **`km_driven`** (~3.9% importance): Usage wear-and-tear.

---

## ⚠️ Limitations

- The model is only as good as the training data. Rare or luxury cars may be
  predicted less accurately.
- The dataset may not reflect current market prices.
- Car condition, accident history, and service records are not captured.
- Location and regional pricing are not in the dataset (v3 version).
- The `torque` column was dropped due to inconsistent formatting.

---

## 🎓 College Viva Questions & Answers

**Q1. What is regression in Machine Learning?**  
Regression is a type of supervised learning where the model predicts a
**continuous numerical value** (like price) rather than a category.

**Q2. Why did you choose these three algorithms?**  
Linear Regression is the simplest baseline. Decision Tree captures non-linear
patterns. Random Forest improves on both by averaging many trees. Comparing them
shows how model complexity affects accuracy.

**Q3. What is overfitting, and how did you prevent it?**  
Overfitting is when a model memorises training data and performs poorly on new data.
We prevented it by limiting Decision Tree depth (`max_depth=8`) and using Random
Forest, which averages many trees to reduce variance.

**Q4. What is a train-test split and why is it important?**  
We split data into 80% for training and 20% for testing. The test set is hidden
from the model during training so we can measure how well it generalises to
new, unseen data.

**Q5. What is One-Hot Encoding?**  
It converts categorical values like "Petrol", "Diesel" into separate binary
(0/1) columns so the ML algorithm can process them as numbers.

**Q6. What is a Scikit-learn Pipeline?**  
A Pipeline chains multiple steps (preprocessing + model) into one object.
This ensures that the same encoding applied during training is automatically
applied when making predictions, preventing data leakage.

**Q7. What does R² score mean?**  
R² (R-squared) measures how much of the variation in car prices the model
explains. An R² of 0.85 means the model explains 85% of the price variation.
A value of 1.0 is perfect.

**Q8. What is feature importance in Random Forest?**  
It measures how much each feature contributes to reducing prediction error
across all trees. Features like `max_power` and `car_age` typically rank high
because they strongly influence the resale value.

**Q9. What is the difference between MAE and RMSE?**  
Both measure average prediction error. RMSE squares the errors before averaging,
so it penalises large errors more heavily than MAE. Use RMSE when large errors
are especially costly.

**Q10. What are the limitations of your model?**  
The model cannot account for a car's physical condition, accident history,
negotiation, or regional demand. It also reflects historical listing prices, not
actual transaction prices. Luxury cars with few data points may be predicted less
accurately.

---

## 🖥️ How to Demonstrate to Your Professor

1. **Start the app:** `streamlit run app.py`
2. **Page 1 – Data Overview:** Explain the dataset size, metrics, and charts.
3. **Page 2 – Predict:** Enter a sample car's details and show the predicted price.
   Try changing the model to show different results.
4. **Page 3 – Model Comparison:** Explain MAE, RMSE, and R² from the table.
   Show the Actual vs Predicted chart and the Feature Importance chart.
5. **Answer viva questions** using the Q&A section above.

---

*Built with Python · Pandas · Scikit-learn · Streamlit · Matplotlib · Seaborn*
