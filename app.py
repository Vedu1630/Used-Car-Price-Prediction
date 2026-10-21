# app.py
# Main Streamlit dashboard for Used Car Price Prediction.
# Run with:  streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import (
    load_data,
    clean_data,
    get_features_and_target,
    get_overview_stats,
)
from model_utils import (
    train_models,
    evaluate_models,
    predict_price,
    get_feature_importance,
    get_actual_vs_predicted,
)

# ─────────────────────────────────────────────
#  Page configuration
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="wide",
)

# ─────────────────────────────────────────────
#  Custom CSS – clean, professional look
# ─────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8f9fa; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }

    /* Section headers */
    h2 { color: #1a1a2e; }
    h3 { color: #16213e; }

    /* Prediction result box */
    .pred-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: white;
        padding: 28px;
        border-radius: 14px;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin: 16px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    /* Info / note box */
    .note-box {
        background-color: #e8f4f8;
        border-left: 4px solid #2196F3;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 10px 0;
        font-size: 0.9rem;
    }

    /* Best model highlight */
    .best-model {
        background-color: #d4edda;
        border: 1px solid #28a745;
        border-radius: 8px;
        padding: 12px 18px;
        margin: 10px 0;
        font-weight: 600;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Load & train  (cached so it runs only once)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Loading and cleaning dataset…")
def load_and_clean():
    """Load the CSV and return a clean DataFrame."""
    raw_df   = load_data("dataset.csv")
    clean_df = clean_data(raw_df)
    return clean_df


@st.cache_data(show_spinner="Training models… (this takes a few seconds)")
def train_all_models(_clean_df):
    """Train all three models and cache the result."""
    X, y      = get_features_and_target(_clean_df)
    trained   = train_models(X, y)
    metrics   = evaluate_models(trained)
    return trained, metrics


# ── Run the cached functions ──
try:
    clean_df = load_and_clean()
    trained, metrics_df = train_all_models(clean_df)
except FileNotFoundError:
    st.error(
        "⚠️  **dataset.csv not found.**\n\n"
        "Please place your Kaggle CSV file (renamed to `dataset.csv`) "
        "in the same folder as `app.py` and restart the app."
    )
    st.stop()


# Shorthand references
pipelines = trained["pipelines"]
X_test    = trained["X_test"]
y_test    = trained["y_test"]


# ─────────────────────────────────────────────
#  Sidebar navigation
# ─────────────────────────────────────────────

st.sidebar.image("https://img.icons8.com/color/96/car.png", width=80)
st.sidebar.title("🚗 Car Price Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to",
    ["📊 Data Overview", "🔮 Predict Car Price", "📈 Model Comparison"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Built with Streamlit · Scikit-learn<br>"
    "NMIMS ML Mini Project</small>",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
#  Helper: format price in INR
# ─────────────────────────────────────────────

def format_inr(amount):
    """Return a human-readable INR string with lakhs/crores."""
    if amount >= 1_00_00_000:
        return f"₹{amount/1_00_00_000:.2f} Cr"
    elif amount >= 1_00_000:
        return f"₹{amount/1_00_000:.2f} Lakhs"
    else:
        return f"₹{amount:,.0f}"


def format_inr_full(amount):
    """Return both Lakhs/Cr format and full integer for detailed cards."""
    if amount >= 1_00_00_000:
        return f"₹{amount/1_00_00_000:.2f} Cr (₹{amount:,.0f})"
    elif amount >= 1_00_000:
        return f"₹{amount/1_00_000:.2f} Lakhs (₹{amount:,.0f})"
    else:
        return f"₹{amount:,.0f}"


# ══════════════════════════════════════════════
#  PAGE 1 – DATA OVERVIEW
# ══════════════════════════════════════════════

if page == "📊 Data Overview":

    st.title("📊 Used Car Dataset Overview")
    st.markdown("Explore the dataset used to train the price prediction models.")
    st.markdown("---")

    # ── Summary metric cards ──────────────────
    stats = get_overview_stats(clean_df)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🚘 Total Records",  f"{stats['total_records']:,}")
    col2.metric("💰 Average Price",  format_inr(stats["avg_price"]))
    col3.metric("📊 Median Price",   format_inr(stats["median_price"]))
    col4.metric("⬇️ Min Price",      format_inr(stats["min_price"]))
    col5.metric("⬆️ Max Price",      format_inr(stats["max_price"]))

    st.markdown("---")

    # ── Row 1: Price distribution + Fuel type breakdown ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("💵 Price Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        # Use log-scale for readability because prices span a wide range
        log_prices = np.log1p(clean_df["selling_price"])
        ax.hist(log_prices, bins=40, color="#4361ee", edgecolor="white", alpha=0.85)
        ax.set_xlabel("log(Selling Price)")
        ax.set_ylabel("Number of Cars")
        ax.set_title("Distribution of Selling Prices (log scale)")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("Log scale is used because car prices vary from ₹50,000 to ₹1 crore+.")

    with col_b:
        st.subheader("⛽ Average Price by Fuel Type")
        fuel_avg = (
            clean_df.groupby("fuel")["selling_price"]
            .mean()
            .sort_values(ascending=False)
        )
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        colors = sns.color_palette("Set2", len(fuel_avg))
        bars = ax2.bar(fuel_avg.index, fuel_avg.values / 1e5, color=colors, edgecolor="white")
        ax2.set_xlabel("Fuel Type")
        ax2.set_ylabel("Average Price (₹ Lakhs)")
        ax2.set_title("Average Selling Price by Fuel Type")
        ax2.spines[["top", "right"]].set_visible(False)
        # Add value labels on bars
        for bar in bars:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"₹{bar.get_height():.1f}L",
                ha="center", va="bottom", fontsize=9
            )
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown("---")

    # ── Row 2: Brand chart + Transmission split ──
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("🏷️ Top 10 Brands by Listing Count")
        top_brands = clean_df["brand"].value_counts().head(10)
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=top_brands.values, y=top_brands.index, palette="Blues_r", ax=ax3)
        ax3.set_xlabel("Number of Cars Listed")
        ax3.set_ylabel("Brand")
        ax3.set_title("Top 10 Car Brands")
        ax3.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col_d:
        st.subheader("⚙️ Transmission Type Split")
        trans_counts = clean_df["transmission"].value_counts()
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        ax4.pie(
            trans_counts.values,
            labels=trans_counts.index,
            autopct="%1.1f%%",
            colors=["#4361ee", "#f72585"],
            startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        ax4.set_title("Manual vs Automatic")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    st.markdown("---")

    # ── Row 3: Km driven vs Price scatter ────
    st.subheader("🛣️  Kilometres Driven vs Selling Price")
    sample = clean_df.sample(min(1500, len(clean_df)), random_state=1)
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    scatter = ax5.scatter(
        sample["km_driven"] / 1000,
        sample["selling_price"] / 1e5,
        alpha=0.45,
        c=sample["car_age"],
        cmap="viridis",
        s=20,
    )
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label("Car Age (years)")
    ax5.set_xlabel("Kilometres Driven (thousands)")
    ax5.set_ylabel("Selling Price (₹ Lakhs)")
    ax5.set_title("Km Driven vs Selling Price  (colour = car age)")
    ax5.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

    st.markdown("---")

    # ── Dataset preview table ─────────────────
    st.subheader("🗃️  Dataset Preview")
    st.dataframe(clean_df.head(50), use_container_width=True)
    st.caption(f"Showing first 50 rows of {len(clean_df):,} total records.")


# ══════════════════════════════════════════════
#  PAGE 2 – PREDICT CAR PRICE
# ══════════════════════════════════════════════

elif page == "🔮 Predict Car Price":

    st.title("🔮 Predict Used Car Price")
    st.markdown("Fill in the car details below and get an estimated resale price.")
    st.markdown("---")

    # ── Gather unique values for dropdowns ───
    brands        = sorted(clean_df["brand"].dropna().unique().tolist())
    fuels         = sorted(clean_df["fuel"].dropna().unique().tolist())
    seller_types  = sorted(clean_df["seller_type"].dropna().unique().tolist())
    transmissions = sorted(clean_df["transmission"].dropna().unique().tolist())
    owners        = sorted(clean_df["owner"].dropna().unique().tolist())

    # Default numeric values from training data medians
    med_km        = int(clean_df["km_driven"].median())
    med_age       = int(clean_df["car_age"].median())
    med_mileage   = round(float(clean_df["mileage"].median()), 1)
    med_engine    = int(clean_df["engine"].median())
    med_max_power = round(float(clean_df["max_power"].median()), 1)
    med_seats     = int(clean_df["seats"].median())

    # ── Input form ───────────────────────────
    with st.form("prediction_form"):
        st.subheader("🚗 Car Details")

        row1_col1, row1_col2, row1_col3 = st.columns(3)

        with row1_col1:
            brand = st.selectbox("Brand", brands, help="Car manufacturer")
            fuel  = st.selectbox("Fuel Type", fuels)

        with row1_col2:
            transmission = st.selectbox("Transmission", transmissions)
            seller_type  = st.selectbox("Seller Type", seller_types)

        with row1_col3:
            owner   = st.selectbox("Owner History", owners)
            seats   = st.number_input(
                "Number of Seats", min_value=2, max_value=10,
                value=med_seats, step=1
            )

        st.markdown("---")
        st.subheader("📐 Technical Specifications")

        row2_col1, row2_col2, row2_col3 = st.columns(3)

        with row2_col1:
            car_age = st.number_input(
                "Car Age (years)", min_value=0, max_value=30,
                value=med_age, step=1,
                help="Number of years since the car was manufactured"
            )
            km_driven = st.number_input(
                "Kilometres Driven", min_value=0, max_value=1_000_000,
                value=med_km, step=1000,
                help="Total kilometres driven"
            )

        with row2_col2:
            mileage = st.number_input(
                "Mileage (kmpl)", min_value=0.0, max_value=50.0,
                value=med_mileage, step=0.5
            )
            engine = st.number_input(
                "Engine Capacity (CC)", min_value=500, max_value=5000,
                value=med_engine, step=50
            )

        with row2_col3:
            max_power = st.number_input(
                "Max Power (bhp)", min_value=10.0, max_value=500.0,
                value=med_max_power, step=1.0
            )

        st.markdown("---")
        st.subheader("🤖 Select Model")

        model_choice = st.radio(
            "Choose the prediction model",
            list(pipelines.keys()),
            horizontal=True,
        )

        submitted = st.form_submit_button("🔍 Predict Price", use_container_width=True)

    # ── Prediction logic ─────────────────────
    if submitted:
        # Basic validation
        errors = []
        if km_driven < 0:
            errors.append("Kilometres driven cannot be negative.")
        if car_age < 0:
            errors.append("Car age cannot be negative.")
        if mileage <= 0:
            errors.append("Mileage must be greater than zero.")

        if errors:
            for err in errors:
                st.error(err)
        else:
            # Build input dict – matches training feature columns exactly
            input_dict = {
                "brand":        brand,
                "car_age":      car_age,
                "km_driven":    km_driven,
                "fuel":         fuel,
                "seller_type":  seller_type,
                "transmission": transmission,
                "owner":        owner,
                "mileage":      mileage,
                "engine":       float(engine),
                "max_power":    max_power,
                "seats":        seats,
            }

            # Remove features not in the trained model
            available_features = trained["cat_cols"] + trained["num_cols"]
            input_dict = {k: v for k, v in input_dict.items() if k in available_features}

            predicted = predict_price(pipelines[model_choice], input_dict)

            # ── Display result ─────────────
            st.markdown(f"""
            <div class="pred-box">
                Estimated Resale Price<br>
                <span style="font-size:2.4rem">{format_inr_full(predicted)}</span>
            </div>
            """, unsafe_allow_html=True)

            # Show breakdown
            result_col1, result_col2, result_col3 = st.columns(3)
            result_col1.metric("Model Used",       model_choice)
            result_col2.metric("Predicted Price",  format_inr_full(predicted))
            result_col3.metric("Car Age",          f"{car_age} year(s)")

            st.markdown("""
            <div class="note-box">
                💡 <strong>Note:</strong> This is an <em>estimated</em> resale price based on
                the patterns learned from the training dataset. Actual market prices may vary
                depending on the car's condition, location, demand, and other factors.
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE 3 – MODEL COMPARISON
# ══════════════════════════════════════════════

elif page == "📈 Model Comparison":

    st.title("📈 Model Performance Comparison")
    st.markdown(
        "All three models are evaluated on the **same held-out test set** "
        "(20% of the data, never seen during training). "
        "This ensures a fair, unbiased comparison."
    )
    st.markdown("---")

    # ── Metrics table ─────────────────────────
    st.subheader("📋 Evaluation Metrics")

    # Highlight best model (highest R²)
    best_idx   = metrics_df["R² Score"].idxmax()
    best_model = metrics_df.loc[best_idx, "Model"]

    st.markdown(f"""
    <div class="best-model">
        🏆 Best performing model on this dataset: <strong>{best_model}</strong>
        (highest R² score on the test set)
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        metrics_df.style.highlight_max(
            subset=["R² Score"], color="#d4edda"
        ).highlight_min(
            subset=["MAE (₹)", "RMSE (₹)"], color="#d4edda"
        ),
        use_container_width=True,
    )

    # ── Metric explanations ───────────────────
    with st.expander("📖 What do these metrics mean?"):
        st.markdown("""
| Metric | Full Name | What it means |
|--------|-----------|---------------|
| **MAE** | Mean Absolute Error | Average ₹ difference between predicted and actual prices. Lower is better. |
| **RMSE** | Root Mean Squared Error | Similar to MAE but penalises large errors more heavily. Lower is better. |
| **R² Score** | Coefficient of Determination | How much of the price variation the model explains. Closer to 1.0 is better. |

> **Important:** These scores reflect performance on the test set only.
> Real-world accuracy may differ due to unseen car conditions or market changes.
        """)

    st.markdown("---")

    # ── Bar chart: R² comparison ──────────────
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.subheader("R² Score Comparison")
        fig_r2, ax_r2 = plt.subplots(figsize=(6, 4))
        colors = ["#4361ee" if m != best_model else "#28a745" for m in metrics_df["Model"]]
        bars = ax_r2.bar(metrics_df["Model"], metrics_df["R² Score"], color=colors, edgecolor="white")
        ax_r2.set_ylim(0, 1.05)
        ax_r2.set_ylabel("R² Score")
        ax_r2.set_title("R² Score by Model  (higher is better)")
        ax_r2.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, metrics_df["R² Score"]):
            ax_r2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=10, fontweight="bold"
            )
        plt.tight_layout()
        st.pyplot(fig_r2)
        plt.close()

    with col_m2:
        st.subheader("MAE Comparison (₹)")
        fig_mae, ax_mae = plt.subplots(figsize=(6, 4))
        colors2 = ["#f72585" if m != best_model else "#28a745" for m in metrics_df["Model"]]
        bars2 = ax_mae.bar(metrics_df["Model"], metrics_df["MAE (₹)"] / 1e5, color=colors2, edgecolor="white")
        ax_mae.set_ylabel("MAE (₹ Lakhs)")
        ax_mae.set_title("Mean Absolute Error  (lower is better)")
        ax_mae.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars2, metrics_df["MAE (₹)"]):
            ax_mae.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"₹{val/1e5:.2f}L", ha="center", fontsize=9, fontweight="bold"
            )
        plt.tight_layout()
        st.pyplot(fig_mae)
        plt.close()

    st.markdown("---")

    # ── Actual vs Predicted scatter ───────────
    st.subheader("🎯 Actual vs Predicted Price Chart")

    selected_model = st.selectbox(
        "Select model to visualise",
        list(pipelines.keys()),
    )

    av_df = get_actual_vs_predicted(trained, selected_model)

    fig_av, ax_av = plt.subplots(figsize=(8, 5))
    ax_av.scatter(
        av_df["Actual"] / 1e5,
        av_df["Predicted"] / 1e5,
        alpha=0.5, s=20, color="#4361ee", label="Predictions"
    )
    # Perfect prediction line
    max_val = max(av_df["Actual"].max(), av_df["Predicted"].max()) / 1e5
    ax_av.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect Prediction")
    ax_av.set_xlabel("Actual Price (₹ Lakhs)")
    ax_av.set_ylabel("Predicted Price (₹ Lakhs)")
    ax_av.set_title(f"Actual vs Predicted – {selected_model}")
    ax_av.legend()
    ax_av.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_av)
    plt.close()
    st.caption(
        "Points closer to the red dashed line indicate more accurate predictions. "
        "A random sample of 300 test points is shown."
    )

    st.markdown("---")

    # ── Feature Importance ────────────────────
    st.subheader("🔍 Feature Importance")
    st.markdown("Feature importance shows which car attributes most influence the predicted price.")

    fi_model = st.selectbox(
        "Select tree-based model",
        ["Decision Tree", "Random Forest"],
        key="fi_model_select"
    )

    fi_df = get_feature_importance(
        pipelines[fi_model],
        trained["cat_cols"],
        trained["num_cols"],
    )

    if fi_df is not None:
        # Show top 15 features for readability
        fi_top = fi_df.head(15)
        fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=fi_top, x="Importance", y="Feature",
            palette="Blues_r", ax=ax_fi
        )
        ax_fi.set_title(f"Top 15 Feature Importances – {fi_model}")
        ax_fi.set_xlabel("Importance Score")
        ax_fi.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_fi)
        plt.close()
    else:
        st.info("Feature importance is only available for Decision Tree and Random Forest.")
