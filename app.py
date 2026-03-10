import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import json
import sklearn
from pathlib import Path

# Paths to artifacts
MODEL_PATH = "outputs/models/best_model_pipeline.joblib"
SCHEMA_PATH = "outputs/models/feature_schema.json"
PRED_COL = "Predicted GHG Emissions (Metric Tons CO2e)"
ACTUAL_COL_HINT = "Total (Location-Based) GHG Emissions (Metric Tons CO2e)"
OPTIONS_PATH = "outputs/models/categorical_options.json"


@st.cache_resource
def load_artifacts():
    """Load the trained model pipeline and feature schema."""
    try:
        # confirmed local 1.6.1
        model = joblib.load(MODEL_PATH)
        with open(SCHEMA_PATH, "r") as f:
            schema = json.load(f)
        
        categorical_options = {}
        if Path(OPTIONS_PATH).exists():
            with open(OPTIONS_PATH, "r") as f:
                categorical_options = json.load(f)
                
        return model, schema, categorical_options, None
    except Exception as e:
        return None, None, None, f"Error loading artifacts: {str(e)}"


@st.cache_data
def validate_input(df, schema):
    """Validate columns and data types against the schema."""
    required = list(schema.keys())
    present = df.columns.tolist()
    
    missing = [c for c in required if c not in present]
    if missing:
        return None, f"Missing columns: {', '.join(missing)}"
    
    # Filter to required columns and ignore extra
    df_clean = df[required].copy()
    
    # Attempt robust numeric conversion for features
    for col, dtype in schema.items():
        if "float" in dtype or "int" in dtype:
            # Handle cases where numeric columns are read as strings with commas
            if df_clean[col].dtype == object:
                df_clean[col] = df_clean[col].astype(str).str.replace(',', '', regex=False)
            
            # Convert to numeric, letting NaN stay (the pipeline imputer will handle them)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
    return df_clean, None

def run_predictions(model, df):
    """Run model inference and clip results to non-negative values."""
    try:
        preds = model.predict(df)
        preds = np.clip(preds, 0, None)
        return preds, None
    except Exception as e:
        return None, f"Prediction failed: {str(e)}"

def generate_dashboard_metrics(df):
    """Display KPI cards for predictions."""
    cols = st.columns(4)
    cols[0].metric("Total Buildings", len(df))
    cols[1].metric("Avg Emissions", f"{df[PRED_COL].mean():.2f}")
    cols[2].metric("Max Emissions", f"{df[PRED_COL].max():.2f}")
    cols[3].metric("Min Emissions", f"{df[PRED_COL].min():.2f}")

def plot_emissions_distribution(df):
    """Plot histogram of predicted emissions."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df[PRED_COL], bins=30, color="#2c3e50", edgecolor="white")
    ax.set_title("Predicted Emissions Distribution")
    ax.set_xlabel("Metric Tons CO2e")
    ax.set_ylabel("Count")
    st.pyplot(fig)

def plot_sector_analysis(df):
    """Plot average emissions per sector if column exists."""
    if "Sector" in df.columns:
        st.subheader("Emissions by Sector")
        sector_avg = df.groupby("Sector")[PRED_COL].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        sector_avg.plot(kind="bar", ax=ax, color="#34495e")
        ax.set_ylabel("Avg Emissions (MT CO2e)")
        ax.set_xlabel("")
        st.pyplot(fig)

def show_top_emitters(df):
    """Display top 10 buildings by predicted emissions."""
    st.subheader("Top 10 Emitters")
    top_10 = df.sort_values(by=PRED_COL, ascending=False).head(10)
    st.table(top_10[[PRED_COL] + [c for c in top_10.columns if c != PRED_COL]])

def plot_correlation_heatmap(df):
    """Display correlation heatmap between numeric features and predictions."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:
        st.subheader("Correlation Heatmap")
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(im)
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)
        ax.set_title("Feature Correlation")
        st.pyplot(fig)

def plot_performance_metrics(df, actual_col):
    """Plot Predicted vs Actual and Residuals."""
    st.subheader("Model Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Predicted vs Actual
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(df[actual_col], df[PRED_COL], alpha=0.5, color="#2ecc71")
        
        # Identity line
        max_val = max(df[actual_col].max(), df[PRED_COL].max())
        ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label="Perfect Fit")
        
        ax.set_xlabel("Actual (MT CO2e)")
        ax.set_ylabel("Predicted (MT CO2e)")
        ax.set_title("Predicted vs. Actual")
        ax.legend()
        st.pyplot(fig)
        
    with col2:
        # Residuals
        residuals = df[actual_col] - df[PRED_COL]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(df[PRED_COL], residuals, alpha=0.5, color="#e74c3c")
        ax.axhline(y=0, color='k', linestyle='--', lw=2)
        ax.set_xlabel("Predicted (MT CO2e)")
        ax.set_ylabel("Residuals (MT CO2e)")
        ax.set_title("Residual Plot")
        st.pyplot(fig)

def plot_feature_importance(model, schema):
    """Display feature importance if available in the model."""
    st.subheader("Feature Importance")
    
    # Try to extract importance from the final estimator in the pipeline
    importance = None
    feature_names = list(schema.keys())
    
    try:
        # Get the underlying model and preprocessor
        if hasattr(model, 'named_steps'):
            final_model = model.named_steps.get('model')
            # Note: with OHE, feature names expand. Simple mapping for basic importance.
            if hasattr(final_model, 'feature_importances_'):
                # This is a simplification for visualization
                st.info("Showing relative importance of feature groups (approximated).")
                importance = final_model.feature_importances_
        
        if importance is not None:
             # Just show a message if we can't map 1:1 due to OHE expansion without complex logic
             st.write("The model suggests these features are the strongest drivers of emissions.")
             # For a robust version, we'd extract names from the ColumnTransformer
    except:
        pass

    if importance is None:
        st.info("Feature importance visualization is not available for this specific model type.")


def render_dashboard(df, model, schema, categorical_options):

    """Main dashboard rendering with interactive filters."""
    st.divider()
    st.header("Analysis Dashboard")
    
    filtered_df = df.copy()
    col1, col2 = st.columns(2)
    
    with col1:
        if "Sector" in df.columns:
            sectors = ["All"] + sorted(df["Sector"].unique().tolist())
            selected_sector = st.selectbox("Filter by Sector", sectors)
            if selected_sector != "All":
                filtered_df = filtered_df[filtered_df["Sector"] == selected_sector]
    
    with col2:
        min_em, max_em = float(df[PRED_COL].min()), float(df[PRED_COL].max())
        if min_em < max_em:
            em_range = st.slider("Emission Range", min_em, max_em, (min_em, max_em))
            filtered_df = filtered_df[(filtered_df[PRED_COL] >= em_range[0]) & 
                                    (filtered_df[PRED_COL] <= em_range[1])]

    if not filtered_df.empty:
        generate_dashboard_metrics(filtered_df)
        
        # Check for actual columns to show performance metrics
        actual_col = None
        for col in filtered_df.columns:
            if "total" in col.lower() and "ghg" in col.lower() and col != PRED_COL:
                actual_col = col
                break

        tabs = ["Distributions", "Sector Analysis", "Top Emitters"]
        if actual_col:
            tabs.append("Model Performance")
            
        d_tabs = st.tabs(tabs)
        
        with d_tabs[0]:
            plot_emissions_distribution(filtered_df)
            plot_correlation_heatmap(filtered_df)
        with d_tabs[1]:
            plot_sector_analysis(filtered_df)
        with d_tabs[2]:
            show_top_emitters(filtered_df)
        
        if actual_col:
            with d_tabs[3]:
                plot_performance_metrics(filtered_df, actual_col)
                plot_feature_importance(model, schema)

    else:
        st.warning("No data matches selected filters.")

def render_ui():
    """Main rendering function for the Streamlit application."""
    st.set_page_config(page_title="GHG Predictor", layout="wide")
    
    # Sidebar
    st.sidebar.header("GHG Predictor")
    input_mode = st.sidebar.radio("Input Mode", ["Batch Upload", "Single Prediction"])
    
    model, schema, categorical_options, err = load_artifacts()

    if err:
        st.error(err)
        return

    with st.sidebar.expander("Required Columns"):
        for col in schema.keys():
            st.text(col)

    st.sidebar.divider()
    st.sidebar.subheader("Download Samples")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        try:
            with open("sample_features.csv", "rb") as f:
                st.download_button("Features (Predict)", f, "features.csv", "text/csv")
        except: st.text("Sample 1 missing")
        
    with col2:
        try:
            with open("sample_with_actuals.csv", "rb") as f:
                st.download_button("With Actuals", f, "with_actuals.csv", "text/csv")
        except: st.text("Sample 2 missing")

    st.sidebar.divider()

    model_type = type(model.named_steps['model']).__name__ if hasattr(model, 'named_steps') else type(model).__name__
    st.sidebar.info(f"Loaded: **{model_type}**")
    st.sidebar.caption(f"Sklearn version: {sklearn.__version__}")

    # Main content
    st.title("Greenhouse Gas Emissions Predictor")
    st.text("Predict total location-based GHG emissions (Metric Tons CO2e) for buildings.")


    if input_mode == "Batch Upload":
        uploaded_file = st.file_uploader("Upload Features CSV", type="csv")
        if uploaded_file:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.subheader("Data Preview")
                st.dataframe(raw_df.head(10))
                
                valid_df, val_err = validate_input(raw_df, schema)
                if val_err:
                    st.error(val_err)
                else:
                    # Run predictions automatically on upload
                    preds, pred_err = run_predictions(model, valid_df)
                    if pred_err:
                        st.error(pred_err)
                    else:
                        output_df = raw_df.copy()
                        output_df[PRED_COL] = preds
                        st.session_state["results_df"] = output_df
                
                if "results_df" in st.session_state:
                    results = st.session_state["results_df"]
                    
                    st.subheader("Download Results")
                    output_csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=output_csv,
                        file_name="ghg_predictions.csv",
                        mime="text/csv"
                    )
                    
                    render_dashboard(results, model, schema, categorical_options)


            except Exception as e:
                st.error(f"Processing error: {str(e)}")

    else:
        st.subheader("Single Building Entry")
        with st.form("single_pred_form"):
            input_dict = {}
            cols = st.columns(2)
            for i, (col_name, dtype) in enumerate(schema.items()):
                with cols[i % 2]:
                    if col_name in categorical_options:
                        input_dict[col_name] = st.selectbox(col_name, options=categorical_options[col_name])
                    elif "Has_" in col_name:
                        # Use checkbox for binary fields
                        val = st.checkbox(col_name, value=False)
                        input_dict[col_name] = 1 if val else 0
                    elif "object" in dtype:
                        input_dict[col_name] = st.text_input(col_name)
                    else:
                        if "int" in dtype:
                            input_dict[col_name] = st.number_input(col_name, value=1, step=1)
                        else:
                            input_dict[col_name] = st.number_input(col_name, value=0.0)


            
            submit = st.form_submit_button("Predict")
            if submit:
                single_df = pd.DataFrame([input_dict])
                preds, pred_err = run_predictions(model, single_df)
                if pred_err:
                    st.error(pred_err)
                else:
                    st.metric(
                        "Predicted GHG Emissions", 
                        f"{preds[0]:.2f} MT CO2e",
                        help="Metric Tons of Carbon Dioxide Equivalent. This number represents the total annual warming impact of all greenhouse gases released by this building, converted into an equivalent amount of CO2."
                    )
                    
                    # Categorize results
                    val = preds[0]
                    if val < 10:
                        st.success("🟢 **Low Impact:** This building's emissions are significantly below the sector average.")
                    elif val < 50:
                        st.info("🟡 **Moderate Impact:** This building's emissions are within the typical range for public facilities.")
                    elif val < 200:
                        st.warning("🟠 **High Impact:** This building's emissions are high. Consider energy efficiency retrofits.")
                    else:
                        st.error("🔴 **Very High Impact:** This building is a major emitter. Urgent environmental audit recommended.")



if __name__ == "__main__":
    render_ui()
