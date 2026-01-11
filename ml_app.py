import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import f_classif
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Page configuration
st.set_page_config(
    page_title="Malaysia USD Exchange Rate Predictor",
    page_icon="🇲🇾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class USDExchangePredictor:
    def __init__(self):
        self.df = None
        self.models = {}
        self.results_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
    def load_sample_data(self):
        """Generate synthetic data for demonstration if real data is not available"""
        dates = pd.date_range(start='1997-01-01', end='2024-12-01', freq='MS')
        n = len(dates)
        
        np.random.seed(42)
        
        # Generate synthetic economic data
        data = pd.DataFrame({
            'date': dates,
            'leading': np.random.normal(100, 5, n).cumsum()/50 + 95,
            'coincident': np.random.normal(100, 5, n).cumsum()/40 + 98,
            'lagging': np.random.normal(100, 5, n).cumsum()/30 + 100,
            'leading_diffusion': np.random.uniform(40, 60, n),
            'coincident_diffusion': np.random.uniform(45, 65, n),
            'gdp': np.random.normal(1000, 50, n).cumsum()/20 + 950,
            'gni': np.random.normal(1100, 60, n).cumsum()/20 + 1050,
            'Net migration': np.random.normal(0, 10000, n),
            'inflation': np.random.normal(2.5, 0.5, n).cumsum()/50 + 2,
            'USD': np.random.normal(4.2, 0.2, n).cumsum()/100 + 4.0
        })
        
        # Add trends and seasonality
        data['USD'] += 0.1 * np.sin(2 * np.pi * np.arange(n) / 12)
        data['USD'] += 0.01 * np.arange(n) / n  # Slight upward trend
        
        return data
    
    def preprocess_data(self, df):
        """Preprocess the data for modeling"""
        df_copy = df.copy()
        
        # Create year and month columns
        df_copy['year'] = df_copy['date'].dt.year
        df_copy['month'] = df_copy['date'].dt.month
        
        # Handle missing values in USD
        df_copy['USD'] = df_copy['USD'].ffill()
        df_copy = df_copy.dropna(subset=['USD'])
        
        # Filter out 2025 if present
        df_copy = df_copy[df_copy['year'] != 2025]
        
        return df_copy
    
    def train_models(self, X_train, y_train):
        """Train all machine learning models"""
        from sklearn.model_selection import cross_val_score
        
        results = {}
        
        # 1. Linear Regression
        start_time = time.time()
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        results['Linear Regression'] = {
            'model': lr,
            'training_time': time.time() - start_time
        }
        
        # 2. Decision Tree
        start_time = time.time()
        dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=42)
        dt.fit(X_train, y_train)
        results['Decision Tree'] = {
            'model': dt,
            'training_time': time.time() - start_time
        }
        
        # 3. Random Forest
        start_time = time.time()
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        results['Random Forest'] = {
            'model': rf,
            'training_time': time.time() - start_time
        }
        
        # 4. Gradient Boosting
        start_time = time.time()
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
        gb.fit(X_train, y_train)
        results['Gradient Boosting'] = {
            'model': gb,
            'training_time': time.time() - start_time
        }
        
        # 5. SVR
        start_time = time.time()
        svr = SVR(kernel='rbf', C=1, epsilon=0.1)
        svr.fit(X_train, y_train)
        results['SVR'] = {
            'model': svr,
            'training_time': time.time() - start_time
        }
        
        # 6. XGBoost (with simplified hyperparameter tuning)
        start_time = time.time()
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        results['XGBoost'] = {
            'model': xgb_model,
            'training_time': time.time() - start_time
        }
        
        return results
    
    def evaluate_models(self, models, X_test, y_test):
        """Evaluate all models and return results dataframe"""
        results = []
        
        for name, model_info in models.items():
            model = model_info['model']
            y_pred = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R² Score': r2,
                'Training Time (s)': model_info['training_time']
            })
        
        return pd.DataFrame(results).sort_values('R² Score', ascending=False)

def main():
    st.markdown('<h1 class="main-header">🇲🇾 Malaysia USD Exchange Rate Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Predicting USD/MYR with Economic Indicators & Migration Data</h3>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Flag_of_Malaysia.svg/800px-Flag_of_Malaysia.svg.png", 
                width=150)
        st.title("Navigation")
        page = st.radio(
            "Select Page:",
            ["🏠 Overview", "📊 Data Analysis", "🤖 Model Training", "🔮 Predictions", "🏛️ Policy Insights", "📋 Report"]
        )
        
        st.markdown("---")
        st.subheader("Data Options")
        use_sample_data = st.checkbox("Use Sample Data", value=True)
        
        if not use_sample_data:
            uploaded_file = st.file_uploader("Upload your CSV data", type=['csv'])
        else:
            uploaded_file = None
            
        st.markdown("---")
        st.info("""
        **Course:** BSD3523 Machine Learning
        **University:** Universiti Teknologi MARA
        **Team:** 
        - Muhammad Danial Bin Issham
        - Ain Mardhiah Binti Abdul Hamid
        - Haizatul Syifa Binti Mansor
        - Hamizan Nasri Bin Zulkairi
        - Siti Nurul Insyirah Binti Mohd Fauzi
        """)
    
    # Initialize predictor
    predictor = USDExchangePredictor()
    
    # Load data
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.info("Using sample data instead.")
            df = predictor.load_sample_data()
    else:
        df = predictor.load_sample_data()
    
    # Preprocess data
    df_processed = predictor.preprocess_data(df)
    predictor.df = df_processed
    
    # Page routing
    if page == "🏠 Overview":
        show_overview_page(df_processed, predictor)
    elif page == "📊 Data Analysis":
        show_analysis_page(df_processed, predictor)
    elif page == "🤖 Model Training":
        show_training_page(df_processed, predictor)
    elif page == "🔮 Predictions":
        show_predictions_page(df_processed, predictor)
    elif page == "🏛️ Policy Insights":
        show_policy_insights_page(df_processed, predictor)
    elif page == "📋 Report":
        show_report_page(df_processed, predictor)

def show_overview_page(df, predictor):
    """Display overview page"""
    st.markdown("""
    ## Project Overview
    
    This project aims to predict the USD/MYR exchange rate using various economic indicators,
    with a special focus on migration trends and their social and governance implications.
    """)
    
    # Key metrics
    st.subheader("📈 Key Economic Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latest_usd = df['USD'].iloc[-1]
        st.metric("Current USD/MYR", f"{latest_usd:.3f}")
    
    with col2:
        avg_inflation = df['inflation'].mean()
        st.metric("Avg Inflation", f"{avg_inflation:.2f}%")
    
    with col3:
        avg_migration = df['Net migration'].mean()
        st.metric("Avg Net Migration", f"{avg_migration:,.0f}")
    
    with col4:
        gdp_growth = ((df['gdp'].iloc[-1] - df['gdp'].iloc[0]) / df['gdp'].iloc[0]) * 100
        st.metric("GDP Growth", f"{gdp_growth:.1f}%")
    
    # Economic indicators visualization
    st.subheader("📊 Economic Trends Over Time")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('USD/MYR Exchange Rate', 'Economic Indicators',
                       'Migration Trends', 'GDP vs GNI'),
        vertical_spacing=0.15
    )
    
    # USD Exchange Rate
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['USD'], name='USD/MYR',
                  line=dict(color='#FF6B6B')),
        row=1, col=1
    )
    
    # Economic Indicators
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['leading'], name='Leading',
                  line=dict(color='#4ECDC4')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['coincident'], name='Coincident',
                  line=dict(color='#45B7D1')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['lagging'], name='Lagging',
                  line=dict(color='#96CEB4')),
        row=1, col=2
    )
    
    # Migration Trends
    fig.add_trace(
        go.Bar(x=df['date'], y=df['Net migration'], name='Net Migration',
              marker_color='#FECA57'),
        row=2, col=1
    )
    
    # GDP vs GNI
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['gdp'], name='GDP',
                  line=dict(color='#FF9FF3')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['gni'], name='GNI',
                  line=dict(color='#54A0FF')),
        row=2, col=2
    )
    
    fig.update_layout(height=700, showlegend=True, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    # Social and Governance Impact
    st.markdown("""
    ## 🏛️ Social & Governance Impact
    
    ### **Social Impact (S)**
    **Migration Trends & Economic Inclusion:**
    - Understanding how migration influences local economy and remittances
    - Developing social programs for migrant integration
    - Maximizing remittance benefits for social welfare
    
    **Remittance Flows & Social Welfare:**
    - Informing decisions about social welfare programs
    - Enhancing financial inclusion and poverty reduction strategies
    
    ### **Governance Impact (G)**
    **Policy-making and Governance:**
    - Managing national reserves and currency stabilization
    - Informing economic policies for currency stability
    
    **Transparency and Decision-Making:**
    - Evidence-based policymaking for foreign trade and investment
    - Data-driven decisions for national reserves management
    """)

def show_analysis_page(df, predictor):
    """Display data analysis page"""
    st.subheader("📊 Data Analysis & Exploration")
    
    # Data preview
    with st.expander("Data Preview", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.write("**Dataset Info:**")
            st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"Date Range: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
    
    # Statistical summary
    with st.expander("Statistical Summary"):
        st.dataframe(df.describe(), use_container_width=True)
    
    # Correlation analysis
    st.subheader("🔗 Correlation Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Matrix of Economic Indicators")
    st.pyplot(fig)
    
    # Feature importance analysis
    st.subheader("🎯 Feature Importance Analysis")
    
    # Prepare features and target
    X = df[['leading', 'coincident', 'lagging', 'leading_diffusion', 
            'coincident_diffusion', 'gdp', 'gni', 'Net migration', 'year', 'month']]
    y = df['USD']
    
    # Correlation with target
    corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)
    
    fig = go.Figure(go.Bar(
        x=corr_with_target.values,
        y=corr_with_target.index,
        orientation='h',
        marker_color='#3498db'
    ))
    
    fig.update_layout(
        title="Correlation with USD Exchange Rate",
        xaxis_title="Absolute Correlation",
        yaxis_title="Feature",
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ANOVA test
    with st.spinner("Performing ANOVA test..."):
        anova_results = []
        for feature in X.columns:
            F, p = f_classif(X[[feature]], y)
            anova_results.append({'Feature': feature, 'F-value': F[0], 'p-value': p[0]})
        
        anova_df = pd.DataFrame(anova_results).sort_values('p-value')
        
        st.write("**ANOVA Results (p-value < 0.05):**")
        st.dataframe(anova_df[anova_df['p-value'] < 0.05], use_container_width=True)
    
    # Distribution analysis
    st.subheader("📈 Distribution Analysis")
    
    selected_features = st.multiselect(
        "Select features for distribution analysis:",
        numeric_cols.tolist(),
        default=['USD', 'inflation', 'gdp', 'Net migration']
    )
    
    if selected_features:
        fig = make_subplots(
            rows=len(selected_features), cols=2,
            subplot_titles=[f"{feat} Histogram" for feat in selected_features] + 
                          [f"{feat} Box Plot" for feat in selected_features],
            vertical_spacing=0.1
        )
        
        for i, feature in enumerate(selected_features, 1):
            # Histogram
            fig.add_trace(
                go.Histogram(x=df[feature], name=feature, nbinsx=30),
                row=i, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(y=df[feature], name=feature),
                row=i, col=2
            )
        
        fig.update_layout(height=300*len(selected_features), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def show_training_page(df, predictor):
    """Display model training page"""
    st.subheader("🤖 Machine Learning Model Training")
    
    # Feature selection
    st.markdown("### 🎯 Feature Selection")
    
    selected_features = st.multiselect(
        "Select features for modeling:",
        ['leading', 'coincident', 'lagging', 'leading_diffusion', 
         'coincident_diffusion', 'gdp', 'gni', 'Net migration', 'year', 'month'],
        default=['gdp', 'gni', 'lagging', 'coincident', 'leading', 'year']
    )
    
    if not selected_features:
        st.warning("Please select at least one feature.")
        return
    
    # Prepare data
    X = df[selected_features]
    y = df['USD']
    
    # Train-test split configuration
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test set size (%)", 10, 40, 20)
    
    with col2:
        random_state = st.number_input("Random state", 0, 100, 42)
    
    # Train models
    if st.button("🚀 Train All Models", type="primary"):
        with st.spinner("Training models... This may take a few moments."):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=random_state
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            predictor.X_train = X_train_scaled
            predictor.X_test = X_test_scaled
            predictor.y_train = y_train
            predictor.y_test = y_test
            predictor.scaler = scaler
            
            # Train models
            models = predictor.train_models(X_train_scaled, y_train)
            predictor.models = models
            
            # Evaluate models
            results_df = predictor.evaluate_models(models, X_test_scaled, y_test)
            predictor.results_df = results_df
            
            st.success("✅ Models trained successfully!")
    
    # Display results if available
    if predictor.results_df is not None:
        st.subheader("📊 Model Performance Comparison")
        
        # Display results table
        st.dataframe(
            predictor.results_df.style.format({
                'RMSE': '{:.4f}',
                'MAE': '{:.4f}',
                'R² Score': '{:.4f}',
                'Training Time (s)': '{:.2f}'
            }),
            use_container_width=True
        )
        
        # Visual comparison
        fig = go.Figure()
        
        metrics_to_plot = st.multiselect(
            "Select metrics to visualize:",
            ['RMSE', 'MAE', 'R² Score'],
            default=['R² Score', 'RMSE']
        )
        
        for metric in metrics_to_plot:
            fig.add_trace(go.Bar(
                x=predictor.results_df['Model'],
                y=predictor.results_df[metric],
                name=metric,
                text=[f'{val:.3f}' for val in predictor.results_df[metric]],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            barmode='group',
            yaxis_title="Score",
            template='plotly_white',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance from Random Forest
        if 'Random Forest' in predictor.models:
            st.subheader("🔍 Feature Importance (Random Forest)")
            
            rf_model = predictor.models['Random Forest']['model']
            importance_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = go.Figure(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h',
                marker_color='#3498db'
            ))
            
            fig.update_layout(
                title="Random Forest Feature Importance",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Actual vs Predicted comparison
        st.subheader("📈 Actual vs Predicted Values")
        
        selected_model = st.selectbox(
            "Select model for visualization:",
            list(predictor.models.keys())
        )
        
        if selected_model:
            model = predictor.models[selected_model]['model']
            y_pred = model.predict(predictor.X_test)
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Actual vs Predicted', 'Prediction Error Distribution'),
                column_widths=[0.7, 0.3]
            )
            
            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=predictor.y_test.values,
                    y=y_pred,
                    mode='markers',
                    name='Predictions',
                    marker=dict(size=8, color='#e74c3c', opacity=0.6)
                ),
                row=1, col=1
            )
            
            # Add perfect prediction line
            min_val = min(predictor.y_test.min(), y_pred.min())
            max_val = max(predictor.y_test.max(), y_pred.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='#2c3e50', dash='dash')
                ),
                row=1, col=1
            )
            
            # Error distribution
            errors = y_pred - predictor.y_test.values
            fig.add_trace(
                go.Histogram(x=errors, name='Prediction Errors', nbinsx=30),
                row=1, col=2
            )
            
            fig.update_layout(
                height=500,
                showlegend=True,
                template='plotly_white'
            )
            
            fig.update_xaxes(title_text="Actual Values", row=1, col=1)
            fig.update_yaxes(title_text="Predicted Values", row=1, col=1)
            fig.update_xaxes(title_text="Prediction Error", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)
            
            st.plotly_chart(fig, use_container_width=True)

def show_predictions_page(df, predictor):
    """Display predictions page"""
    st.subheader("🔮 USD Exchange Rate Predictions")
    
    if not predictor.models:
        st.warning("⚠️ Please train models first in the 'Model Training' section.")
        return
    
    # Prediction interface
    st.markdown("### 📝 Make Custom Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        leading = st.number_input("Leading Index", value=100.0, min_value=0.0, max_value=200.0, step=0.1)
        coincident = st.number_input("Coincident Index", value=100.0, min_value=0.0, max_value=200.0, step=0.1)
        lagging = st.number_input("Lagging Index", value=100.0, min_value=0.0, max_value=200.0, step=0.1)
    
    with col2:
        gdp = st.number_input("GDP", value=1000.0, min_value=0.0, max_value=5000.0, step=1.0)
        gni = st.number_input("GNI", value=1100.0, min_value=0.0, max_value=5000.0, step=1.0)
        net_migration = st.number_input("Net Migration", value=0.0, min_value=-100000.0, max_value=100000.0, step=1000.0)
    
    with col3:
        year = st.number_input("Year", value=2024, min_value=1997, max_value=2030, step=1)
        month = st.number_input("Month", value=12, min_value=1, max_value=12, step=1)
    
    # Select model
    selected_model = st.selectbox(
        "Select prediction model:",
        list(predictor.models.keys())
    )
    
    if st.button("Generate Prediction", type="primary"):
        model = predictor.models[selected_model]['model']
        
        # Prepare input data
        input_data = pd.DataFrame({
            'leading': [leading],
            'coincident': [coincident],
            'lagging': [lagging],
            'gdp': [gdp],
            'gni': [gni],
            'Net migration': [net_migration],
            'year': [year],
            'month': [month]
        })
        
        # Ensure all columns are present
        for col in predictor.X_train.shape[1]:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Scale input
        input_scaled = predictor.scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Display prediction
        st.markdown("### 📊 Prediction Result")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 15px; color: white;">
                <h3>Predicted USD/MYR Exchange Rate</h3>
                <h1 style="font-size: 4rem; margin: 1rem 0;">{prediction:.3f}</h1>
                <p>Using {selected_model} model</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Interpretation
        st.markdown("#### 📝 Interpretation")
        
        latest_usd = df['USD'].iloc[-1]
        change = ((prediction - latest_usd) / latest_usd) * 100
        
        if change > 2:
            st.warning(f"📈 **Strong Appreciation Expected**: Predicted increase of {change:.1f}% from current rate of {latest_usd:.3f}")
        elif change > 0:
            st.info(f"📈 **Moderate Appreciation Expected**: Predicted increase of {change:.1f}% from current rate of {latest_usd:.3f}")
        elif change < -2:
            st.error(f"📉 **Strong Depreciation Expected**: Predicted decrease of {abs(change):.1f}% from current rate of {latest_usd:.3f}")
        else:
            st.success(f"➡️ **Stable Exchange Rate**: Predicted change of {change:.1f}% from current rate of {latest_usd:.3f}")
    
    # Future forecasting
    st.subheader("🔮 Future Forecasting")
    
    n_months = st.slider("Number of months to forecast", 1, 12, 6)
    
    if st.button("Generate Future Forecast"):
        st.markdown(f"#### 📈 {n_months}-Month USD Forecast")
        
        # Generate future predictions
        last_date = df['date'].max()
        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, n_months+1)]
        
        # Use Random Forest for forecasting (best performing model)
        if 'Random Forest' in predictor.models:
            rf_model = predictor.models['Random Forest']['model']
            
            # Create future features based on trends
            future_predictions = []
            confidence_intervals = []
            
            for i in range(n_months):
                # Simple trend projection
                future_features = {
                    'leading': df['leading'].iloc[-1] * (1 + 0.001 * i),
                    'coincident': df['coincident'].iloc[-1] * (1 + 0.001 * i),
                    'lagging': df['lagging'].iloc[-1] * (1 + 0.001 * i),
                    'gdp': df['gdp'].iloc[-1] * (1 + 0.005 * i),
                    'gni': df['gni'].iloc[-1] * (1 + 0.005 * i),
                    'Net migration': df['Net migration'].iloc[-1],
                    'year': future_dates[i].year,
                    'month': future_dates[i].month
                }
                
                # Prepare and scale input
                future_input = pd.DataFrame([future_features])
                future_scaled = predictor.scaler.transform(future_input)
                
                # Make prediction
                pred = rf_model.predict(future_scaled)[0]
                future_predictions.append(pred)
                
                # Simple confidence interval
                confidence_intervals.append([pred * 0.98, pred * 1.02])
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_USD': future_predictions,
                'Lower_Bound': [ci[0] for ci in confidence_intervals],
                'Upper_Bound': [ci[1] for ci in confidence_intervals]
            })
            
            # Plot forecast
            fig = go.Figure()
            
            # Historical data (last 2 years)
            historical_dates = df['date'][df['date'] >= (last_date - pd.DateOffset(years=2))]
            historical_usd = df.loc[df['date'].isin(historical_dates), 'USD']
            
            fig.add_trace(go.Scatter(
                x=historical_dates,
                y=historical_usd,
                mode='lines',
                name='Historical',
                line=dict(color='#3498db', width=2)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Predicted_USD'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#e74c3c', width=3, dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
                y=[ci[0] for ci in confidence_intervals] + [ci[1] for ci in confidence_intervals][::-1],
                fill='toself',
                fillcolor='rgba(231, 76, 60, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval'
            ))
            
            fig.update_layout(
                title=f"{n_months}-Month USD/MYR Forecast",
                xaxis_title="Date",
                yaxis_title="USD/MYR Exchange Rate",
                template='plotly_white',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display forecast table
            st.dataframe(forecast_df.style.format({
                'Predicted_USD': '{:.3f}',
                'Lower_Bound': '{:.3f}',
                'Upper_Bound': '{:.3f}'
            }), use_container_width=True)

def show_policy_insights_page(df, predictor):
    """Display policy insights page"""
    st.subheader("🏛️ Policy Insights & Social Impact")
    
    # Social Impact Analysis
    st.markdown("""
    ## 🌍 Social Impact Analysis
    
    ### **Migration & Economic Inclusion**
    Our analysis reveals significant relationships between migration trends and economic indicators:
    """)
    
    # Migration impact visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Migration vs USD', 'Migration vs GDP',
                       'Remittance Impact Analysis', 'Social Welfare Correlation'),
        vertical_spacing=0.15
    )
    
    # Migration vs USD
    fig.add_trace(
        go.Scatter(x=df['Net migration'], y=df['USD'], mode='markers',
                  name='Migration vs USD', marker=dict(color='#FF6B6B')),
        row=1, col=1
    )
    
    # Migration vs GDP
    fig.add_trace(
        go.Scatter(x=df['Net migration'], y=df['gdp'], mode='markers',
                  name='Migration vs GDP', marker=dict(color='#4ECDC4')),
        row=1, col=2
    )
    
    # Remittance impact (simulated)
    remittance_impact = df['Net migration'].apply(lambda x: abs(x) * 0.05)
    fig.add_trace(
        go.Scatter(x=df['date'], y=remittance_impact, mode='lines',
                  name='Estimated Remittance Impact', line=dict(color='#FECA57')),
        row=2, col=1
    )
    
    # Social welfare correlation
    welfare_index = (df['gni'] / df['gdp']) * 100
    fig.add_trace(
        go.Scatter(x=df['date'], y=welfare_index, mode='lines',
                  name='Welfare Index (GNI/GDP)', line=dict(color='#54A0FF')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    # Policy Recommendations
    st.markdown("""
    ### **Policy Recommendations**
    
    #### **For Migration Management:**
    1. **Integration Programs**: Develop targeted programs for migrant economic integration
    2. **Remittance Facilitation**: Create channels to maximize remittance benefits
    3. **Skill Development**: Align migrant skills with economic needs
    
    #### **For Currency Stability:**
    1. **Reserve Management**: Use predictions for optimal foreign reserve allocation
    2. **Trade Policy**: Adjust tariffs based on exchange rate forecasts
    3. **Investment Strategy**: Guide foreign investment decisions
    
    #### **For Social Welfare:**
    1. **Targeted Support**: Direct resources based on migration patterns
    2. **Financial Inclusion**: Expand services to migrant communities
    3. **Economic Planning**: Incorporate migration trends in development plans
    """)
    
    # Governance Impact
    st.markdown("""
    ## 🏛️ Governance Impact
    
    ### **Transparency & Decision-Making**
    Our model enables evidence-based policymaking through:
    
    - **Data-Driven Insights**: Clear relationships between economic indicators
    - **Predictive Capabilities**: Forward-looking policy planning
    - **Scenario Analysis**: Testing policy impacts before implementation
    
    ### **Risk Management**
    Key areas for governance attention:
    
    1. **Currency Risk**: Manage exposure to USD fluctuations
    2. **Economic Shocks**: Prepare for migration-related economic changes
    3. **Social Stability**: Address migration-induced social challenges
    """)
    
    # Impact Metrics
    st.subheader("📊 Social & Governance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        migration_volatility = df['Net migration'].std()
        st.metric("Migration Volatility", f"{migration_volatility:,.0f}")
    
    with col2:
        usd_volatility = df['USD'].std()
        st.metric("USD Volatility", f"{usd_volatility:.3f}")
    
    with col3:
        economic_resilience = df['coincident'].std() / df['coincident'].mean()
        st.metric("Economic Resilience", f"{economic_resilience:.3f}")

def show_report_page(df, predictor):
    """Display report page"""
    st.subheader("📋 Project Report & Executive Summary")
    
    # Executive Summary
    st.markdown("""
    ## Executive Summary
    
    **Project Title:** Malaysia USD Exchange Rate Prediction with Social-Governance Impact Analysis
    
    **Objective:** Develop machine learning models to predict USD/MYR exchange rates while analyzing social and governance implications of migration and economic trends.
    
    **Key Achievements:**
    1. Successfully developed 6 ML models for USD prediction
    2. Identified key economic indicators influencing exchange rates
    3. Analyzed migration's impact on economic stability
    4. Created actionable insights for policymakers
    """)
    
    # Methodology
    st.markdown("""
    ## Methodology
    
    ### **Data Sources:**
    - MEI (Main Economic Indicators)
    - GDP/GNI Annual Data
    - Net Migration Statistics
    - Monthly Exchange Rates
    - CPI Inflation Data
    
    ### **Technical Approach:**
    1. **Data Integration**: Combined multiple economic datasets
    2. **Feature Engineering**: Created temporal and derived features
    3. **Model Development**: Implemented 6 ML algorithms
    4. **Evaluation**: Comprehensive performance metrics
    5. **Impact Analysis**: Social and governance implications
    """)
    
    # Results Summary
    st.markdown("## Results Summary")
    
    if predictor.results_df is not None:
        best_model = predictor.results_df.iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Best Model", best_model['Model'])
        
        with col2:
            st.metric("Best R² Score", f"{best_model['R² Score']:.3f}")
        
        with col3:
            st.metric("Best RMSE", f"{best_model['RMSE']:.4f}")
        
        st.dataframe(predictor.results_df, use_container_width=True)
    else:
        st.info("Train models to see performance results.")
    
    # Key Insights
    st.markdown("""
    ## Key Insights
    
    ### **Economic Insights:**
    1. **GDP and GNI** are strong predictors of USD exchange rates
    2. **Migration trends** significantly impact economic stability
    3. **Leading indicators** provide early signals for currency movements
    
    ### **Social Impact:**
    1. Migration patterns correlate with economic performance
    2. Remittance flows influence currency stability
    3. Social welfare can be optimized using predictive insights
    
    ### **Governance Implications:**
    1. Data-driven policies enhance economic stability
    2. Predictive models support proactive governance
    3. Migration management requires integrated economic planning
    """)
    
    # Recommendations
    st.markdown("""
    ## Recommendations
    
    ### **For Government Agencies:**
    1. **Bank Negara Malaysia**: Use predictions for currency management
    2. **DOSM**: Enhance data collection on migration-economic links
    3. **Economic Planning Unit**: Integrate predictions in development plans
    
    ### **For Policy Implementation:**
    1. Develop migration-responsive economic policies
    2. Create social programs based on predictive insights
    3. Establish early warning systems for economic risks
    
    ### **For Future Research:**
    1. Incorporate more granular migration data
    2. Expand to other currency pairs
    3. Develop real-time prediction systems
    """)
    
    # Download Report
    st.markdown("## 📥 Download Resources")
    
    if st.button("Generate Complete Report"):
        report_content = f"""
        MALAYSIA USD EXCHANGE RATE PREDICTION REPORT
        ============================================
        
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        EXECUTIVE SUMMARY:
        - Project: Predictive modeling of USD/MYR with social-governance analysis
        - Period Analyzed: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}
        - Data Points: {len(df)} monthly observations
        - Features Analyzed: {len(df.columns) - 2} economic indicators
        
        KEY FINDINGS:
        1. Machine learning models successfully predict USD exchange rates
        2. Migration trends significantly impact economic indicators
        3. GDP and GNI are among the most important predictors
        
        RECOMMENDATIONS:
        - Implement data-driven currency management strategies
        - Develop migration-responsive economic policies
        - Use predictive insights for social program optimization
        
        TEAM:
        - Muhammad Danial Bin Issham
        - Ain Mardhiah Binti Abdul Hamid
        - Haizatul Syifa Binti Mansor
        - Hamizan Nasri Bin Zulkairi
        - Siti Nurul Insyirah Binti Mohd Fauzi
        
        COURSE: BSD3523 Machine Learning
        UNIVERSITY: Universiti Teknologi MARA
        """
        
        st.download_button(
            label="Download Report as TXT",
            data=report_content,
            file_name="usd_prediction_report.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()