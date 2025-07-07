import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
import shap

# Set page config
st.set_page_config(
    page_title="Clinical Trial Criteria Optimizer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_synthetic_patient_data(n_patients=1000, seed=42):
    """Generate synthetic patient data with realistic clinical characteristics."""
    np.random.seed(seed)
    
    # Generate realistic patient data
    data = {
        'patient_id': range(1, n_patients + 1),
        'age': np.random.normal(65, 15, n_patients).clip(18, 95),
        'bmi': np.random.normal(28, 6, n_patients).clip(18, 50),
        'creatinine': np.random.normal(1.2, 0.4, n_patients).clip(0.5, 3.0),
        'hba1c': np.random.normal(7.5, 1.5, n_patients).clip(5.0, 12.0),
        'systolic_bp': np.random.normal(140, 20, n_patients).clip(90, 200),
        'diastolic_bp': np.random.normal(85, 12, n_patients).clip(60, 120),
        'cholesterol': np.random.normal(200, 40, n_patients).clip(120, 300),
        'triglycerides': np.random.normal(150, 80, n_patients).clip(50, 500),
        'smoking_status': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
        'diabetes_history': np.random.choice([0, 1], n_patients, p=[0.6, 0.4]),
        'cardiovascular_history': np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),
        'kidney_disease': np.random.choice([0, 1], n_patients, p=[0.9, 0.1]),
        'liver_disease': np.random.choice([0, 1], n_patients, p=[0.95, 0.05]),
        'medication_count': np.random.poisson(3, n_patients).clip(0, 10)
    }
    
    # Create event rate based on risk factors (simplified model)
    risk_score = (
        (data['age'] - 65) / 15 * 0.3 +
        (data['bmi'] - 28) / 6 * 0.2 +
        (data['creatinine'] - 1.2) / 0.4 * 0.4 +
        (data['hba1c'] - 7.5) / 1.5 * 0.3 +
        data['smoking_status'] * 0.5 +
        data['diabetes_history'] * 0.4 +
        data['cardiovascular_history'] * 0.6
    )
    
    # Convert risk score to probability (0.05 to 0.25 range)
    event_probability = 0.05 + 0.20 * (1 / (1 + np.exp(-risk_score)))
    data['event_occurred'] = np.random.binomial(1, event_probability)
    
    return pd.DataFrame(data)

def apply_inclusion_exclusion_criteria(patients, criteria_config):
    """Apply inclusion/exclusion criteria to patient population."""
    eligible_patients = patients.copy()
    
    for criterion, config in criteria_config.items():
        if config['active']:
            if config['type'] == 'continuous':
                if config['operator'] == '>':
                    eligible_patients = eligible_patients[
                        eligible_patients[config['variable']] > config['threshold']
                    ]
                elif config['operator'] == '<':
                    eligible_patients = eligible_patients[
                        eligible_patients[config['variable']] < config['threshold']
                    ]
                elif config['operator'] == '>=':
                    eligible_patients = eligible_patients[
                        eligible_patients[config['variable']] >= config['threshold']
                    ]
                elif config['operator'] == '<=':
                    eligible_patients = eligible_patients[
                        eligible_patients[config['variable']] <= config['threshold']
                    ]
            elif config['type'] == 'binary':
                eligible_patients = eligible_patients[
                    eligible_patients[config['variable']] == config['threshold']
                ]
    
    return eligible_patients

def run_monte_carlo_simulation(patients, criteria_config, n_simulations=1000):
    """Run Monte Carlo simulation with varying criteria thresholds."""
    results = []
    
    # Define parameter ranges for each criterion
    param_ranges = {}
    for criterion, config in criteria_config.items():
        if config['active']:
            if config['type'] == 'continuous':
                # Generate range around current threshold
                current_val = config['threshold']
                range_width = config.get('range_width', 0.5)
                min_val = current_val * (1 - range_width)
                max_val = current_val * (1 + range_width)
                param_ranges[criterion] = np.linspace(min_val, max_val, 10)
            else:  # binary
                param_ranges[criterion] = [0, 1]
    
    # Generate all combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    for i in range(min(n_simulations, np.prod([len(vals) for vals in param_values]))):
        # Randomly sample parameter values
        sampled_params = {}
        for j, param_name in enumerate(param_names):
            sampled_params[param_name] = np.random.choice(param_values[j])
        
        # Create temporary criteria config with sampled parameters
        temp_config = criteria_config.copy()
        for criterion, config in temp_config.items():
            if criterion in sampled_params:
                temp_config[criterion]['threshold'] = sampled_params[criterion]
        
        # Apply criteria and calculate outcomes
        eligible_patients = apply_inclusion_exclusion_criteria(patients, temp_config)
        
        population_size = len(eligible_patients)
        event_rate = eligible_patients['event_occurred'].mean() if population_size > 0 else 0
        
        # Store results
        result = {
            'simulation_id': i,
            'population_size': population_size,
            'event_rate': event_rate
        }
        
        # Add parameter values
        for criterion in param_names:
            result[f'param_{criterion}'] = sampled_params[criterion]
        
        results.append(result)
    
    return pd.DataFrame(results)

def calculate_shapley_values(simulation_results, criteria_config):
    """Calculate Shapley values for each criterion's impact on outcomes."""
    # Prepare features for SHAP analysis
    feature_cols = [col for col in simulation_results.columns if col.startswith('param_')]
    target_cols = ['population_size', 'event_rate']
    
    shap_values_dict = {}
    
    for target in target_cols:
        # Prepare data
        X = simulation_results[feature_cols].copy()
        y = simulation_results[target]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train a model to explain
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        
        # Calculate mean absolute SHAP values for each feature
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        
        shap_values_dict[target] = {
            'features': feature_cols,
            'shap_values': mean_shap_values,
            'model': model,
            'explainer': explainer
        }
    
    return shap_values_dict

def train_optimization_model(simulation_results, criteria_config):
    """Train a model to predict outcomes from criteria parameters for optimization."""
    # Prepare features for optimization
    feature_cols = [col for col in simulation_results.columns if col.startswith('param_')]
    target_cols = ['population_size', 'event_rate']
    
    X = simulation_results[feature_cols].copy()
    y = simulation_results[target_cols].copy()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train a multi-output model
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_scaled, y)
    
    return model, scaler, feature_cols

def objective_function(params, model, scaler, feature_cols, target_population, target_event_rate, 
                      population_weight=0.5, event_rate_weight=0.5):
    """Objective function for optimization."""
    # Reshape parameters for prediction with proper feature names
    X = pd.DataFrame([params], columns=list(feature_cols))
    
    # Suppress warnings for scaler transform
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_scaled = scaler.transform(X)
    
    # Predict outcomes
    prediction = model.predict(X_scaled)[0]
    predicted_population, predicted_event_rate = prediction
    
    # Calculate weighted loss
    population_loss = abs(predicted_population - target_population) / max(target_population, 1)
    event_rate_loss = abs(predicted_event_rate - target_event_rate) / max(target_event_rate, 0.01)
    
    total_loss = (population_weight * population_loss + 
                  event_rate_weight * event_rate_loss)
    
    return total_loss

def find_optimal_criteria(simulation_results, criteria_config, target_population, target_event_rate,
                         population_weight=0.5, event_rate_weight=0.5, n_optimizations=10):
    """Find optimal criteria values for target outcomes."""
    # Train optimization model
    model, scaler, feature_cols = train_optimization_model(simulation_results, criteria_config)
    
    # Get parameter bounds
    param_bounds = []
    for criterion, config in criteria_config.items():
        if config['active']:
            if config['type'] == 'continuous':
                current_val = config['threshold']
                range_width = config.get('range_width', 0.5)
                min_val = current_val * (1 - range_width)
                max_val = current_val * (1 + range_width)
                param_bounds.append((min_val, max_val))
            else:  # binary
                param_bounds.append((0, 1))
    
    best_result = None
    best_loss = float('inf')
    
    # Run multiple optimizations with different starting points
    for i in range(n_optimizations):
        # Random starting point
        x0 = []
        for bounds in param_bounds:
            x0.append(np.random.uniform(bounds[0], bounds[1]))
        
        try:
            result = minimize(
                objective_function,
                x0,
                args=(model, scaler, feature_cols, target_population, target_event_rate,
                      population_weight, event_rate_weight),
                bounds=param_bounds,
                method='L-BFGS-B',
                options={'maxiter': 1000}
            )
            
            if result.success and result.fun < best_loss:
                best_result = result
                best_loss = result.fun
                
        except Exception as e:
            continue
    
    if best_result is None:
        return None
    
    # Get optimal parameters
    optimal_params = best_result.x
    
    # Predict outcomes with optimal parameters
    X_opt = pd.DataFrame([optimal_params], columns=list(feature_cols))
    
    # Suppress warnings for scaler transform
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_opt_scaled = scaler.transform(X_opt)
    optimal_outcomes = model.predict(X_opt_scaled)[0]
    
    return {
        'optimal_params': optimal_params,
        'predicted_population': optimal_outcomes[0],
        'predicted_event_rate': optimal_outcomes[1],
        'loss': best_loss,
        'feature_cols': feature_cols
    }

def find_similar_populations(simulation_results, target_population, target_event_rate, 
                           population_tolerance=0.1, event_rate_tolerance=0.01, max_results=20):
    """Find simulation results with similar population size and event rate."""
    # Calculate distances to target
    population_diff = abs(simulation_results['population_size'] - target_population) / max(target_population, 1)
    event_rate_diff = abs(simulation_results['event_rate'] - target_event_rate) / max(target_event_rate, 0.01)
    
    # Find results within tolerance
    within_tolerance = (population_diff <= population_tolerance) & (event_rate_diff <= event_rate_tolerance)
    
    if not within_tolerance.any():
        # If no exact matches, find closest results
        distances = np.sqrt(population_diff**2 + event_rate_diff**2)
        closest_indices = np.argsort(distances)[:max_results]
        similar_results = simulation_results.iloc[closest_indices].copy()
    else:
        similar_results = simulation_results[within_tolerance].copy()
        if len(similar_results) > max_results:
            # Sort by distance and take top results
            distances = np.sqrt(population_diff[within_tolerance]**2 + event_rate_diff[within_tolerance]**2)
            closest_indices = np.argsort(distances)[:max_results]
            similar_results = similar_results.iloc[closest_indices]
    
    # Add distance information
    similar_results['population_distance'] = abs(similar_results['population_size'] - target_population)
    similar_results['event_rate_distance'] = abs(similar_results['event_rate'] - target_event_rate)
    similar_results['total_distance'] = np.sqrt(
        (similar_results['population_distance'] / max(target_population, 1))**2 + 
        (similar_results['event_rate_distance'] / max(target_event_rate, 0.01))**2
    )
    
    return similar_results.sort_values('total_distance')

def create_optimization_plots(simulation_results, optimal_result, target_population, target_event_rate):
    """Create plots for optimization results."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Population Size vs Event Rate (All Simulations)',
            'Parameter Distribution for Similar Outcomes',
            'Optimization Results Comparison',
            'Parameter Sensitivity'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: All simulations with target point
    fig.add_trace(
        go.Scatter(
            x=simulation_results['population_size'],
            y=simulation_results['event_rate'],
            mode='markers',
            marker=dict(color='lightblue', opacity=0.6, size=4),
            name='All Simulations',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add target point
    fig.add_trace(
        go.Scatter(
            x=[target_population],
            y=[target_event_rate],
            mode='markers',
            marker=dict(color='red', size=12, symbol='star'),
            name='Target',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add optimal point if available
    if optimal_result:
        fig.add_trace(
            go.Scatter(
                x=[optimal_result['predicted_population']],
                y=[optimal_result['predicted_event_rate']],
                mode='markers',
                marker=dict(color='green', size=12, symbol='diamond'),
                name='Optimal',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Plot 2: Parameter distribution for similar outcomes
    if optimal_result:
        similar_results = find_similar_populations(
            simulation_results, 
            optimal_result['predicted_population'], 
            optimal_result['predicted_event_rate'],
            population_tolerance=0.05,
            event_rate_tolerance=0.005,
            max_results=50
        )
        
        param_cols = [col for col in similar_results.columns if col.startswith('param_')]
        for i, param_col in enumerate(param_cols):
            param_name = param_col.replace('param_', '').replace('_criteria', '')
            fig.add_trace(
                go.Histogram(
                    x=similar_results[param_col],
                    name=param_name,
                    opacity=0.7,
                    showlegend=False
                ),
                row=1, col=2
            )
    
    # Plot 3: Optimization comparison
    if optimal_result:
        fig.add_trace(
            go.Bar(
                x=['Target', 'Optimal'],
                y=[target_population, optimal_result['predicted_population']],
                name='Population Size',
                marker_color=['red', 'green'],
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=['Target', 'Optimal'],
                y=[target_event_rate, optimal_result['predicted_event_rate']],
                name='Event Rate',
                marker_color=['red', 'green'],
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Plot 4: Parameter sensitivity (placeholder)
    if optimal_result:
        param_names = [col.replace('param_', '').replace('_criteria', '') 
                      for col in optimal_result['feature_cols']]
        fig.add_trace(
            go.Bar(
                x=param_names,
                y=optimal_result['optimal_params'],
                name='Optimal Parameters',
                marker_color='lightgreen',
                showlegend=False
            ),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title_text="Optimization Analysis")
    return fig

def main():
    st.markdown('<h1 class="main-header">🏥 Clinical Trial Criteria Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("""
    This application simulates clinical trial pipelines using Monte Carlo simulations to optimize 
    inclusion/exclusion criteria. Adjust criteria thresholds and analyze their impact on population 
    size and event rates using Shapley values.
    """)
    
    # Sidebar configuration
    st.sidebar.markdown('<div class="sidebar-header">📊 Simulation Parameters</div>', unsafe_allow_html=True)
    
    n_patients = st.sidebar.slider("Number of Patients", 500, 5000, 1000, 100)
    n_simulations = st.sidebar.slider("Number of Monte Carlo Simulations", 100, 5000, 1000, 100)
    
    # Generate synthetic data
    patients = generate_synthetic_patient_data(n_patients)
    
    # Define inclusion/exclusion criteria
    criteria_config = {
        'age_criteria': {
            'name': 'Age',
            'variable': 'age',
            'type': 'continuous',
            'operator': '>=',
            'threshold': 18,
            'active': True,
            'range_width': 0.3
        },
        'bmi_criteria': {
            'name': 'BMI',
            'variable': 'bmi',
            'type': 'continuous',
            'operator': '<=',
            'threshold': 35,
            'active': True,
            'range_width': 0.4
        },
        'creatinine_criteria': {
            'name': 'Creatinine',
            'variable': 'creatinine',
            'type': 'continuous',
            'operator': '<=',
            'threshold': 2.0,
            'active': True,
            'range_width': 0.5
        },
        'smoking_criteria': {
            'name': 'Non-Smoker',
            'variable': 'smoking_status',
            'type': 'binary',
            'operator': '==',
            'threshold': 0,
            'active': True
        },
        'diabetes_criteria': {
            'name': 'Diabetes History',
            'variable': 'diabetes_history',
            'type': 'binary',
            'operator': '==',
            'threshold': 0,
            'active': True
        }
    }
    
    # Criteria configuration in sidebar
    st.sidebar.markdown('<div class="sidebar-header">🎯 Criteria Configuration</div>', unsafe_allow_html=True)
    
    for criterion_key, config in criteria_config.items():
        st.sidebar.markdown(f"**{config['name']}**")
        
        config['active'] = st.sidebar.checkbox(
            f"Enable {config['name']}", 
            value=config['active'], 
            key=f"active_{criterion_key}"
        )
        
        if config['active']:
            if config['type'] == 'continuous':
                config['threshold'] = st.sidebar.number_input(
                    f"Threshold ({config['operator']})",
                    value=float(config['threshold']),
                    step=0.1,
                    key=f"threshold_{criterion_key}"
                )
            else:  # binary
                config['threshold'] = st.sidebar.selectbox(
                    f"Value",
                    options=[0, 1],
                    index=0 if config['threshold'] == 0 else 1,
                    key=f"threshold_{criterion_key}"
                )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Patient Population Overview")
        
        # Apply current criteria
        eligible_patients = apply_inclusion_exclusion_criteria(patients, criteria_config)
        
        # Display metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("Total Patients", len(patients))
        
        with metric_col2:
            st.metric("Eligible Patients", len(eligible_patients))
        
        with metric_col3:
            eligibility_rate = len(eligible_patients) / len(patients) * 100
            st.metric("Eligibility Rate", f"{eligibility_rate:.1f}%")
        
        # Patient characteristics distribution
        st.markdown("### 📊 Patient Characteristics Distribution")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        variables = ['age', 'bmi', 'creatinine', 'hba1c', 'systolic_bp', 'cholesterol']
        titles = ['Age', 'BMI', 'Creatinine', 'HbA1c', 'Systolic BP', 'Cholesterol']
        
        for i, (var, title) in enumerate(zip(variables, titles)):
            axes[i].hist(patients[var], bins=30, alpha=0.7, label='All Patients', color='lightblue')
            axes[i].hist(eligible_patients[var], bins=30, alpha=0.7, label='Eligible Patients', color='orange')
            axes[i].set_title(title)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("### 🎯 Current Criteria Summary")
        
        for criterion_key, config in criteria_config.items():
            if config['active']:
                st.markdown(f"""
                **{config['name']}**  
                {config['operator']} {config['threshold']}
                """)
        
        st.markdown("### 📋 Event Rate Analysis")
        
        if len(eligible_patients) > 0:
            overall_event_rate = patients['event_occurred'].mean()
            eligible_event_rate = eligible_patients['event_occurred'].mean()
            
            st.metric("Overall Event Rate", f"{overall_event_rate:.3f}")
            st.metric("Eligible Event Rate", f"{eligible_event_rate:.3f}")
            
            if eligible_event_rate > overall_event_rate:
                st.success("✅ Higher event rate in eligible population")
            else:
                st.warning("⚠️ Lower event rate in eligible population")
    
    # Monte Carlo Simulation Section
    st.markdown("---")
    st.markdown("### 🔄 Monte Carlo Simulation")
    
    if st.button("Run Monte Carlo Simulation", type="primary"):
        with st.spinner("Running Monte Carlo simulation..."):
            # Run simulation
            simulation_results = run_monte_carlo_simulation(patients, criteria_config, n_simulations)
            
            # Calculate Shapley values
            shap_values_dict = calculate_shapley_values(simulation_results, criteria_config)
            
                        # Store results in session state for download
            st.session_state.simulation_results = simulation_results
            st.session_state.shap_values = shap_values_dict
    
    # Display Monte Carlo results persistently
    if 'simulation_results' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Monte Carlo Simulation Results")
        
        # Display the simulation results that were just calculated
        simulation_results = st.session_state.simulation_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Simulation Results Distribution")
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Population Size Distribution', 'Event Rate Distribution'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Histogram(x=simulation_results['population_size'], name='Population Size'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Histogram(x=simulation_results['event_rate'], name='Event Rate'),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 Population Size vs Event Rate")
            
            fig = px.scatter(
                simulation_results,
                x='population_size',
                y='event_rate',
                title='Population Size vs Event Rate',
                labels={'population_size': 'Population Size', 'event_rate': 'Event Rate'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Shapley Values Analysis (if available)
        if 'shap_values' in st.session_state:
            shap_values_dict = st.session_state.shap_values
            
            st.markdown("### 🎯 Shapley Values Analysis")
            
            # Educational section about Shapley values
            with st.expander("📚 What are Shapley Values?", expanded=False):
                st.markdown("""
                **Shapley values** are a concept from cooperative game theory that provide a principled way to attribute the contribution of each feature to a model's prediction.
                
                #### 🎮 Game Theory Foundation
                - Named after Lloyd Shapley (Nobel Prize in Economics, 2012)
                - Originally developed to fairly distribute payoffs among players in cooperative games
                - Applied to machine learning to explain feature contributions
                
                #### 🔍 How They Work
                1. **Marginal Contribution**: For each feature, calculate how much it adds to the prediction
                2. **All Possible Combinations**: Consider every possible subset of features
                3. **Average Contribution**: Average the marginal contributions across all combinations
                4. **Fair Attribution**: Each feature gets credit for its unique contribution
                
                #### 📊 In Our Context
                - **Features** = Inclusion/Exclusion criteria thresholds
                - **Outcomes** = Population size and event rate
                - **Shapley Value** = How much each criterion contributes to the outcome
                - **Higher Values** = Greater impact on the result
                
                #### 🎯 Why Use Shapley Values?
                - **Fair Attribution**: Each criterion gets credit for its unique contribution
                - **Additive**: Sum of all Shapley values equals the total prediction
                - **Model Agnostic**: Works with any machine learning model
                - **Interpretable**: Easy to understand and explain
                """)
            
            with st.expander("🔬 How We Calculate Shapley Values", expanded=False):
                st.markdown("""
                #### 📈 Our Implementation Process
                
                **Step 1: Model Training**
                - Train a Random Forest model to predict outcomes from criteria parameters
                - Use standardized features for consistent scaling
                - Model learns the relationship between criteria thresholds and outcomes
                
                **Step 2: SHAP Explainer Creation**
                - Create a TreeExplainer for the trained Random Forest model
                - Calculate expected values (baseline predictions)
                - Prepare for SHAP value computation
                
                **Step 3: SHAP Value Calculation**
                - For each simulation result, compute SHAP values for all features
                - Each SHAP value represents the feature's contribution to that specific prediction
                - Positive values = feature increases the outcome
                - Negative values = feature decreases the outcome
                
                **Step 4: Aggregation**
                - Average absolute SHAP values across all simulations
                - This gives us the overall importance of each criterion
                - Higher average = more important criterion
                
                #### 🧮 Mathematical Foundation
                For feature i, the Shapley value is:
                
                ```
                φᵢ = Σ_{S⊆N\\{i}} |S|!(|N|-|S|-1)!/|N|! × [f(S∪{i}) - f(S)]
                ```
                
                Where:
                - N = set of all features
                - S = subset of features (excluding feature i)
                - f(S) = model prediction using only features in S
                - f(S∪{i}) = model prediction using features in S plus feature i
                
                #### 📊 Interpretation Guide
                - **High Positive SHAP**: Criterion strongly increases the outcome
                - **High Negative SHAP**: Criterion strongly decreases the outcome
                - **Low SHAP**: Criterion has minimal impact
                - **Variable SHAP**: Criterion's effect depends on other criteria
                """)
            
            # Impact visualization
            st.markdown("**Impact of each criterion on outcomes (higher values = greater impact):**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Impact on Population Size")
                
                population_shap = shap_values_dict['population_size']
                fig = px.bar(
                    x=[col.replace('param_', '').replace('_criteria', '') for col in population_shap['features']],
                    y=population_shap['shap_values'],
                    title='Criterion Impact on Population Size',
                    labels={'x': 'Criteria', 'y': 'SHAP Value (Impact)'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Impact on Event Rate")
                
                event_shap = shap_values_dict['event_rate']
                fig = px.bar(
                    x=[col.replace('param_', '').replace('_criteria', '') for col in event_shap['features']],
                    y=event_shap['shap_values'],
                    title='Criterion Impact on Event Rate',
                    labels={'x': 'Criteria', 'y': 'SHAP Value (Impact)'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Advanced SHAP visualizations
            with st.expander("🔬 Advanced SHAP Visualizations", expanded=False):
                st.markdown("#### 📈 Beeswarm Plot")
                st.markdown("Shows the distribution of SHAP values for each feature across all simulations:")
                
                # Create beeswarm plot for population size
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                # Population size beeswarm
                feature_names = [col.replace('param_', '').replace('_criteria', '') for col in population_shap['features']]
                
                # Get SHAP values
                population_shap_values = population_shap['explainer'].shap_values(simulation_results[population_shap['features']])
                
                # Create beeswarm plot manually since summary_plot has API issues
                for i, feature in enumerate(population_shap['features']):
                    feature_name = feature.replace('param_', '').replace('_criteria', '')
                    shap_values = population_shap_values[:, i]
                    feature_values = simulation_results[feature]
                    
                    # Create scatter plot with color coding
                    scatter = axes[0].scatter(
                        feature_values, 
                        shap_values,
                        c=shap_values, 
                        cmap='RdBu_r',
                        alpha=0.6,
                        s=20
                    )
                    
                    if i == 0:  # Only add colorbar once
                        plt.colorbar(scatter, ax=axes[0], label='SHAP Value')
                
                axes[0].set_xlabel('Feature Values')
                axes[0].set_ylabel('SHAP Values')
                axes[0].set_title('Population Size - SHAP Values Distribution')
                axes[0].grid(True, alpha=0.3)
                
                # Event rate beeswarm
                event_shap_values = event_shap['explainer'].shap_values(simulation_results[event_shap['features']])
                
                for i, feature in enumerate(event_shap['features']):
                    feature_name = feature.replace('param_', '').replace('_criteria', '')
                    shap_values = event_shap_values[:, i]
                    feature_values = simulation_results[feature]
                    
                    # Create scatter plot with color coding
                    scatter = axes[1].scatter(
                        feature_values, 
                        shap_values,
                        c=shap_values, 
                        cmap='RdBu_r',
                        alpha=0.6,
                        s=20
                    )
                    
                    if i == 0:  # Only add colorbar once
                        plt.colorbar(scatter, ax=axes[1], label='SHAP Value')
                
                axes[1].set_xlabel('Feature Values')
                axes[1].set_ylabel('SHAP Values')
                axes[1].set_title('Event Rate - SHAP Values Distribution')
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("#### 🎯 Waterfall Plot Examples")
                st.markdown("Shows how each feature contributes to the prediction for specific cases:")
                
                # Select a few interesting cases for force plots
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**High Population Size Case:**")
                    high_pop_idx = simulation_results['population_size'].idxmax()
                    high_pop_data = simulation_results.loc[high_pop_idx, population_shap['features']]
                    
                    # Create waterfall plot instead of force plot
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    # Get SHAP values for this case
                    high_pop_shap_values = population_shap['explainer'].shap_values(simulation_results[population_shap['features']])[high_pop_idx]
                    
                    # Create waterfall plot
                    feature_names_short = [col.replace('param_', '').replace('_criteria', '') for col in population_shap['features']]
                    y_pos = np.arange(len(feature_names_short))
                    
                    colors = ['red' if x < 0 else 'blue' for x in high_pop_shap_values]
                    ax.barh(y_pos, high_pop_shap_values, color=colors, alpha=0.7)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(feature_names_short)
                    ax.set_xlabel('SHAP Value')
                    ax.set_title(f'High Population Case ({int(simulation_results.loc[high_pop_idx, "population_size"])} patients)')
                    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                
                with col2:
                    st.markdown("**High Event Rate Case:**")
                    high_event_idx = simulation_results['event_rate'].idxmax()
                    high_event_data = simulation_results.loc[high_event_idx, event_shap['features']]
                    
                    # Create waterfall plot instead of force plot
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    # Get SHAP values for this case
                    high_event_shap_values = event_shap['explainer'].shap_values(simulation_results[event_shap['features']])[high_event_idx]
                    
                    # Create waterfall plot
                    feature_names_short = [col.replace('param_', '').replace('_criteria', '') for col in event_shap['features']]
                    y_pos = np.arange(len(feature_names_short))
                    
                    colors = ['red' if x < 0 else 'blue' for x in high_event_shap_values]
                    ax.barh(y_pos, high_event_shap_values, color=colors, alpha=0.7)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(feature_names_short)
                    ax.set_xlabel('SHAP Value')
                    ax.set_title(f'High Event Rate Case ({simulation_results.loc[high_event_idx, "event_rate"]:.3f})')
                    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
            
            # Interactive SHAP exploration
            st.markdown("#### 🔍 Interactive SHAP Exploration")
            st.markdown("Explore SHAP values for specific simulation cases using waterfall plots:")
            
            # Let user select a case
            case_options = {
                "Highest Population": simulation_results['population_size'].idxmax(),
                "Lowest Population": simulation_results['population_size'].idxmin(),
                "Highest Event Rate": simulation_results['event_rate'].idxmax(),
                "Lowest Event Rate": simulation_results['event_rate'].idxmin(),
                "Random Case": simulation_results.sample(1).index[0]
            }
            
            selected_case = st.selectbox(
                "Choose a case to explore:",
                list(case_options.keys()),
                key="shap_case_exploration"
            )
            
            if selected_case:
                case_idx = case_options[selected_case]
                case_data = simulation_results.loc[case_idx]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Case Details:**")
                    st.write(f"Population Size: {int(case_data['population_size'])}")
                    st.write(f"Event Rate: {case_data['event_rate']:.3f}")
                    
                    # Show parameter values for this case
                    st.markdown("**Criteria Values:**")
                    for col in population_shap['features']:
                        param_name = col.replace('param_', '').replace('_criteria', '')
                        param_value = case_data[col]
                        st.write(f"{param_name}: {param_value:.2f}")
                
                with col2:
                    # Create waterfall plots for selected case
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    
                    # Population size waterfall plot
                    pop_shap_values = population_shap['explainer'].shap_values(simulation_results[population_shap['features']])[case_idx]
                    feature_names_short = [col.replace('param_', '').replace('_criteria', '') for col in population_shap['features']]
                    y_pos = np.arange(len(feature_names_short))
                    
                    colors_pop = ['red' if x < 0 else 'blue' for x in pop_shap_values]
                    ax1.barh(y_pos, pop_shap_values, color=colors_pop, alpha=0.7)
                    ax1.set_yticks(y_pos)
                    ax1.set_yticklabels(feature_names_short)
                    ax1.set_xlabel('SHAP Value')
                    ax1.set_title(f'Population Size SHAP Values')
                    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    ax1.grid(True, alpha=0.3)
                    
                    # Event rate waterfall plot
                    event_shap_values = event_shap['explainer'].shap_values(simulation_results[event_shap['features']])[case_idx]
                    colors_event = ['red' if x < 0 else 'blue' for x in event_shap_values]
                    ax2.barh(y_pos, event_shap_values, color=colors_event, alpha=0.7)
                    ax2.set_yticks(y_pos)
                    ax2.set_yticklabels(feature_names_short)
                    ax2.set_xlabel('SHAP Value')
                    ax2.set_title(f'Event Rate SHAP Values')
                    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Summary table
            st.markdown("### 📋 Impact Summary Table")
            
            summary_data = []
            for i, feature in enumerate(population_shap['features']):
                criterion_name = feature.replace('param_', '').replace('_criteria', '')
                summary_data.append({
                    'Criterion': criterion_name,
                    'Population Size Impact': population_shap['shap_values'][i],
                    'Event Rate Impact': event_shap['shap_values'][i],
                    'Total Impact': population_shap['shap_values'][i] + event_shap['shap_values'][i]
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('Total Impact', ascending=False)
            
            st.dataframe(summary_df, use_container_width=True)
            
            # Practical interpretation guide
            with st.expander("💡 How to Interpret These Results", expanded=False):
                st.markdown("""
                #### 🎯 Understanding Your Results
                
                **High Impact Criteria (Top of the list):**
                - These criteria have the strongest influence on your outcomes
                - Small changes to these criteria will have large effects
                - Focus optimization efforts on these criteria first
                
                **Low Impact Criteria (Bottom of the list):**
                - These criteria have minimal influence on outcomes
                - Changes to these criteria won't significantly affect results
                - Consider simplifying or removing these criteria
                
                #### 📊 Population Size vs Event Rate
                
                **High Population Impact, Low Event Rate Impact:**
                - Criterion affects how many patients are eligible
                - Doesn't strongly affect the risk profile of eligible patients
                - Example: Age criteria might exclude many patients but not change event rate much
                
                **Low Population Impact, High Event Rate Impact:**
                - Criterion doesn't exclude many patients
                - But strongly affects the risk profile of remaining patients
                - Example: Diabetes history might exclude few patients but strongly predict events
                
                **High Impact on Both:**
                - Criterion is critical for both enrollment and risk stratification
                - Most important criteria to optimize carefully
                - Example: Creatinine levels might exclude many patients AND predict events
                
                #### 🔧 Practical Recommendations
                
                1. **Start with High-Impact Criteria**: Focus optimization on criteria with highest total impact
                2. **Consider Trade-offs**: Population size vs event rate priorities
                3. **Test Sensitivity**: Small changes to high-impact criteria
                4. **Simplify Protocol**: Consider removing low-impact criteria
                5. **Validate Results**: Test optimized criteria on new data
                """)
    
    # Optimization Section
    st.markdown("---")
    st.markdown("### 🎯 Criteria Optimization")
    
    if 'simulation_results' in st.session_state:
        simulation_results = st.session_state.simulation_results
        st.markdown("Find optimal inclusion/exclusion criteria values for your target outcomes.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Target Outcomes")
            target_population = st.number_input(
                "Target Population Size",
                min_value=1,
                max_value=len(patients),
                value=len(eligible_patients),
                step=10,
                key="target_population_opt"
            )
            
            target_event_rate = st.number_input(
                "Target Event Rate",
                min_value=0.0,
                max_value=1.0,
                value=float(eligible_patients['event_occurred'].mean()) if len(eligible_patients) > 0 else 0.1,
                step=0.01,
                format="%.3f",
                key="target_event_rate_opt"
            )
        
        with col2:
            st.markdown("#### ⚖️ Optimization Weights")
            population_weight = st.slider(
                "Population Size Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="How much to prioritize population size vs event rate",
                key="population_weight_opt"
            )
            
            event_rate_weight = 1.0 - population_weight
            st.metric("Event Rate Weight", f"{event_rate_weight:.1f}")
            
            n_optimizations = st.slider(
                "Number of Optimization Runs",
                min_value=1,
                max_value=20,
                value=10,
                step=1,
                help="More runs = better chance of finding global optimum",
                key="n_optimizations_opt"
            )
        
        if st.button("Find Optimal Criteria", type="primary"):
            with st.spinner("Finding optimal criteria values..."):
                optimal_result = find_optimal_criteria(
                    simulation_results,
                    criteria_config,
                    target_population,
                    target_event_rate,
                    population_weight,
                    event_rate_weight,
                    n_optimizations
                )
                
                if optimal_result:
                    st.session_state.optimal_result = optimal_result
                    st.success("✅ Optimal criteria found!")
                else:
                    st.error("❌ Could not find optimal criteria. Try adjusting targets or weights.")
        
        # Display optimization results
        if 'optimal_result' in st.session_state and 'simulation_results' in st.session_state:
            optimal_result = st.session_state.optimal_result
            
            st.markdown("#### 🎯 Optimal Criteria Values")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Predicted Outcomes:**")
                st.metric("Population Size", f"{optimal_result['predicted_population']:.0f}")
                st.metric("Event Rate", f"{optimal_result['predicted_event_rate']:.3f}")
                st.metric("Optimization Loss", f"{optimal_result['loss']:.4f}")
            
            with col2:
                st.markdown("**Optimal Parameter Values:**")
                for i, feature_col in enumerate(optimal_result['feature_cols']):
                    param_name = feature_col.replace('param_', '').replace('_criteria', '')
                    param_value = optimal_result['optimal_params'][i]
                    
                    # Format based on parameter type
                    if param_value < 0.5:  # Likely binary
                        formatted_value = "0 (Exclude)" if param_value < 0.5 else "1 (Include)"
                    else:
                        formatted_value = f"{param_value:.2f}"
                    
                    st.metric(param_name, formatted_value)
            
            # Optimization plots
            st.markdown("#### 📈 Optimization Analysis")
            opt_fig = create_optimization_plots(
                simulation_results, 
                optimal_result, 
                target_population, 
                target_event_rate
            )
            st.plotly_chart(opt_fig, use_container_width=True)
            
            # Similar populations analysis
            st.markdown("#### 🔍 Explore Similar Populations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                population_tolerance = st.slider(
                    "Population Size Tolerance (%)",
                    min_value=1,
                    max_value=50,
                    value=10,
                    step=1,
                    key="population_tolerance_similar"
                ) / 100
                
                event_rate_tolerance = st.slider(
                    "Event Rate Tolerance",
                    min_value=0.001,
                    max_value=0.1,
                    value=0.01,
                    step=0.001,
                    format="%.3f",
                    key="event_rate_tolerance_similar"
                )
            
            with col2:
                max_results = st.slider(
                    "Maximum Results to Show",
                    min_value=5,
                    max_value=50,
                    value=20,
                    step=5,
                    key="max_results_similar"
                )
            
            if st.button("Find Similar Populations"):
                with st.spinner("Finding similar populations..."):
                    similar_results = find_similar_populations(
                        simulation_results,
                        optimal_result['predicted_population'],
                        optimal_result['predicted_event_rate'],
                        population_tolerance,
                        event_rate_tolerance,
                        max_results
                    )
                    
                    st.session_state.similar_results = similar_results
                    st.success(f"✅ Found {len(similar_results)} similar populations!")
            
            # Display similar populations
            if 'similar_results' in st.session_state:
                similar_results = st.session_state.similar_results
                
                st.markdown("**Similar Population Results:**")
                
                # Create a summary table
                summary_data = []
                for _, row in similar_results.head(10).iterrows():
                    param_values = {}
                    for col in similar_results.columns:
                        if col.startswith('param_'):
                            param_name = col.replace('param_', '').replace('_criteria', '')
                            param_values[param_name] = row[col]
                    
                    summary_data.append({
                        'Population Size': int(row['population_size']),
                        'Event Rate': f"{row['event_rate']:.3f}",
                        'Distance': f"{row['total_distance']:.3f}",
                        **param_values
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Parameter distribution for similar results
                st.markdown("**Parameter Distribution for Similar Outcomes:**")
                
                param_cols = [col for col in similar_results.columns if col.startswith('param_')]
                if param_cols:
                    fig, axes = plt.subplots(1, len(param_cols), figsize=(5*len(param_cols), 4))
                    if len(param_cols) == 1:
                        axes = [axes]
                    
                    for i, param_col in enumerate(param_cols):
                        param_name = param_col.replace('param_', '').replace('_criteria', '')
                        axes[i].hist(similar_results[param_col], bins=20, alpha=0.7, color='lightblue')
                        axes[i].set_title(f'{param_name} Distribution')
                        axes[i].set_xlabel('Parameter Value')
                        axes[i].set_ylabel('Frequency')
                        axes[i].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
    else:
        st.info("💡 **Run the Monte Carlo simulation above first to enable optimization features.**")
    
    # Download section
    if 'simulation_results' in st.session_state:
        simulation_results = st.session_state.simulation_results
        
        st.markdown("---")
        st.markdown("### 💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = simulation_results.to_csv(index=False)
            st.download_button(
                label="Download Simulation Results (CSV)",
                data=csv,
                file_name="clinical_trial_simulation_results.csv",
                mime="text/csv"
            )
        
        with col2:
            # Create summary report
            summary_report = f"""
Clinical Trial Criteria Optimization Report

Simulation Parameters:
- Total Patients: {len(patients)}
- Number of Simulations: {len(simulation_results)}
- Active Criteria: {sum(1 for config in criteria_config.values() if config['active'])}

Current Results:
- Eligible Patients: {len(eligible_patients)}
- Eligibility Rate: {len(eligible_patients) / len(patients) * 100:.1f}%
- Event Rate: {eligible_patients['event_occurred'].mean():.3f}

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            st.download_button(
                label="Download Summary Report (TXT)",
                data=summary_report,
                file_name="clinical_trial_summary_report.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main() 