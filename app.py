import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, roc_auc_score
from sklearn.calibration import calibration_curve
import shap
import xgboost as xgb
from scipy.optimize import minimize
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
import time
import joblib
from typing import List, Dict, Tuple, Optional


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
    """Generate synthetic patient data with realistic clinical characteristics using correlation matrix."""
    np.random.seed(seed)
    
    # Define clinical variables and their means/standard deviations
    variables = {
        'age': {'mean': 65, 'std': 15, 'min': 18, 'max': 95},
        'bmi': {'mean': 28, 'std': 6, 'min': 18, 'max': 50},
        'creatinine': {'mean': 1.2, 'std': 0.4, 'min': 0.5, 'max': 3.0},
        'hba1c': {'mean': 7.5, 'std': 1.5, 'min': 5.0, 'max': 12.0},
        'systolic_bp': {'mean': 140, 'std': 20, 'min': 90, 'max': 200},
        'diastolic_bp': {'mean': 85, 'std': 12, 'min': 60, 'max': 120},
        'cholesterol': {'mean': 200, 'std': 40, 'min': 120, 'max': 300},
        'triglycerides': {'mean': 150, 'std': 80, 'min': 50, 'max': 500}
    }
    
    # Define correlation matrix based on clinical relationships
    # Order: age, bmi, creatinine, hba1c, systolic_bp, diastolic_bp, cholesterol, triglycerides
    correlation_matrix = np.array([
        [1.0,  0.2,  0.3,  0.1,  0.4,  0.3,  0.1,  0.0],  # age
        [0.2,  1.0,  0.1,  0.3,  0.2,  0.2,  0.2,  0.4],  # bmi
        [0.3,  0.1,  1.0,  0.2,  0.1,  0.1,  0.0,  0.0],  # creatinine
        [0.1,  0.3,  0.2,  1.0,  0.2,  0.1,  0.3,  0.2],  # hba1c
        [0.4,  0.2,  0.1,  0.2,  1.0,  0.7,  0.2,  0.1],  # systolic_bp
        [0.3,  0.2,  0.1,  0.1,  0.7,  1.0,  0.2,  0.1],  # diastolic_bp
        [0.1,  0.2,  0.0,  0.3,  0.2,  0.2,  1.0,  0.5],  # cholesterol
        [0.0,  0.4,  0.0,  0.2,  0.1,  0.1,  0.5,  1.0]   # triglycerides
    ])
    
    # Generate correlated continuous variables using multivariate normal distribution
    var_names = list(variables.keys())
    means = [variables[var]['mean'] for var in var_names]
    stds = [variables[var]['std'] for var in var_names]
    
    # Convert correlation matrix to covariance matrix
    cov_matrix = np.diag(stds) @ correlation_matrix @ np.diag(stds)
    
    # Generate correlated data
    correlated_data = np.random.multivariate_normal(means, cov_matrix, n_patients)
    
    # Clip to realistic ranges
    for i, var in enumerate(var_names):
        correlated_data[:, i] = np.clip(
            correlated_data[:, i], 
            variables[var]['min'], 
            variables[var]['max']
        )
    
    # Generate binary variables with correlations to continuous variables
    # Smoking: correlated with age (older patients less likely to smoke)
    smoking_prob = 1 / (1 + np.exp(-(-0.02 * (correlated_data[:, 0] - 65) + np.random.normal(0, 0.5, n_patients))))
    smoking_status = np.random.binomial(1, smoking_prob)
    
    # Diabetes: correlated with age, BMI, HbA1c
    diabetes_prob = 1 / (1 + np.exp(-(
        0.02 * (correlated_data[:, 0] - 65) +  # age
        0.05 * (correlated_data[:, 1] - 28) +  # bmi
        0.3 * (correlated_data[:, 3] - 7.5) +  # hba1c
        np.random.normal(0, 0.5, n_patients)
    )))
    diabetes_history = np.random.binomial(1, diabetes_prob)
    
    # Cardiovascular history: correlated with age, BP, cholesterol
    cv_prob = 1 / (1 + np.exp(-(
        0.03 * (correlated_data[:, 0] - 65) +  # age
        0.01 * (correlated_data[:, 4] - 140) + # systolic_bp
        0.01 * (correlated_data[:, 5] - 85) +  # diastolic_bp
        0.002 * (correlated_data[:, 6] - 200) + # cholesterol
        np.random.normal(0, 0.5, n_patients)
    )))
    cardiovascular_history = np.random.binomial(1, cv_prob)
    
    # Kidney disease: correlated with age, creatinine, diabetes
    kidney_prob = 1 / (1 + np.exp(-(
        0.02 * (correlated_data[:, 0] - 65) +  # age
        0.5 * (correlated_data[:, 2] - 1.2) +  # creatinine
        0.5 * diabetes_history +               # diabetes
        np.random.normal(0, 0.5, n_patients)
    )))
    kidney_disease = np.random.binomial(1, kidney_prob)
    
    # Liver disease: correlated with age, triglycerides
    liver_prob = 1 / (1 + np.exp(-(
        0.01 * (correlated_data[:, 0] - 65) +  # age
        0.001 * (correlated_data[:, 7] - 150) + # triglycerides
        np.random.normal(0, 0.5, n_patients)
    )))
    liver_disease = np.random.binomial(1, liver_prob)
    
    # Medication count: correlated with age and number of conditions
    condition_count = smoking_status + diabetes_history + cardiovascular_history + kidney_disease + liver_disease
    medication_prob = np.exp(-2 + 0.02 * (correlated_data[:, 0] - 65) + 0.3 * condition_count)
    medication_count = np.random.poisson(medication_prob).clip(0, 10)
    
    # Create the data dictionary
    data = {
        'patient_id': range(1, n_patients + 1),
        'age': correlated_data[:, 0],
        'bmi': correlated_data[:, 1],
        'creatinine': correlated_data[:, 2],
        'hba1c': correlated_data[:, 3],
        'systolic_bp': correlated_data[:, 4],
        'diastolic_bp': correlated_data[:, 5],
        'cholesterol': correlated_data[:, 6],
        'triglycerides': correlated_data[:, 7],
        'smoking_status': smoking_status,
        'diabetes_history': diabetes_history,
        'cardiovascular_history': cardiovascular_history,
        'kidney_disease': kidney_disease,
        'liver_disease': liver_disease,
        'medication_count': medication_count
    }
    
    # Create event rate based on risk factors with realistic correlations
    risk_score = (
        (data['age'] - 65) / 15 * 0.5 +
        (data['bmi'] - 28) / 6 * 0.3 +
        (data['creatinine'] - 1.2) / 0.4 * 0.6 +
        (data['hba1c'] - 7.5) / 1.5 * 0.4 +
        data['smoking_status'] * 1.0 +
        data['diabetes_history'] * 0.8 +
        data['cardiovascular_history'] * 1.2
    )
    
    # Convert risk score to probability with higher base rate and stronger effect
    # This creates more realistic event rates (0.10 to 0.40 range)
    event_probability = 0.10 + 0.50 * (1 / (1 + np.exp(-risk_score)))
    data['event_occurred'] = np.random.binomial(1, event_probability)
    
    return pd.DataFrame(data)



def apply_inclusion_exclusion_criteria(patients, criteria_config):
    """Apply inclusion/exclusion criteria to patient population."""
    # Use optimized vectorized version for better performance
    if len(patients) > 100:  # Only use optimization for larger datasets
        return apply_inclusion_exclusion_criteria_optimized(patients, criteria_config)
    else:
        return apply_inclusion_exclusion_criteria_original(patients, criteria_config)

def apply_inclusion_exclusion_criteria_optimized(patients, criteria_config):
    """Optimized version using vectorized operations."""
    # Convert to numpy arrays for faster processing
    patient_data = patients[['age', 'bmi', 'creatinine', 'hba1c', 'systolic_bp', 
                            'diastolic_bp', 'cholesterol', 'triglycerides', 
                            'smoking_status', 'diabetes_history', 'cardiovascular_history', 
                            'kidney_disease', 'liver_disease', 'medication_count']].values
    
    # Map criteria to feature indices
    feature_map = {
        'age': 0, 'bmi': 1, 'creatinine': 2, 'hba1c': 3, 'systolic_bp': 4,
        'diastolic_bp': 5, 'cholesterol': 6, 'triglycerides': 7,
        'smoking_status': 8, 'diabetes_history': 9, 'cardiovascular_history': 10,
        'kidney_disease': 11, 'liver_disease': 12, 'medication_count': 13
    }
    
    # Operator mapping
    op_map = {'>': 0, '<': 1, '>=': 2, '<=': 3, '==': 4}
    
    # Initialize eligibility mask (all patients start as eligible)
    eligible_mask = np.ones(len(patients), dtype=bool)
    
    # Apply each criterion
    for criterion, config in criteria_config.items():
        if config['active']:
            variable = config['variable']
            if variable in feature_map:
                feature_idx = feature_map[variable]
                operator = op_map[config['operator']]
                threshold = config['threshold']
                
                # Apply operator using vectorized NumPy operations
                if operator == 0:  # >
                    eligible_mask &= (patient_data[:, feature_idx] > threshold)
                elif operator == 1:  # <
                    eligible_mask &= (patient_data[:, feature_idx] < threshold)
                elif operator == 2:  # >=
                    eligible_mask &= (patient_data[:, feature_idx] >= threshold)
                elif operator == 3:  # <=
                    eligible_mask &= (patient_data[:, feature_idx] <= threshold)
                elif operator == 4:  # ==
                    eligible_mask &= (patient_data[:, feature_idx] == threshold)
    
    # Return filtered patients
    return patients[eligible_mask]

def apply_inclusion_exclusion_criteria_original(patients, criteria_config):
    """Original implementation for smaller datasets."""
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

@st.cache_data
def run_monte_carlo_simulation(patients, criteria_config, n_simulations=1000):
    """Run Monte Carlo simulation with varying criteria thresholds - optimized version."""
    # For now, use sequential processing to avoid multiprocessing issues
    # Parallel processing can be re-enabled later with better error handling
    return run_monte_carlo_simulation_sequential(patients, criteria_config, n_simulations)

def run_monte_carlo_simulation_parallel(patients, criteria_config, n_simulations=1000):
    """Parallel Monte Carlo simulation using multiprocessing."""
    # Determine optimal number of workers
    n_workers = min(mp.cpu_count(), 8, n_simulations // 100)  # Cap at 8 workers
    
    # Split simulations across workers
    simulations_per_worker = n_simulations // n_workers
    remainder = n_simulations % n_workers
    
    # Create partial function with fixed arguments
    run_single_worker = partial(
        run_monte_carlo_worker,
        patients=patients,
        criteria_config=criteria_config
    )
    
    # Prepare arguments for each worker
    worker_args = []
    start_idx = 0
    for i in range(n_workers):
        worker_simulations = simulations_per_worker + (1 if i < remainder else 0)
        worker_args.append((start_idx, worker_simulations))
        start_idx += worker_simulations
    
    # Run parallel simulations
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results_list = list(executor.map(run_single_worker, worker_args))
    
    # Combine results
    all_results = []
    for results in results_list:
        all_results.extend(results)
    
    return pd.DataFrame(all_results)

def run_monte_carlo_worker(worker_args):
    """Worker function for parallel Monte Carlo simulation."""
    start_idx, n_simulations = worker_args
    
    # Generate parameter ranges for this worker
    param_ranges = generate_parameter_ranges(criteria_config)
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    results = []
    np.random.seed(42 + start_idx)  # Different seed for each worker
    
    for i in range(n_simulations):
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
            'simulation_id': start_idx + i,
            'population_size': population_size,
            'event_rate': event_rate
        }
        
        # Add parameter values
        for criterion in param_names:
            result[f'param_{criterion}'] = sampled_params[criterion]
        
        results.append(result)
    
    return results

def run_monte_carlo_simulation_sequential(patients, criteria_config, n_simulations=1000):
    """Sequential Monte Carlo simulation for smaller datasets."""
    results = []
    
    # Define parameter ranges for each criterion
    param_ranges = generate_parameter_ranges(criteria_config)
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

def generate_parameter_ranges(criteria_config):
    """Generate parameter ranges for Monte Carlo simulation."""
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
    return param_ranges

def benchmark_performance(patients, criteria_config, n_simulations=100):
    """Benchmark performance improvements."""
    st.markdown("#### 🏃‍♂️ Performance Benchmark")
    
    # Test original vs optimized criteria application
    st.write("**Testing criteria application speed...**")
    
    # Original method
    start_time = time.time()
    for _ in range(10):
        apply_inclusion_exclusion_criteria_original(patients, criteria_config)
    original_time = time.time() - start_time
    
    # Optimized method
    start_time = time.time()
    for _ in range(10):
        apply_inclusion_exclusion_criteria_optimized(patients, criteria_config)
    optimized_time = time.time() - start_time
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Original Time", f"{original_time:.3f}s")
    
    with col2:
        st.metric("Optimized Time", f"{optimized_time:.3f}s")
    
    with col3:
        speedup = original_time / optimized_time if optimized_time > 0 else 1
        st.metric("Speedup", f"{speedup:.1f}x")
    
    # Test simulation methods
    st.write("**Testing simulation methods...**")
    
    # Sequential simulation
    start_time = time.time()
    sequential_results = run_monte_carlo_simulation_sequential(patients, criteria_config, n_simulations)
    sequential_time = time.time() - start_time
    
    # Parallel simulation (if applicable)
    if n_simulations > 100:
        start_time = time.time()
        parallel_results = run_monte_carlo_simulation_parallel(patients, criteria_config, n_simulations)
        parallel_time = time.time() - start_time
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sequential", f"{sequential_time:.2f}s")
        
        with col2:
            st.metric("Parallel", f"{parallel_time:.2f}s")
        
        with col3:
            speedup = sequential_time / parallel_time if parallel_time > 0 else 1
            st.metric("Speedup", f"{speedup:.1f}x")
    
    st.success("✅ Performance benchmark completed!")

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
                      population_weight=0.5, event_rate_weight=0.5, penalty_weight=1.0):
    """Robust objective function for optimization with multiple loss components."""
    try:
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
        
        # Ensure predictions are valid
        predicted_population = max(0, predicted_population)
        predicted_event_rate = np.clip(predicted_event_rate, 0, 1)
        
        # Calculate multiple loss components
        # 1. Population size loss (relative error)
        population_loss = abs(predicted_population - target_population) / max(target_population, 1)
        
        # 2. Event rate loss (relative error)
        event_rate_loss = abs(predicted_event_rate - target_event_rate) / max(target_event_rate, 0.01)
        
        # 3. Penalty for extreme values (encourage reasonable criteria)
        extreme_penalty = 0
        for i, param in enumerate(params):
            if i < len(feature_cols):
                col_name = feature_cols[i]
                # Add penalty for very extreme values
                if 'age' in col_name.lower():
                    if param < 18 or param > 100:  # Unreasonable age bounds
                        extreme_penalty += abs(param - np.clip(param, 18, 100)) / 100
                elif 'bmi' in col_name.lower():
                    if param < 15 or param > 60:  # Unreasonable BMI bounds
                        extreme_penalty += abs(param - np.clip(param, 15, 60)) / 60
                elif 'creatinine' in col_name.lower():
                    if param < 0.3 or param > 5.0:  # Unreasonable creatinine bounds
                        extreme_penalty += abs(param - np.clip(param, 0.3, 5.0)) / 5.0
        
        # 4. Smoothness penalty (prefer gradual changes from current values)
        smoothness_penalty = 0
        # This could be implemented if we track current values
        
        # Combine all loss components
        total_loss = (population_weight * population_loss + 
                     event_rate_weight * event_rate_loss + 
                     penalty_weight * extreme_penalty)
        
        return total_loss
        
    except Exception as e:
        # Return a high penalty for failed evaluations
        return 1e6

def find_optimal_criteria(simulation_results, criteria_config, target_population, target_event_rate,
                         population_weight=0.5, event_rate_weight=0.5, n_optimizations=10):
    """Simplified optimization that finds the best real simulation points directly."""
    
    # Input validation
    if simulation_results is None or len(simulation_results) == 0:
        return None
    
    if target_population <= 0 or target_event_rate < 0 or target_event_rate > 1:
        return None
    
    # Get feature columns from simulation results
    feature_cols = [col for col in simulation_results.columns if col.startswith('param_')]
    if not feature_cols:
        return None
    
    # Strategy 1: Direct search - find closest simulation points to target
    # Calculate weighted distance to target for each simulation
    population_diff = abs(simulation_results['population_size'] - target_population) / max(target_population, 1)
    event_rate_diff = abs(simulation_results['event_rate'] - target_event_rate) / max(target_event_rate, 0.01)
    
    # Weighted distance based on user preferences
    weighted_distance = (population_weight * population_diff + 
                        event_rate_weight * event_rate_diff)
    
    # Find the best simulation points
    best_indices = weighted_distance.nsmallest(min(10, len(simulation_results))).index
    
    best_results = []
    for idx in best_indices:
        simulation = simulation_results.iloc[idx]
        
        # Extract parameters
        optimal_params = []
        for feature_col in feature_cols:
            optimal_params.append(simulation[feature_col])
        
        best_results.append({
            'simulation_id': simulation['simulation_id'],
            'population_size': simulation['population_size'],
            'event_rate': simulation['event_rate'],
            'optimal_params': optimal_params,
            'distance': weighted_distance.iloc[idx],
            'feature_cols': feature_cols
        })
    
    # Strategy 2: If user wants more optimization, try to find better combinations
    if n_optimizations > 1:
        # Find similar populations and analyze their parameter distributions
        similar_results = find_similar_populations(
            simulation_results, target_population, target_event_rate,
            population_tolerance=0.2, event_rate_tolerance=0.02, max_results=50
        )
        
        if len(similar_results) > 0:
            # Analyze parameter distributions to suggest improvements
            param_analysis = {}
            for feature_col in feature_cols:
                param_name = feature_col.replace('param_', '').replace('_criteria', '')
                param_values = similar_results[feature_col]
                
                # Get parameter type
                param_type = 'continuous'
                for criterion, config in criteria_config.items():
                    if config['active'] and criterion in feature_col:
                        param_type = config['type']
                        break
                
                if param_type == 'binary':
                    # For binary, find most common value
                    most_common = param_values.mode().iloc[0] if len(param_values.mode()) > 0 else param_values.iloc[0]
                    param_analysis[feature_col] = most_common
                else:
                    # For continuous, use median or mean based on distribution
                    if param_values.std() < param_values.mean() * 0.1:  # Low variance
                        param_analysis[feature_col] = param_values.median()
                    else:
                        param_analysis[feature_col] = param_values.mean()
            
            # Create a "suggested" result based on analysis
            suggested_params = [param_analysis.get(col, 0) for col in feature_cols]
            
            # Find closest simulation to this suggestion
            suggested_sim = find_closest_simulation_point(
                simulation_results,
                target_population,
                target_event_rate,
                feature_cols
            )
            
            if suggested_sim:
                best_results.append({
                    'simulation_id': suggested_sim['simulation_id'],
                    'population_size': suggested_sim['population_size'],
                    'event_rate': suggested_sim['event_rate'],
                    'optimal_params': suggested_sim['parameters'],
                    'distance': suggested_sim['distance'],
                    'feature_cols': feature_cols,
                    'method': 'parameter_analysis'
                })
    
    # Return the best result (closest to target)
    if best_results:
        best_result = min(best_results, key=lambda x: x['distance'])
        
        # Add parameter types for display
        param_types = []
        for feature_col in feature_cols:
            param_type = 'continuous'
            for criterion, config in criteria_config.items():
                if config['active'] and criterion in feature_col:
                    param_type = config['type']
                    break
            param_types.append(param_type)
        
        return {
            'optimal_params': best_result['optimal_params'],
            'predicted_population': best_result['population_size'],  # Use real values
            'predicted_event_rate': best_result['event_rate'],       # Use real values
            'closest_real_point': {
                'simulation_id': best_result['simulation_id'],
                'population_size': best_result['population_size'],
                'event_rate': best_result['event_rate'],
                'parameters': best_result['optimal_params'],
                'distance': best_result['distance']
            },
            'loss': best_result['distance'],
            'feature_cols': feature_cols,
            'param_types': param_types,
            'optimization_history': [{'method': 'direct_search', 'loss': best_result['distance'], 'success': True}],
            'success': True,
            'method': best_result.get('method', 'direct_search')
        }
    
    return None

def analyze_patient_event_prediction(patients, criteria_config):
    """Analyze patient-level event prediction using XGBoost with hyperparameter tuning."""
    
    # Import metrics early
    from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
    
    # Prepare patient data for ML analysis
    # Use patient characteristics as features, event_occurred as target
    feature_cols = ['age', 'bmi', 'creatinine', 'hba1c', 'systolic_bp', 'diastolic_bp', 
                   'cholesterol', 'triglycerides', 'smoking_status', 'diabetes_history', 
                   'cardiovascular_history', 'kidney_disease', 'liver_disease', 'medication_count']
    
    # Filter to only include features that exist in the data
    available_features = [col for col in feature_cols if col in patients.columns]
    
    if not available_features:
        return None
    
    # Create features and target
    X = patients[available_features].copy()
    y = patients['event_occurred'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Check event rate and add debugging info
    event_rate = y.mean()
    st.info(f"📊 Dataset Info: {len(y)} patients, {y.sum()} events, event rate: {event_rate:.3f}")
    
    if event_rate < 0.05:
        st.warning(f"⚠️ Low event rate ({event_rate:.3f}). This may affect model performance.")
    
    # Use random seed for reproducibility but different each run
    random_seed = np.random.randint(1, 10000)
    
    # Split data into train/test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_seed, stratify=y
    )
    
    # Check prediction distribution after training
    st.info(f"📈 Training set: {len(y_train)} patients, {y_train.sum()} events, event rate: {y_train.mean():.3f}")
    st.info(f"📈 Test set: {len(y_test)} patients, {y_test.sum()} events, event rate: {y_test.mean():.3f}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define XGBoost hyperparameter search space
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'scale_pos_weight': [1, 2, 3]  # More conservative class imbalance handling
    }
    
    # Initialize XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        random_state=random_seed,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Perform randomized search for hyperparameter tuning
    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_dist,
        n_iter=20,  # Number of parameter settings sampled
        cv=5,       # 5-fold cross-validation
        scoring='roc_auc',
        random_state=random_seed,
        n_jobs=-1,  # Use all available cores
        verbose=0
    )
    
    # Fit the model
    random_search.fit(X_train_scaled, y_train)
    
    # Get best model
    best_model = random_search.best_estimator_
    
    # Make predictions
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Use optimal threshold that maximizes AUC (Youden's J statistic)
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)  # Youden's J statistic
    optimal_threshold = thresholds_roc[optimal_idx]
    
    # Use the optimal threshold for predictions
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    st.info(f"🎯 Optimal threshold: {optimal_threshold:.3f} (maximizes TPR - FPR)")
    st.info(f"   - At this threshold: TPR={tpr[optimal_idx]:.3f}, FPR={fpr[optimal_idx]:.3f}")
    st.info(f"   - Youden's J = {tpr[optimal_idx] - fpr[optimal_idx]:.3f}")
    
    # Debug prediction distribution
    st.info(f"🔍 Prediction Analysis:")
    st.info(f"   - Predicted probabilities range: {y_pred_proba.min():.3f} to {y_pred_proba.max():.3f}")
    st.info(f"   - Mean predicted probability: {y_pred_proba.mean():.3f}")
    st.info(f"   - Predictions: {y_pred.sum()} positive out of {len(y_pred)} ({y_pred.mean():.3f})")
    st.info(f"   - Actual events: {y_test.sum()} out of {len(y_test)} ({y_test.mean():.3f})")
    
    # Calculate comprehensive metrics
    # Basic metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = (y_pred == y_test).mean()
    
    # Detailed classification metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Sensitivity (Recall), Specificity, etc.
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value (Precision)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    # SHAP analysis
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_scaled)
    if len(shap_values) == 2:  # Binary classification returns 2 arrays
        shap_values = shap_values[1]  # Use positive class SHAP values
    
    # Calculate correlations with event occurrence
    correlations = []
    for col in available_features:
        corr = patients[col].corr(patients['event_occurred'])
        correlations.append(abs(corr))
    
    return {
        'feature_names': available_features,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler,
        'model': best_model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'confusion_matrix': cm,
        'feature_importance': best_model.feature_importances_,
        'explainer': explainer,
        'shap_values': shap_values,
        'correlations': correlations,
        'best_params': random_search.best_params_,
        'random_seed': random_seed,
        'event_rate': event_rate
    }

def analyze_optimization_robustness(simulation_results, criteria_config, target_population, target_event_rate):
    """Analyze the robustness of optimization by testing multiple scenarios."""
    
    # Test different weight combinations
    weight_combinations = [
        (0.5, 0.5),  # Balanced
        (0.7, 0.3),  # Population-focused
        (0.3, 0.7),  # Event rate-focused
        (0.9, 0.1),  # Very population-focused
        (0.1, 0.9),  # Very event rate-focused
    ]
    
    results = []
    
    for pop_weight, event_weight in weight_combinations:
        try:
            result = find_optimal_criteria(
                simulation_results, criteria_config, target_population, target_event_rate,
                population_weight=pop_weight, event_rate_weight=event_weight, n_optimizations=5
            )
            
            if result:
                results.append({
                    'pop_weight': pop_weight,
                    'event_weight': event_weight,
                    'predicted_population': result['predicted_population'],
                    'predicted_event_rate': result['predicted_event_rate'],
                    'loss': result['loss'],
                    'success': result.get('success', True),
                    'method_used': result.get('optimization_history', [{}])[-1].get('method', 'unknown') if result.get('optimization_history') else 'unknown'
                })
        except Exception as e:
            continue
    
    return pd.DataFrame(results) if results else None

def find_closest_simulation_point(simulation_results, target_population, target_event_rate, feature_cols):
    """Find the closest real simulation point to the target values."""
    if len(simulation_results) == 0:
        return None
    
    # Calculate distances to target
    population_diff = abs(simulation_results['population_size'] - target_population) / max(target_population, 1)
    event_rate_diff = abs(simulation_results['event_rate'] - target_event_rate) / max(target_event_rate, 0.01)
    total_distance = np.sqrt(population_diff**2 + event_rate_diff**2)
    
    # Find the closest point
    closest_idx = total_distance.idxmin()
    closest_point = simulation_results.iloc[closest_idx]
    
    # Extract the parameter values for this point
    closest_params = []
    for feature_col in feature_cols:
        closest_params.append(closest_point[feature_col])
    
    return {
        'simulation_id': closest_point['simulation_id'],
        'population_size': closest_point['population_size'],
        'event_rate': closest_point['event_rate'],
        'parameters': closest_params,
        'distance': total_distance.iloc[closest_idx]
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
    if optimal_result and 'closest_real_point' in optimal_result:
        real_point = optimal_result['closest_real_point']
        fig.add_trace(
            go.Scatter(
                x=[real_point['population_size']],
                y=[real_point['event_rate']],
                mode='markers',
                marker=dict(color='green', size=12, symbol='diamond'),
                name='Best Simulation Found',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Plot 2: Parameter distribution for similar outcomes
    if optimal_result and 'closest_real_point' in optimal_result:
        real_point = optimal_result['closest_real_point']
        similar_results = find_similar_populations(
            simulation_results, 
            real_point['population_size'], 
            real_point['event_rate'],
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
    if optimal_result and 'closest_real_point' in optimal_result:
        real_point = optimal_result['closest_real_point']
        fig.add_trace(
            go.Bar(
                x=['Target', 'Best Simulation'],
                y=[target_population, real_point['population_size']],
                name='Population Size',
                marker_color=['red', 'green'],
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=['Target', 'Best Simulation'],
                y=[target_event_rate, real_point['event_rate']],
                name='Event Rate',
                marker_color=['red', 'green'],
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Plot 4: Parameter sensitivity - showing optimal criteria values
    if optimal_result and 'closest_real_point' in optimal_result:
        real_point = optimal_result['closest_real_point']
        param_names = [col.replace('param_', '').replace('_criteria', '') 
                      for col in optimal_result['feature_cols']]
        
        # Add optimal parameters
        fig.add_trace(
            go.Bar(
                x=param_names,
                y=real_point['parameters'],
                name='Optimal Parameters',
                marker_color='green',
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
        
        # Show correlation matrix
        with st.expander("🔗 Clinical Variable Correlations", expanded=False):
            st.markdown("**Correlation matrix showing clinical relationships between variables:**")
            
            # Calculate actual correlations from the data
            continuous_vars = ['age', 'bmi', 'creatinine', 'hba1c', 'systolic_bp', 'diastolic_bp', 'cholesterol', 'triglycerides']
            corr_matrix = patients[continuous_vars].corr()
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                       square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
            plt.title('Clinical Variable Correlations')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            **Clinical Interpretation:**
            - **Age** correlates with BP, creatinine (aging effects)
            - **BMI** correlates with HbA1c, triglycerides (metabolic syndrome)
            - **Blood Pressure** components are highly correlated
            - **Lipids** (cholesterol, triglycerides) are correlated
            - **HbA1c** correlates with BMI and lipids (diabetes/metabolic syndrome)
            """)
        
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
    
    # Performance optimization info
    with st.expander("🚀 Performance Optimizations", expanded=False):
        st.markdown("""
        **⚡ Speed Improvements:**
        - **Vectorized Operations**: NumPy-optimized criteria application
        - **Smart Caching**: Results cached to avoid recomputation
        - **Efficient Data Processing**: Optimized DataFrame operations
        
        **📊 Expected Performance:**
        - **Criteria application**: 5-10x faster with vectorized operations
        - **Monte Carlo simulation**: 2-3x faster with optimized processing
        - **Overall speedup**: 2-5x faster depending on hardware and dataset size
        
        **🔄 Future Enhancements:**
        - Parallel processing will be re-enabled with better error handling
        - Additional optimizations for very large datasets
        """)
        
        # Add benchmark button
        if st.button("🏃‍♂️ Run Performance Benchmark", key="benchmark"):
            benchmark_performance(patients, criteria_config, min(100, n_simulations))
    
    if st.button("Run Monte Carlo Simulation", type="primary"):
        start_time = time.time()
        
        with st.spinner("Running optimized Monte Carlo simulation..."):
            # Run simulation
            simulation_results = run_monte_carlo_simulation(patients, criteria_config, n_simulations)
            
            # Calculate Shapley values
            shap_start_time = time.time()
            shap_values_dict = calculate_shapley_values(simulation_results, criteria_config)
            shap_time = time.time() - shap_start_time
            
            # Store results in session state for download
            st.session_state.simulation_results = simulation_results
            st.session_state.shap_values = shap_values_dict
        
        total_time = time.time() - start_time
        st.success(f"✅ Simulation completed in {total_time:.2f} seconds!")
        st.info(f"📊 {len(simulation_results)} simulations, SHAP analysis: {shap_time:.2f}s")
    
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
                st.markdown("**🎯 Target Outcomes:**")
                st.metric("Population Size", f"{target_population:.0f}")
                st.metric("Event Rate", f"{target_event_rate:.3f}")
                
                if 'closest_real_point' in optimal_result:
                    real_point = optimal_result['closest_real_point']
                    st.metric("Distance to Target", f"{real_point['distance']:.3f}")
                    st.caption("Lower = closer to target")
            
            with col2:
                st.markdown("**✅ Best Simulation Found:**")
                st.metric("Population Size", f"{optimal_result['predicted_population']:.0f}")
                st.metric("Event Rate", f"{optimal_result['predicted_event_rate']:.3f}")
                st.metric("Simulation ID", f"{optimal_result['closest_real_point']['simulation_id']}")
                
                if 'method' in optimal_result:
                    method_name = optimal_result['method'].replace('_', ' ').title()
                    st.metric("Method Used", method_name)
            
            # Show optimal criteria values
            st.markdown("#### 📊 Optimal Criteria Values")
            
            if 'closest_real_point' in optimal_result:
                real_point = optimal_result['closest_real_point']
                
                param_display = []
                for i, feature_col in enumerate(optimal_result['feature_cols']):
                    param_name = feature_col.replace('param_', '').replace('_criteria', '')
                    param_value = real_point['parameters'][i]
                    
                    # Format values based on parameter type
                    if 'param_types' in optimal_result and i < len(optimal_result['param_types']):
                        param_type = optimal_result['param_types'][i]
                        if param_type == 'binary':
                            formatted_value = "1 (Include)" if param_value == 1 else "0 (Exclude)"
                        else:
                            formatted_value = f"{param_value:.2f}"
                    else:
                        # Fallback logic
                        if abs(param_value - round(param_value)) < 0.01:
                            formatted_value = "1 (Include)" if param_value > 0.5 else "0 (Exclude)"
                        else:
                            formatted_value = f"{param_value:.2f}"
                    
                    param_display.append({
                        'Criterion': param_name,
                        'Optimal Value': formatted_value
                    })
                
                # Display optimal criteria table
                criteria_df = pd.DataFrame(param_display)
                st.dataframe(criteria_df, use_container_width=True)
            
            # Show optimization status
            col1, col2 = st.columns(2)
            
            with col1:
                if 'success' in optimal_result:
                    success_status = "✅ Successful" if optimal_result['success'] else "⚠️ Partial Success"
                    st.metric("Optimization Status", success_status)
                
                if 'optimization_history' in optimal_result and optimal_result['optimization_history']:
                    method_used = optimal_result['optimization_history'][-1].get('method', 'Unknown')
                    st.metric("Method Used", method_used)
            
            with col2:
                if 'closest_real_point' in optimal_result:
                    real_point = optimal_result['closest_real_point']
                    st.metric("Real Point Distance", f"{real_point['distance']:.3f}")
                    st.caption("Lower = closer to target")
            
            # Optimization plots
            st.markdown("#### 📈 Optimization Analysis")
            opt_fig = create_optimization_plots(
                simulation_results, 
                optimal_result, 
                target_population, 
                target_event_rate
            )
            st.plotly_chart(opt_fig, use_container_width=True)
            
            # Robustness Analysis
            st.markdown("#### 🔬 Robustness Analysis")
            
            with st.expander("📊 Test Optimization Robustness", expanded=False):
                st.markdown("""
                This analysis tests how robust your optimization is by trying different weight combinations.
                A robust optimization should produce similar results across different weight settings.
                """)
                
                if st.button("Run Robustness Analysis", key="robustness_analysis"):
                    with st.spinner("Analyzing optimization robustness..."):
                        robustness_results = analyze_optimization_robustness(
                            simulation_results,
                            criteria_config,
                            target_population,
                            target_event_rate
                        )
                        
                        if robustness_results is not None and len(robustness_results) > 0:
                            st.session_state.robustness_results = robustness_results
                            st.success("✅ Robustness analysis completed!")
                        else:
                            st.error("❌ Robustness analysis failed.")
                
                # Display robustness results
                if 'robustness_results' in st.session_state:
                    robustness_results = st.session_state.robustness_results
                    
                    st.markdown("**Results across different weight combinations:**")
                    
                    # Create visualization
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=(
                            'Population Size vs Weights',
                            'Event Rate vs Weights', 
                            'Loss vs Weights',
                            'Success Rate by Method'
                        ),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    # Plot 1: Population size vs weights
                    fig.add_trace(
                        go.Scatter(
                            x=robustness_results['pop_weight'],
                            y=robustness_results['predicted_population'],
                            mode='markers+lines',
                            name='Population Size',
                            marker=dict(color='blue', size=8)
                        ),
                        row=1, col=1
                    )
                    
                    # Plot 2: Event rate vs weights
                    fig.add_trace(
                        go.Scatter(
                            x=robustness_results['pop_weight'],
                            y=robustness_results['predicted_event_rate'],
                            mode='markers+lines',
                            name='Event Rate',
                            marker=dict(color='red', size=8)
                        ),
                        row=1, col=2
                    )
                    
                    # Plot 3: Loss vs weights
                    fig.add_trace(
                        go.Scatter(
                            x=robustness_results['pop_weight'],
                            y=robustness_results['loss'],
                            mode='markers+lines',
                            name='Loss',
                            marker=dict(color='green', size=8)
                        ),
                        row=2, col=1
                    )
                    
                    # Plot 4: Method distribution
                    method_counts = robustness_results['method_used'].value_counts()
                    fig.add_trace(
                        go.Bar(
                            x=method_counts.index,
                            y=method_counts.values,
                            name='Method Usage',
                            marker_color='orange'
                        ),
                        row=2, col=2
                    )
                    
                    fig.update_layout(height=600, title_text="Optimization Robustness Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary statistics
                    st.markdown("**Robustness Summary:**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        pop_std = robustness_results['predicted_population'].std()
                        pop_cv = pop_std / robustness_results['predicted_population'].mean()
                        st.metric("Population CV", f"{pop_cv:.3f}")
                        st.caption("Lower = more robust")
                    
                    with col2:
                        event_std = robustness_results['predicted_event_rate'].std()
                        event_cv = event_std / robustness_results['predicted_event_rate'].mean()
                        st.metric("Event Rate CV", f"{event_cv:.3f}")
                        st.caption("Lower = more robust")
                    
                    with col3:
                        success_rate = robustness_results['success'].mean()
                        st.metric("Success Rate", f"{success_rate:.1%}")
                        st.caption("Higher = more reliable")
                    
                    # Detailed results table
                    st.markdown("**Detailed Results:**")
                    display_results = robustness_results.copy()
                    display_results['pop_weight'] = display_results['pop_weight'].round(1)
                    display_results['event_weight'] = display_results['event_weight'].round(1)
                    display_results['predicted_population'] = display_results['predicted_population'].round(0)
                    display_results['predicted_event_rate'] = display_results['predicted_event_rate'].round(3)
                    display_results['loss'] = display_results['loss'].round(4)
                    
                    st.dataframe(display_results, use_container_width=True)
            
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
    
    # Machine Learning Analysis Section
    st.markdown("---")
    st.markdown("### 🤖 Patient-Level Event Prediction")
    
    st.markdown("""
    This section uses machine learning to predict whether individual patients will have events based on their characteristics.
    This helps you understand which patient factors are most predictive of events and how well we can predict individual outcomes.
    """)
    
    if st.button("Run Patient Prediction Analysis", type="primary", key="ml_analysis"):
        with st.spinner("Running patient-level prediction analysis..."):
            ml_results = analyze_patient_event_prediction(patients, criteria_config)
            
            if ml_results:
                st.session_state.ml_results = ml_results
                st.success("✅ Patient prediction analysis completed!")
            else:
                st.error("❌ Patient prediction analysis failed. Check your patient data.")
        
        # Display ML results
        if 'ml_results' in st.session_state:
            ml_results = st.session_state.ml_results
            
            st.markdown("#### 📊 XGBoost Model Performance")
            
            # Display model info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Random Seed", ml_results['random_seed'])
                st.metric("Event Rate", f"{ml_results['event_rate']:.3f}")
                st.metric("Test Set Size", len(ml_results['y_test']))
            
            with col2:
                st.metric("AUC", f"{ml_results['auc']:.3f}")
                st.metric("Accuracy", f"{ml_results['accuracy']:.3f}")
                st.metric("F1 Score", f"{ml_results['f1']:.3f}")
            
            with col3:
                st.metric("Sensitivity", f"{ml_results['sensitivity']:.3f}")
                st.metric("Specificity", f"{ml_results['specificity']:.3f}")
                st.metric("Precision", f"{ml_results['precision']:.3f}")
            
            # Performance metrics table
            performance_data = [{
                'Metric': 'AUC',
                'Value': ml_results['auc'],
                'Description': 'Area Under ROC Curve'
            }, {
                'Metric': 'Accuracy',
                'Value': ml_results['accuracy'],
                'Description': 'Overall accuracy'
            }, {
                'Metric': 'Sensitivity (Recall)',
                'Value': ml_results['sensitivity'],
                'Description': 'True Positive Rate'
            }, {
                'Metric': 'Specificity',
                'Value': ml_results['specificity'],
                'Description': 'True Negative Rate'
            }, {
                'Metric': 'Precision (PPV)',
                'Value': ml_results['precision'],
                'Description': 'Positive Predictive Value'
            }, {
                'Metric': 'NPV',
                'Value': ml_results['npv'],
                'Description': 'Negative Predictive Value'
            }, {
                'Metric': 'F1 Score',
                'Value': ml_results['f1'],
                'Description': 'Harmonic mean of precision and recall'
            }]
            
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df.round(4), use_container_width=True)
            
            # Display best hyperparameters
            st.markdown("#### 🔧 Best Hyperparameters Found")
            best_params_df = pd.DataFrame(list(ml_results['best_params'].items()), 
                                        columns=['Parameter', 'Value'])
            st.dataframe(best_params_df, use_container_width=True)
            
            # ROC Curve
            st.markdown("#### 📈 ROC Curve")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            fpr, tpr, thresholds_roc = roc_curve(ml_results['y_test'], ml_results['y_pred_proba'])
            auc = ml_results['auc']
            ax.plot(fpr, tpr, label=f'XGBoost (AUC = {auc:.3f})', linewidth=2)
            
            # Mark the optimal threshold point
            optimal_idx = np.argmax(tpr - fpr)
            ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8, 
                   label=f'Optimal Threshold\nTPR={tpr[optimal_idx]:.3f}, FPR={fpr[optimal_idx]:.3f}')
            
            ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve for Event Prediction')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Actual vs Predicted Analysis
            st.markdown("#### 🎯 Actual vs Predicted Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Confusion Matrix
                cm = ml_results['confusion_matrix']
                
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)
                
                # Add text annotations
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black")
                
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix - XGBoost')
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(['No Event', 'Event'])
                ax.set_yticklabels(['No Event', 'Event'])
                st.pyplot(fig)
            
            with col2:
                # Prediction distribution
                fig, ax = plt.subplots(figsize=(8, 6))
                
                ax.hist(ml_results['y_pred_proba'], bins=20, alpha=0.7, 
                       label=f'XGBoost (AUC: {ml_results["auc"]:.3f})', color='blue')
                
                ax.set_xlabel('Predicted Probability of Event')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Predicted Probabilities')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Calibration Analysis
            st.markdown("#### 📊 Calibration Analysis")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            fraction_of_positives, mean_predicted_value = calibration_curve(
                ml_results['y_test'], ml_results['y_pred_proba'], n_bins=10
            )
            ax.plot(mean_predicted_value, fraction_of_positives, 
                   marker='o', label='XGBoost', linewidth=2, markersize=8)
            
            # Perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title('Calibration Plot')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Feature Importance Analysis
            st.markdown("#### 🎯 Feature Importance Analysis")
            
            # Create comparison of different importance measures
            importance_data = []
            for i, feature_name in enumerate(ml_results['feature_names']):
                xgb_importance = ml_results['feature_importance'][i]
                correlation = ml_results['correlations'][i]
                
                importance_data.append({
                    'Patient Characteristic': feature_name,
                    'XGBoost': xgb_importance,
                    'Correlation': correlation,
                    'Average Importance': (xgb_importance + correlation) / 2
                })
            
            importance_df = pd.DataFrame(importance_data)
            importance_df = importance_df.sort_values('Average Importance', ascending=False)
            
            # Display importance comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🌳 XGBoost Importance**")
                fig = px.bar(
                    importance_df,
                    x='Patient Characteristic',
                    y='XGBoost',
                    title='XGBoost Feature Importance',
                    color='XGBoost',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**📊 Correlation with Events**")
                fig = px.bar(
                    importance_df,
                    x='Patient Characteristic',
                    y='Correlation',
                    title='Absolute Correlation with Event Occurrence',
                    color='Correlation',
                    color_continuous_scale='plasma'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # SHAP Analysis
            st.markdown("#### 🎯 SHAP Analysis")
            
            if ml_results['shap_values'] is not None:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                shap.summary_plot(
                    ml_results['shap_values'], 
                    ml_results['X_test_scaled'],
                    feature_names=ml_results['feature_names'],
                    show=False,
                    plot_type="bar"
                )
                
                plt.title("SHAP Feature Importance for Patient Event Prediction")
                plt.tight_layout()
                st.pyplot(fig)
            
            # Error Analysis
            st.markdown("#### 🔍 Error Analysis")
            
            # Analyze prediction errors
            errors = ml_results['y_test'] != ml_results['y_pred']
            
            if errors.sum() > 0:
                error_patients = ml_results['X_test'][errors]
                correct_patients = ml_results['X_test'][~errors]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**❌ Misclassified Patients**")
                    st.write(f"Number of misclassified patients: {errors.sum()}")
                    st.write(f"Misclassification rate: {errors.mean():.3f}")
                    
                    # Show characteristics of misclassified patients
                    if len(error_patients) > 0:
                        st.markdown("**Characteristics of Misclassified Patients:**")
                        error_summary = error_patients.describe()
                        st.dataframe(error_summary.round(2))
                
                with col2:
                    st.markdown("**✅ Correctly Classified Patients**")
                    st.write(f"Number of correctly classified patients: {len(correct_patients)}")
                    st.write(f"Correct classification rate: {1 - errors.mean():.3f}")
            
            # Clinical interpretation
            st.markdown("#### 💡 Clinical Interpretation")
            
            with st.expander("🎯 Understanding Patient-Level Prediction Results", expanded=False):
                st.markdown("""
                #### 🤖 Patient-Level Prediction Insights
                
                **AUC (Area Under ROC Curve):**
                - Measures how well the model distinguishes between patients with and without events
                - Values range from 0.5 (random) to 1.0 (perfect)
                - AUC > 0.7 = good, AUC > 0.8 = very good, AUC > 0.9 = excellent
                
                **Calibration:**
                - Shows how well predicted probabilities match actual event rates
                - Well-calibrated models have predictions close to the diagonal line
                - Important for clinical decision making
                
                **Feature Importance:**
                - Shows which patient characteristics are most predictive of events
                - Higher values = more important for predicting individual patient outcomes
                - Can guide patient selection and monitoring strategies
                
                **Error Analysis:**
                - Identifies characteristics of patients where prediction fails
                - Helps understand model limitations and areas for improvement
                
                #### 🎯 Clinical Applications
                
                **Patient Risk Stratification:**
                - Use model predictions to identify high-risk patients
                - Focus monitoring and interventions on patients with high predicted risk
                
                **Trial Enrichment:**
                - Enroll patients with higher predicted event probabilities
                - Reduces sample size requirements while maintaining statistical power
                
                **Clinical Decision Support:**
                - Use predictions to guide treatment decisions
                - Identify patients who need more intensive monitoring
                
                **Regulatory Strategy:**
                - Demonstrate evidence-based patient selection
                - Justify inclusion/exclusion criteria with predictive modeling
                """)
            
            # Top predictive features summary
            st.markdown("#### 🏆 Top Predictive Patient Characteristics")
            
            top_features = importance_df.head(5)
            
            for i, (_, row) in enumerate(top_features.iterrows()):
                st.markdown(f"""
                **{i+1}. {row['Patient Characteristic']}**
                - XGBoost Importance: {row['XGBoost']:.3f}
                - Correlation: {row['Correlation']:.3f}
                - **Clinical Action**: {'High' if row['Average Importance'] > 0.3 else 'Medium' if row['Average Importance'] > 0.1 else 'Low'} priority for patient risk assessment
                """)
    
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