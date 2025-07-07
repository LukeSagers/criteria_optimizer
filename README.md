# Clinical Trial Criteria Optimizer

An interactive web application that simulates clinical trial pipelines using Monte Carlo simulations to optimize inclusion/exclusion criteria. The application uses Shapley values to determine the impact of each criterion on population size and event rates.

## Features

### 🏥 Synthetic Patient Data Generation
- Generates realistic patient data with clinical characteristics
- 1000+ patients with age, BMI, creatinine, HbA1c, blood pressure, and more
- Includes binary variables like smoking status, diabetes history, cardiovascular history
- Calculates event rates based on risk factors

### 🎯 Configurable Inclusion/Exclusion Criteria
- **5 predefined criteria**: Age, BMI, Creatinine, Smoking Status, Diabetes History
- **Continuous criteria**: Adjustable thresholds with operators (>, <, >=, <=)
- **Binary criteria**: Include/exclude based on presence/absence
- **Real-time filtering**: See immediate impact on eligible population

### 🔄 Monte Carlo Simulation
- Runs thousands of parameter combinations
- Varies criteria thresholds systematically
- Tracks population size and event rates for each combination
- Configurable number of simulations (100-5000)

### 📊 Shapley Values Analysis
- Calculates feature importance using SHAP (SHapley Additive exPlanations)
- Determines impact of each criterion on:
  - **Population Size**: How many patients remain eligible
  - **Event Rate**: Probability of clinical events in eligible population
- Provides interpretable machine learning explanations
- **Educational Content**: Detailed explanations of Shapley values and methodology
- **Interactive Visualizations**: Beeswarm plots, force plots, and case-by-case analysis
- **Practical Interpretation Guide**: How to understand and apply the results

### 🎯 Criteria Optimization
- **Multi-objective optimization**: Find optimal criteria values for target outcomes
- **Weighted optimization**: Balance population size vs event rate priorities
- **Global optimization**: Uses multiple optimization runs to find best solutions
- **Similar population analysis**: Explore criteria combinations that produce similar outcomes

### 📈 Interactive Visualizations
- Patient characteristics distribution (eligible vs. all patients)
- Simulation results distribution
- Population size vs. event rate scatter plots
- Shapley values bar charts
- Real-time metrics and summaries

## Installation

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd criteria_optimizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## How to Use

### 1. Configure Simulation Parameters
- **Number of Patients**: Adjust the synthetic population size (500-5000)
- **Number of Simulations**: Set Monte Carlo simulation iterations (100-5000)

### 2. Set Inclusion/Exclusion Criteria
In the sidebar, configure each criterion:
- **Enable/Disable**: Toggle criteria on/off
- **Threshold**: Set the cutoff value for continuous criteria
- **Value**: Choose 0 or 1 for binary criteria

### 3. Run Monte Carlo Simulation
- Click "Run Monte Carlo Simulation" button
- Wait for the simulation to complete
- View results in interactive charts

### 4. Analyze Results
- **Population Overview**: See current eligible population metrics
- **Simulation Results**: Distribution of population sizes and event rates
- **Shapley Values**: Impact of each criterion on outcomes
- **Educational Content**: Learn about Shapley values and methodology
- **Interactive Visualizations**: Explore beeswarm plots and force plots
- **Summary Table**: Ranked list of criterion importance

### 5. Optimize Criteria
- **Set Target Outcomes**: Specify desired population size and event rate
- **Adjust Weights**: Balance importance of population size vs event rate
- **Find Optimal Values**: Get recommended criteria thresholds
- **Explore Similar Populations**: See other criteria combinations with similar outcomes

### 6. Download Results
- **CSV Export**: Download all simulation results
- **Summary Report**: Get a text report with key findings

## Statistical Methodology

### Monte Carlo Simulation
The application uses Monte Carlo methods to explore the parameter space of inclusion/exclusion criteria:

1. **Parameter Sampling**: Randomly samples threshold values for each criterion
2. **Population Filtering**: Applies criteria to synthetic patient data
3. **Outcome Calculation**: Measures population size and event rate
4. **Iteration**: Repeats thousands of times to build comprehensive dataset

### Shapley Values
Shapley values provide a principled way to attribute the contribution of each feature to the model's prediction:

1. **Model Training**: Uses Random Forest to predict outcomes from criteria parameters
2. **SHAP Analysis**: Calculates Shapley values for each criterion
3. **Impact Assessment**: Measures absolute contribution to population size and event rate
4. **Interpretation**: Higher values indicate greater impact on outcomes

**Educational Features:**
- **Theory Explanation**: Game theory foundation and mathematical formulation
- **Implementation Details**: Step-by-step process of how we calculate SHAP values
- **Interactive Examples**: Explore specific cases with force plots
- **Visualization Types**: Beeswarm plots, force plots, and summary plots
- **Practical Guide**: How to interpret and apply the results

### Multi-Objective Optimization
The application uses advanced optimization techniques to find optimal criteria values:

1. **Objective Function**: Weighted combination of population size and event rate targets
2. **Global Optimization**: Multiple optimization runs with different starting points
3. **Parameter Bounds**: Respects realistic ranges for each criterion
4. **Similar Population Analysis**: Finds criteria combinations producing similar outcomes

## Example Use Cases

### Clinical Trial Design
- **Optimize enrollment**: Balance population size with event rate
- **Risk stratification**: Identify criteria that most affect outcomes
- **Protocol refinement**: Test different inclusion/exclusion combinations
- **Target optimization**: Find criteria values for specific population/event rate targets

### Regulatory Planning
- **Sample size estimation**: Understand how criteria affect eligible population
- **Risk assessment**: Evaluate impact on expected event rates
- **Protocol justification**: Provide data-driven rationale for criteria

### Research Planning
- **Feasibility studies**: Assess trial feasibility with different criteria
- **Sensitivity analysis**: Test robustness of criteria choices
- **Comparative analysis**: Compare different trial designs

## Technical Details

### Data Generation
- **Synthetic Patients**: Realistic clinical characteristics based on medical literature
- **Risk Modeling**: Event rates calculated from composite risk scores
- **Correlations**: Realistic relationships between clinical variables

### Performance
- **Caching**: Streamlit caching for efficient data generation
- **Vectorization**: NumPy operations for fast computation
- **Parallel Processing**: Efficient Monte Carlo simulation

### Visualization
- **Plotly**: Interactive charts for exploration
- **Matplotlib**: Static plots for publication
- **Responsive Design**: Works on desktop and mobile

## Limitations

- **Synthetic Data**: Results are illustrative, not clinical
- **Simplified Model**: Event rates use simplified risk calculations
- **Limited Criteria**: Currently supports 5 criteria (expandable)
- **No External Validation**: Results should be validated with real data

## Future Enhancements

- **More Criteria**: Add additional clinical variables
- **Advanced Models**: Implement more sophisticated risk models
- **Real Data Integration**: Support for real patient datasets
- **Cost Analysis**: Include economic impact of criteria choices
- **Multi-Objective Optimization**: Balance multiple outcomes simultaneously

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 