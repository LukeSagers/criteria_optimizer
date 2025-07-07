#!/usr/bin/env python3
"""
Example usage of Clinical Trial Criteria Optimizer functions
This script demonstrates the core functionality without the Streamlit interface.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from app import (
    generate_synthetic_patient_data,
    apply_inclusion_exclusion_criteria,
    run_monte_carlo_simulation,
    calculate_shapley_values
)

def main():
    """Demonstrate the core functionality."""
    print("🏥 Clinical Trial Criteria Optimizer - Example Usage")
    print("=" * 60)
    
    # 1. Generate synthetic patient data
    print("\n1. Generating synthetic patient data...")
    patients = generate_synthetic_patient_data(n_patients=1000, seed=42)
    print(f"✅ Generated {len(patients)} patients")
    print(f"   Overall event rate: {patients['event_occurred'].mean():.3f}")
    
    # 2. Define inclusion/exclusion criteria
    print("\n2. Setting up inclusion/exclusion criteria...")
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
    
    # 3. Apply criteria and see initial results
    print("\n3. Applying inclusion/exclusion criteria...")
    eligible_patients = apply_inclusion_exclusion_criteria(patients, criteria_config)
    print(f"✅ Eligible patients: {len(eligible_patients)}")
    print(f"   Eligibility rate: {len(eligible_patients) / len(patients) * 100:.1f}%")
    print(f"   Event rate: {eligible_patients['event_occurred'].mean():.3f}")
    
    # 4. Run Monte Carlo simulation
    print("\n4. Running Monte Carlo simulation...")
    print("   This may take a few moments...")
    simulation_results = run_monte_carlo_simulation(patients, criteria_config, n_simulations=500)
    print(f"✅ Completed {len(simulation_results)} simulations")
    
    # 5. Calculate Shapley values
    print("\n5. Calculating Shapley values...")
    shap_values_dict = calculate_shapley_values(simulation_results, criteria_config)
    print("✅ Shapley values calculated")
    
    # 6. Display results
    print("\n6. Results Summary:")
    print("-" * 40)
    
    # Population size impact
    population_shap = shap_values_dict['population_size']
    print("\nImpact on Population Size:")
    for i, feature in enumerate(population_shap['features']):
        criterion_name = feature.replace('param_', '').replace('_criteria', '')
        impact = population_shap['shap_values'][i]
        print(f"   {criterion_name}: {impact:.4f}")
    
    # Event rate impact
    event_shap = shap_values_dict['event_rate']
    print("\nImpact on Event Rate:")
    for i, feature in enumerate(event_shap['features']):
        criterion_name = feature.replace('param_', '').replace('_criteria', '')
        impact = event_shap['shap_values'][i]
        print(f"   {criterion_name}: {impact:.4f}")
    
    # 7. Create summary plots
    print("\n7. Creating summary plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Population size distribution
    axes[0, 0].hist(simulation_results['population_size'], bins=30, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Population Size Distribution')
    axes[0, 0].set_xlabel('Population Size')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Event rate distribution
    axes[0, 1].hist(simulation_results['event_rate'], bins=30, alpha=0.7, color='lightcoral')
    axes[0, 1].set_title('Event Rate Distribution')
    axes[0, 1].set_xlabel('Event Rate')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Population size vs Event rate
    axes[1, 0].scatter(simulation_results['population_size'], simulation_results['event_rate'], 
                      alpha=0.6, color='green')
    axes[1, 0].set_title('Population Size vs Event Rate')
    axes[1, 0].set_xlabel('Population Size')
    axes[1, 0].set_ylabel('Event Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Shapley values comparison
    criterion_names = [col.replace('param_', '').replace('_criteria', '') 
                      for col in population_shap['features']]
    x_pos = np.arange(len(criterion_names))
    
    axes[1, 1].bar(x_pos - 0.2, population_shap['shap_values'], 0.4, 
                   label='Population Size', alpha=0.7, color='skyblue')
    axes[1, 1].bar(x_pos + 0.2, event_shap['shap_values'], 0.4, 
                   label='Event Rate', alpha=0.7, color='lightcoral')
    axes[1, 1].set_title('Shapley Values Comparison')
    axes[1, 1].set_xlabel('Criteria')
    axes[1, 1].set_ylabel('SHAP Value')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(criterion_names, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_results.png', dpi=300, bbox_inches='tight')
    print("✅ Saved plots to 'example_results.png'")
    
    # 8. Save results to CSV
    print("\n8. Saving results...")
    simulation_results.to_csv('example_simulation_results.csv', index=False)
    print("✅ Saved simulation results to 'example_simulation_results.csv'")
    
    print("\n🎉 Example completed successfully!")
    print("=" * 60)
    print("\nFiles created:")
    print("   - example_results.png (summary plots)")
    print("   - example_simulation_results.csv (simulation data)")
    print("\nTo run the interactive version, use:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    main() 