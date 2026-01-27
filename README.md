# A/B Testing Framework

A comprehensive Python framework for designing, executing, and analyzing A/B tests with statistical rigor. Features an interactive Streamlit dashboard for experimentation and visualization.

## Features

- **Frequentist Statistical Testing**: Two-sample t-tests, z-tests for proportions, chi-square tests
- **Bayesian Analysis**: Posterior probability calculations with credible intervals
- **Power Analysis**: Sample size calculations and minimum detectable effect estimation
- **Heterogeneous Treatment Effects (HTE)**: Analyze treatment effects across user segments
- **Data Validation**: Sample ratio mismatch detection and data quality checks
- **Interactive Dashboard**: Streamlit-based UI for real-time experimentation

## Project Structure

```
ABTestingFramework/
├── config.py                 # Configuration parameters
├── streamlit_app.py          # Interactive Streamlit dashboard
├── test_analysis.py          # Unit tests
├── requirements.txt          # Python dependencies
├── data/                     # Sample datasets
└── src/
    ├── data_generation.py    # Synthetic data generation
    ├── data_loading.py       # Data loading utilities
    ├── data_validation.py    # Data quality checks
    └── statistical_test.py   # Statistical testing methods
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ABTestingFramework.git
cd ABTestingFramework
```

2. Create and activate a virtual environment:
```bash
python3 -m venv ab_venv
source ab_venv/bin/activate  # On Windows: ab_venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard provides three data source options:
- **Use Existing Data**: Load pre-generated experiment data
- **Upload CSV**: Upload your own A/B test data
- **Generate Synthetic Data**: Create customizable test data

### Configuration

Modify `config.py` to customize experiment parameters:

```python
EXPERIMENT_CONFIG = {
    'experiment_name': 'checkout_optimization_v1',
    'test_duration_days': 21,
    'n_users': 50000,
    'allocation_ratio': 0.5,
}

STATS_CONFIG = {
    'alpha': 0.05,              # Significance level
    'power': 0.80,              # Statistical power
    'mde': 0.15,                # Minimum detectable effect
    'baseline_conversion': 0.08,
}
```

## Dashboard Features

1. **Experiment Configuration**: Set sample sizes, test duration, and allocation ratios
2. **Statistical Settings**: Configure alpha, power, and MDE
3. **Results Analysis**: View conversion rates, lift, and statistical significance
4. **Bayesian Analysis**: Posterior distributions and probability of improvement
5. **Segment Analysis**: Heterogeneous treatment effects by device type, user segment

## Key Metrics

- Conversion Rate Lift
- Statistical Significance (p-value)
- Confidence Intervals
- Bayesian Probability of Improvement
- Revenue Impact

## Dependencies

- streamlit
- pandas
- numpy
- scipy
- statsmodels
- scikit-learn
- plotly
- seaborn

## License

MIT License
