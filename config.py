"""
Configuration file for A/B Testing Framework
Modify these parameters to customize your experiment
"""


EXPERIMENT_CONFIG = {
    'experiment_name': 'checkout_optimization_v1',
    'start_date': '2024-11-01',
    'test_duration_days': 21,
    'n_users': 50000,
    'allocation_ratio': 0.5,  # 50/50 split between control and treatment
}

STATS_CONFIG = {
    'alpha': 0.05,              # Significance level
    'power': 0.80,              # Statistical power
    'mde': 0.15,                # Minimum detectable effect (15% relative lift)
    'baseline_conversion': 0.08, # Expected control conversion rate
}

EFFECT_CONFIG = {
    'control_conversion_rate': 0.08,
    'treatment_conversion_rate': 0.092,  # 15% relative lift
    'control_avg_order_value': 75,
    'treatment_avg_order_value': 78,
    'order_value_std': 25,
}

USER_SEGMENTS = {
    'device_type': {
        'mobile': 0.5,
        'desktop': 0.4,
        'tablet': 0.1
    },
    'user_segment': {
        'new': 0.4,
        'returning': 0.4,
        'loyal': 0.2
    }
}

HTE_CONFIG = {
    'mobile_treatment_boost': 1.10,  # Treatment works 10% better on mobile
    'new_user_penalty': 0.85,        # New users convert 15% less
}


DATABASE_CONFIG = {
    'database_url': 'sqlite:///data/ab_testing.db',
    # For PostgreSQL: 'postgresql://username:password@localhost:5432/ab_testing'
}


DATA_QUALITY_CONFIG = {
    'srm_alpha': 0.001,              # Sample ratio mismatch threshold
    'max_allocation_deviation': 0.02, # Max deviation from 50/50
    'min_test_duration_days': 14,    # Minimum test duration
}
