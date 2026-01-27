"""
Synthetic A/B Test Data Generation
Generates realistic e-commerce experiment data with heterogeneous treatment effects
"""

import os
import sys
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EXPERIMENT_CONFIG, EFFECT_CONFIG, USER_SEGMENTS, HTE_CONFIG


# Set seeds for reproducibility
np.random.seed(42)
fake = Faker()
Faker.seed(42)


def generate_ab_test_data():
    """Generate complete A/B test dataset"""
    
    print("="*70)
    print("GENERATING A/B TEST DATA")
    print("="*70)
    print(f"\nExperiment: {EXPERIMENT_CONFIG['experiment_name']}")
    print(f"Users: {EXPERIMENT_CONFIG['n_users']:,}")
    print(f"Duration: {EXPERIMENT_CONFIG['test_duration_days']} days")
    print(f"Control conversion: {EFFECT_CONFIG['control_conversion_rate']:.2%}")
    print(f"Treatment conversion: {EFFECT_CONFIG['treatment_conversion_rate']:.2%}")
    print(f"Expected lift: {(EFFECT_CONFIG['treatment_conversion_rate']/EFFECT_CONFIG['control_conversion_rate'] - 1):.2%}")
    print()
    
    data = []
    start_date = datetime.strptime(EXPERIMENT_CONFIG['start_date'], '%Y-%m-%d')
    
    for i in range(EXPERIMENT_CONFIG['n_users']):
        if (i + 1) % 10000 == 0:
            print(f"Generated {i+1:,} users...")
        
        # User assignment
        user_id = f'user_{i:06d}'
        variant = np.random.choice(['control', 'treatment'], 
                                   p=[EXPERIMENT_CONFIG['allocation_ratio'], 
                                      1 - EXPERIMENT_CONFIG['allocation_ratio']])
        
        # Entry timestamp
        days_offset = np.random.randint(0, EXPERIMENT_CONFIG['test_duration_days'])
        hours_offset = np.random.randint(0, 24)
        minutes_offset = np.random.randint(0, 60)
        entry_timestamp = start_date + timedelta(days=days_offset, 
                                                  hours=hours_offset, 
                                                  minutes=minutes_offset)
        
        # User characteristics
        device_type = np.random.choice(
            list(USER_SEGMENTS['device_type'].keys()),
            p=list(USER_SEGMENTS['device_type'].values())
        )
        
        user_segment = np.random.choice(
            list(USER_SEGMENTS['user_segment'].keys()),
            p=list(USER_SEGMENTS['user_segment'].values())
        )
        
        # Calculate conversion probability with HTE
        if variant == 'control':
            base_rate = EFFECT_CONFIG['control_conversion_rate']
        else:
            base_rate = EFFECT_CONFIG['treatment_conversion_rate']
        
        # Apply heterogeneous treatment effects
        if variant == 'treatment' and device_type == 'mobile':
            base_rate *= HTE_CONFIG['mobile_treatment_boost']
        
        if user_segment == 'new':
            base_rate *= HTE_CONFIG['new_user_penalty']
        
        # Add random noise (±5%)
        conversion_rate = base_rate * np.random.uniform(0.95, 1.05)
        conversion_rate = min(max(conversion_rate, 0), 1)
        
        # Determine if converted
        converted = int(np.random.random() < conversion_rate)
        
        # Generate order metrics
        if converted:
            if variant == 'control':
                mean_aov = EFFECT_CONFIG['control_avg_order_value']
            else:
                mean_aov = EFFECT_CONFIG['treatment_avg_order_value']
            
            order_value = max(10, np.random.normal(mean_aov, EFFECT_CONFIG['order_value_std']))
            order_id = f'ORD_{fake.uuid4()[:8]}'
        else:
            order_value = 0
            order_id = None
        
        # Engagement metrics
        time_on_page = np.random.lognormal(mean=3.5, sigma=0.8)
        pages_viewed = max(1, np.random.poisson(lam=3.2))
        bounced = int(np.random.random() < 0.35)
        
        # Compile record
        data.append({
            'user_id': user_id,
            'variant': variant,
            'entry_timestamp': entry_timestamp,
            'device_type': device_type,
            'user_segment': user_segment,
            'converted': converted,
            'order_value': round(order_value, 2),
            'order_id': order_id,
            'time_on_page_sec': round(time_on_page, 1),
            'pages_viewed': pages_viewed,
            'bounced': bounced
        })
    
    df = pd.DataFrame(data)
    
    # Summary statistics
    print("\n" + "="*70)
    print("GENERATION SUMMARY")
    print("="*70)
    print(f"\nTotal users: {len(df):,}")
    print(f"\nVariant distribution:")
    print(df['variant'].value_counts())
    
    control_conv = df[df['variant']=='control']['converted'].mean()
    treatment_conv = df[df['variant']=='treatment']['converted'].mean()
    observed_lift = (treatment_conv / control_conv - 1) if control_conv > 0 else 0
    
    print(f"\nControl conversion: {control_conv:.4f}")
    print(f"Treatment conversion: {treatment_conv:.4f}")
    print(f"Observed lift: {observed_lift:.2%}")
    
    control_aov = df[df['variant']=='control']['order_value'].mean()
    treatment_aov = df[df['variant']=='treatment']['order_value'].mean()
    
    print(f"\nControl AOV: ${control_aov:.2f}")
    print(f"Treatment AOV: ${treatment_aov:.2f}")
    
    print(f"\nDevice distribution:")
    print(df['device_type'].value_counts())
    
    print(f"\nUser segment distribution:")
    print(df['user_segment'].value_counts())
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    output_path = 'data/ab_test_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Data saved to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")
    print("="*70 + "\n")
    
    return df


if __name__ == "__main__":
    df = generate_ab_test_data()
    print("First 10 rows:")
    print(df.head(10))
    print("\nData types:")
    print(df.dtypes)
