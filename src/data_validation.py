"""
Data Quality Validation for A/B Tests
Implements critical checks for experiment validity
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import chisquare

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_QUALITY_CONFIG


def check_sample_ratio_mismatch(df, expected_ratio=0.5, alpha=0.001):
    """
    Critical: Detect Sample Ratio Mismatch (SRM)
    SRM indicates bugs in randomization implementation
    """
    print("\n" + "="*70)
    print("SAMPLE RATIO MISMATCH CHECK")
    print("="*70)
    
    observed = df['variant'].value_counts()
    total = len(df)
    expected = [total * expected_ratio, total * (1 - expected_ratio)]
    
    chi2, p_value = chisquare(observed.values, expected)
    
    control_pct = observed.get('control', 0) / total
    treatment_pct = observed.get('treatment', 0) / total
    
    print(f"Expected allocation: {expected_ratio:.1%} / {1-expected_ratio:.1%}")
    print(f"Observed allocation: {control_pct:.2%} / {treatment_pct:.2%}")
    print(f"Chi-square: {chi2:.4f}")
    print(f"P-value: {p_value:.6f}")
    
    if p_value < alpha:
        print("\n⚠️  WARNING: SAMPLE RATIO MISMATCH DETECTED!")
        print("Check randomization implementation for bugs")
        return False
    else:
        print("\n✓ No SRM detected - randomization is valid")
        return True


def check_duplicates(df):
    """Check for duplicate user IDs"""
    print("\n" + "="*70)
    print("DUPLICATE USER CHECK")
    print("="*70)
    
    duplicates = df['user_id'].duplicated().sum()
    
    if duplicates > 0:
        print(f"⚠️  WARNING: Found {duplicates} duplicate user IDs!")
        return False
    else:
        print("✓ No duplicate users found")
        return True


def check_test_duration(df, min_days=14):
    """Verify test ran for minimum duration"""
    print("\n" + "="*70)
    print("TEST DURATION CHECK")
    print("="*70)
    
    df['entry_timestamp'] = pd.to_datetime(df['entry_timestamp'])
    duration = (df['entry_timestamp'].max() - df['entry_timestamp'].min()).days
    
    print(f"Test duration: {duration} days")
    print(f"Minimum required: {min_days} days")
    
    if duration < min_days:
        print(f"\n⚠️  WARNING: Test duration too short!")
        return False
    else:
        print("\n✓ Test duration is sufficient")
        return True


def check_conversion_logic(df):
    """Verify conversion logic consistency"""
    print("\n" + "="*70)
    print("CONVERSION LOGIC CHECK")
    print("="*70)
    
    # Non-converted users should have 0 order value
    non_converted_with_value = df[(df['converted'] == 0) & (df['order_value'] > 0)]
    
    # Converted users should have positive order value
    converted_no_value = df[(df['converted'] == 1) & (df['order_value'] <= 0)]
    
    issues = []
    if len(non_converted_with_value) > 0:
        issues.append(f"{len(non_converted_with_value)} non-converted users have order values")
    
    if len(converted_no_value) > 0:
        issues.append(f"{len(converted_no_value)} converted users have no order value")
    
    if issues:
        for issue in issues:
            print(f"⚠️  WARNING: {issue}")
        return False
    else:
        print("✓ Conversion logic is consistent")
        return True


def check_temporal_distribution(df):
    """Check for unexpected temporal patterns"""
    print("\n" + "="*70)
    print("TEMPORAL DISTRIBUTION CHECK")
    print("="*70)
    
    df['hour'] = pd.to_datetime(df['entry_timestamp']).dt.hour
    hourly_dist = df['hour'].value_counts().sort_index()
    
    # Check if traffic is reasonably distributed
    observed_dist = hourly_dist.values / hourly_dist.sum()
    min_pct = observed_dist.min()
    max_pct = observed_dist.max()
    
    print(f"Hourly traffic range: {min_pct:.2%} - {max_pct:.2%}")
    
    if min_pct < 0.02 or max_pct > 0.08:
        print("\n⚠️  NOTE: Uneven temporal distribution (may be expected)")
    else:
        print("\n✓ Temporal distribution looks reasonable")
    
    return True


def check_outliers(df):
    """Check for outliers in order values"""
    print("\n" + "="*70)
    print("OUTLIER CHECK")
    print("="*70)
    
    converted = df[df['converted'] == 1]
    
    if len(converted) > 0:
        Q1 = converted['order_value'].quantile(0.25)
        Q3 = converted['order_value'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR
        
        outliers = converted[(converted['order_value'] < lower) | 
                            (converted['order_value'] > upper)]
        
        print(f"Order value range: ${converted['order_value'].min():.2f} - ${converted['order_value'].max():.2f}")
        print(f"Expected range: ${lower:.2f} - ${upper:.2f}")
        print(f"Outliers: {len(outliers)} ({len(outliers)/len(converted):.2%})")
        
        if len(outliers) / len(converted) > 0.05:
            print(f"\n⚠️  NOTE: {len(outliers)/len(converted):.2%} of orders are outliers")
        else:
            print("\n✓ Outlier rate is acceptable")
    
    return True


def check_segment_balance(df):
    """Ensure randomization is balanced across segments"""
    print("\n" + "="*70)
    print("SEGMENT BALANCE CHECK")
    print("="*70)
    
    balanced = True
    
    for segment in ['device_type', 'user_segment']:
        print(f"\nChecking {segment}:")
        balance = pd.crosstab(df[segment], df['variant'], normalize='index')
        print(balance)
        
        for seg_val in balance.index:
            control_pct = balance.loc[seg_val, 'control']
            if control_pct < 0.45 or control_pct > 0.55:
                print(f"  ⚠️  Imbalance in {segment}={seg_val}: {control_pct:.2%} control")
                balanced = False
    
    if balanced:
        print("\n✓ Variant allocation is balanced across segments")
    
    return balanced


def run_all_validations(csv_path='data/ab_test_data.csv'):
    """Run complete validation suite"""
    print("\n" + "="*70)
    print("STARTING DATA QUALITY VALIDATION")
    print("="*70)
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} records from {csv_path}")
    
    issues = []
    
    # Run all checks
    if not check_sample_ratio_mismatch(df):
        issues.append("Sample Ratio Mismatch detected")
    
    if not check_duplicates(df):
        issues.append("Duplicate users found")
    
    if not check_test_duration(df):
        issues.append("Test duration too short")
    
    if not check_conversion_logic(df):
        issues.append("Conversion logic inconsistent")
    
    check_temporal_distribution(df)
    check_outliers(df)
    check_segment_balance(df)
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    if len(issues) == 0:
        print("\n✓ ALL CRITICAL CHECKS PASSED!")
        print("Data quality is good - proceed with analysis")
        return True
    else:
        print(f"\n⚠️  Found {len(issues)} critical issues:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        print("\nReview these issues before proceeding")
        return False


if __name__ == "__main__":
    passed = run_all_validations()
    sys.exit(0 if passed else 1)
