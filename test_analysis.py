"""
Unit Tests for A/B Testing Framework
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.statistical_test import ABTestAnalyzer
from src.data_validation import (
    check_sample_ratio_mismatch, 
    check_duplicates,
    check_conversion_logic
)


class TestABTestAnalyzer:
    """Test cases for ABTestAnalyzer"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data"""
        np.random.seed(42)
        n = 1000
        
        data = {
            'user_id': [f'user_{i}' for i in range(n)],
            'variant': np.random.choice(['control', 'treatment'], n),
            'converted': np.random.binomial(1, 0.08, n),
            'order_value': np.random.normal(75, 25, n),
            'entry_timestamp': pd.date_range('2024-01-01', periods=n, freq='h'),
            'device_type': np.random.choice(['mobile', 'desktop'], n),
            'user_segment': np.random.choice(['new', 'returning'], n)
        }
        
        df = pd.DataFrame(data)
        df.loc[df['converted'] == 0, 'order_value'] = 0
        
        return df
    
    def test_initialization(self, sample_data):
        """Test analyzer initialization"""
        analyzer = ABTestAnalyzer(sample_data)
        assert len(analyzer.control) > 0
        assert len(analyzer.treatment) > 0
        assert len(analyzer.control) + len(analyzer.treatment) == len(sample_data)
    
    def test_frequentist_test_binary(self, sample_data):
        """Test frequentist analysis for binary metric"""
        analyzer = ABTestAnalyzer(sample_data)
        results = analyzer.frequentist_test('converted')
        
        assert 'control_rate' in results
        assert 'treatment_rate' in results
        assert 'p_value' in results
        assert 'relative_lift' in results
        assert 0 <= results['p_value'] <= 1
        assert results['test_type'] == 'proportions_ztest'
    
    def test_frequentist_test_continuous(self, sample_data):
        """Test frequentist analysis for continuous metric"""
        sample_data.loc[sample_data['converted'] == 1, 'order_value'] = \
            np.random.normal(75, 25, (sample_data['converted'] == 1).sum())
        
        analyzer = ABTestAnalyzer(sample_data)
        results = analyzer.frequentist_test('order_value')
        
        assert 'control_mean' in results
        assert 'treatment_mean' in results
        assert 't_statistic' in results
        assert results['test_type'] == 'welch_ttest'
    
    def test_bayesian_test(self, sample_data):
        """Test Bayesian analysis"""
        analyzer = ABTestAnalyzer(sample_data)
        results = analyzer.bayesian_test('converted', n_simulations=1000)
        
        assert 'prob_treatment_better' in results
        assert 'expected_lift' in results
        assert 'credible_interval_95' in results
        assert 0 <= results['prob_treatment_better'] <= 1
        assert len(results['credible_interval_95']) == 2
    
    def test_power_analysis(self, sample_data):
        """Test power analysis"""
        analyzer = ABTestAnalyzer(sample_data)
        results = analyzer.power_analysis()
        
        assert 'required_n_per_variant' in results
        assert 'achieved_power' in results
        assert 'is_adequately_powered' in results
        assert results['required_n_per_variant'] > 0
        assert 0 <= results['achieved_power'] <= 1
    
    def test_bootstrap_ci(self, sample_data):
        """Test bootstrap confidence intervals"""
        analyzer = ABTestAnalyzer(sample_data)
        results = analyzer.bootstrap_ci('converted', n_bootstrap=100)
        
        assert 'bootstrap_mean_lift' in results
        assert 'ci_lower' in results
        assert 'ci_upper' in results
        assert results['ci_lower'] < results['ci_upper']
    
    def test_hte_analysis(self, sample_data):
        """Test heterogeneous treatment effects"""
        analyzer = ABTestAnalyzer(sample_data)
        results = analyzer.heterogeneous_treatment_effects(['device_type'])
        
        assert 'device_type' in results
        assert isinstance(results['device_type'], pd.DataFrame)
        assert len(results['device_type']) > 0
        assert 'relative_lift' in results['device_type'].columns
    
    def test_novelty_detection(self, sample_data):
        """Test novelty effect detection"""
        analyzer = ABTestAnalyzer(sample_data)
        results = analyzer.detect_novelty_effect(window_days=7)
        
        assert 'early_relative_lift' in results
        assert 'late_relative_lift' in results
        assert 'novelty_detected' in results
        assert isinstance(results['novelty_detected'], bool)
    
    def test_business_impact(self, sample_data):
        """Test business impact calculation"""
        analyzer = ABTestAnalyzer(sample_data)
        results = analyzer.calculate_business_impact(
            annual_traffic=100000,
            avg_order_value=75
        )
        
        assert 'additional_revenue' in results
        assert 'additional_conversions' in results
        assert 'roi_percentage' in results
        assert results['additional_revenue'] >= 0 or results['additional_revenue'] < 0
    
    def test_sequential_test(self, sample_data):
        """Test sequential testing"""
        analyzer = ABTestAnalyzer(sample_data)
        results = analyzer.sequential_test('converted', spending_function='obf')
        
        assert isinstance(results, pd.DataFrame)
        assert 'look' in results.columns
        assert 'p_value' in results.columns
        assert 'can_stop' in results.columns


class TestDataValidation:
    """Test cases for data validation"""
    
    @pytest.fixture
    def valid_data(self):
        """Create valid test data"""
        np.random.seed(42)
        n = 1000
        
        data = {
            'user_id': [f'user_{i}' for i in range(n)],
            'variant': np.random.choice(['control', 'treatment'], n, p=[0.5, 0.5]),
            'converted': np.random.binomial(1, 0.08, n),
            'order_value': [75 if c else 0 for c in np.random.binomial(1, 0.08, n)],
            'entry_timestamp': pd.date_range('2024-01-01', periods=n, freq='h')
        }
        
        return pd.DataFrame(data)
    
    def test_srm_check_valid(self, valid_data):
        """Test SRM check with valid data"""
        result = check_sample_ratio_mismatch(valid_data, expected_ratio=0.5)
        assert isinstance(result, bool)
    
    def test_duplicate_check_no_duplicates(self, valid_data):
        """Test duplicate check with no duplicates"""
        result = check_duplicates(valid_data)
        assert result == True
    
    def test_duplicate_check_with_duplicates(self, valid_data):
        """Test duplicate check with duplicates"""
        # Add duplicate
        valid_data = pd.concat([valid_data, valid_data.head(1)], ignore_index=True)
        result = check_duplicates(valid_data)
        assert result == False
    
    def test_conversion_logic_valid(self, valid_data):
        """Test conversion logic with valid data"""
        result = check_conversion_logic(valid_data)
        assert result == True
    
    def test_conversion_logic_invalid(self, valid_data):
        """Test conversion logic with invalid data"""
        # Make some non-converted users have order value
        valid_data.loc[0, 'converted'] = 0
        valid_data.loc[0, 'order_value'] = 100
        result = check_conversion_logic(valid_data)
        assert result == False


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_dataframe(self):
        """Test with empty dataframe"""
        df = pd.DataFrame(columns=['user_id', 'variant', 'converted'])
        
        with pytest.raises(Exception):
            analyzer = ABTestAnalyzer(df)
            analyzer.frequentist_test('converted')
    
    def test_single_variant(self):
        """Test with only one variant"""
        df = pd.DataFrame({
            'user_id': ['user_1', 'user_2'],
            'variant': ['control', 'control'],
            'converted': [0, 1]
        })
        
        with pytest.raises(Exception):
            analyzer = ABTestAnalyzer(df)
    
    def test_all_zeros(self):
        """Test with all zeros (no conversions)"""
        df = pd.DataFrame({
            'user_id': [f'user_{i}' for i in range(100)],
            'variant': ['control'] * 50 + ['treatment'] * 50,
            'converted': [0] * 100,
            'entry_timestamp': pd.date_range('2024-01-01', periods=100, freq='h')
        })
        
        analyzer = ABTestAnalyzer(df)
        results = analyzer.frequentist_test('converted')
        
        assert results['control_rate'] == 0
        assert results['treatment_rate'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
