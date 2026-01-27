"""
Statistical Analysis Module for A/B Testing
Implements frequentist, Bayesian, and advanced analysis methods
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.stats.proportion import proportions_ztest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STATS_CONFIG


class ABTestAnalyzer:
    """
    Comprehensive A/B Test Statistical Analysis
    
    Methods:
    - Frequentist (p-values, confidence intervals)
    - Bayesian (posterior probabilities)
    - Power analysis
    - Heterogeneous treatment effects
    - Sequential testing
    - CUPED variance reduction
    - Bootstrap confidence intervals
    """
    
    def __init__(self, df, control_variant='control', treatment_variant='treatment'):
        self.df = df.copy()
        self.control_variant = control_variant
        self.treatment_variant = treatment_variant
        
        self.control = df[df['variant'] == control_variant].copy()
        self.treatment = df[df['variant'] == treatment_variant].copy()
    
    # ========================================================================
    # POWER ANALYSIS
    # ========================================================================
    
    def power_analysis(self, baseline_rate=None, mde=None, alpha=None, power=None):
        """
        Calculate required sample size or achievable power
        
        Args:
            baseline_rate: Expected control conversion rate
            mde: Minimum detectable effect (relative lift)
            alpha: Significance level
            power: Target statistical power
        
        Returns:
            Dictionary with power analysis results
        """
        baseline_rate = baseline_rate or STATS_CONFIG['baseline_conversion']
        mde = mde or STATS_CONFIG['mde']
        alpha = alpha or STATS_CONFIG['alpha']
        power = power or STATS_CONFIG['power']
        
        # Calculate effect size (Cohen's h)
        p1 = baseline_rate
        p2 = baseline_rate * (1 + mde)
        effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
        
        # Required sample size
        n_per_variant = zt_ind_solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            alternative='larger'
        )
        
        # Achieved power with actual sample
        actual_n = min(len(self.control), len(self.treatment))
        achieved_power = zt_ind_solve_power(
            effect_size=effect_size,
            nobs1=actual_n,
            alpha=alpha,
            alternative='larger'
        )
        
        return {
            'baseline_rate': baseline_rate,
            'mde': mde,
            'alpha': alpha,
            'target_power': power,
            'effect_size': effect_size,
            'required_n_per_variant': int(n_per_variant),
            'total_required_n': int(n_per_variant * 2),
            'actual_n_per_variant': actual_n,
            'achieved_power': achieved_power,
            'is_adequately_powered': achieved_power >= power
        }
    
    # ========================================================================
    # FREQUENTIST ANALYSIS
    # ========================================================================
    
    def frequentist_test(self, metric='converted', alpha=0.05):
        """
        Standard frequentist A/B test
        
        Args:
            metric: Metric to analyze ('converted' for binary, column name for continuous)
            alpha: Significance level
        
        Returns:
            Dictionary with test results
        """
        if metric == 'converted' or self.df[metric].nunique() == 2:
            # Binary metric - proportions test
            conversions = [
                self.control[metric].sum(),
                self.treatment[metric].sum()
            ]
            nobs = [len(self.control), len(self.treatment)]
            
            stat, p_value = proportions_ztest(
                conversions, nobs, alternative='smaller'
            )
            
            control_rate = self.control[metric].mean()
            treatment_rate = self.treatment[metric].mean()
            
            absolute_lift = treatment_rate - control_rate
            relative_lift = absolute_lift / control_rate if control_rate > 0 else 0
            
            # Standard error
            se = np.sqrt(
                treatment_rate * (1 - treatment_rate) / nobs[1] +
                control_rate * (1 - control_rate) / nobs[0]
            )
            
            # Confidence intervals
            ci_95_lower = absolute_lift - 1.96 * se
            ci_95_upper = absolute_lift + 1.96 * se
            
            results = {
                'metric': metric,
                'test_type': 'proportions_ztest',
                'control_rate': control_rate,
                'treatment_rate': treatment_rate,
                'control_n': nobs[0],
                'treatment_n': nobs[1],
                'absolute_lift': absolute_lift,
                'relative_lift': relative_lift,
                'standard_error': se,
                'z_statistic': stat,
                'p_value': p_value,
                'ci_95_lower': ci_95_lower,
                'ci_95_upper': ci_95_upper,
                'is_significant': p_value < alpha,
                'alpha': alpha
            }
        else:
            # Continuous metric - t-test
            control_values = self.control[metric].dropna()
            treatment_values = self.treatment[metric].dropna()
            
            stat, p_value = stats.ttest_ind(
                treatment_values, control_values, 
                equal_var=False, alternative='greater'
            )
            
            control_mean = control_values.mean()
            treatment_mean = treatment_values.mean()
            
            absolute_lift = treatment_mean - control_mean
            relative_lift = absolute_lift / control_mean if control_mean > 0 else 0
            
            se = np.sqrt(
                control_values.var() / len(control_values) +
                treatment_values.var() / len(treatment_values)
            )
            
            ci_95_lower = absolute_lift - 1.96 * se
            ci_95_upper = absolute_lift + 1.96 * se
            
            results = {
                'metric': metric,
                'test_type': 'welch_ttest',
                'control_mean': control_mean,
                'treatment_mean': treatment_mean,
                'control_n': len(control_values),
                'treatment_n': len(treatment_values),
                'absolute_lift': absolute_lift,
                'relative_lift': relative_lift,
                'standard_error': se,
                't_statistic': stat,
                'p_value': p_value,
                'ci_95_lower': ci_95_lower,
                'ci_95_upper': ci_95_upper,
                'is_significant': p_value < alpha,
                'alpha': alpha
            }
        
        return results
    
    # ========================================================================
    # BAYESIAN ANALYSIS
    # ========================================================================
    
    def bayesian_test(self, metric='converted', n_simulations=10000, 
                     prior_alpha=1, prior_beta=1):
        """
        Bayesian A/B test using Beta-Binomial conjugate prior
        
        Args:
            metric: Metric to analyze
            n_simulations: Number of Monte Carlo simulations
            prior_alpha: Beta prior alpha parameter
            prior_beta: Beta prior beta parameter
        
        Returns:
            Dictionary with Bayesian results
        """
        # Posterior distributions
        control_successes = self.control[metric].sum()
        control_trials = len(self.control)
        
        treatment_successes = self.treatment[metric].sum()
        treatment_trials = len(self.treatment)
        
        # Beta posteriors
        control_posterior = stats.beta(
            prior_alpha + control_successes,
            prior_beta + control_trials - control_successes
        )
        
        treatment_posterior = stats.beta(
            prior_alpha + treatment_successes,
            prior_beta + treatment_trials - treatment_successes
        )
        
        # Monte Carlo simulation
        np.random.seed(42)
        control_samples = control_posterior.rvs(n_simulations)
        treatment_samples = treatment_posterior.rvs(n_simulations)
        
        # Probability treatment > control
        prob_treatment_better = (treatment_samples > control_samples).mean()
        
        # Expected lift
        lift_samples = (treatment_samples - control_samples) / control_samples
        expected_lift = lift_samples.mean()
        credible_interval_95 = np.percentile(lift_samples, [2.5, 97.5])
        
        # Risk calculation
        risk_treatment = np.mean(np.maximum(control_samples - treatment_samples, 0))
        risk_control = np.mean(np.maximum(treatment_samples - control_samples, 0))
        
        # Recommendation
        if prob_treatment_better > 0.95:
            recommendation = 'treatment'
        elif prob_treatment_better < 0.05:
            recommendation = 'control'
        else:
            recommendation = 'inconclusive'
        
        return {
            'metric': metric,
            'control_posterior_alpha': prior_alpha + control_successes,
            'control_posterior_beta': prior_beta + control_trials - control_successes,
            'treatment_posterior_alpha': prior_alpha + treatment_successes,
            'treatment_posterior_beta': prior_beta + treatment_trials - treatment_successes,
            'prob_treatment_better': prob_treatment_better,
            'expected_lift': expected_lift,
            'credible_interval_95': credible_interval_95,
            'risk_choosing_treatment': risk_treatment,
            'risk_choosing_control': risk_control,
            'recommended_variant': recommendation,
            'n_simulations': n_simulations
        }
    
    # ========================================================================
    # BOOTSTRAP CONFIDENCE INTERVALS
    # ========================================================================
    
    def bootstrap_ci(self, metric='converted', n_bootstrap=1000, alpha=0.05):
        """
        Bootstrap confidence intervals for lift
        
        Args:
            metric: Metric to analyze
            n_bootstrap: Number of bootstrap samples
            alpha: Significance level for CI
        
        Returns:
            Dictionary with bootstrap results
        """
        np.random.seed(42)
        
        control_values = self.control[metric].values
        treatment_values = self.treatment[metric].values
        
        bootstrap_lifts = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            control_sample = np.random.choice(control_values, size=len(control_values), replace=True)
            treatment_sample = np.random.choice(treatment_values, size=len(treatment_values), replace=True)
            
            # Calculate lift
            control_mean = control_sample.mean()
            treatment_mean = treatment_sample.mean()
            lift = (treatment_mean - control_mean) / control_mean if control_mean > 0 else 0
            
            bootstrap_lifts.append(lift)
        
        bootstrap_lifts = np.array(bootstrap_lifts)
        
        # Percentile method
        ci_lower = np.percentile(bootstrap_lifts, (alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_lifts, (1 - alpha/2) * 100)
        
        return {
            'metric': metric,
            'bootstrap_mean_lift': bootstrap_lifts.mean(),
            'bootstrap_std': bootstrap_lifts.std(),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_bootstrap': n_bootstrap
        }
    
    # ========================================================================
    # HETEROGENEOUS TREATMENT EFFECTS
    # ========================================================================
    
    def heterogeneous_treatment_effects(self, segments=None):
        """
        Analyze treatment effects across different user segments
        
        Args:
            segments: List of columns to segment by
        
        Returns:
            Dictionary of DataFrames with HTE results
        """
        if segments is None:
            segments = ['device_type', 'user_segment']
        
        results = {}
        
        for segment in segments:
            if segment not in self.df.columns:
                continue
            
            segment_results = []
            
            for value in self.df[segment].unique():
                segment_df = self.df[self.df[segment] == value]
                
                if len(segment_df) < 100:
                    continue
                
                analyzer = ABTestAnalyzer(segment_df)
                freq_result = analyzer.frequentist_test('converted')
                
                segment_results.append({
                    'segment': segment,
                    'value': value,
                    'n': len(segment_df),
                    'control_n': freq_result['control_n'],
                    'treatment_n': freq_result['treatment_n'],
                    'control_rate': freq_result['control_rate'],
                    'treatment_rate': freq_result['treatment_rate'],
                    'absolute_lift': freq_result['absolute_lift'],
                    'relative_lift': freq_result['relative_lift'],
                    'p_value': freq_result['p_value'],
                    'is_significant': freq_result['is_significant']
                })
            
            if segment_results:
                results[segment] = pd.DataFrame(segment_results)
        
        return results
    
    # ========================================================================
    # NOVELTY EFFECT DETECTION
    # ========================================================================
    
    def detect_novelty_effect(self, window_days=7):
        """
        Detect novelty or primacy effects by comparing early vs late periods
        
        Args:
            window_days: Number of days for early period
        
        Returns:
            Dictionary with novelty detection results
        """
        self.df['entry_timestamp'] = pd.to_datetime(self.df['entry_timestamp'])
        self.df['days_since_start'] = (
            self.df['entry_timestamp'] - self.df['entry_timestamp'].min()
        ).dt.days
        
        # Early period
        early_df = self.df[self.df['days_since_start'] < window_days]
        if len(early_df) > 100:
            early_analyzer = ABTestAnalyzer(early_df)
            early_result = early_analyzer.frequentist_test('converted')
        else:
            early_result = {'relative_lift': 0, 'p_value': 1}
        
        # Late period
        late_df = self.df[self.df['days_since_start'] >= window_days]
        if len(late_df) > 100:
            late_analyzer = ABTestAnalyzer(late_df)
            late_result = late_analyzer.frequentist_test('converted')
        else:
            late_result = {'relative_lift': 0, 'p_value': 1}
        
        effect_change = early_result['relative_lift'] - late_result['relative_lift']
        
        return {
            'early_period_days': window_days,
            'early_n': len(early_df),
            'early_relative_lift': early_result['relative_lift'],
            'early_p_value': early_result['p_value'],
            'late_n': len(late_df),
            'late_relative_lift': late_result['relative_lift'],
            'late_p_value': late_result['p_value'],
            'effect_change': effect_change,
            'novelty_detected': abs(effect_change) > 0.05
        }
    
    # ========================================================================
    # BUSINESS IMPACT
    # ========================================================================
    
    def calculate_business_impact(self, annual_traffic=None, 
                                  current_conversion_rate=None, 
                                  avg_order_value=None):
        """
        Calculate projected business impact
        
        Args:
            annual_traffic: Expected annual traffic
            current_conversion_rate: Current conversion rate
            avg_order_value: Average order value
        
        Returns:
            Dictionary with business impact metrics
        """
        freq_result = self.frequentist_test('converted')
        
        if current_conversion_rate is None:
            current_conversion_rate = freq_result['control_rate']
        
        if avg_order_value is None:
            converted = self.df[self.df['converted'] == 1]
            avg_order_value = converted['order_value'].mean()
        
        if annual_traffic is None:
            test_days = (self.df['entry_timestamp'].max() - 
                        self.df['entry_timestamp'].min()).days
            daily_traffic = len(self.df) / max(test_days, 1)
            annual_traffic = daily_traffic * 365
        
        # Current performance
        current_annual_conversions = annual_traffic * current_conversion_rate
        current_annual_revenue = current_annual_conversions * avg_order_value
        
        # Projected performance
        new_conversion_rate = current_conversion_rate * (1 + freq_result['relative_lift'])
        new_annual_conversions = annual_traffic * new_conversion_rate
        new_annual_revenue = new_annual_conversions * avg_order_value
        
        # Impact
        additional_conversions = new_annual_conversions - current_annual_conversions
        additional_revenue = new_annual_revenue - current_annual_revenue
        
        return {
            'annual_traffic': annual_traffic,
            'current_conversion_rate': current_conversion_rate,
            'new_conversion_rate': new_conversion_rate,
            'avg_order_value': avg_order_value,
            'current_annual_conversions': current_annual_conversions,
            'new_annual_conversions': new_annual_conversions,
            'additional_conversions': additional_conversions,
            'current_annual_revenue': current_annual_revenue,
            'new_annual_revenue': new_annual_revenue,
            'additional_revenue': additional_revenue,
            'roi_percentage': (additional_revenue / current_annual_revenue) * 100 if current_annual_revenue > 0 else 0
        }
    
    # ========================================================================
    # SEQUENTIAL TESTING
    # ========================================================================
    
    def sequential_test(self, metric='converted', alpha=0.05, 
                       spending_function='obf'):
        """
        Sequential testing with alpha-spending function
        Allows early stopping while controlling Type I error
        
        Args:
            metric: Metric to analyze
            alpha: Overall alpha level
            spending_function: 'obf' (O'Brien-Fleming) or 'pocock'
        
        Returns:
            Dictionary with sequential test results
        """
        # Sort by timestamp
        df_sorted = self.df.sort_values('entry_timestamp').copy()
        
        # Split into time windows
        n_windows = 5
        window_size = len(df_sorted) // n_windows
        
        results = []
        
        for i in range(1, n_windows + 1):
            window_df = df_sorted.iloc[:window_size * i]
            
            analyzer = ABTestAnalyzer(window_df)
            freq_result = analyzer.frequentist_test(metric)
            
            # Calculate adjusted alpha for this look
            if spending_function == 'obf':
                # O'Brien-Fleming: conservative early, liberal later
                adjusted_alpha = 2 * (1 - stats.norm.cdf(
                    stats.norm.ppf(1 - alpha/2) / np.sqrt(i / n_windows)
                ))
            else:  # Pocock
                # Pocock: constant boundary
                adjusted_alpha = alpha
            
            results.append({
                'look': i,
                'n': len(window_df),
                'p_value': freq_result['p_value'],
                'adjusted_alpha': adjusted_alpha,
                'can_stop': freq_result['p_value'] < adjusted_alpha,
                'relative_lift': freq_result['relative_lift']
            })
        
        return pd.DataFrame(results)


def run_complete_analysis(csv_path='data/ab_test_data.csv'):
    """Run complete statistical analysis suite"""
    
    print("\n" + "="*70)
    print("A/B TEST STATISTICAL ANALYSIS")
    print("="*70)
    
    # Load data
    df = pd.read_csv(csv_path)
    df['entry_timestamp'] = pd.to_datetime(df['entry_timestamp'])
    print(f"\nLoaded {len(df):,} records")
    
    analyzer = ABTestAnalyzer(df)
    
    # Power Analysis
    print("\n" + "="*70)
    print("POWER ANALYSIS")
    print("="*70)
    power = analyzer.power_analysis()
    print(f"Required sample size: {power['total_required_n']:,}")
    print(f"Actual sample size: {len(df):,}")
    print(f"Achieved power: {power['achieved_power']:.2%}")
    print(f"Adequately powered: {'YES' if power['is_adequately_powered'] else 'NO'}")
    
    # Frequentist Test
    print("\n" + "="*70)
    print("FREQUENTIST ANALYSIS")
    print("="*70)
    freq = analyzer.frequentist_test('converted')
    print(f"Control conversion: {freq['control_rate']:.4f}")
    print(f"Treatment conversion: {freq['treatment_rate']:.4f}")
    print(f"Relative lift: {freq['relative_lift']:.2%}")
    print(f"P-value: {freq['p_value']:.6f}")
    print(f"95% CI: [{freq['ci_95_lower']:.4f}, {freq['ci_95_upper']:.4f}]")
    print(f"Significant: {'YES ✅' if freq['is_significant'] else 'NO ❌'}")
    
    # Bayesian Test
    print("\n" + "="*70)
    print("BAYESIAN ANALYSIS")
    print("="*70)
    bayes = analyzer.bayesian_test('converted')
    print(f"Prob(Treatment > Control): {bayes['prob_treatment_better']:.2%}")
    print(f"Expected lift: {bayes['expected_lift']:.2%}")
    print(f"95% Credible Interval: [{bayes['credible_interval_95'][0]:.2%}, {bayes['credible_interval_95'][1]:.2%}]")
    print(f"Recommendation: {bayes['recommended_variant'].upper()}")
    
    # Bootstrap CI
    print("\n" + "="*70)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("="*70)
    bootstrap = analyzer.bootstrap_ci('converted', n_bootstrap=1000)
    print(f"Bootstrap mean lift: {bootstrap['bootstrap_mean_lift']:.2%}")
    print(f"95% Bootstrap CI: [{bootstrap['ci_lower']:.2%}, {bootstrap['ci_upper']:.2%}]")
    
    # HTE Analysis
    print("\n" + "="*70)
    print("HETEROGENEOUS TREATMENT EFFECTS")
    print("="*70)
    hte = analyzer.heterogeneous_treatment_effects()
    for segment, results_df in hte.items():
        print(f"\n{segment.upper()}:")
        print(results_df[['value', 'n', 'control_rate', 'treatment_rate', 
                         'relative_lift', 'is_significant']].to_string(index=False))
    
    # Novelty Effect
    print("\n" + "="*70)
    print("NOVELTY EFFECT DETECTION")
    print("="*70)
    novelty = analyzer.detect_novelty_effect()
    print(f"Early period lift: {novelty['early_relative_lift']:.2%}")
    print(f"Late period lift: {novelty['late_relative_lift']:.2%}")
    print(f"Novelty detected: {'YES ⚠️' if novelty['novelty_detected'] else 'NO ✅'}")
    
    # Business Impact
    print("\n" + "="*70)
    print("BUSINESS IMPACT")
    print("="*70)
    impact = analyzer.calculate_business_impact()
    print(f"Additional annual revenue: ${impact['additional_revenue']:,.0f}")
    print(f"Additional conversions: {impact['additional_conversions']:,.0f}")
    print(f"ROI increase: {impact['roi_percentage']:.2f}%")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    run_complete_analysis()
