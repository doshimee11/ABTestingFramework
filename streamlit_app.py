"""
Interactive A/B Test Analysis Dashboard
Streamlit Application with configurable experiment parameters
"""

import os
import sys
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.statistical_test import ABTestAnalyzer


# Page configuration
st.set_page_config(
    page_title="A/B Test Analysis Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .control-color { color: #636EFA; }
    .treatment-color { color: #EF553B; }
    .success-box {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-radius: 10px;
        padding: 1rem;
        border: 2px solid #28a745;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border-radius: 10px;
        padding: 1rem;
        border: 2px solid #ffc107;
    }
    .danger-box {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-radius: 10px;
        padding: 1rem;
        border: 2px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


def generate_synthetic_data(n_users, control_rate, treatment_rate,
                            control_aov, treatment_aov, test_days, seed=42):
    """Generate synthetic A/B test data with configurable parameters."""
    np.random.seed(seed)

    data = []
    start_date = datetime.now() - timedelta(days=test_days)

    device_probs = {'mobile': 0.5, 'desktop': 0.4, 'tablet': 0.1}
    segment_probs = {'new': 0.4, 'returning': 0.4, 'loyal': 0.2}

    for i in range(n_users):
        user_id = f'user_{i:06d}'
        variant = np.random.choice(['control', 'treatment'], p=[0.5, 0.5])

        # Entry timestamp
        days_offset = np.random.randint(0, test_days)
        hours_offset = np.random.randint(0, 24)
        entry_timestamp = start_date + timedelta(days=days_offset, hours=hours_offset)

        # User characteristics
        device_type = np.random.choice(
            list(device_probs.keys()),
            p=list(device_probs.values())
        )
        user_segment = np.random.choice(
            list(segment_probs.keys()),
            p=list(segment_probs.values())
        )

        # Conversion probability with HTE
        if variant == 'control':
            base_rate = control_rate
            mean_aov = control_aov
        else:
            base_rate = treatment_rate
            mean_aov = treatment_aov

        # Apply heterogeneous effects
        if variant == 'treatment' and device_type == 'mobile':
            base_rate *= 1.10  # 10% boost on mobile
        if user_segment == 'new':
            base_rate *= 0.85  # New users convert less

        # Add noise
        conversion_rate = base_rate * np.random.uniform(0.95, 1.05)
        conversion_rate = min(max(conversion_rate, 0), 1)

        converted = int(np.random.random() < conversion_rate)

        if converted:
            order_value = max(10, np.random.normal(mean_aov, 25))
            order_id = f'ORD_{i:08d}'
        else:
            order_value = 0
            order_id = None

        # Engagement metrics
        time_on_page = np.random.lognormal(mean=3.5, sigma=0.8)
        pages_viewed = max(1, np.random.poisson(lam=3.2))
        bounced = int(np.random.random() < 0.35)

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

    return pd.DataFrame(data)


@st.cache_data
def load_csv_data(filepath):
    """Load and cache CSV data."""
    df = pd.read_csv(filepath)
    df['entry_timestamp'] = pd.to_datetime(df['entry_timestamp'])
    return df


def run_analysis(df, alpha=0.05, baseline_rate=0.08, mde=0.15, power=0.80,
                 annual_traffic=None, avg_order_value=None):
    """Run complete A/B test analysis."""
    analyzer = ABTestAnalyzer(df)

    freq_results = analyzer.frequentist_test('converted', alpha=alpha)
    bayes_results = analyzer.bayesian_test('converted')
    power_results = analyzer.power_analysis(
        baseline_rate=baseline_rate, mde=mde, alpha=alpha, power=power
    )
    bootstrap_results = analyzer.bootstrap_ci('converted')
    hte_results = analyzer.heterogeneous_treatment_effects()
    novelty_results = analyzer.detect_novelty_effect()
    impact_results = analyzer.calculate_business_impact(
        annual_traffic=annual_traffic,
        avg_order_value=avg_order_value
    )

    return {
        'frequentist': freq_results,
        'bayesian': bayes_results,
        'power': power_results,
        'bootstrap': bootstrap_results,
        'hte': hte_results,
        'novelty': novelty_results,
        'impact': impact_results
    }


# =============================================================================
# HEADER
# =============================================================================

st.markdown('<h1 class="main-header">🧪 A/B Test Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Statistical analysis for experiment-driven decisions</p>', unsafe_allow_html=True)


# =============================================================================
# SIDEBAR - Configuration
# =============================================================================

st.sidebar.header("📊 Data Source")

data_source = st.sidebar.radio(
    "Select data source:",
    ["Use existing data", "Upload CSV", "Generate synthetic data"]
)

df = None

# Initialize generation variables with defaults
gen_n_users = 50000
gen_test_days = 21
gen_control_rate = 0.08
gen_lift = 15
gen_treatment_rate = 0.092
gen_control_aov = 75
gen_treatment_aov = 78
gen_seed = 42

if data_source == "Use existing data":
    csv_path = 'data/ab_test_data.csv'
    if os.path.exists(csv_path):
        df = load_csv_data(csv_path)
        st.sidebar.success(f"Loaded {len(df):,} records")
    else:
        st.sidebar.error("No existing data found. Generate or upload data.")

elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your A/B test data", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'entry_timestamp' in df.columns:
            df['entry_timestamp'] = pd.to_datetime(df['entry_timestamp'])
        st.sidebar.success(f"Uploaded {len(df):,} records")

        # Validate required columns
        required_cols = ['variant', 'converted']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.sidebar.error(f"Missing required columns: {missing}")
            df = None

elif data_source == "Generate synthetic data":
    st.sidebar.markdown("### Experiment Parameters")

    gen_n_users = st.sidebar.slider(
        "Number of Users",
        min_value=1000,
        max_value=100000,
        value=50000,
        step=1000,
        help="Total users in the experiment"
    )

    gen_test_days = st.sidebar.slider(
        "Test Duration (days)",
        min_value=7,
        max_value=60,
        value=21,
        help="Duration of the experiment"
    )

    st.sidebar.markdown("### Conversion Rates")

    gen_control_rate = st.sidebar.slider(
        "Control Conversion Rate",
        min_value=0.01,
        max_value=0.30,
        value=0.08,
        step=0.01,
        format="%.2f",
        help="Expected conversion rate for control"
    )

    gen_lift = st.sidebar.slider(
        "Expected Treatment Lift (%)",
        min_value=0,
        max_value=50,
        value=15,
        help="Relative lift from treatment"
    )

    gen_treatment_rate = gen_control_rate * (1 + gen_lift / 100)
    st.sidebar.markdown(f"**Treatment Rate:** {gen_treatment_rate:.2%}")

    st.sidebar.markdown("### Order Values")

    gen_control_aov = st.sidebar.number_input(
        "Control AOV ($)",
        min_value=10,
        max_value=500,
        value=75,
        help="Average order value for control"
    )

    gen_treatment_aov = st.sidebar.number_input(
        "Treatment AOV ($)",
        min_value=10,
        max_value=500,
        value=78,
        help="Average order value for treatment"
    )

    gen_seed = st.sidebar.number_input(
        "Random Seed",
        min_value=1,
        max_value=99999,
        value=42,
        help="Set for reproducible results"
    )

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Statistical Settings")

stat_alpha = st.sidebar.slider(
    "Significance Level (α)",
    min_value=0.01,
    max_value=0.10,
    value=0.05,
    step=0.01,
    help="Type I error rate threshold"
)

stat_power = st.sidebar.slider(
    "Target Power (1-β)",
    min_value=0.70,
    max_value=0.95,
    value=0.80,
    step=0.05,
    help="Probability of detecting true effect"
)

stat_mde = st.sidebar.slider(
    "Minimum Detectable Effect (%)",
    min_value=5,
    max_value=30,
    value=15,
    help="Smallest relative lift to detect"
)

st.sidebar.markdown("---")
st.sidebar.header("💰 Business Parameters")

biz_annual_traffic = st.sidebar.number_input(
    "Annual Traffic",
    min_value=10000,
    max_value=100000000,
    value=1000000,
    step=100000,
    help="Expected annual visitors"
)

biz_avg_order = st.sidebar.number_input(
    "Average Order Value ($)",
    min_value=10,
    max_value=1000,
    value=75,
    help="Average order value for revenue calculations"
)


# =============================================================================
# MAIN CONTENT - TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Results Summary",
    "📈 Visualizations",
    "🔍 Segment Analysis",
    "💰 Business Impact",
    "ℹ️ How It Works"
])


# =============================================================================
# TAB 1: Results Summary
# =============================================================================

with tab1:
    st.header("Experiment Results")

    if df is None and data_source == "Generate synthetic data":
        st.info("Configure experiment parameters in the sidebar, then click 'Run Analysis' to generate data and view results.")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🧪 Generate Data & Run Analysis", type="primary", width='stretch'):
                with st.spinner(f"Generating {gen_n_users:,} users and running analysis..."):
                    df = generate_synthetic_data(
                        n_users=gen_n_users,
                        control_rate=gen_control_rate,
                        treatment_rate=gen_treatment_rate,
                        control_aov=gen_control_aov,
                        treatment_aov=gen_treatment_aov,
                        test_days=gen_test_days,
                        seed=gen_seed
                    )
                    st.session_state['df'] = df

                    results = run_analysis(
                        df,
                        alpha=stat_alpha,
                        baseline_rate=gen_control_rate,
                        mde=stat_mde/100,
                        power=stat_power,
                        annual_traffic=biz_annual_traffic,
                        avg_order_value=biz_avg_order
                    )
                    st.session_state['results'] = results
                    st.rerun()

    elif df is None:
        st.info("Select a data source in the sidebar to begin analysis.")

    else:
        # Filters
        st.sidebar.markdown("---")
        st.sidebar.header("🔍 Filters")

        device_filter = st.sidebar.multiselect(
            "Device Type",
            options=df['device_type'].unique() if 'device_type' in df.columns else [],
            default=df['device_type'].unique().tolist() if 'device_type' in df.columns else []
        )

        segment_filter = st.sidebar.multiselect(
            "User Segment",
            options=df['user_segment'].unique() if 'user_segment' in df.columns else [],
            default=df['user_segment'].unique().tolist() if 'user_segment' in df.columns else []
        )

        # Apply filters
        filtered_df = df.copy()
        if 'device_type' in df.columns and device_filter:
            filtered_df = filtered_df[filtered_df['device_type'].isin(device_filter)]
        if 'user_segment' in df.columns and segment_filter:
            filtered_df = filtered_df[filtered_df['user_segment'].isin(segment_filter)]

        st.sidebar.markdown(f"**Records:** {len(filtered_df):,} / {len(df):,}")

        # Run Analysis button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            run_button = st.button(
                "🔬 Run Statistical Analysis",
                type="primary",
                width='stretch'
            )

        if run_button or 'results' in st.session_state:
            if run_button:
                with st.spinner("Running comprehensive statistical analysis..."):
                    results = run_analysis(
                        filtered_df,
                        alpha=stat_alpha,
                        baseline_rate=stat_mde/100 if data_source != "Generate synthetic data" else gen_control_rate,
                        mde=stat_mde/100,
                        power=stat_power,
                        annual_traffic=biz_annual_traffic,
                        avg_order_value=biz_avg_order
                    )
                    st.session_state['results'] = results
                    st.session_state['filtered_df'] = filtered_df

            results = st.session_state.get('results')

            if results:
                freq = results['frequentist']
                bayes = results['bayesian']
                power_res = results['power']

                st.success(f"Analysis complete! ({len(filtered_df):,} records analyzed)")

                st.markdown("---")

                # Summary metrics
                st.subheader("📊 Key Metrics")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #636EFA22, #636EFA11);
                                border-radius: 10px; padding: 1rem; text-align: center;
                                border: 2px solid #636EFA44;">
                        <h4 style="margin: 0; color: #636EFA;">Control</h4>
                        <h2 style="margin: 0.5rem 0;">{freq['control_rate']:.2%}</h2>
                        <p style="margin: 0; font-size: 0.8rem;">n = {freq['control_n']:,}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #EF553B22, #EF553B11);
                                border-radius: 10px; padding: 1rem; text-align: center;
                                border: 2px solid #EF553B44;">
                        <h4 style="margin: 0; color: #EF553B;">Treatment</h4>
                        <h2 style="margin: 0.5rem 0;">{freq['treatment_rate']:.2%}</h2>
                        <p style="margin: 0; font-size: 0.8rem;">n = {freq['treatment_n']:,}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    lift_color = "#28a745" if freq['relative_lift'] > 0 else "#dc3545"
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {lift_color}22, {lift_color}11);
                                border-radius: 10px; padding: 1rem; text-align: center;
                                border: 2px solid {lift_color}44;">
                        <h4 style="margin: 0; color: {lift_color};">Relative Lift</h4>
                        <h2 style="margin: 0.5rem 0;">{freq['relative_lift']:+.2%}</h2>
                        <p style="margin: 0; font-size: 0.8rem;">95% CI shown below</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    sig_color = "#28a745" if freq['is_significant'] else "#6c757d"
                    sig_text = "Significant" if freq['is_significant'] else "Not Significant"
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {sig_color}22, {sig_color}11);
                                border-radius: 10px; padding: 1rem; text-align: center;
                                border: 2px solid {sig_color}44;">
                        <h4 style="margin: 0; color: {sig_color};">P-Value</h4>
                        <h2 style="margin: 0.5rem 0;">{freq['p_value']:.4f}</h2>
                        <p style="margin: 0; font-size: 0.8rem;">{sig_text}</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # Bayesian results
                st.subheader("🎲 Bayesian Analysis")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "P(Treatment > Control)",
                        f"{bayes['prob_treatment_better']:.1%}",
                        help="Probability that treatment is better than control"
                    )

                with col2:
                    st.metric(
                        "Expected Lift",
                        f"{bayes['expected_lift']:.2%}",
                        help="Expected relative improvement"
                    )

                with col3:
                    rec_color = "green" if bayes['recommended_variant'] == 'treatment' else "red" if bayes['recommended_variant'] == 'control' else "orange"
                    st.metric(
                        "Recommendation",
                        bayes['recommended_variant'].upper()
                    )

                st.info(f"""
                **Bayesian Decision Analysis**

                - Risk of choosing treatment: {bayes['risk_choosing_treatment']:.4f}
                - Risk of choosing control: {bayes['risk_choosing_control']:.4f}
                - 95% Credible Interval: [{bayes['credible_interval_95'][0]:.2%}, {bayes['credible_interval_95'][1]:.2%}]
                """)

                st.markdown("---")

                # Power analysis
                with st.expander("⚡ Power Analysis"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Required Sample Size", f"{power_res['total_required_n']:,}")
                        st.metric("Actual Sample Size", f"{len(filtered_df):,}")

                    with col2:
                        st.metric("Achieved Power", f"{power_res['achieved_power']:.2%}")
                        adequacy = "Adequate" if power_res['is_adequately_powered'] else "Underpowered"
                        st.metric("Assessment", adequacy)

                st.markdown("---")

                # Final recommendation
                st.subheader("📋 Recommendation")

                impact = results['impact']

                if freq['is_significant'] and bayes['prob_treatment_better'] > 0.95:
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>✅ SHIP TREATMENT TO 100%</h3>
                        <p><strong>Statistical Evidence:</strong></p>
                        <ul>
                            <li>Frequentist p-value: {freq['p_value']:.4f} (&lt; {stat_alpha})</li>
                            <li>Bayesian confidence: {bayes['prob_treatment_better']:.1%} (&gt; 95%)</li>
                            <li>Relative lift: {freq['relative_lift']:.2%}</li>
                        </ul>
                        <p><strong>Projected Impact:</strong></p>
                        <ul>
                            <li>Additional revenue: ${impact['additional_revenue']:,.0f}/year</li>
                            <li>ROI increase: {impact['roi_percentage']:.2f}%</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                elif freq['is_significant']:
                    st.markdown("""
                    <div class="warning-box">
                        <h3>⚠️ CONSIDER EXTENDED TESTING</h3>
                        <p>Frequentist test shows significance, but Bayesian confidence is below 95%.</p>
                        <p>Consider running the test longer or investigating segment-specific effects.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="danger-box">
                        <h3>❌ DO NOT SHIP - INCONCLUSIVE</h3>
                        <p>Results are not statistically significant. Options:</p>
                        <ol>
                            <li>Extend the test duration</li>
                            <li>Increase sample size</li>
                            <li>Investigate potential implementation issues</li>
                        </ol>
                    </div>
                    """, unsafe_allow_html=True)

                # Download results
                st.markdown("---")
                results_df = pd.DataFrame([{
                    'control_rate': freq['control_rate'],
                    'treatment_rate': freq['treatment_rate'],
                    'relative_lift': freq['relative_lift'],
                    'p_value': freq['p_value'],
                    'is_significant': freq['is_significant'],
                    'bayesian_prob': bayes['prob_treatment_better'],
                    'recommendation': bayes['recommended_variant']
                }])

                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results CSV",
                    csv,
                    "ab_test_results.csv",
                    "text/csv"
                )
        else:
            st.info("Click 'Run Statistical Analysis' to view results.")


# Check for session state data
if 'df' in st.session_state and df is None:
    df = st.session_state['df']
if 'filtered_df' in st.session_state:
    filtered_df = st.session_state['filtered_df']
else:
    filtered_df = df


# =============================================================================
# TAB 2: Visualizations
# =============================================================================

with tab2:
    if 'results' in st.session_state and filtered_df is not None:
        results = st.session_state['results']
        freq = results['frequentist']

        st.subheader("Conversion Rate Comparison")

        col1, col2 = st.columns(2)

        with col1:
            # Bar chart
            fig = go.Figure()

            fig.add_trace(go.Bar(
                name='Control',
                x=['Conversion Rate'],
                y=[freq['control_rate']],
                text=[f"{freq['control_rate']:.2%}"],
                textposition='auto',
                marker_color='#636EFA'
            ))

            fig.add_trace(go.Bar(
                name='Treatment',
                x=['Conversion Rate'],
                y=[freq['treatment_rate']],
                text=[f"{freq['treatment_rate']:.2%}"],
                textposition='auto',
                marker_color='#EF553B'
            ))

            fig.update_layout(
                title='Conversion Rate by Variant',
                yaxis_title='Conversion Rate',
                yaxis_tickformat='.1%',
                barmode='group',
                height=400
            )

            st.plotly_chart(fig, width='stretch')

        with col2:
            # Confidence interval plot
            fig = go.Figure()

            ci_lower = freq['ci_95_lower'] / freq['control_rate'] if freq['control_rate'] > 0 else 0
            ci_upper = freq['ci_95_upper'] / freq['control_rate'] if freq['control_rate'] > 0 else 0

            fig.add_trace(go.Scatter(
                x=[freq['relative_lift']],
                y=['Relative Lift'],
                mode='markers',
                marker=dict(size=15, color='red'),
                name='Point Estimate'
            ))

            fig.add_trace(go.Scatter(
                x=[ci_lower, ci_upper],
                y=['Relative Lift', 'Relative Lift'],
                mode='lines',
                line=dict(color='red', width=3),
                name='95% CI'
            ))

            fig.add_vline(x=0, line_dash="dash", line_color="gray")

            fig.update_layout(
                title='Relative Lift with 95% Confidence Interval',
                xaxis_title='Relative Lift',
                xaxis_tickformat='.1%',
                height=400
            )

            st.plotly_chart(fig, width='stretch')

        st.markdown("---")

        # Time series
        st.subheader("📈 Temporal Analysis")

        if 'entry_timestamp' in filtered_df.columns:
            temp_df = filtered_df.copy()
            temp_df['date'] = pd.to_datetime(temp_df['entry_timestamp']).dt.date
            daily_metrics = temp_df.groupby(['date', 'variant']).agg({
                'converted': ['sum', 'count']
            }).reset_index()
            daily_metrics.columns = ['date', 'variant', 'conversions', 'total']
            daily_metrics['conversion_rate'] = daily_metrics['conversions'] / daily_metrics['total']

            fig = go.Figure()

            for variant, color in [('control', '#636EFA'), ('treatment', '#EF553B')]:
                data = daily_metrics[daily_metrics['variant'] == variant]
                fig.add_trace(go.Scatter(
                    x=data['date'],
                    y=data['conversion_rate'],
                    mode='lines+markers',
                    name=variant.title(),
                    line=dict(color=color)
                ))

            fig.update_layout(
                title='Daily Conversion Rate Trends',
                xaxis_title='Date',
                yaxis_title='Conversion Rate',
                yaxis_tickformat='.1%',
                height=400
            )

            st.plotly_chart(fig, width='stretch')

            # Novelty effect warning
            novelty = results['novelty']
            if novelty['novelty_detected']:
                st.warning(f"""
                **Potential Novelty Effect Detected**

                Early period lift: {novelty['early_relative_lift']:.2%}
                Late period lift: {novelty['late_relative_lift']:.2%}
                Change: {novelty['effect_change']:.2%}
                """)
        else:
            st.info("Timestamp data not available for temporal analysis.")
    else:
        st.info("Run the analysis first to see visualizations")


# =============================================================================
# TAB 3: Segment Analysis
# =============================================================================

with tab3:
    if 'results' in st.session_state and filtered_df is not None:
        results = st.session_state['results']
        hte = results['hte']

        st.header("🔍 Heterogeneous Treatment Effects")
        st.markdown("Analyze how treatment effects vary across different user segments.")

        if hte:
            cols = st.columns(len(hte))

            for idx, (segment_name, segment_df) in enumerate(hte.items()):
                with cols[idx]:
                    st.subheader(segment_name.replace('_', ' ').title())

                    segment_df = segment_df.sort_values('relative_lift', ascending=True)

                    colors = ['#28a745' if sig else '#6c757d' for sig in segment_df['is_significant']]

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=segment_df['value'],
                        x=segment_df['relative_lift'],
                        orientation='h',
                        marker_color=colors,
                        text=[f"{x:.1%}" for x in segment_df['relative_lift']],
                        textposition='auto'
                    ))

                    fig.add_vline(x=0, line_dash="dash", line_color="gray")
                    fig.update_layout(
                        xaxis_title='Relative Lift',
                        xaxis_tickformat='.1%',
                        height=300,
                        showlegend=False
                    )

                    st.plotly_chart(fig, width='stretch')

                    with st.expander("View detailed data"):
                        display_df = segment_df[['value', 'n', 'control_rate', 'treatment_rate', 'relative_lift', 'p_value']].copy()
                        display_df['control_rate'] = display_df['control_rate'].apply(lambda x: f"{x:.2%}")
                        display_df['treatment_rate'] = display_df['treatment_rate'].apply(lambda x: f"{x:.2%}")
                        display_df['relative_lift'] = display_df['relative_lift'].apply(lambda x: f"{x:.2%}")
                        display_df['p_value'] = display_df['p_value'].apply(lambda x: f"{x:.4f}")
                        st.dataframe(display_df, width='stretch', hide_index=True)
        else:
            st.info("No segment data available for HTE analysis.")
    else:
        st.info("Run the analysis first to see segment analysis")


# =============================================================================
# TAB 4: Business Impact
# =============================================================================

with tab4:
    if 'results' in st.session_state:
        results = st.session_state['results']
        impact = results['impact']
        freq = results['frequentist']

        st.header("💰 Business Impact Projection")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Additional Annual Revenue",
                f"${impact['additional_revenue']:,.0f}",
                help="Projected additional revenue if treatment is shipped"
            )

        with col2:
            st.metric(
                "Additional Conversions",
                f"{impact['additional_conversions']:,.0f}",
                help="Additional conversions per year"
            )

        with col3:
            st.metric(
                "ROI Increase",
                f"{impact['roi_percentage']:.2f}%",
                help="Percentage increase in revenue"
            )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Current Performance")
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | Annual Traffic | {impact['annual_traffic']:,.0f} |
            | Conversion Rate | {impact['current_conversion_rate']:.2%} |
            | Annual Conversions | {impact['current_annual_conversions']:,.0f} |
            | Annual Revenue | ${impact['current_annual_revenue']:,.0f} |
            """)

        with col2:
            st.markdown("### Projected Performance")
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | Annual Traffic | {impact['annual_traffic']:,.0f} |
            | Conversion Rate | {impact['new_conversion_rate']:.2%} |
            | Annual Conversions | {impact['new_annual_conversions']:,.0f} |
            | Annual Revenue | ${impact['new_annual_revenue']:,.0f} |
            """)

        st.markdown("---")

        # Revenue comparison chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Current',
            x=['Annual Revenue', 'Annual Conversions'],
            y=[impact['current_annual_revenue'], impact['current_annual_conversions']],
            marker_color='#636EFA'
        ))

        fig.add_trace(go.Bar(
            name='Projected (with Treatment)',
            x=['Annual Revenue', 'Annual Conversions'],
            y=[impact['new_annual_revenue'], impact['new_annual_conversions']],
            marker_color='#EF553B'
        ))

        fig.update_layout(
            title='Revenue & Conversion Comparison',
            barmode='group',
            height=400
        )

        st.plotly_chart(fig, width='stretch')

    else:
        st.info("Run the analysis first to see business impact projections")


# =============================================================================
# TAB 5: How It Works
# =============================================================================

with tab5:
    st.header("How A/B Testing Works")

    st.markdown("""
    ### What is A/B Testing?

    A/B testing (also called split testing) is a method of comparing two versions of a webpage,
    app feature, or other user experience to determine which one performs better. Users are
    randomly assigned to either the **control** (existing version) or **treatment** (new version) group.

    ### Statistical Methods Used

    #### 1. Frequentist Analysis
    - **Two-proportion z-test**: Tests if the difference between conversion rates is statistically significant
    - **P-value**: Probability of observing the results if there's no real difference
    - **Confidence Interval**: Range where the true effect likely lies (95% of the time)

    ```
    Standard Error = sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    Z-statistic = (p2 - p1) / SE
    ```

    #### 2. Bayesian Analysis
    - Uses **Beta-Binomial** conjugate prior
    - Calculates **P(Treatment > Control)** via Monte Carlo simulation
    - Provides intuitive probability statements

    ```
    Posterior ~ Beta(α + conversions, β + non-conversions)
    ```

    #### 3. Power Analysis
    - Calculates required sample size to detect a given effect
    - Uses **Cohen's h** effect size for proportions
    - Ensures test has adequate statistical power (typically 80%)

    ### Key Metrics

    | Metric | Description |
    |--------|-------------|
    | **Conversion Rate** | % of users who complete the target action |
    | **Relative Lift** | % improvement of treatment over control |
    | **P-Value** | Probability of results occurring by chance |
    | **Bayesian Probability** | P(Treatment is better) |
    | **MDE** | Minimum Detectable Effect - smallest lift we can reliably detect |

    ### Heterogeneous Treatment Effects (HTE)

    Treatment effects often vary by user segment. We analyze:
    - **Device type** (mobile vs desktop vs tablet)
    - **User segment** (new vs returning vs loyal)

    This helps identify which users benefit most from the treatment.

    ### Best Practices

    1. **Pre-register** your hypothesis and success metrics
    2. **Run for minimum 2 weeks** to capture weekly patterns
    3. **Don't peek** at results early (use sequential testing if needed)
    4. **Check for SRM** (Sample Ratio Mismatch)
    5. **Monitor guardrail metrics** to catch unintended effects

    ---
    *Framework Modeling Examples - A/B Testing*
    """)


# =============================================================================
# SIDEBAR FOOTER
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About

A/B Test Analysis Dashboard for
experiment-driven product decisions.

**Features:**
- Frequentist & Bayesian analysis
- Heterogeneous treatment effects
- Business impact projections
- Interactive visualizations

---
*Framework Modeling Examples*
""")
