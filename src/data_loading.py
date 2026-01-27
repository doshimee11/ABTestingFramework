"""
ETL Pipeline - Load A/B test data into database
"""

import os
import sys
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATABASE_CONFIG, EXPERIMENT_CONFIG


def create_database_schema():
    """Create database schema from SQL file"""
    print("\n" + "="*70)
    print("CREATING DATABASE SCHEMA")
    print("="*70)
    
    engine = create_engine(DATABASE_CONFIG['database_url'])
    
    # Read schema file
    schema_path = 'data/schema.sql'
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    
    # Execute schema statements
    with engine.connect() as conn:
        statements = [s.strip() for s in schema_sql.split(';') if s.strip()]
        for statement in statements:
            try:
                conn.execute(text(statement))
                conn.commit()
            except Exception as e:
                pass  # Table may already exist
    
    print("✓ Schema created successfully")
    return engine


def load_experiment_metadata(engine):
    """Load experiment metadata"""
    print("\n" + "="*70)
    print("LOADING EXPERIMENT METADATA")
    print("="*70)
    
    experiment_data = pd.DataFrame([{
        'experiment_name': EXPERIMENT_CONFIG['experiment_name'],
        'description': 'Simplified checkout flow vs. standard 3-step checkout',
        'start_date': EXPERIMENT_CONFIG['start_date'],
        'end_date': pd.Timestamp(EXPERIMENT_CONFIG['start_date']) + 
                   pd.Timedelta(days=EXPERIMENT_CONFIG['test_duration_days']),
        'status': 'completed',
        'created_at': datetime.now()
    }])
    
    experiment_data.to_sql('experiments', engine, if_exists='append', index=False)
    
    # Get experiment_id
    with engine.connect() as conn:
        result = conn.execute(text("SELECT experiment_id FROM experiments ORDER BY experiment_id DESC LIMIT 1"))
        experiment_id = result.fetchone()[0]
    
    print(f"✓ Loaded experiment: {EXPERIMENT_CONFIG['experiment_name']}")
    print(f"  Experiment ID: {experiment_id}")
    
    return experiment_id


def load_user_assignments(df, engine, experiment_id):
    """Load user assignments"""
    print("\n" + "="*70)
    print("LOADING USER ASSIGNMENTS")
    print("="*70)
    
    user_data = df[['user_id', 'variant', 'entry_timestamp', 'device_type', 'user_segment']].copy()
    user_data['experiment_id'] = experiment_id
    user_data.rename(columns={'entry_timestamp': 'assignment_timestamp'}, inplace=True)
    
    user_data.to_sql('experiment_users', engine, if_exists='append', index=False)
    
    print(f"✓ Loaded {len(user_data):,} user assignments")
    print(f"  Control: {len(user_data[user_data['variant']=='control']):,}")
    print(f"  Treatment: {len(user_data[user_data['variant']=='treatment']):,}")


def load_conversions(df, engine):
    """Load conversion events"""
    print("\n" + "="*70)
    print("LOADING CONVERSIONS")
    print("="*70)
    
    conversions = df[df['converted'] == 1][['user_id', 'entry_timestamp', 'order_value', 'order_id']].copy()
    conversions.rename(columns={'entry_timestamp': 'conversion_timestamp'}, inplace=True)
    
    conversions.to_sql('conversions', engine, if_exists='append', index=False)
    
    print(f"✓ Loaded {len(conversions):,} conversions")
    print(f"  Total revenue: ${conversions['order_value'].sum():,.2f}")
    print(f"  Average order value: ${conversions['order_value'].mean():.2f}")


def load_session_metrics(df, engine):
    """Load session engagement metrics"""
    print("\n" + "="*70)
    print("LOADING SESSION METRICS")
    print("="*70)
    
    session_data = df[['user_id', 'entry_timestamp', 'time_on_page_sec', 'pages_viewed', 'bounced']].copy()
    session_data.rename(columns={'entry_timestamp': 'session_timestamp'}, inplace=True)
    session_data.to_sql('session_metrics', engine, if_exists='append', index=False)
    
    print(f"✓ Loaded {len(session_data):,} session records")


def load_events(df, engine):
    """Generate and load event stream"""
    print("\n" + "="*70)
    print("GENERATING EVENT STREAM")
    print("="*70)
    
    events = []
    
    for _, row in df.iterrows():
        # Page view event
        events.append({
            'user_id': row['user_id'],
            'event_type': 'checkout_page_view',
            'event_timestamp': row['entry_timestamp'],
            'event_properties': f'{{"variant": "{row["variant"]}", "device": "{row["device_type"]}"}}'
        })
        
        # Checkout started (if not bounced)
        if row['bounced'] == 0:
            events.append({
                'user_id': row['user_id'],
                'event_type': 'checkout_start',
                'event_timestamp': row['entry_timestamp'] + pd.Timedelta(seconds=30),
                'event_properties': f'{{"variant": "{row["variant"]}"}}'
            })
        
        # Purchase event (if converted)
        if row['converted'] == 1:
            events.append({
                'user_id': row['user_id'],
                'event_type': 'purchase',
                'event_timestamp': row['entry_timestamp'] + pd.Timedelta(seconds=120),
                'event_properties': f'{{"order_id": "{row["order_id"]}", "order_value": {row["order_value"]}}}'
            })
    
    events_df = pd.DataFrame(events)
    events_df.to_sql('events', engine, if_exists='append', index=False)
    
    print(f"✓ Loaded {len(events_df):,} events")


def verify_data_load(engine):
    """Verify data was loaded correctly"""
    print("\n" + "="*70)
    print("VERIFYING DATA LOAD")
    print("="*70)
    
    with engine.connect() as conn:
        # Check tables
        result = conn.execute(text("SELECT COUNT(*) FROM experiments"))
        print(f"\nExperiments: {result.fetchone()[0]}")
        
        result = conn.execute(text("SELECT COUNT(*) FROM experiment_users"))
        print(f"Users: {result.fetchone()[0]:,}")
        
        result = conn.execute(text("SELECT COUNT(*) FROM conversions"))
        print(f"Conversions: {result.fetchone()[0]:,}")
        
        result = conn.execute(text("SELECT COUNT(*) FROM events"))
        print(f"Events: {result.fetchone()[0]:,}")
        
        # Check conversion rates
        print("\nConversion rates by variant:")
        result = conn.execute(text("SELECT * FROM v_conversion_rates"))
        for row in result:
            print(f"  {row[2]}: {row[5]:.4f} ({row[4]:,}/{row[3]:,} users)")
    
    print("\n✓ Data load verification complete")


def run_full_etl(csv_path='data/ab_test_data.csv'):
    """Run complete ETL pipeline"""
    print("\n" + "="*70)
    print("STARTING ETL PIPELINE")
    print("="*70)
    
    # Create schema
    engine = create_database_schema()
    
    # Load CSV
    print(f"\nReading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['entry_timestamp'] = pd.to_datetime(df['entry_timestamp'])
    print(f"✓ Loaded {len(df):,} records")
    
    # Load data
    experiment_id = load_experiment_metadata(engine)
    load_user_assignments(df, engine, experiment_id)
    load_conversions(df, engine)
    load_session_metrics(df, engine)
    load_events(df, engine)
    
    # Verify
    verify_data_load(engine)
    
    print("\n" + "="*70)
    print("ETL PIPELINE COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_full_etl()
