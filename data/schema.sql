-- A/B Testing Framework Database Schema
-- Compatible with both PostgreSQL and SQLite

-- Experiments table - stores metadata about each experiment
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name VARCHAR(100) NOT NULL,
    description TEXT,
    hypothesis TEXT,
    start_date DATE NOT NULL,
    end_date DATE,
    status VARCHAR(20) DEFAULT 'draft',  -- draft, running, completed, stopped
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Experiment metrics - defines success metrics for each experiment
CREATE TABLE IF NOT EXISTS experiment_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(20) NOT NULL,  -- primary, secondary, guardrail
    description TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

-- Experiment users - tracks user assignments to variants
CREATE TABLE IF NOT EXISTS experiment_users (
    user_id VARCHAR(50) PRIMARY KEY,
    experiment_id INTEGER NOT NULL,
    variant VARCHAR(20) NOT NULL,  -- control, treatment, treatment_b, etc.
    assignment_timestamp TIMESTAMP NOT NULL,
    device_type VARCHAR(20),
    user_segment VARCHAR(20),
    location VARCHAR(100),
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

-- Events table - tracks user interactions
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(50) NOT NULL,
    event_type VARCHAR(50) NOT NULL,  -- page_view, click, add_to_cart, checkout_start, etc.
    event_timestamp TIMESTAMP NOT NULL,
    event_properties TEXT,  -- JSON string of additional properties
    FOREIGN KEY (user_id) REFERENCES experiment_users(user_id)
);

-- Conversions table - tracks conversion events
CREATE TABLE IF NOT EXISTS conversions (
    conversion_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(50) NOT NULL,
    conversion_timestamp TIMESTAMP NOT NULL,
    order_value DECIMAL(10, 2),
    order_id VARCHAR(50),
    FOREIGN KEY (user_id) REFERENCES experiment_users(user_id)
);

-- Session metrics - engagement and guardrail metrics
CREATE TABLE IF NOT EXISTS session_metrics (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(50) NOT NULL,
    session_timestamp TIMESTAMP NOT NULL,
    time_on_page_sec DECIMAL(10, 2),
    pages_viewed INTEGER,
    bounced INTEGER,  -- 0 or 1
    FOREIGN KEY (user_id) REFERENCES experiment_users(user_id)
);

-- Experiment results - stores computed statistical results
CREATE TABLE IF NOT EXISTS experiment_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    variant VARCHAR(20) NOT NULL,
    sample_size INTEGER,
    mean_value DECIMAL(10, 6),
    std_value DECIMAL(10, 6),
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

-- Create indices for performance optimization
CREATE INDEX IF NOT EXISTS idx_experiment_users_variant ON experiment_users(variant);
CREATE INDEX IF NOT EXISTS idx_experiment_users_experiment ON experiment_users(experiment_id);
CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(event_timestamp);
CREATE INDEX IF NOT EXISTS idx_conversions_user ON conversions(user_id);
CREATE INDEX IF NOT EXISTS idx_session_metrics_user ON session_metrics(user_id);

-- Create views for common queries

-- View: Conversion rates by variant
CREATE VIEW IF NOT EXISTS v_conversion_rates AS
SELECT 
    e.experiment_id,
    e.experiment_name,
    eu.variant,
    COUNT(DISTINCT eu.user_id) as total_users,
    COUNT(DISTINCT c.user_id) as converted_users,
    CAST(COUNT(DISTINCT c.user_id) AS FLOAT) / COUNT(DISTINCT eu.user_id) as conversion_rate,
    AVG(c.order_value) as avg_order_value,
    SUM(c.order_value) as total_revenue
FROM experiments e
JOIN experiment_users eu ON e.experiment_id = eu.experiment_id
LEFT JOIN conversions c ON eu.user_id = c.user_id
GROUP BY e.experiment_id, e.experiment_name, eu.variant;

-- View: Session metrics by variant
CREATE VIEW IF NOT EXISTS v_session_metrics AS
SELECT 
    e.experiment_id,
    e.experiment_name,
    eu.variant,
    COUNT(DISTINCT eu.user_id) as total_users,
    AVG(sm.time_on_page_sec) as avg_time_on_page,
    AVG(sm.pages_viewed) as avg_pages_viewed,
    AVG(sm.bounced) as bounce_rate
FROM experiments e
JOIN experiment_users eu ON e.experiment_id = eu.experiment_id
LEFT JOIN session_metrics sm ON eu.user_id = sm.user_id
GROUP BY e.experiment_id, e.experiment_name, eu.variant;

-- View: Heterogeneous treatment effects by device type
CREATE VIEW IF NOT EXISTS v_hte_by_device AS
SELECT 
    e.experiment_id,
    e.experiment_name,
    eu.variant,
    eu.device_type,
    COUNT(DISTINCT eu.user_id) as total_users,
    COUNT(DISTINCT c.user_id) as converted_users,
    CAST(COUNT(DISTINCT c.user_id) AS FLOAT) / COUNT(DISTINCT eu.user_id) as conversion_rate
FROM experiments e
JOIN experiment_users eu ON e.experiment_id = eu.experiment_id
LEFT JOIN conversions c ON eu.user_id = c.user_id
GROUP BY e.experiment_id, e.experiment_name, eu.variant, eu.device_type;

-- View: Heterogeneous treatment effects by user segment
CREATE VIEW IF NOT EXISTS v_hte_by_segment AS
SELECT 
    e.experiment_id,
    e.experiment_name,
    eu.variant,
    eu.user_segment,
    COUNT(DISTINCT eu.user_id) as total_users,
    COUNT(DISTINCT c.user_id) as converted_users,
    CAST(COUNT(DISTINCT c.user_id) AS FLOAT) / COUNT(DISTINCT eu.user_id) as conversion_rate
FROM experiments e
JOIN experiment_users eu ON e.experiment_id = eu.experiment_id
LEFT JOIN conversions c ON eu.user_id = c.user_id
GROUP BY e.experiment_id, e.experiment_name, eu.variant, eu.user_segment;