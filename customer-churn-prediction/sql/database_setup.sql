-- Database Setup for Customer Churn Analysis
-- This script sets up the database and creates tables for the Telco Customer Churn dataset

-- Create database (uncomment based on your DB system)
-- CREATE DATABASE customer_churn_db;
-- USE customer_churn_db;

-- Drop table if exists
DROP TABLE IF EXISTS telco_customers;

-- Create the main customer table
CREATE TABLE telco_customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    gender VARCHAR(10),
    senior_citizen INT,
    partner VARCHAR(5),
    dependents VARCHAR(5),
    tenure INT,
    phone_service VARCHAR(5),
    multiple_lines VARCHAR(20),
    internet_service VARCHAR(20),
    online_security VARCHAR(20),
    online_backup VARCHAR(20),
    device_protection VARCHAR(20),
    tech_support VARCHAR(20),
    streaming_tv VARCHAR(20),
    streaming_movies VARCHAR(20),
    contract VARCHAR(20),
    paperless_billing VARCHAR(5),
    payment_method VARCHAR(30),
    monthly_charges DECIMAL(8,2),
    total_charges VARCHAR(20), -- Initially as VARCHAR due to empty strings
    churn VARCHAR(5)
);

-- Create indexes for better query performance
CREATE INDEX idx_churn ON telco_customers(churn);
CREATE INDEX idx_contract ON telco_customers(contract);
CREATE INDEX idx_tenure ON telco_customers(tenure);
CREATE INDEX idx_internet_service ON telco_customers(internet_service);
CREATE INDEX idx_tech_support ON telco_customers(tech_support);

-- Create a view for cleaned data (converting total_charges to numeric)
CREATE VIEW telco_customers_clean AS
SELECT 
    customer_id,
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    CASE 
        WHEN total_charges = '' OR total_charges IS NULL THEN 0
        ELSE CAST(total_charges AS DECIMAL(8,2))
    END AS total_charges_numeric,
    churn
FROM telco_customers
WHERE total_charges != '' AND total_charges IS NOT NULL;