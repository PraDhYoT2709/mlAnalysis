-- Business Analysis Queries for Customer Churn Prediction
-- These queries provide insights into customer behavior and churn patterns

-- 1. Overall Churn Rate
SELECT 
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_percent
FROM telco_customers_clean;

-- 2. Churn Rate by Contract Type
SELECT 
    contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_percent
FROM telco_customers_clean
GROUP BY contract
ORDER BY churn_rate_percent DESC;

-- 3. Average Tenure: Churned vs Non-Churned Customers
SELECT 
    churn,
    COUNT(*) AS customer_count,
    ROUND(AVG(tenure), 2) AS avg_tenure_months,
    MIN(tenure) AS min_tenure,
    MAX(tenure) AS max_tenure
FROM telco_customers_clean
GROUP BY churn;

-- 4. Customer Segmentation by Payment Frequency and Churn
SELECT 
    CASE 
        WHEN contract = 'Month-to-month' THEN 'Monthly'
        WHEN contract = 'One year' THEN 'Annual'
        WHEN contract = 'Two year' THEN 'Biennial'
        ELSE 'Other'
    END AS payment_frequency,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_percent,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(AVG(total_charges_numeric), 2) AS avg_total_charges
FROM telco_customers_clean
GROUP BY payment_frequency
ORDER BY churn_rate_percent DESC;

-- 5. Tech Support Impact on Churn
SELECT 
    tech_support,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_percent
FROM telco_customers_clean
WHERE internet_service != 'No'  -- Only customers with internet service
GROUP BY tech_support
ORDER BY churn_rate_percent DESC;

-- 6. Internet Service Type Analysis
SELECT 
    internet_service,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_percent,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM telco_customers_clean
GROUP BY internet_service
ORDER BY churn_rate_percent DESC;

-- 7. Senior Citizens Churn Analysis
SELECT 
    CASE WHEN senior_citizen = 1 THEN 'Senior' ELSE 'Non-Senior' END AS customer_type,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_percent,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM telco_customers_clean
GROUP BY senior_citizen
ORDER BY churn_rate_percent DESC;

-- 8. Payment Method Impact on Churn
SELECT 
    payment_method,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_percent
FROM telco_customers_clean
GROUP BY payment_method
ORDER BY churn_rate_percent DESC;

-- 9. Monthly Charges Distribution by Churn Status
SELECT 
    churn,
    COUNT(*) AS customer_count,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(MIN(monthly_charges), 2) AS min_monthly_charges,
    ROUND(MAX(monthly_charges), 2) AS max_monthly_charges,
    ROUND(AVG(total_charges_numeric), 2) AS avg_total_charges
FROM telco_customers_clean
GROUP BY churn;

-- 10. High-Risk Customer Segments (Multiple Risk Factors)
SELECT 
    COUNT(*) AS high_risk_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_high_risk,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_percent
FROM telco_customers_clean
WHERE contract = 'Month-to-month'
    AND tech_support = 'No'
    AND internet_service = 'Fiber optic'
    AND tenure <= 12;

-- 11. Paperless Billing and Churn Correlation
SELECT 
    paperless_billing,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_percent
FROM telco_customers_clean
GROUP BY paperless_billing
ORDER BY churn_rate_percent DESC;

-- 12. Tenure Buckets Analysis
SELECT 
    CASE 
        WHEN tenure <= 12 THEN '0-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 36 THEN '25-36 months'
        WHEN tenure <= 48 THEN '37-48 months'
        ELSE '48+ months'
    END AS tenure_bucket,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_percent,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM telco_customers_clean
GROUP BY tenure_bucket
ORDER BY churn_rate_percent DESC;