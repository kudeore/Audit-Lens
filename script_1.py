# Create dummy issues dataset
issues_data = []

# Issue types and categories for generating realistic issues
issue_categories = [
    'Access Control', 'Data Security', 'Process Control', 'Documentation', 
    'Compliance', 'Risk Management', 'Financial Control', 'IT Security',
    'Vendor Management', 'Inventory Control', 'Revenue Recognition', 
    'Payroll Control', 'Cash Management', 'Regulatory', 'Quality Control'
]

issue_types = [
    'Inadequate', 'Insufficient', 'Weak', 'Missing', 'Outdated', 
    'Incomplete', 'Unauthorized', 'Improper', 'Lack of', 'Poor'
]

# Generate issues for each audit
issue_id = 1
for _, audit in audits_df.iterrows():
    # Generate 3-6 issues per audit
    num_issues = random.randint(3, 6)
    
    for i in range(num_issues):
        issue_title_components = [
            random.choice(issue_types),
            random.choice(issue_categories),
            'Procedures' if random.random() > 0.5 else 'Controls'
        ]
        
        issue_title = ' '.join(issue_title_components)
        
        # Generate realistic issue descriptions
        descriptions = [
            f"Control mechanisms for {random.choice(['user access', 'data processing', 'financial reporting'])} are not functioning as designed",
            f"Policies related to {random.choice(['vendor selection', 'inventory management', 'revenue recording'])} lack proper implementation",
            f"Documentation for {random.choice(['approval processes', 'reconciliation procedures', 'risk assessments'])} is incomplete or missing",
            f"Monitoring activities for {random.choice(['compliance requirements', 'security protocols', 'quality standards'])} are not performed regularly",
            f"Training programs for {random.choice(['staff members', 'management personnel', 'system users'])} are outdated or insufficient",
            f"Segregation of duties in {random.choice(['payment processing', 'data entry', 'approval workflows'])} is not properly maintained",
            f"System configurations for {random.choice(['access controls', 'data backup', 'audit trails'])} do not meet security requirements",
            f"Review procedures for {random.choice(['financial statements', 'operational reports', 'compliance documentation'])} lack management oversight"
        ]
        
        issues_data.append({
            'issue_id': f'ISS-{str(issue_id).zfill(4)}',
            'audit_title': audit['audit_title'],
            'issue_title': issue_title,
            'issue_description': random.choice(descriptions),
            'severity': random.choice(['High', 'Medium', 'Low']),
            'status': random.choice(['Open', 'In Progress', 'Closed']),
            'assigned_to': f'Auditor_{random.randint(1, 10)}',
            'created_date': f'2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}'
        })
        issue_id += 1

# Create issues dataframe and save to CSV
issues_df = pd.DataFrame(issues_data)
issues_df.to_csv('issues_dataset.csv', index=False)

print("Issues dataset created successfully!")
print(f"Shape: {issues_df.shape}")
print(f"Number of unique audits with issues: {issues_df['audit_title'].nunique()}")
print("\nFirst 5 rows:")
print(issues_df.head())