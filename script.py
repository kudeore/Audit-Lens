# Create dummy audit and issues datasets as CSV files
import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Create dummy audits dataset
audits_data = {
    'audit_id': [f'AUD-2024-{str(i).zfill(3)}' for i in range(1, 51)],
    'audit_title': [
        'Financial Controls Assessment Manufacturing Operations',
        'IT Security Data Privacy Compliance Review',
        'Procurement Process Vendor Management Audit',
        'Revenue Recognition Sales Process Review', 
        'Internal Controls Financial Reporting Assessment',
        'Cybersecurity Framework Implementation Review',
        'Inventory Management Warehouse Operations Audit',
        'Human Resources Payroll System Review',
        'Treasury Cash Management Controls Assessment',
        'Regulatory Compliance Risk Management Review',
        'Supply Chain Logistics Operations Audit',
        'Information Technology General Controls Review',
        'Credit Risk Management Assessment',
        'Operational Risk Controls Evaluation',
        'Anti Money Laundering Controls Review',
        'Data Governance Privacy Protection Audit',
        'Business Continuity Disaster Recovery Review',
        'Fraud Risk Assessment Investigation',
        'Environmental Health Safety Compliance Audit',
        'Quality Management System Review',
        'Customer Data Protection Privacy Audit',
        'Third Party Risk Management Assessment',
        'Investment Portfolio Management Review',
        'Insurance Claims Processing Audit',
        'Marketing Campaign Effectiveness Review',
        'Employee Benefits Administration Audit',
        'Fixed Assets Management Review',
        'Accounts Payable Processing Controls',
        'Accounts Receivable Collection Procedures',
        'Budget Planning Financial Controls',
        'Tax Compliance Reporting Review',
        'Contract Management Legal Compliance',
        'Product Development Quality Assurance',
        'Sales Commission Calculation Audit',
        'Expense Management Reimbursement Review',
        'Capital Expenditure Approval Process',
        'Bank Reconciliation Controls Assessment',
        'Petty Cash Management Review',
        'Foreign Exchange Risk Management',
        'Derivative Instruments Trading Controls',
        'Customer Credit Assessment Procedures',
        'Supplier Performance Evaluation Review',
        'Project Management Controls Audit',
        'Research Development Expenditure Review',
        'Intellectual Property Management Assessment',
        'Corporate Governance Compliance Review',
        'Executive Compensation Review Process',
        'Board Meeting Documentation Audit',
        'Shareholder Communications Review',
        'Compliance Training Program Assessment'
    ],
    'audit_findings': [
        'Multiple control weaknesses identified in inventory valuation and revenue timing',
        'Significant gaps in user access controls and data encryption standards',
        'Inadequate vendor screening procedures and contract oversight mechanisms',
        'Revenue cut-off issues and insufficient sales process documentation',
        'Material weaknesses in month-end closing and journal entry controls',
        'Incomplete security policy implementation and incident response gaps',
        'Physical count discrepancies and obsolete inventory write-down issues',
        'Unauthorized payroll access and incomplete termination procedures',
        'Cash handling irregularities and delayed bank reconciliation processes',
        'Non-compliance with industry regulations and inadequate risk assessments',
        'Supplier concentration risks and insufficient logistics monitoring',
        'Change management deficiencies and inadequate IT governance structure',
        'Credit scoring model limitations and insufficient collateral documentation',
        'Operational risk identification gaps and inadequate mitigation strategies',
        'AML transaction monitoring weaknesses and incomplete customer due diligence',
        'Data classification issues and insufficient privacy impact assessments',
        'Business continuity plan gaps and inadequate disaster recovery testing',
        'Fraud detection control weaknesses and insufficient investigation procedures',
        'Environmental compliance gaps and incomplete safety documentation',
        'Quality control process deficiencies and inadequate corrective actions',
        'Customer data handling violations and insufficient consent management',
        'Third party risk assessment gaps and inadequate monitoring procedures',
        'Investment policy violations and insufficient portfolio diversification',
        'Claims processing delays and inadequate documentation requirements',
        'Marketing compliance issues and insufficient campaign tracking',
        'Benefits enrollment errors and inadequate employee communication',
        'Asset capitalization issues and incomplete depreciation calculations',
        'Duplicate payment risks and insufficient three-way matching',
        'Collection procedure gaps and inadequate credit limit monitoring',
        'Budget variance analysis weaknesses and insufficient management reporting',
        'Tax calculation errors and incomplete regulatory filing procedures',
        'Contract approval bypasses and insufficient legal review processes',
        'Product testing gaps and inadequate quality documentation',
        'Commission calculation errors and insufficient sales validation',
        'Expense policy violations and inadequate receipt documentation',
        'Capital approval threshold bypasses and insufficient project tracking',
        'Reconciliation timing issues and unexplained variance resolution',
        'Cash custody weaknesses and inadequate petty cash replenishment',
        'FX exposure monitoring gaps and insufficient hedging documentation',
        'Trading limit violations and inadequate derivative valuation',
        'Credit assessment inconsistencies and insufficient financial analysis',
        'Supplier evaluation gaps and inadequate performance metrics',
        'Project milestone tracking issues and insufficient budget controls',
        'R&D cost allocation errors and inadequate project documentation',
        'IP portfolio gaps and insufficient trademark protection',
        'Governance policy violations and insufficient board oversight',
        'Compensation benchmark gaps and insufficient performance linkage',
        'Meeting documentation gaps and insufficient action item tracking',
        'Communication policy violations and insufficient disclosure controls',
        'Training completion gaps and insufficient compliance testing'
    ]
}

# Create audits dataframe and save to CSV
audits_df = pd.DataFrame(audits_data)
audits_df.to_csv('audits_dataset.csv', index=False)

print("Audits dataset created successfully!")
print(f"Shape: {audits_df.shape}")
print("\nFirst 5 rows:")
print(audits_df.head())