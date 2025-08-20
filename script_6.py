# Create a simple test/demo script
demo_script = """
#!/usr/bin/env python3
\"\"\"
Demo script for Smart Audit Analysis Tool
Tests the main functionality of the Flask API
\"\"\"

import requests
import json
import time

# Configuration
API_BASE_URL = "http://localhost:5000/api"

def test_health_check():
    \"\"\"Test if the API is running\"\"\"
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ API Health Check Passed")
            print(f"   Status: {data['status']}")
            print(f"   Audits loaded: {data['audits_loaded']}")
            print(f"   Issues loaded: {data['issues_loaded']}")
            return True
        else:
            print("❌ API Health Check Failed")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure Flask app is running on localhost:5000")
        return False

def test_full_analysis():
    \"\"\"Test the complete audit analysis workflow\"\"\"
    print("\\n🔍 Testing Full Analysis...")
    
    # Test data
    test_audit_titles = [
        "Financial Controls Assessment Manufacturing Operations",
        "IT Security Data Privacy Compliance Review", 
        "Procurement Process Vendor Management Audit",
        "Revenue Recognition Sales Process Review"
    ]
    
    for i, audit_title in enumerate(test_audit_titles, 1):
        print(f"\\nTest {i}: {audit_title}")
        print("-" * 60)
        
        payload = {
            "audit_title": audit_title,
            "threshold": 0.3,
            "n_clusters": 4
        }
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/full-analysis",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Analysis successful")
                print(f"   Similar audits found: {data['matches_found']}")
                print(f"   Issue themes identified: {data['themes_found']}")
                
                # Show top similar audit
                if data['similar_audits']:
                    top_match = data['similar_audits'][0]
                    print(f"   Top match: {top_match['audit_id']} ({top_match['similarity_score']:.2f} similarity)")
                
                # Show themes
                if data['clustered_issues']:
                    print("   Themes found:")
                    for theme_name, theme_data in data['clustered_issues'].items():
                        print(f"     - {theme_name}: {theme_data['issue_count']} issues")
                        
            else:
                print(f"❌ Analysis failed: {response.json().get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Request failed: {str(e)}")
        
        # Add delay between requests
        time.sleep(1)

def test_individual_endpoints():
    \"\"\"Test individual API endpoints\"\"\"
    print("\\n🧪 Testing Individual Endpoints...")
    
    # Test similar audits search
    print("\\nTesting similarity search...")
    payload = {
        "audit_title": "Financial Controls Assessment Manufacturing Operations",
        "threshold": 0.2
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/search-similar-audits",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Similarity search successful: {data['matches_found']} matches")
        else:
            print(f"❌ Similarity search failed: {response.json().get('error')}")
            
    except Exception as e:
        print(f"❌ Similarity search request failed: {str(e)}")
    
    # Test issue clustering
    print("\\nTesting issue clustering...")
    payload = {
        "audit_titles": [
            "Financial Controls Assessment Manufacturing Operations",
            "IT Security Data Privacy Compliance Review"
        ],
        "n_clusters": 3
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze-issues",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Issue clustering successful: {data['themes_found']} themes")
        else:
            print(f"❌ Issue clustering failed: {response.json().get('error')}")
            
    except Exception as e:
        print(f"❌ Issue clustering request failed: {str(e)}")

def main():
    \"\"\"Run all tests\"\"\"
    print("🚀 Smart Audit Analysis Tool - API Demo")
    print("=" * 60)
    
    # Test API connectivity
    if not test_health_check():
        print("\\n💡 Make sure to start the Flask application first:")
        print("   python app.py")
        return
    
    # Test full analysis workflow
    test_full_analysis()
    
    # Test individual endpoints
    test_individual_endpoints()
    
    print("\\n✨ Demo completed!")
    print("\\n💡 To start the web interface:")
    print("   1. Run: python app.py")
    print("   2. Open: http://localhost:5000")

if __name__ == "__main__":
    main()
"""

# Create project structure summary
structure_summary = """
# 📁 Smart Audit Analysis Tool - Complete File Structure

## Generated Files Overview

### Core Application Files
```
📂 smart-audit-analyzer/
├── 🐍 app.py                    # Flask backend with ML algorithms
├── 📋 requirements.txt          # Python dependencies  
├── 📖 README.md                 # Complete setup guide
├── 🧪 demo.py                   # API testing script
├── 📊 audits_dataset.csv        # Sample audit data (50 records)
├── 📋 issues_dataset.csv        # Sample issues data (220+ records)
├── 📂 templates/
│   └── 🌐 index.html           # Frontend HTML interface
└── 📂 static/
    └── 🎨 style.css            # Professional CSS styling
```

## Key Features Implemented

### 🧠 Backend Intelligence (app.py)
✅ **TF-IDF Vectorization** - Text to numerical conversion  
✅ **Cosine Similarity** - Audit matching algorithm  
✅ **K-Means Clustering** - Issue theme identification  
✅ **Dynamic Theme Naming** - Auto-generated cluster names  
✅ **RESTful API** - Clean endpoint structure  
✅ **Error Handling** - Robust exception management  
✅ **Data Preprocessing** - Text cleaning and normalization  

### 🎨 Frontend Experience (HTML/CSS/JS)
✅ **Responsive Design** - Mobile-friendly interface  
✅ **Interactive Slider** - Adjustable similarity threshold  
✅ **Real-time Updates** - Dynamic result rendering  
✅ **Modal Dialogs** - Detailed issue exploration  
✅ **Professional UI** - Business-grade styling  
✅ **Loading States** - Smooth user feedback  
✅ **Error Handling** - User-friendly error messages  

### 📊 Sample Data
✅ **50 Audit Records** - Realistic audit scenarios  
✅ **220+ Issues** - Comprehensive issue database  
✅ **Multiple Categories** - Various audit types covered  
✅ **CSV Format** - Easy to replace with your data  

## 🚀 Quick Start Commands

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Backend
```bash
python app.py
```

### 3. Open Frontend
```
http://localhost:5000
```

### 4. Test API (Optional)
```bash
python demo.py
```

## 🔧 Customization Points

### Replace Sample Data
- Update `audits_dataset.csv` with your audit data
- Update `issues_dataset.csv` with your issues
- Columns must match the expected format

### Database Integration  
- Modify data loading in `app.py`
- Replace CSV reading with SQL queries
- Maintain the same DataFrame structure

### ML Parameter Tuning
- Adjust TF-IDF parameters in `AuditAnalyzer` class
- Modify clustering parameters in `cluster_issues` method
- Customize text preprocessing steps

## 📈 API Endpoints Summary

| Endpoint | Method | Purpose |
|----------|---------|---------|
| `/api/health` | GET | Check API status |
| `/api/search-similar-audits` | POST | Find similar audits |
| `/api/analyze-issues` | POST | Cluster issues by theme |
| `/api/full-analysis` | POST | Complete analysis workflow |

## 🎯 Business Value

### For Audit Professionals
- **Historical Intelligence** - Learn from past audits
- **Risk Pattern Recognition** - Identify recurring issues  
- **Efficient Planning** - Focus on known problem areas
- **Knowledge Transfer** - Leverage institutional memory

### For IT Teams
- **Scalable Architecture** - Easy to extend and modify
- **Modern Tech Stack** - Flask, scikit-learn, responsive design
- **API-First Design** - Integration-ready structure
- **Maintainable Code** - Clear separation of concerns
"""

# Save files
with open('demo.py', 'w') as f:
    f.write(demo_script)

with open('PROJECT_STRUCTURE.md', 'w') as f:
    f.write(structure_summary)

print("Demo script and project structure created!")
print("Files created:")
print("- demo.py (API testing script)")
print("- PROJECT_STRUCTURE.md (Complete overview)")

# Display final summary
print("\n" + "="*60)
print("🎉 SMART AUDIT ANALYSIS TOOL - COMPLETE!")
print("="*60)
print("\n📦 Files Generated:")
files_generated = [
    "app.py (Flask Backend with ML)",
    "templates/index.html (Frontend Interface)", 
    "static/style.css (Professional Styling)",
    "audits_dataset.csv (50 Sample Audits)",
    "issues_dataset.csv (220+ Sample Issues)",
    "requirements.txt (Dependencies)",
    "README.md (Setup Guide)",
    "demo.py (API Testing Script)",
    "PROJECT_STRUCTURE.md (Overview)"
]

for i, file in enumerate(files_generated, 1):
    print(f"{i:2}. ✅ {file}")

print(f"\n📊 Dataset Summary:")
print(f"   • {len(audits_df)} audit records across diverse categories")
print(f"   • {len(issues_df)} issues with severity levels and assignments") 
print(f"   • Ready-to-use CSV format for easy data replacement")

print(f"\n🚀 To Start:")
print(f"   1. pip install -r requirements.txt")
print(f"   2. python app.py") 
print(f"   3. Open http://localhost:5000")

print(f"\n🧪 To Test API:")
print(f"   python demo.py")