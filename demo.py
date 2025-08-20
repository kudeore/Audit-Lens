
#!/usr/bin/env python3
"""
Demo script for Smart Audit Analysis Tool
Tests the main functionality of the Flask API
"""

import requests
import json
import time

# Configuration
API_BASE_URL = "http://localhost:5000/api"

def test_health_check():
    """Test if the API is running"""
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
    """Test the complete audit analysis workflow"""
    print("\n🔍 Testing Full Analysis...")

    # Test data
    test_audit_titles = [
        "Financial Controls Assessment Manufacturing Operations",
        "IT Security Data Privacy Compliance Review", 
        "Procurement Process Vendor Management Audit",
        "Revenue Recognition Sales Process Review"
    ]

    for i, audit_title in enumerate(test_audit_titles, 1):
        print(f"\nTest {i}: {audit_title}")
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
    """Test individual API endpoints"""
    print("\n🧪 Testing Individual Endpoints...")

    # Test similar audits search
    print("\nTesting similarity search...")
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
    print("\nTesting issue clustering...")
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
    """Run all tests"""
    print("🚀 Smart Audit Analysis Tool - API Demo")
    print("=" * 60)

    # Test API connectivity
    if not test_health_check():
        print("\n💡 Make sure to start the Flask application first:")
        print("   python app.py")
        return

    # Test full analysis workflow
    test_full_analysis()

    # Test individual endpoints
    test_individual_endpoints()

    print("\n✨ Demo completed!")
    print("\n💡 To start the web interface:")
    print("   1. Run: python app.py")
    print("   2. Open: http://localhost:5000")

if __name__ == "__main__":
    main()
