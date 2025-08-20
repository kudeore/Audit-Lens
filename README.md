# Smart Audit Analysis Tool

An AI-powered full-stack web application for intelligent audit matching and issue clustering using machine learning algorithms.

## 🚀 Features

- **Intelligent Audit Matching**: Uses TF-IDF vectorization and cosine similarity to find historically similar audits
- **Advanced Issue Clustering**: K-means clustering algorithm groups related issues by themes
- **Dynamic Theme Generation**: Automatically generates meaningful theme names from cluster analysis
- **Adjustable Similarity Threshold**: Fine-tune matching sensitivity (0.1 to 1.0)
- **Professional UI**: Modern, responsive design built for business professionals
- **Real-time Analysis**: Fast processing with comprehensive result visualization

## 🏗️ Architecture

### Backend (Python Flask)
- **Flask API**: RESTful endpoints for audit analysis
- **scikit-learn**: Machine learning algorithms (TF-IDF, cosine similarity, K-means clustering)
- **Pandas**: Data processing and manipulation
- **CORS Support**: Cross-origin resource sharing for frontend-backend communication

### Frontend (HTML/CSS/JavaScript)
- **Responsive Design**: Modern UI with professional styling
- **Interactive Components**: Dynamic forms, sliders, and modal dialogs
- **Real-time Updates**: AJAX calls to backend API
- **Data Visualization**: Clean presentation of analysis results

## 📦 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Data Files
The application includes sample datasets:
- `audits_dataset.csv` - 50 sample audit records
- `issues_dataset.csv` - 220+ sample issues across different audit types

**To use your own data:**
1. Replace `audits_dataset.csv` with your audit data (columns: audit_id, audit_title, audit_findings)
2. Replace `issues_dataset.csv` with your issues data (columns: issue_id, audit_title, issue_title, issue_description, severity, status, assigned_to, created_date)

### Step 3: Run the Application
```bash
python app.py
```

The application will start on `http://localhost:5000`

## 📊 API Endpoints

### Health Check
```
GET /api/health
```

### Find Similar Audits
```
POST /api/search-similar-audits
Content-Type: application/json

{
    "audit_title": "Your audit title here",
    "threshold": 0.3
}
```

### Cluster Issues
```
POST /api/analyze-issues
Content-Type: application/json

{
    "audit_titles": ["Audit Title 1", "Audit Title 2"],
    "n_clusters": 5
}
```

### Full Analysis (Recommended)
```
POST /api/full-analysis
Content-Type: application/json

{
    "audit_title": "Your audit title here",
    "threshold": 0.3,
    "n_clusters": 5
}
```

## 🎯 Usage Guide

### 1. Enter Current Audit Title
- Input your planned audit title in the text area
- Be descriptive for better matching results

### 2. Adjust Similarity Threshold
- **0.1-0.3**: Broad matching (more results, less precise)
- **0.4-0.6**: Balanced matching (recommended)
- **0.7-1.0**: Strict matching (fewer, more precise results)

### 3. Analyze Results
- Review similar historical audits with similarity scores
- Explore clustered issues organized by themes
- Click on themes to see detailed issue breakdowns

## 🔧 Customization

### Adding New Data Sources
To connect to a database instead of CSV files, modify the data loading section in `app.py`:

```python
# Replace CSV loading with database connection
import sqlite3  # or your preferred database connector

def load_data_from_db():
    conn = sqlite3.connect('your_database.db')
    audits_df = pd.read_sql_query("SELECT * FROM audits", conn)
    issues_df = pd.read_sql_query("SELECT * FROM issues", conn)
    conn.close()
    return audits_df, issues_df
```

### Modifying Clustering Parameters
Adjust clustering behavior in the `AuditAnalyzer` class:

```python
# In cluster_issues method
kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10,
    max_iter=300  # Add more parameters as needed
)
```

### Customizing TF-IDF Vectorization
Modify text analysis parameters:

```python
self.tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,      # Increase for more detailed analysis
    stop_words='english',   # Add custom stop words
    ngram_range=(1, 3),     # Include trigrams
    min_df=2,              # Minimum document frequency
    max_df=0.8             # Maximum document frequency
)
```

## 🛠️ Technical Details

### Machine Learning Algorithms Used

1. **TF-IDF Vectorization**
   - Converts text to numerical vectors
   - Handles term frequency and inverse document frequency
   - Supports n-grams for better context understanding

2. **Cosine Similarity**
   - Measures similarity between audit titles
   - Range: 0 (completely different) to 1 (identical)
   - Efficient for high-dimensional sparse vectors

3. **K-Means Clustering**
   - Groups similar issues together
   - Automatically determines themes from cluster centroids
   - Configurable number of clusters

### Performance Considerations

- **TF-IDF Vectors**: Cached for quick similarity calculations
- **Preprocessing**: Text normalization and stop word removal
- **Clustering**: Optimized for small to medium datasets (up to 10,000 issues)

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Change port in app.py
   app.run(debug=True, host='0.0.0.0', port=5001)
   ```

2. **CORS Errors**
   - Ensure Flask-CORS is installed
   - Check that the API URL matches your backend address

3. **Empty Results**
   - Try lowering the similarity threshold
   - Verify your audit title contains relevant keywords
   - Check that your data files are properly formatted

4. **Clustering Errors**
   - Ensure you have enough issues for clustering (minimum 3-5)
   - Check for empty or null values in issue descriptions

## 📁 Project Structure

```
smart-audit-analyzer/
├── app.py                    # Flask backend application
├── requirements.txt          # Python dependencies
├── audits_dataset.csv       # Sample audit data
├── issues_dataset.csv       # Sample issues data
├── templates/
│   └── index.html           # Frontend HTML template
├── static/
│   └── style.css           # CSS styling
└── README.md               # This file
```

## 🔄 Future Enhancements

- [ ] Database integration (PostgreSQL, MySQL)
- [ ] User authentication and session management
- [ ] Advanced visualization with charts and graphs
- [ ] Export functionality (PDF reports, Excel files)
- [ ] Real-time collaboration features
- [ ] Machine learning model training interface
- [ ] Advanced text preprocessing (stemming, lemmatization)
- [ ] Integration with enterprise audit management systems

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the troubleshooting section above

---

**Built with ❤️ for the audit and compliance community**
