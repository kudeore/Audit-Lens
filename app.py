
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import json
from collections import Counter
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Load datasets
try:
    audits_df = pd.read_csv('audits_dataset.csv')
    issues_df = pd.read_csv('issues_dataset.csv')
    print(f"Loaded {len(audits_df)} audits and {len(issues_df)} issues")
except FileNotFoundError as e:
    print(f"Error loading datasets: {e}")
    audits_df = pd.DataFrame()
    issues_df = pd.DataFrame()

class AuditAnalyzer:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.audit_vectors = None
        self.prepare_audit_vectors()

    def preprocess_text(self, text):
        """Clean and preprocess text for similarity analysis"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def prepare_audit_vectors(self):
        """Prepare TF-IDF vectors for all audit titles"""
        if audits_df.empty:
            return

        # Preprocess audit titles
        processed_titles = [self.preprocess_text(title) for title in audits_df['audit_title']]

        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )

        # Fit and transform audit titles
        self.audit_vectors = self.tfidf_vectorizer.fit_transform(processed_titles)
        print(f"Created TF-IDF vectors with shape: {self.audit_vectors.shape}")

    def find_similar_audits(self, query_title, threshold=0.1):
        """Find audits similar to the query title"""
        if self.tfidf_vectorizer is None or audits_df.empty:
            return []

        # Preprocess query
        processed_query = self.preprocess_text(query_title)

        # Transform query to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([processed_query])

        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, self.audit_vectors).flatten()

        # Find matches above threshold
        similar_indices = np.where(similarities >= threshold)[0]

        # Create results
        results = []
        for idx in similar_indices:
            results.append({
                'audit_id': audits_df.iloc[idx]['audit_id'],
                'audit_title': audits_df.iloc[idx]['audit_title'],
                'audit_findings': audits_df.iloc[idx]['audit_findings'],
                'similarity_score': float(similarities[idx])
            })

        # Sort by similarity score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results

    def cluster_issues(self, audit_titles, n_clusters=None):
        """Cluster issues from matched audits"""
        if issues_df.empty:
            return []

        # Get issues from matched audits
        matched_issues = issues_df[issues_df['audit_title'].isin(audit_titles)]

        if matched_issues.empty:
            return []

        # Combine issue title and description for clustering
        issue_texts = (matched_issues['issue_title'] + ' ' + matched_issues['issue_description']).tolist()
        processed_texts = [self.preprocess_text(text) for text in issue_texts]

        # Create TF-IDF vectors for issues
        issue_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9
        )

        try:
            issue_vectors = issue_vectorizer.fit_transform(processed_texts)

            # Determine optimal number of clusters
            if n_clusters is None:
                n_clusters = min(5, max(2, len(matched_issues) // 3))

            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(issue_vectors)

            # Group issues by cluster
            clustered_issues = {}
            feature_names = issue_vectorizer.get_feature_names_out()

            for cluster_id in range(n_clusters):
                cluster_issues = matched_issues[cluster_labels == cluster_id]

                # Generate theme name from cluster centroid
                centroid = kmeans.cluster_centers_[cluster_id]
                top_features_idx = centroid.argsort()[-5:][::-1]
                top_features = [feature_names[i] for i in top_features_idx]
                theme_name = f"Theme: {' & '.join(top_features[:3]).title()}"

                clustered_issues[theme_name] = {
                    'theme_id': cluster_id,
                    'issue_count': len(cluster_issues),
                    'issues': cluster_issues.to_dict('records')
                }

            return clustered_issues

        except Exception as e:
            print(f"Clustering error: {e}")
            return self.fallback_clustering(matched_issues)

    def fallback_clustering(self, issues):
        """Fallback clustering based on issue titles"""
        clusters = {}
        for _, issue in issues.iterrows():
            # Simple clustering based on first word of issue title
            theme = issue['issue_title'].split()[0] if issue['issue_title'] else 'Other'
            theme_name = f"Theme: {theme} Issues"

            if theme_name not in clusters:
                clusters[theme_name] = {
                    'theme_id': len(clusters),
                    'issue_count': 0,
                    'issues': []
                }

            clusters[theme_name]['issues'].append(issue.to_dict())
            clusters[theme_name]['issue_count'] += 1

        return clusters

# Initialize analyzer
analyzer = AuditAnalyzer()

@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'audits_loaded': len(audits_df),
        'issues_loaded': len(issues_df)
    })

@app.route('/api/search-similar-audits', methods=['POST'])
def search_similar_audits():
    """Find similar audits based on title"""
    try:
        data = request.get_json()

        if not data or 'audit_title' not in data:
            return jsonify({'error': 'audit_title is required'}), 400

        audit_title = data['audit_title']
        threshold = float(data.get('threshold', 0.1))

        if threshold < 0.0 or threshold > 1.0:
            return jsonify({'error': 'threshold must be between 0.0 and 1.0'}), 400

        # Find similar audits
        similar_audits = analyzer.find_similar_audits(audit_title, threshold)

        return jsonify({
            'query_title': audit_title,
            'threshold': threshold,
            'matches_found': len(similar_audits),
            'similar_audits': similar_audits
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-issues', methods=['POST'])
def analyze_issues():
    """Cluster issues from similar audits"""
    try:
        data = request.get_json()

        if not data or 'audit_titles' not in data:
            return jsonify({'error': 'audit_titles list is required'}), 400

        audit_titles = data['audit_titles']
        n_clusters = data.get('n_clusters', None)

        if not isinstance(audit_titles, list) or len(audit_titles) == 0:
            return jsonify({'error': 'audit_titles must be a non-empty list'}), 400

        # Cluster issues
        clustered_issues = analyzer.cluster_issues(audit_titles, n_clusters)

        return jsonify({
            'audit_titles': audit_titles,
            'themes_found': len(clustered_issues),
            'clustered_issues': clustered_issues
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/full-analysis', methods=['POST'])
def full_analysis():
    """Complete analysis: find similar audits and cluster their issues"""
    try:
        data = request.get_json()

        if not data or 'audit_title' not in data:
            return jsonify({'error': 'audit_title is required'}), 400

        audit_title = data['audit_title']
        threshold = float(data.get('threshold', 0.1))
        n_clusters = data.get('n_clusters', None)

        # Step 1: Find similar audits
        similar_audits = analyzer.find_similar_audits(audit_title, threshold)

        if not similar_audits:
            return jsonify({
                'query_title': audit_title,
                'threshold': threshold,
                'similar_audits': [],
                'clustered_issues': {},
                'message': 'No similar audits found with the current threshold'
            })

        # Step 2: Extract audit titles for issue clustering
        audit_titles_for_clustering = [audit['audit_title'] for audit in similar_audits]

        # Step 3: Cluster issues
        clustered_issues = analyzer.cluster_issues(audit_titles_for_clustering, n_clusters)

        return jsonify({
            'query_title': audit_title,
            'threshold': threshold,
            'matches_found': len(similar_audits),
            'themes_found': len(clustered_issues),
            'similar_audits': similar_audits,
            'clustered_issues': clustered_issues
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-theme-details/<theme_name>', methods=['GET'])
def get_theme_details(theme_name):
    """Get detailed issues for a specific theme"""
    try:
        # This would typically be cached from previous analysis
        # For demo purposes, return a message
        return jsonify({
            'theme_name': theme_name,
            'message': 'Use /api/full-analysis endpoint to get complete theme details'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
