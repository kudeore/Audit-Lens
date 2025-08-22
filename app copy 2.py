# --- ADD/REPLACE these imports at the top of app.py ---
import os
import re
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from datetime import datetime

# --- NEW LANGCHAIN IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

load_dotenv()
app = Flask(__name__)
CORS(app)

# --- LLM Configuration ---
try:
    api_key = os.getenv("GOOGLE_API")
    if not api_key:
        print("Warning: GOOGLE_API environment variable not set.")
except Exception as e:
    print(f"Error accessing environment variables: {e}")

# --- Load Datasets ---
try:
    audits_df = pd.read_csv('audits_dataset.csv')
    issues_df = pd.read_csv('issues_dataset.csv')
    print(f"Loaded {len(audits_df)} audits and {len(issues_df)} issues")
except FileNotFoundError as e:
    print(f"Error loading datasets: {e}")
    audits_df = pd.DataFrame(columns=['audit_id', 'audit_title', 'audit_findings'])
    issues_df = pd.DataFrame(columns=['issue_id', 'audit_title', 'issue_title', 'issue_description'])

# ==============================================================================
# --- Data Analysis and Retrieval Class ---
# ==============================================================================
class AuditAnalyzer:
    """Handles text processing, similarity search, and issue clustering."""
    def __init__(self):
        self.tfidf_vectorizer = None
        self.audit_vectors = None
        self.prepare_audit_vectors()

    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = ' '.join(text.split())
        return text

    def prepare_audit_vectors(self):
        if audits_df.empty:
            print("Audit dataframe is empty. Skipping vector preparation.")
            return
        processed_titles = audits_df['audit_title'].apply(self.preprocess_text)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        self.audit_vectors = self.tfidf_vectorizer.fit_transform(processed_titles)
        print(f"Created TF-IDF vectors with shape: {self.audit_vectors.shape}")

    def find_similar_audits(self, query_title, threshold=0.1, top_n=5):
        if self.tfidf_vectorizer is None or self.audit_vectors is None or audits_df.empty:
            return []
        processed_query = self.preprocess_text(query_title)
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.audit_vectors).flatten()
        similar_indices = np.where(similarities >= threshold)[0]
        if len(similar_indices) == 0:
            return []
        results = [{
            'audit_id': audits_df.iloc[idx]['audit_id'],
            'audit_title': audits_df.iloc[idx]['audit_title'],
            'audit_findings': audits_df.iloc[idx]['audit_findings'],
            'similarity_score': float(similarities[idx])
        } for idx in similar_indices]
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_n]

    def cluster_issues(self, audit_titles, n_clusters=None):
        if issues_df.empty or not audit_titles:
            return []
        matched_issues = issues_df[issues_df['audit_title'].isin(audit_titles)]
        if matched_issues.empty:
            return []
        issue_texts = (matched_issues['issue_title'].fillna('') + ' ' + matched_issues['issue_description'].fillna('')).tolist()
        processed_texts = [self.preprocess_text(text) for text in issue_texts]
        if not any(processed_texts): return []
        issue_vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
        try:
            issue_vectors = issue_vectorizer.fit_transform(processed_texts)
            if issue_vectors.shape[0] < 2: return self.fallback_clustering(matched_issues)
            if n_clusters is None:
                n_clusters = min(5, max(2, len(matched_issues) // 3))
            if issue_vectors.shape < n_clusters:
                n_clusters = issue_vectors.shape
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(issue_vectors)
            clustered_issues = {}
            feature_names = issue_vectorizer.get_feature_names_out()
            for cluster_id in range(n_clusters):
                cluster_mask = (cluster_labels == cluster_id)
                cluster_issues_df = matched_issues[cluster_mask]
                top_features_idx = kmeans.cluster_centers_[cluster_id].argsort()[-5:][::-1]
                top_features = [feature_names[i] for i in top_features_idx]
                theme_name = f"Theme: {' & '.join(top_features[:3]).title()}"
                clustered_issues[theme_name] = {
                    'theme_id': cluster_id,
                    'issue_count': len(cluster_issues_df),
                    'issues': cluster_issues_df.to_dict('records')
                }
            return clustered_issues
        except Exception as e:
            print(f"Clustering error: {e}")
            return self.fallback_clustering(matched_issues)

    def fallback_clustering(self, issues):
        clusters = {}
        for _, issue in issues.iterrows():
            theme = issue['issue_title'].split()[0] if issue['issue_title'] else 'General'
            theme_name = f"Theme: {theme}"
            if theme_name not in clusters:
                clusters[theme_name] = {'theme_id': len(clusters), 'issue_count': 0, 'issues': []}
            clusters[theme_name]['issues'].append(issue.to_dict())
            clusters[theme_name]['issue_count'] += 1
        return clusters

# ==============================================================================
# --- Pydantic Data Models for Structured LLM Output (MODIFIED) ---
# ==============================================================================

class SubStep(BaseModel):
    sub_step: str = Field(description="A granular action within the main process step.")
    root_cause_analysis: List[str] = Field(description="A list of 'Why?' questions and answers exploring the purpose of this sub-step.")
    identified_risks: List[str] = Field(description="A list of potential risks associated with this sub-step.")
    leading_questions: List[str] = Field(description="A list of probing questions for the user to consider for this sub-step.")

class ProcessStep(BaseModel):
    step_number: str = Field(description="The sequential number of the main process step (e.g., '1', '2').")
    step_description: str = Field(description="A high-level description of the main process step.")
    sub_steps: List[SubStep] = Field(description="A list of detailed sub-steps that make up this main step.")

class AuditAnalysis(BaseModel):
    intro: str = Field(description="An introductory statement from the AI Principal Auditor persona.")
    process_breakdown: List[ProcessStep] = Field(description="A list of all the broken-down process steps.")

# ==============================================================================
# --- LLM-Powered Function (The Core AI Logic) (MODIFIED) ---
# ==============================================================================
def get_detailed_audit_plan(process_description: str) -> dict:
    if not api_key:
        return {"error": "GOOGLE_API environment variable not set. Please configure your .env file."}
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.1, google_api_key=api_key, max_output_tokens=8192)
        parser = JsonOutputParser(pydantic_object=AuditAnalysis)

        prompt_template = PromptTemplate(
            template="""
Persona:
You are a highly experienced Principal Auditor in Financial Markets. Your defining trait is professional skepticism combined with methodical risk analysis. You systematically dissect every process step and sub-step, challenge assumptions, and apply root cause analysis to uncover hidden risks and dependencies.

Core Mission:
Your task is to guide a user through comprehensive audit preparation. You will break down the provided process into granular steps, identify risks at each stage, and generate leading questions.

Your Thought Process (Follow these steps systematically):

Phase 1: Process Decomposition
- Break down the user's process description into detailed steps and sub-steps.
- For each sub-step, identify what systems, departments, or external parties interact and note these connections.

Phase 2: Risk Identification
- For each sub-step, identify risks by applying root cause analysis. Consider risks from the sub-step itself and from its connections to other systems or processes.
- Stick to the following risk categories: OPERATIONAL, DATA, COMPLIANCE, FINANCIAL, STRATEGIC, CYBERSECURITY, and REGULATORY/LEGAL RISKS.

Phase 3: Leading Question Generation
- For each identified risk, consider its root cause and create multiple potential scenarios.
- For each scenario, create probing questions that will uncover overlooked areas and ensure complete preparedness.

Focus Areas:
- Be exhaustive in breaking down every step and sub-step.
- Apply systematic questioning to uncover all related areas and dependencies.
- Identify risks comprehensively at each stage.
- Generate insightful leading questions that drive thorough self-assessment.
- Maintain a banking/financial markets context throughout.

User's Process Description:
"{description}"

Action:
Based on your systematic risk-based analysis, generate your response in the specified JSON format. Ensure the breakdown is complete for all identified steps.

\n{format_instructions}\n
""",
            input_variables=["description"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt_template | llm | StrOutputParser()
        raw_response = chain.invoke({"description": process_description})
        
        # Clean the raw string by removing markdown fences
        cleaned_response = re.sub(r"^\s*``````\s*$", "", raw_response, flags=re.MULTILINE).strip()
        
        # Manually parse the cleaned string
        ai_response = parser.parse(cleaned_response)

        return ai_response

    except Exception as e:
        print(f"An error occurred in get_detailed_audit_plan: {e}")
        error_payload = {"error": str(e)}
        if 'raw_response' in locals():
            error_payload['raw_response'] = raw_response
        return error_payload

# ==============================================================================
# --- Flask API Endpoints ---
# ==============================================================================
analyzer = AuditAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/generate-audit-plan', methods=['POST'])
def generate_audit_plan():
    try:
        data = request.get_json()
        process_description = data.get('process_description')
        if not process_description:
            return jsonify({'error': 'process_description is required'}), 400
        analysis_result = get_detailed_audit_plan(process_description)
        if 'error' in analysis_result:
            return jsonify(analysis_result), 500
        return jsonify(analysis_result)
    except Exception as e:
        print(f"Error in /api/generate-audit-plan: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/api/find-historical-audits', methods=['POST'])
def find_historical_audits():
    try:
        data = request.get_json()
        if 'audit_title' not in data:
            return jsonify({'error': 'audit_title is required'}), 400
        audit_title = data['audit_title']
        threshold = float(data.get('threshold', 0.1))
        similar_audits = analyzer.find_similar_audits(audit_title, threshold)
        if not similar_audits:
            return jsonify({
                'message': 'No sufficiently similar historical audits were found.',
                'query_title': audit_title,
                'matches_found': 0,
                'similar_audits': [],
                'clustered_issues': {}
            })
        audit_titles = [audit['audit_title'] for audit in similar_audits]
        clustered_issues = analyzer.cluster_issues(audit_titles)
        return jsonify({
            'query_title': audit_title,
            'matches_found': len(similar_audits),
            'themes_found': len(clustered_issues),
            'similar_audits': similar_audits,
            'clustered_issues': clustered_issues
        })
    except Exception as e:
        print(f"Error in /api/find-historical-audits: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
