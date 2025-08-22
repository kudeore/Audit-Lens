# --- ADD/REPLACE these imports at the top of app.py ---
import os
import re
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from dotenv import load_dotenv
load_dotenv()

# --- NEW LANGCHAIN IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field # New imports for data modeling
from typing import List # New import for type hinting

app = Flask(__name__)
CORS(app)


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# --- LLM Integration (Google Gemini) ---
# Ensure you have your Google API key set as an environment variable
# e.g., export GOOGLE_API_KEY='your_key_here'
try:
    api_key = os.getenv("GOOGLE_API")
    if not api_key:
        print("Warning: GOOGLE_API_KEY environment variable not set.")
    else:
        genai.configure(api_key=api_key)
except Exception as e:
    print(f"Error configuring Google Generative AI client: {e}")

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
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text

    def prepare_audit_vectors(self):
        """Prepare TF-IDF vectors for all audit titles"""
        if audits_df.empty: return
        processed_titles = [self.preprocess_text(title) for title in audits_df['audit_title']]
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        self.audit_vectors = self.tfidf_vectorizer.fit_transform(processed_titles)
        print(f"Created TF-IDF vectors with shape: {self.audit_vectors.shape}")

    def find_similar_audits(self, query_title, threshold=0.1, top_n=5):
        """
        Stricter search: Find top N audits similar to the query title
        only if they are above the specified threshold.
        """
        if self.tfidf_vectorizer is None or audits_df.empty: return []
        
        processed_query = self.preprocess_text(query_title)
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.audit_vectors).flatten()
        
        # Get indices of all matches strictly above the threshold
        similar_indices = np.where(similarities >= threshold)[0]
        
        if len(similar_indices) == 0:
            return [] # Return empty if no matches are found

        results = [{'audit_id': audits_df.iloc[idx]['audit_id'], 'audit_title': audits_df.iloc[idx]['audit_title'], 'audit_findings': audits_df.iloc[idx]['audit_findings'], 'similarity_score': float(similarities[idx])} for idx in similar_indices]
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_n]


    def cluster_issues(self, audit_titles, n_clusters=None):
        """Cluster issues from matched audits"""
        if issues_df.empty or not audit_titles: return []
        matched_issues = issues_df[issues_df['audit_title'].isin(audit_titles)]
        if matched_issues.empty: return []
        issue_texts = (matched_issues['issue_title'] + ' ' + matched_issues['issue_description']).tolist()
        processed_texts = [self.preprocess_text(text) for text in issue_texts]
        issue_vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
        try:
            issue_vectors = issue_vectorizer.fit_transform(processed_texts)
            if n_clusters is None: n_clusters = min(5, max(2, len(matched_issues) // 3))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(issue_vectors)
            clustered_issues, feature_names = {}, issue_vectorizer.get_feature_names_out()
            for cluster_id in range(n_clusters):
                cluster_issues_df = matched_issues[cluster_labels == cluster_id]
                centroid, top_features_idx = kmeans.cluster_centers_[cluster_id], kmeans.cluster_centers_[cluster_id].argsort()[-5:][::-1]
                top_features = [feature_names[i] for i in top_features_idx]
                theme_name = f"Theme: {' & '.join(top_features[:3]).title()}"
                clustered_issues[theme_name] = {'theme_id': cluster_id, 'issue_count': len(cluster_issues_df), 'issues': cluster_issues_df.to_dict('records')}
            return clustered_issues
        except Exception as e:
            print(f"Clustering error: {e}")
            return self.fallback_clustering(matched_issues)

    def fallback_clustering(self, issues):
        clusters = {}
        for _, issue in issues.iterrows():
            theme = issue['issue_title'].split()[0] if issue['issue_title'] else 'Other'
            theme_name = f"Theme: {theme} Issues"
            if theme_name not in clusters: clusters[theme_name] = {'theme_id': len(clusters), 'issue_count': 0, 'issues': []}
            clusters[theme_name]['issues'].append(issue.to_dict())
            clusters[theme_name]['issue_count'] += 1
        return clusters
class RiskArea(BaseModel):
    area: str = Field(description="Name of a specific, tangible risk area.")
    explanation: str = Field(description="Why this is a risk for the user's process, based on the provided context or general knowledge.")
    historical_evidence_count: int = Field(description="Number of historical issues provided that support this risk area. If no historical data was provided, this should be 0.")

class SuggestedIssue(BaseModel):
    issue_title: str = Field(description="A concise, actionable title for a potential issue an auditor might raise.")
    issue_description: str = Field(description="A detailed description of the potential weakness or control gap.")
    potential_impact: str = Field(description="What could go wrong if this issue is not addressed (e.g., financial loss, non-compliance).")
    rationale: str = Field(description="Your reasoning for suggesting this, explicitly linking back to historical issues if available, or to general audit principles if not.")

class DataDrivenResponse(BaseModel):
    analysis_summary: str = Field(description="A brief, high-level summary of your findings, emphasizing risks seen in historical data.")
    key_risk_areas: List[RiskArea] = Field(description="A list of key risk areas identified from the historical data. You MUST identify at least 2.")
    suggested_issues_to_self_raise: List[SuggestedIssue] = Field(description="A list of potential issues the user could proactively investigate. You MUST suggest at least 2.")

class GeneralGuidanceResponse(BaseModel):
    analysis_summary: str = Field(description="A summary explaining that while no internal data was found, you are offering general guidance on common risks for this topic.")
    general_risk_areas: List[RiskArea] = Field(description="A list of broad, common risk areas for this topic based on your general knowledge. You MUST identify at least 2.")
    questions_to_consider: List[SuggestedIssue] = Field(description="A list of thought-provoking questions an auditor might ask, framed as suggested issues. You MUST provide at least 2.")


analyzer = AuditAnalyzer()

# --- Helper function for LLM call (Updated for Google Gemini) ---
def call_llm_for_analysis(prompt):
    """Calls the Google Generative AI API to get analysis."""
    if not os.getenv("GOOGLE_API"):
        raise ValueError("Google API key is not configured.")
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        
        # Configuration for safety and deterministic output
        generation_config = genai.GenerationConfig(
            temperature=0.2,
        )
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        # Gemini can sometimes wrap the JSON in markdown, so we clean it.
        cleaned_text = response.text.strip().replace('```json', '').replace('```', '').strip()
        
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM response: {e}")
        print(f"LLM Raw Response: {response.text}")
        raise ValueError("The AI model returned a response that was not valid JSON.")
    except Exception as e:
        print(f"Error calling Google Generative AI API: {e}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# Existing endpoints are unchanged...

@app.route('/api/full-analysis', methods=['POST'])
def full_analysis():
    try:
        data = request.get_json()
        audit_title = data['audit_title']
        threshold = float(data.get('threshold', 0.1))
        similar_audits = analyzer.find_similar_audits(audit_title, threshold)
        if not similar_audits: return jsonify({'message': 'No similar audits found'})
        audit_titles = [audit['audit_title'] for audit in similar_audits]
        clustered_issues = analyzer.cluster_issues(audit_titles)
        return jsonify({'query_title': audit_title, 'matches_found': len(similar_audits), 'themes_found': len(clustered_issues), 'similar_audits': similar_audits, 'clustered_issues': clustered_issues})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- AI Auditor Assistant Endpoint (using Google Gemini) ---

@app.route('/api/ai-audit-assist', methods=['POST'])
def ai_audit_assist():
    """Provides AI-driven audit preparation advice using a robust LangChain Pydantic chain."""
    try:
        data = request.get_json()
        process_description = data.get('process_description')
        if not process_description:
            return jsonify({'error': 'process_description is required'}), 400

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API")
        )
        
        similar_audits = analyzer.find_similar_audits(process_description, threshold=0.1, top_n=5)
        
        relevant_issues = pd.DataFrame()
        if similar_audits:
            relevant_audit_titles = [audit['audit_title'] for audit in similar_audits]
            relevant_issues = issues_df[issues_df['audit_title'].isin(relevant_audit_titles)]

        if not relevant_issues.empty:
            # --- PATH 1: Data-Driven Analysis (RAG) ---
            parser = JsonOutputParser(pydantic_object=DataDrivenResponse)
            issues_context = "\n".join([f"- Issue: '{row['issue_title']}' | Description: '{row['issue_description']}'" for _, row in relevant_issues.head(20).iterrows()])
            
            prompt_template = PromptTemplate(
                template="""You are a Principal Auditor, an expert at providing actionable advice. Your task is to analyze the user's process and the provided historical data to create a proactive audit preparation plan.
                You MUST generate a response that follows the provided JSON schema exactly. Do not leave any fields blank.
                \n{format_instructions}\n
                User's process description: {description}
                Historical issues from similar audits: {context}""",
                input_variables=["description", "context"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            chain = prompt_template | llm | parser
            ai_response = chain.invoke({"description": process_description, "context": issues_context})
            ai_response['analysis_type'] = 'data_driven'

        else:
            # --- PATH 2: General Guidance (No RAG) ---
            parser = JsonOutputParser(pydantic_object=GeneralGuidanceResponse)
            
            prompt_template = PromptTemplate(
                template="""You are a Principal Auditor, an expert at providing actionable advice. A user needs help preparing for an audit. No specific internal data was found, so you must provide general guidance based on your broad knowledge of auditing this topic.
                You MUST generate a response that follows the provided JSON schema exactly. Do not leave any fields blank.
                \n{format_instructions}\n
                User's process or audit topic: {description}""",
                input_variables=["description"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            chain = prompt_template | llm | parser
            ai_response = chain.invoke({"description": process_description})
            # To match the frontend, we rename the keys before sending
            ai_response['key_risk_areas'] = ai_response.pop('general_risk_areas')
            ai_response['suggested_issues_to_self_raise'] = ai_response.pop('questions_to_consider')
            ai_response['analysis_type'] = 'general_guidance'

        return jsonify(ai_response)

    except Exception as e:
        print(f"Error in AI Audit Assist (LangChain): {e}")
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)