# Create CSS styling
css_content = """
/* Smart Audit Analysis Tool - CSS Styles */

/* Reset and Base Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

/* Header Styles */
.header {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
    padding: 2rem 0;
    margin-bottom: 2rem;
}

.header-content {
    text-align: center;
}

.logo {
    font-size: 2.5rem;
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
}

.tagline {
    font-size: 1.1rem;
    color: #7f8c8d;
    font-weight: 300;
}

/* Main Content */
.main-content {
    padding: 0 0 4rem 0;
}

/* Card Styles */
.card {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    margin-bottom: 2rem;
    overflow: hidden;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
}

.card-header {
    background: linear-gradient(135deg, #3498db, #2980b9);
    color: white;
    padding: 1.5rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.card-header h2 {
    font-size: 1.5rem;
    font-weight: 600;
}

.card-header p {
    margin-top: 0.5rem;
    opacity: 0.9;
    font-size: 0.9rem;
}

.card-body {
    padding: 2rem;
}

/* Form Styles */
.form-group {
    margin-bottom: 1.5rem;
}

.form-label {
    display: flex;
    align-items: center;
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 0.5rem;
    font-size: 1rem;
}

.label-icon {
    margin-right: 0.5rem;
    font-size: 1.2rem;
}

.form-control {
    width: 100%;
    padding: 12px 16px;
    border: 2px solid #e0e6ed;
    border-radius: 8px;
    font-size: 1rem;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
    background-color: #fff;
}

.form-control:focus {
    outline: none;
    border-color: #3498db;
    box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
}

.form-control::placeholder {
    color: #95a5a6;
}

/* Threshold Slider */
.threshold-slider {
    width: 100%;
    height: 8px;
    border-radius: 4px;
    background: #e0e6ed;
    outline: none;
    -webkit-appearance: none;
    appearance: none;
    cursor: pointer;
}

.threshold-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #3498db;
    cursor: pointer;
    box-shadow: 0 2px 6px rgba(52, 152, 219, 0.3);
}

.threshold-slider::-moz-range-thumb {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #3498db;
    cursor: pointer;
    border: none;
    box-shadow: 0 2px 6px rgba(52, 152, 219, 0.3);
}

.threshold-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 0.5rem;
    font-size: 0.8rem;
    color: #7f8c8d;
}

/* Button Styles */
.btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 12px 24px;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    text-decoration: none;
    cursor: pointer;
    transition: all 0.3s ease;
    gap: 8px;
}

.btn-primary {
    background: linear-gradient(135deg, #3498db, #2980b9);
    color: white;
    box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
}

.btn-primary:hover {
    background: linear-gradient(135deg, #2980b9, #1f5f8b);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
}

.btn-secondary {
    background: #95a5a6;
    color: white;
}

.btn-secondary:hover {
    background: #7f8c8d;
}

.btn-icon {
    font-size: 1.1rem;
}

/* Badge */
.badge {
    background: rgba(255, 255, 255, 0.2);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* Loading Section */
.loading-section {
    text-align: center;
    padding: 4rem 2rem;
}

.loading-content {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 16px;
    padding: 3rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    display: inline-block;
}

.spinner {
    width: 50px;
    height: 50px;
    border: 4px solid #e0e6ed;
    border-top: 4px solid #3498db;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 1rem auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Results Section */
.results-section {
    animation: fadeInUp 0.6s ease;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Similar Audits Grid */
.similar-audits-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 1.5rem;
}

.audit-card {
    border: 1px solid #e0e6ed;
    border-radius: 12px;
    padding: 1.5rem;
    background: #f8f9fa;
    transition: all 0.3s ease;
}

.audit-card:hover {
    background: #fff;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
}

.audit-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}

.audit-header h4 {
    color: #2c3e50;
    font-weight: 600;
}

.similarity-score {
    background: linear-gradient(135deg, #27ae60, #2ecc71);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.audit-title {
    color: #34495e;
    margin-bottom: 0.75rem;
    font-size: 1rem;
    font-weight: 600;
}

.audit-findings {
    color: #7f8c8d;
    font-size: 0.9rem;
    line-height: 1.5;
}

/* Issue Clusters Grid */
.clusters-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
}

.cluster-card {
    border: 2px solid #e0e6ed;
    border-radius: 12px;
    padding: 1.5rem;
    background: linear-gradient(135deg, #f8f9fa, #fff);
    cursor: pointer;
    transition: all 0.3s ease;
}

.cluster-card:hover {
    border-color: #3498db;
    box-shadow: 0 8px 24px rgba(52, 152, 219, 0.15);
    transform: translateY(-4px);
}

.cluster-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}

.cluster-header h4 {
    color: #2c3e50;
    font-weight: 600;
    font-size: 1.1rem;
}

.issue-count {
    background: linear-gradient(135deg, #e74c3c, #c0392b);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.cluster-preview {
    color: #7f8c8d;
    font-size: 0.9rem;
    font-style: italic;
}

/* Modal Styles */
.modal {
    display: none;
    position: fixed;
    z-index: 1000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(5px);
    justify-content: center;
    align-items: center;
}

.modal-content {
    background: white;
    border-radius: 16px;
    width: 90%;
    max-width: 800px;
    max-height: 80vh;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    animation: modalFadeIn 0.3s ease;
}

@keyframes modalFadeIn {
    from {
        opacity: 0;
        transform: scale(0.9);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

.modal-header {
    background: linear-gradient(135deg, #9b59b6, #8e44ad);
    color: white;
    padding: 1.5rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.modal-header h3 {
    font-size: 1.3rem;
    font-weight: 600;
}

.close-btn {
    background: none;
    border: none;
    color: white;
    font-size: 1.5rem;
    cursor: pointer;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    transition: background-color 0.3s ease;
}

.close-btn:hover {
    background-color: rgba(255, 255, 255, 0.2);
}

.modal-body {
    padding: 2rem;
    max-height: 60vh;
    overflow-y: auto;
}

/* Theme Issues */
.theme-issues {
    space-y: 1.5rem;
}

.issue-item {
    border: 1px solid #e0e6ed;
    border-radius: 12px;
    padding: 1.5rem;
    background: #f8f9fa;
    margin-bottom: 1.5rem;
}

.issue-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1rem;
}

.issue-header h5 {
    color: #2c3e50;
    font-weight: 600;
    flex: 1;
    margin-right: 1rem;
}

.severity {
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
}

.severity-high {
    background: #e74c3c;
    color: white;
}

.severity-medium {
    background: #f39c12;
    color: white;
}

.severity-low {
    background: #27ae60;
    color: white;
}

.issue-description {
    color: #34495e;
    line-height: 1.6;
    margin-bottom: 1rem;
}

.issue-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    font-size: 0.8rem;
    color: #7f8c8d;
}

.issue-meta strong {
    color: #2c3e50;
}

/* Error Section */
.error-section {
    text-align: center;
    padding: 4rem 2rem;
}

.error-content {
    background: rgba(231, 76, 60, 0.1);
    border: 2px solid #e74c3c;
    border-radius: 16px;
    padding: 3rem;
    display: inline-block;
    color: #c0392b;
}

.error-content h3 {
    margin-bottom: 1rem;
    font-size: 1.5rem;
}

.no-data {
    text-align: center;
    color: #7f8c8d;
    font-style: italic;
    padding: 2rem;
}

/* Footer */
.footer {
    background: rgba(44, 62, 80, 0.95);
    color: white;
    text-align: center;
    padding: 2rem 0;
    margin-top: 4rem;
}

/* Responsive Design */
@media (max-width: 768px) {
    .container {
        padding: 0 15px;
    }
    
    .logo {
        font-size: 2rem;
    }
    
    .card-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.5rem;
    }
    
    .similar-audits-grid,
    .clusters-grid {
        grid-template-columns: 1fr;
    }
    
    .audit-header,
    .cluster-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.5rem;
    }
    
    .modal-content {
        width: 95%;
        max-height: 90vh;
    }
    
    .modal-header {
        padding: 1rem 1.5rem;
    }
    
    .modal-body {
        padding: 1.5rem;
    }
    
    .issue-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.5rem;
    }
    
    .issue-meta {
        flex-direction: column;
        gap: 0.5rem;
    }
}

@media (max-width: 480px) {
    .header {
        padding: 1.5rem 0;
    }
    
    .logo {
        font-size: 1.8rem;
    }
    
    .tagline {
        font-size: 1rem;
    }
    
    .card-body {
        padding: 1.5rem;
    }
    
    .audit-card,
    .cluster-card,
    .issue-item {
        padding: 1rem;
    }
}

/* Utility Classes */
.text-center {
    text-align: center;
}

.mt-1 { margin-top: 0.5rem; }
.mt-2 { margin-top: 1rem; }
.mt-3 { margin-top: 1.5rem; }
.mt-4 { margin-top: 2rem; }

.mb-1 { margin-bottom: 0.5rem; }
.mb-2 { margin-bottom: 1rem; }
.mb-3 { margin-bottom: 1.5rem; }
.mb-4 { margin-bottom: 2rem; }

.p-1 { padding: 0.5rem; }
.p-2 { padding: 1rem; }
.p-3 { padding: 1.5rem; }
.p-4 { padding: 2rem; }

/* Scrollbar Styles */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
}
"""

# Create static directory and save CSS
os.makedirs('static', exist_ok=True)

with open('static/style.css', 'w') as f:
    f.write(css_content)

print("CSS stylesheet created successfully!")
print("Filename: static/style.css")