import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request
import json

# Task 1
df = pd.read_csv("G2 software - CRM Category Product Overviews (1).csv")
df = df[['product_name', 'main_category', 'Features', 'description', 'rating']].dropna()
df['product_name'] = df['product_name'].str.strip()
df['main_category'] = df['main_category'].str.strip().str.lower()
df['Features'] = df['Features'].str.strip().str.lower()
df['description'] = df['description'].str.strip().str.lower()

def assign_custom_categories(row):
    text = (row['main_category'] + " " + row['description']).lower()
    categories = []
    if any(word in text for word in ["crm", "customer relationship"]):
        categories.append("crm software")
    if any(word in text for word in ["finance", "accounting", "budget", "sales"]):
        categories.append("accounting & finance software")
    if any(word in text for word in ["erp", "inventory", "supply chain", "manufacturing"]):
        categories.append("erp software")
    if any(word in text for word in ["hr", "payroll", "recruit", "talent"]):
        categories.append("hr software")
    if not categories:
        categories.append(row['main_category'])
    return categories

def join_categories(cats):
    return " ".join(cats)

df['custom_categories'] = df.apply(assign_custom_categories, axis=1)
df['combined_text'] = df['Features'] + " " + df['description'] + " " + df['custom_categories'].apply(join_categories)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

# Task 2
def match_capabilities(software_category, capabilities, threshold=0.01):
    def category_match(cats):
        return software_category.lower() in cats
    candidates = df[df['custom_categories'].apply(category_match)]#fast efficient searching
    if candidates.empty:
        candidates = df
    query_text = " ".join(capabilities).lower()
    query_vector = vectorizer.transform([query_text])
    similarities = cosine_similarity(query_vector, vectorizer.transform(candidates['combined_text'])).flatten()
    candidates = candidates.copy()#mitigates pandas warning
    candidates['similarity'] = similarities
    filtered = candidates[candidates['similarity'] >= threshold]
    if filtered.empty:
        filtered = candidates
    return filtered

# Task 3
def rank_vendors(candidates, top_n=10):
    candidates = candidates.copy()
    candidates['score'] = candidates['rating'] * 0.7 + candidates['similarity'] * 0.3#popular weight heuristic
    ranked = candidates.sort_values(by='score', ascending=False).head(top_n)
    return ranked[['product_name', 'main_category', 'custom_categories', 'score', 'similarity']]

def get_ranked_vendors(software_category, capabilities, threshold=0.01):
    matched = match_capabilities(software_category, capabilities, threshold)
    return rank_vendors(matched).to_dict(orient='records')

# Task 4
app = Flask(__name__)

@app.route('/vendor_qualification', methods=['POST'])
def vendor_qualification():
    data = request.get_json()
    software_category = data.get("software_category", "").lower()
    capabilities = []
    for c in data.get("capabilities", []):
        capabilities.append(c.lower())
    results = get_ranked_vendors(software_category, capabilities)
    return app.response_class(
        response=json.dumps({"results": results}, indent=2),
        status=200,
        mimetype="application/json"
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
