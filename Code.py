import random
import networkx as nx
import matplotlib.pyplot as plt
from textblob import TextBlob
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from node2vec import Node2Vec
import numpy as np
import pandas as pd
import feedparser
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# --- Parameters ---
NUM_USERS = 300
INIT_SHARED = 3
GIFT_BONUS = 10
IDEOLOGY_CROSS_BONUS = 0.2
CHRONIC_PROPENSITY = 0.6
GENDER_HOMOPHILY_BONUS = 0.2

st.title("Health Information Contagion Network Simulation with Improved Features")

# --- Step 1: Network Setup ---
G = nx.erdos_renyi_graph(NUM_USERS, 0.05, seed=42)
nx.set_node_attributes(G, False, 'shared')
nx.set_node_attributes(G, 0, 'score')
nx.set_node_attributes(G, False, 'gifted')
nx.set_node_attributes(G, 0, 'triggered_count')
nx.set_node_attributes(G, '', 'gender')
nx.set_node_attributes(G, False, 'has_chronic_disease')
nx.set_node_attributes(G, '', 'ideology')
nx.set_node_attributes(G, '', 'sentiment')
nx.set_node_attributes(G, 0.0, 'sentiment_polarity')

# --- Step 2: Sentiment Analyzer ---
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.5:
        sentiment = 'pro-health'
    elif polarity < -0.5:
        sentiment = 'anti-health'
    else:
        sentiment = 'neutral'
    return sentiment, polarity

# --- Step 3: Fetch Podcasts ---
def get_podcasts_from_rss(feed_url, max_items=5):
    feed = feedparser.parse(feed_url)
    podcasts = []
    for entry in feed.entries[:max_items]:
        podcasts.append({
            "user": entry.get('author', 'podcaster'),
            "content": entry.title,
            "platform": "RSS",
            "url": entry.link
        })
    return podcasts

rss_urls = [
    "https://feeds.npr.org/510307/rss.xml",
    "https://feeds.simplecast.com/54nAGcIl",
    "https://rss.art19.com/the-daily",
    "https://feeds.megaphone.fm/ADL9840290619",
]

podcast_items = []
for url in rss_urls:
    try:
        podcast_items.extend(get_podcasts_from_rss(url))
    except Exception:
        pass

podcast_sentiments = []
podcast_polarities = []
for p in podcast_items:
    s, polarity = analyze_sentiment(p['content'])
    podcast_sentiments.append(s)
    podcast_polarities.append(polarity)

if not podcast_sentiments:
    podcast_sentiments = ['neutral'] * 10
    podcast_polarities = [0.0] * 10

counts = {
    'pro-health': podcast_sentiments.count('pro-health'),
    'anti-health': podcast_sentiments.count('anti-health'),
    'neutral': podcast_sentiments.count('neutral')
}
total = sum(counts.values()) or 1
weights = {k: v / total for k, v in counts.items()}

# --- Step 4: Assign User Attributes and stronger ideology signal ---
for node in G.nodes:
    G.nodes[node]['gender'] = random.choice(['Male', 'Female'])
    G.nodes[node]['has_chronic_disease'] = random.random() < 0.5
    
    # Assign initial ideology randomly with weights
    initial_ideo = random.choices(
        population=['pro-health', 'anti-health', 'neutral'],
        weights=[weights.get('pro-health', 0.33), weights.get('anti-health', 0.33), weights.get('neutral', 0.33)],
        k=1
    )[0]
    G.nodes[node]['ideology'] = initial_ideo

# Improve ideology by assigning based on majority of neighbors (simulate homophily)
for node in G.nodes:
    neighbors = list(G.neighbors(node))
    if neighbors:
        neighbor_ideos = [G.nodes[n]['ideology'] for n in neighbors]
        majority_ideo = max(set(neighbor_ideos), key=neighbor_ideos.count)
        # With 70% probability, assign neighbor majority ideology
        if random.random() < 0.7:
            G.nodes[node]['ideology'] = majority_ideo

# Assign sentiment & polarity matching ideology (simulate real signal)
for node in G.nodes:
    ideo = G.nodes[node]['ideology']
    if ideo == 'pro-health':
        pol = random.uniform(0.6, 1.0)
        sentiment = 'pro-health'
    elif ideo == 'anti-health':
        pol = random.uniform(-1.0, -0.6)
        sentiment = 'anti-health'
    else:
        pol = random.uniform(-0.5, 0.5)
        sentiment = 'neutral'
    G.nodes[node]['sentiment'] = sentiment
    G.nodes[node]['sentiment_polarity'] = pol

# --- Step 5: Calculate network features ---
degree_centrality = nx.degree_centrality(G)
clustering_coeff = nx.clustering(G)
betweenness_centrality = nx.betweenness_centrality(G)

# --- Step 6: Node2Vec Embeddings ---
node2vec = Node2Vec(G, dimensions=16, walk_length=10, num_walks=50, workers=1, seed=42)
model = node2vec.fit(window=5, min_count=1, batch_words=4)

# --- Step 7: Build feature matrix ---
user_features = []
user_labels = []
for node in G.nodes:
    u = G.nodes[node]
    
    # Basic features
    gender_f = 1 if u['gender'] == 'Female' else 0
    chronic_f = 1 if u['has_chronic_disease'] else 0
    sentiment_p = u['sentiment_polarity']
    deg_cent = degree_centrality[node]
    cluster_c = clustering_coeff[node]
    between_c = betweenness_centrality[node]
    
    # Interaction features
    gender_sentiment = gender_f * sentiment_p
    chronic_sentiment = chronic_f * sentiment_p
    
    # Node2Vec embeddings as list
    embedding = model.wv[str(node)]
    
    features = [
        gender_f, chronic_f, sentiment_p,
        deg_cent, cluster_c, between_c,
        gender_sentiment, chronic_sentiment
    ] + list(embedding)
    
    user_features.append(features)
    user_labels.append(u['ideology'])

# --- Step 8: Encode and scale features ---
le = LabelEncoder()
y_encoded = le.fit_transform(user_labels)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(user_features)

# --- Step 9: Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- Step 10: Logistic Regression ---
logreg = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# --- Step 11: Evaluation ---
test_accuracy = accuracy_score(y_test, y_pred)
st.subheader("Model Evaluation (Logistic Regression with Enhanced Features)")
st.write(f"**Test Accuracy:** {test_accuracy:.2%}")

report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose().round(2)
st.dataframe(report_df)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(logreg, X_train, y_train, cv=skf)
st.write(f"**Cross-validated Accuracy (train set):** {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

# You can continue with your contagion simulation and visualization here...

