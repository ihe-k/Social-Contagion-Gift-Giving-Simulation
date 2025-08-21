import random
import networkx as nx
import matplotlib.pyplot as plt
from textblob import TextBlob
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
import feedparser
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from xgboost import XGBClassifier
from collections import Counter

# --- Parameters ---
NUM_USERS = 300
INIT_SHARED = 3
GIFT_BONUS = 10
BONUS_POINTS = 5
IDEOLOGY_CROSS_BONUS = 0.2
CHRONIC_PROPENSITY = 0.6
GENDER_HOMOPHILY_BONUS = 0.2

st.title("Health Information Contagion Network Simulation")

# --- Step 1: Network Setup (Users Only) ---
G = nx.erdos_renyi_graph(NUM_USERS, 0.5, seed=42)
nx.set_node_attributes(G, False, 'shared')
nx.set_node_attributes(G, False, 'gifted')
nx.set_node_attributes(G, 0, 'triggered_count')
nx.set_node_attributes(G, '', 'gender')
nx.set_node_attributes(G, False, 'has_chronic_disease')
nx.set_node_attributes(G, '', 'ideology')
nx.set_node_attributes(G, '', 'sentiment')

# For demonstration, randomly assign some gender, ideology, chronic disease, and sentiment
genders = ['Male', 'Female']
ideologies = ['pro-health', 'anti-health', 'neutral']

for node in G.nodes:
    G.nodes[node]['gender'] = random.choice(genders)
    G.nodes[node]['has_chronic_disease'] = random.random() < 0.3  # 30% chance
    G.nodes[node]['ideology'] = random.choices(ideologies, weights=[0.4, 0.2, 0.4])[0]
    G.nodes[node]['sentiment'] = G.nodes[node]['ideology']  # Simplified for now

# --- Step 2: Sentiment Analyzer ---
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return 'pro-health'
    elif polarity < -0.2:
        return 'anti-health'
    else:
        return 'neutral'

# --- Step 3: Fetch Podcasts via RSS ---
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
    # Add your RSS URLs here...
    "https://feeds.npr.org/510307/rss.xml",           # NPR Life Kit Health
    "https://feeds.simplecast.com/54nAGcIl",          # Stuff You Should Know
    "https://rss.art19.com/the-daily",                 # The Daily by NYT
    "https://feeds.megaphone.fm/ADL9840290619",       # Revisionist History
    "https://drhyman.com/feed/podcast/",                      # The Doctor’s Farmacy
    "https://feeds.megaphone.fm/nutritiondiva",               # Nutrition Diva
    "https://feeds.megaphone.fm/foundmyfitness",              # FoundMyFitness
    "https://themodelhealthshow.libsyn.com/rss",              # The Model Health Show
    "https://wellnessmama.com/feed/podcast/",                 # Wellness Mama Podcast
    "https://mindbodygreen.com/feed/podcast",                 # Mindbodygreen Podcast
    "https://peterattiamd.com/feed/podcast/",                 # The Peter Attia Drive
    "https://ultimatehealthpodcast.com/feed/podcast/",        # The Ultimate Health Podcast
    "https://feeds.megaphone.fm/sem-podcast",                  # Seminars in Integrative Medicine
    "https://feeds.simplecast.com/2fo6fiz5",                   # The Plant Proof Podcast
    "https://feeds.megaphone.fm/mindpump",                     # Mind Pump: Raw Fitness Truth
    # Additional pro-health podcasts:
    "https://feeds.simplecast.com/6SZWJjdx",                  # FoundMyFitness Deep Dives
    "https://anchor.fm/s/7a0e3b4c/podcast/rss",               # The Balanced Life with Robin Long
    "https://feeds.feedburner.com/WellnessForce",             # Wellness Force Podcast
    "https://feeds.simplecast.com/WU9gBqT3",                  # The Health Code
    "https://feeds.megaphone.fm/HSW1741400476",               # Happier with Gretchen Rubin
    "https://feeds.simplecast.com/tOjNXec5",                  # The Rich Roll Podcast
    "https://feeds.megaphone.fm/NFL7271905056",               # The Ultimate Health Podcast
    "https://feeds.soundcloud.com/users/soundcloud:users:32216449/sounds.rss",  # NutritionFacts.org Podcast
    "https://podcast.wellness.com/feed.xml",                   # Wellness.com Podcast
]

podcast_items = []
for url in rss_urls:
    try:
        podcast_items.extend(get_podcasts_from_rss(url))
    except Exception:
        pass  # silently ignore feeds that fail

# --- Step 4: Assign User Attributes ---

podcast_sentiments = [analyze_sentiment(p['content']) for p in podcast_items]
if not podcast_sentiments:
    podcast_sentiments = ['neutral'] * 10

counts = {
    'pro-health': podcast_sentiments.count('pro-health'),
    'anti-health': podcast_sentiments.count('anti-health'),
    'neutral': podcast_sentiments.count('neutral')
}
total = sum(counts.values())
weights = {k: v / total for k, v in counts.items()}

# --- Calculate centralities ---
betweenness_centrality = nx.betweenness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
degree_centrality = nx.degree_centrality(G)
pagerank = nx.pagerank(G)
closeness = nx.closeness_centrality(G)

# Map sentiment strings to numeric for modeling
sentiment_map = {'pro-health': 1, 'neutral': 0, 'anti-health': -1}
sentiment_trends = {node: sentiment_map.get(G.nodes[node]['sentiment'], 0) for node in G.nodes}

# --- Prepare features and labels ---
user_features = []
user_labels = []

for node in G.nodes:
    u = G.nodes[node]
    
    gender_female = 1 if u['gender'] == 'Female' else 0
    has_chronic = 1 if u['has_chronic_disease'] else 0
    
    features = [
        gender_female,
        has_chronic,
        sentiment_trends[node],
        betweenness_centrality[node],
        eigenvector_centrality[node],
        degree_centrality[node],
        pagerank[node],
        closeness[node]
    ]
    
    neighbors = list(G.neighbors(node))
    if neighbors:
        pro_health_neighbors = sum(1 for n in neighbors if G.nodes[n]['ideology'] == 'pro-health') / len(neighbors)
        anti_health_neighbors = sum(1 for n in neighbors if G.nodes[n]['ideology'] == 'anti-health') / len(neighbors)
    else:
        pro_health_neighbors = 0
        anti_health_neighbors = 0

    features.extend([pro_health_neighbors, anti_health_neighbors])

    user_features.append(features)
    user_labels.append(u['ideology'])

feature_names = [
    'gender_female',
    'has_chronic',
    'sentiment_trend',
    'betweenness_centrality',
    'eigenvector_centrality',
    'degree_centrality',
    'pagerank',
    'closeness',
    'pro_health_neighbors',
    'anti_health_neighbors'
]

# --- Feature scaling for continuous features (indexes 3 to 7) ---
scaler = StandardScaler()
continuous_features = np.array(user_features)[:, 3:8]  # centrality + sentiment_trend (index 2 is int but can be scaled)
scaled_continuous = scaler.fit_transform(continuous_features)

# Replace scaled features back into user_features
user_features_np = np.array(user_features)
user_features_np[:, 3:8] = scaled_continuous

user_labels_np = np.array(user_labels)

st.write(f"Number of features in your feature matrix: {user_features_np.shape[1]}")
st.write(f"Number of feature names: {len(feature_names)}")

if user_features_np.shape[1] != len(feature_names):
    st.error("Mismatch detected! Number of features does NOT match number of feature names.")
else:
    st.success("Feature matrix columns and feature names length MATCH.")

# Stratified split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(sss.split(user_features_np, user_labels_np))

X_train, X_test = user_features_np[train_index], user_features_np[test_index]
y_train, y_test = user_labels_np[train_index], user_labels_np[test_index]

# Encode labels
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

# --- Train and Evaluate Random Forest ---
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf.fit(X_train, y_train_enc)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test_enc, y_pred)
st.write(f"Random Forest Accuracy: {accuracy:.2f}")

st.write("Classification Report:")
st.text(classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_))

ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test_enc, display_labels=label_encoder.classes_)
plt.title("Random Forest Confusion Matrix")
st.pyplot(plt.gcf())

# --- Train and Evaluate XGBoost ---
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb.fit(X_train, y_train_enc)

y_pred_xgb = xgb.predict(X_test)
accuracy_xgb = accuracy_score(y_test_enc, y_pred_xgb)
st.write(f"XGBoost Accuracy: {accuracy_xgb:.2f}")

st.write("XGBoost Classification Report:")
st.text(classification_report(y_test_enc, y_pred_xgb, target_names=label_encoder.classes_))

ConfusionMatrixDisplay.from_estimator(xgb, X_test, y_test_enc, display_labels=label_encoder.classes_)
plt.title("XGBoost Confusion Matrix")
st.pyplot(plt.gcf())

# --- Visualization: Ideology Distribution ---
ideology_counts = Counter(user_labels)
labels_ = list(ideology_counts.keys())
sizes = list(ideology_counts.values())
colors = ['green', 'red', 'gray']

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels_, autopct='%1.1f%%', colors=colors, startangle=90)
ax1.axis('equal')
st.pyplot(fig1)

# --- Visualization: Network ---
color_map = {'pro-health': 'green', 'anti-health': 'red', 'neutral': 'gray'}
node_colors = [color_map[G.nodes[n]['ideology']] for n in G.nodes]

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50)
nx.draw_networkx_edges(G, pos, alpha=0.1)
plt.title("User Ideology Network Visualization")
plt.axis('off')
st.pyplot(plt.gcf())
