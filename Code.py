import random
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import feedparser
from textblob import TextBlob
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# --- Parameters ---
NUM_USERS = 30
INIT_SHARED = 3
SHARE_PROB = 0.3
GIFT_BONUS = 10
IDEOLOGY_CROSS_BONUS = 0.2
CHRONIC_PROPENSITY = 0.6
GENDER_HOMOPHILY_BONUS = 0.2

# --- Step 1: Network Setup (Users only) ---
G = nx.erdos_renyi_graph(NUM_USERS, 0.1, seed=42)
nx.set_node_attributes(G, False, 'shared')
nx.set_node_attributes(G, 0, 'score')
nx.set_node_attributes(G, False, 'gifted')
nx.set_node_attributes(G, 0, 'triggered_count')
nx.set_node_attributes(G, '', 'gender')
nx.set_node_attributes(G, False, 'has_chronic_disease')
nx.set_node_attributes(G, '', 'ideology')
nx.set_node_attributes(G, '', 'sentiment')

# --- Step 2: Sentiment Analyzer ---
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.5:
        return 'pro-health'
    elif polarity < -0.5:
        return 'anti-health'
    else:
        return 'neutral'

# --- Step 3: Fetch Podcasts via RSS (Content Source Only) ---
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

# Example feeds (including non-health podcasts)
rss_urls = [
    "https://feeds.npr.org/510307/rss.xml",  # NPR Life Kit Health
    "https://rss.art19.com/the-daily",      # The Daily (News)
    "https://rss.art19.com/planet-money",   # Planet Money (Economics)
    "https://rss.art19.com/startup",        # StartUp Podcast (Business)
    "https://rss.art19.com/serial",         # Serial (True Crime)
    # Removed problematic feeds like apology-line
]

podcast_items = []
for url in rss_urls:
    try:
        podcast_items.extend(get_podcasts_from_rss(url))
    except Exception:
        # silently skip failed feeds to avoid clutter
        pass

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

for node in G.nodes:
    G.nodes[node]['gender'] = random.choice(['Male', 'Female'])
    G.nodes[node]['has_chronic_disease'] = random.choice([True, False])
    G.nodes[node]['ideology'] = random.choices(
        population=['pro-health', 'anti-health', 'neutral'],
        weights=[weights.get('pro-health', 0.33), weights.get('anti-health', 0.33), weights.get('neutral', 0.33)],
        k=1
    )[0]
    G.nodes[node]['sentiment'] = G.nodes[node]['ideology']
    G.nodes[node]['shared'] = False
    G.nodes[node]['score'] = 0
    G.nodes[node]['triggered_count'] = 0
    G.nodes[node]['gifted'] = False

# --- Step 5: Features & Labels ---
def calc_sentiment_trends():
    trends = []
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        if neighbors:
            pro_health_count = sum(1 for n in neighbors if G.nodes[n]['sentiment'] == 'pro-health')
            trends.append(pro_health_count / len(neighbors))
        else:
            trends.append(0)
    return trends

sentiment_trends = calc_sentiment_trends()
betweenness_centrality = nx.betweenness_centrality(G)

user_features = []
user_labels = []
for node in G.nodes:
    u = G.nodes[node]
    features = [
        1 if u['gender'] == 'Female' else 0,
        1 if u['has_chronic_disease'] else 0,
        1 if u['ideology'] == 'pro-health' else 0,
        1 if u['ideology'] == 'anti-health' else 0,
        1 if u['ideology'] == 'neutral' else 0,
        sentiment_trends[node],
        betweenness_centrality[node]
    ]
    user_features.append(features)
    user_labels.append(u['ideology'])

X_train, X_test, y_train, y_test = train_test_split(
    user_features, user_labels, test_size=0.2, random_state=42
)

# --- Step 6: Model Training ---
param_grid = {'n_estimators': [100], 'max_depth': [10], 'min_samples_split': [2]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=2, n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# --- Step 7: Evaluation ---
st.subheader("Model Evaluation")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
st.text(classification_report(y_test, y_pred))

# --- Step 8: Contagion Simulation ---
pos = nx.spring_layout(G, seed=42)
seed_nodes = random.sample(list(G.nodes), INIT_SHARED)
for node in seed_nodes:
    G.nodes[node]['shared'] = True
    G.nodes[node]['gifted'] = True

contagion, current = [set(seed_nodes)], set(seed_nodes)
while current:
    next_step = set()
    for u in current:
        for v in G.neighbors(u):
            if not G.nodes[v]['shared']:
                prob = SHARE_PROB + (GIFT_BONUS / 100 if G.nodes[u]['gifted'] else 0)
                if G.nodes[u]['ideology'] != G.nodes[v]['ideology']:
                    prob += IDEOLOGY_CROSS_BONUS
                if G.nodes[v]['has_chronic_disease']:
                    prob = max(prob, CHRONIC_PROPENSITY)
                if G.nodes[u]['gender'] == G.nodes[v]['gender']:
                    prob += GENDER_HOMOPHILY_BONUS
                prob = min(max(prob, 0), 1)
                if random.random() < prob:
                    G.nodes[v]['shared'] = True
                    G.nodes[v]['triggered_count'] += 1
                    next_step.add(v)
    if not next_step:
        break
    contagion.append(next_step)
    current = next_step

# --- Step 9: Visualization with enhanced clarity ---
st.subheader("User Network Contagion Simulation with Clusters")

betweenness = nx.betweenness_centrality(G)

fig, ax = plt.subplots(figsize=(10, 8))

# Node colors by gender
node_colors = ['lightgreen' if G.nodes[n]['gender'] == 'Male' else 'lightblue' for n in G.nodes]

# Node shapes by gender: circle for Male, square for Female
node_shapes = {'Male': 'o', 'Female': 's'}

# Edge colors by gender similarity
edge_colors = []
for u, v in G.edges:
    if G.nodes[u]['gender'] == 'Male' and G.nodes[v]['gender'] == 'Male':
        edge_colors.append('lightgreen')
    elif G.nodes[u]['gender'] == 'Female' and G.nodes[v]['gender'] == 'Female':
        edge_colors.append('lightblue')
    else:
        edge_colors.append('gray')

# Draw edges
nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, alpha=0.5)

# Draw nodes by gender with sizes and border widths (betweenness)
for gender in ['Male', 'Female']:
    nodelist = [n for n in G.nodes if G.nodes[n]['gender'] == gender]
    sizes = [300 + 150 * G.nodes[n]['triggered_count'] for n in nodelist]
    widths = [3 + 10 * betweenness[n] for n in nodelist]
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=nodelist,
        node_color=['lightgreen' if gender == 'Male' else 'lightblue'] * len(nodelist),
        node_shape=node_shapes[gender],
        node_size=sizes,
        edgecolors='black',
        linewidths=widths,
        ax=ax,
        alpha=0.9
    )

# Add labels (user ID and ideology abbreviation)
labels = {n: f"{n}\n{G.nodes[n]['ideology'][:3]}" for n in G.nodes}
nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

# Legend
male_patch = mpatches.Patch(color='lightgreen', label='Male')
female_patch
