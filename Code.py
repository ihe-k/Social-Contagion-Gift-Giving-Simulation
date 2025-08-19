import random
import networkx as nx
import matplotlib.pyplot as plt
from textblob import TextBlob
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import numpy as np
import requests
from bs4 import BeautifulSoup
import feedparser

# --- Parameters ---
NUM_USERS = 30
INIT_SHARED = 3
SHARE_PROB = 0.3
GIFT_BONUS = 10
IDEOLOGY_CROSS_BONUS = 0.2
CHRONIC_PROPENSITY = 0.6
GENDER_HOMOPHILY_BONUS = 0.2

# --- Step 1: Network Setup (Users Only) ---
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

# Example feeds (You can add/remove valid RSS feeds here)
rss_urls = [
    "https://feeds.npr.org/510307/rss.xml",  # NPR Life Kit Health
    # Add more valid podcast RSS URLs here
]

podcast_items = []
for url in rss_urls:
    try:
        podcast_items.extend(get_podcasts_from_rss(url))
    except Exception as e:
        st.warning(f"Failed to fetch or parse feed: {url}")

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
weights = {k: v/total for k, v in counts.items()}

for node in G.nodes:
    G.nodes[node]['gender'] = random.choice(['Male', 'Female'])
    G.nodes[node]['has_chronic_disease'] = random.choice([True, False])
    G.nodes[node]['ideology'] = random.choices(
        population=['pro-health', 'anti-health', 'neutral'],
        weights=[weights.get('pro-health',0.33), weights.get('anti-health',0.33), weights.get('neutral',0.33)],
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
            pro_health_count = sum(1 for n in neighbors if G.nodes[n]['sentiment']=='pro-health')
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
fig_cm, ax_cm = plt.subplots()
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=ax_cm)
st.pyplot(fig_cm)

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

# --- Step 9: Visualization & Leaderboard ---

st.subheader("User Network Contagion Simulation")

col1, col2 = st.columns([3,1])

with col1:
    fig_net, ax_net = plt.subplots(figsize=(8, 6))

    # Edges colored by gender pairs
    male_edges = [(u,v) for u,v in G.edges if G.nodes[u]['gender']=='Male' and G.nodes[v]['gender']=='Male']
    female_edges = [(u,v) for u,v in G.edges if G.nodes[u]['gender']=='Female' and G.nodes[v]['gender']=='Female']
    mixed_edges = [(u,v) for u,v in G.edges if G.nodes[u]['gender'] != G.nodes[v]['gender']]

    nx.draw_networkx_edges(G, pos, edgelist=male_edges, edge_color='lightgreen', alpha=0.7, ax=ax_net)
    nx.draw_networkx_edges(G, pos, edgelist=female_edges, edge_color='lightblue', alpha=0.7, ax=ax_net)
    nx.draw_networkx_edges(G, pos, edgelist=mixed_edges, edge_color='gray', alpha=0.4, ax=ax_net)

    # Nodes by gender and shape
    male_nodes = [n for n in G.nodes if G.nodes[n]['gender']=='Male']
    female_nodes = [n for n in G.nodes if G.nodes[n]['gender']=='Female']

    nx.draw_networkx_nodes(G, pos, nodelist=male_nodes,
                           node_color='green',
                           node_shape='o',
                           node_size=[300 + 50 * G.nodes[n]['triggered_count'] for n in male_nodes],
                           alpha=0.9, ax=ax_net)

    nx.draw_networkx_nodes(G, pos, nodelist=female_nodes,
                           node_color='blue',
                           node_shape='s',
                           node_size=[300 + 50 * G.nodes[n]['triggered_count'] for n in female_nodes],
                           alpha=0.9, ax=ax_net)

    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax_net)
    ax_net.axis('off')
    st.pyplot(fig_net)

with col2:
    st.subheader("🏆 Top Influencers")

    sorted_nodes = sorted(G.nodes, key=lambda n: G.nodes[n]['score'], reverse=True)
    top_n = st.slider("Number of Top Influencers", min_value=1, max_value=10, value=3)
    top_nodes = sorted_nodes[:top_n]

    for rank, node in enumerate(top_nodes, 1):
        gender = G.nodes[node]['gender']
        score = G.nodes[node]['score']
        triggered = G.nodes[node]['triggered_count']
        st.markdown(f"**Rank {rank}: User {node} — Score: {score},** Triggered: {triggered}, Gender: {gender}")

    male_triggered = sum(G.nodes[n]['triggered_count'] for n in G.nodes if G.nodes[n]['gender']=='Male')
    female_triggered = sum(G.nodes[n]['triggered_count'] for n in G.nodes if G.nodes[n]['gender']=='Female')

    st.markdown("---")
    st.markdown(f"**Male Users Triggered Shares:** {male_triggered}")
    st.markdown(f"**Female Users Triggered Shares:** {female_triggered}")

    st.markdown("---")
    st.markdown("Use the slider to adjust the number of top influencers shown.")
