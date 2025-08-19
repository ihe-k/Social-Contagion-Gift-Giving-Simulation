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

# --- Step 1: Network Setup ---
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
    return 'pro-health' if polarity > 0.5 else ('anti-health' if polarity < -0.5 else 'neutral')

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

rss_url = "https://feeds.npr.org/510307/rss.xml"  # Example RSS feed
podcast_items = get_podcasts_from_rss(rss_url)

# --- Step 4: Scrape Health Podcast Metadata from Listen Notes Pages ---
def scrape_listennotes_show(show_url):
    resp = requests.get(show_url)
    if resp.status_code != 200:
        return None
    soup = BeautifulSoup(resp.text, 'html.parser')
    title = soup.find('h1').text if soup.find('h1') else "Pod Title"
    desc = soup.find('p').text if soup.find('p') else ""
    return {"user": title.split()[0], "content": desc, "platform": "Web", "url": show_url}

ln_url = "https://www.listennotes.com/podcasts/health-insights-podcast-wellness-and-BTgZb84DPEH/"
scraped = scrape_listennotes_show(ln_url)
if scraped:
    podcast_items.append(scraped)

# --- Combine Content ---
all_content = podcast_items  # Only podcasts

# --- Step 5: Assign User Attributes ---
user_data = []
for content in all_content:
    sentiment = analyze_sentiment(content["content"])
    user_data.append({
        'user': content['user'],
        'gender': random.choice(['Male', 'Female']),
        'sentiment': sentiment,
        'ideology': sentiment,
        'has_chronic_disease': random.choice([True, False])
    })

for i, u in enumerate(user_data):
    if i >= NUM_USERS:
        break
    for k in u:
        G.nodes[i][k] = u[k]
    G.nodes[i]['score'] = 0
    G.nodes[i]['triggered_count'] = 0
    G.nodes[i]['shared'] = False

# --- Step 6: Features & Labels ---
def calc_sentiment_trends():
    return [np.mean([1 if G.nodes[n]['sentiment']=='pro-health' else 0 for n in G.neighbors(node)]) if list(G.neighbors(node)) else 0 for node in G.nodes]

sent_trends = calc_sentiment_trends()
centrality = nx.betweenness_centrality(G)

features, labels = [], []
for n in G.nodes:
    u = G.nodes[n]
    features.append([
        1 if u['gender']=='Female' else 0,
        1 if u['has_chronic_disease'] else 0,
        1 if u['ideology']=='pro-health' else 0,
        1 if u['ideology']=='anti-health' else 0,
        1 if u['ideology']=='neutral' else 0,
        sent_trends[n],
        centrality[n]
    ])
    labels.append(u['ideology'])

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# --- Step 7: Model Training ---
param_grid = {'n_estimators':[100], 'max_depth':[10], 'min_samples_split':[2]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=2, n_jobs=-1)
grid.fit(X_train, y_train)
best = grid.best_estimator_
y_pred = best.predict(X_test)

# --- Step 8: Evaluation ---
st.subheader("Model Evaluation")
st.write(f"Accuracy: {accuracy_score(y_test,y_pred):.2%}")
st.text(classification_report(y_test,y_pred))
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_estimator(best, X_test, y_test, ax=ax)
st.pyplot(fig)

# --- Step 9: Contagion Simulation ---
pos = nx.spring_layout(G, seed=42)
seed = random.sample(list(G.nodes), INIT_SHARED)
for node in seed:
    G.nodes[node]['shared'] = True
    G.nodes[node]['gifted'] = True

contagion, current = [set(seed)], set(seed)
while current:
    next_step = set()
    for u in current:
        for v in G.neighbors(u):
            if not G.nodes[v]['shared']:
                prob = SHARE_PROB + (GIFT_BONUS/100 if G.nodes[u]['gifted'] else 0)
                if G.nodes[u]['ideology'] != G.nodes[v]['ideology']:
                    prob += IDEOLOGY_CROSS_BONUS
                if G.nodes[v]['has_chronic_disease']:
                    prob = max(prob, CHRONIC_PROPENSITY)
                if G.nodes[u]['gender'] == G.nodes[v]['gender']:
                    prob += GENDER_HOMOPHILY_BONUS
                if random.random() < min(max(prob,0),1):
                    G.nodes[v]['shared'] = True
                    G.nodes[v]['triggered_count'] += 1
                    next_step.add(v)
    if not next_step: break
    contagion.append(next_step)
    current = next_step

# --- Step 10: Static Contagion Plot ---
st.subheader("Podcast-Based Health Info Spread Simulation (Static)")

fig, ax = plt.subplots(figsize=(10,7))

# Color nodes by gender
node_colors = ['lightgreen' if G.nodes[n]['gender']=='Male' else 'lightblue' for n in G.nodes]
# Size nodes by influence (triggered_count)
node_sizes = [300 + 100 * G.nodes[n]['triggered_count'] for n in G.nodes]

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5, edge_color='gray', ax=ax)

# Show ideology labels on nodes
labels = {n: G.nodes[n]['ideology'] for n in G.nodes}
nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black', ax=ax)

ax.set_title("Health Information Contagion Spread (Static)")
ax.axis('off')
st.pyplot(fig)
