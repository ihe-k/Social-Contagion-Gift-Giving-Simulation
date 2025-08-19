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
            "content": entry.title + " " + (entry.get('summary', '') if 'summary' in entry else ''),
            "platform": "RSS",
            "url": entry.link
        })
    return podcasts

# --- Step 4: Scrape ListenNotes show page fallback ---
def scrape_listennotes_show(show_url):
    try:
        resp = requests.get(show_url, timeout=10)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, 'html.parser')
        title = soup.find('h1').text.strip() if soup.find('h1') else "Pod Title"
        desc = soup.find('p').text.strip() if soup.find('p') else ""
        return {"user": title.split()[0], "content": desc, "platform": "Web", "url": show_url}
    except Exception:
        return None

# --- Step 5: Podcast feeds to include ---
rss_feeds = [
    "https://joeroganexp.joerogan.libsynpro.com/rss",           # Joe Rogan Experience
    "https://feeds.simplecast.com/54nAGcIl",                    # Call Her Daddy
    "https://feeds.megaphone.fm/WWO3519750118",                 # This Past Weekend w/ Theo Von
    "https://feeds.npr.org/510318/podcast.xml",                 # Up First (NPR)
    "https://feeds.npr.org/510307/rss.xml",                     # The Daily (NPR)
    "https://melrobbinspodcast.libsyn.com/rss",                 # The Mel Robbins Podcast
    "https://hubermanlab.libsyn.com/rss",                        # Huberman Lab Podcast
    "https://feeds.simplecast.com/8sYogDlc",                     # SmartLess Podcast
    "https://the-diary-of-a-ceo.simplecast.com/rss",            # The Diary Of A CEO
    "https://tuckercarlsonshow.libsyn.com/rss",                 # Tucker Carlson Show (unofficial)
]

podcast_items = []
for feed_url in rss_feeds:
    try:
        podcast_items.extend(get_podcasts_from_rss(feed_url, max_items=5))
    except Exception as e:
        st.warning(f"Failed to fetch or parse feed: {feed_url}")

# Fallback scrape for New Heights with Jason and Travis Kelce (no official RSS found)
new_heights_url = "https://www.listennotes.com/c/2a7e215622c94b66a6b4f15b2275c09c/"
scraped = scrape_listennotes_show(new_heights_url)
if scraped:
    podcast_items.append(scraped)

# --- Step 6: Assign User Attributes ---
user_data = []
for content in podcast_items:
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
    G.nodes[i]['gifted'] = False

# --- Step 7: Features & Labels ---
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

# --- Step 8: Model Training ---
param_grid = {'n_estimators':[100], 'max_depth':[10], 'min_samples_split':[2]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=2, n_jobs=-1)
grid.fit(X_train, y_train)
best = grid.best_estimator_
y_pred = best.predict(X_test)

# --- Step 9: Evaluation ---
st.subheader("Model Evaluation")
st.write(f"Accuracy: {accuracy_score(y_test,y_pred):.2%}")
st.text(classification_report(y_test,y_pred))
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_estimator(best, X_test, y_test, ax=ax)
st.pyplot(fig)

# --- Step 10: Contagion Simulation ---
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
                prob = min(max(prob, 0), 1)
                if random.random() < prob:
                    G.nodes[v]['shared'] = True
                    G.nodes[v]['triggered_count'] += 1
                    next_step.add(v)
    if not next_step: break
    contagion.append(next_step)
    current = next_step

# --- Step 11: Display Graph ---
st.subheader("Podcast-Based Health Info Spread Simulation")
fig, ax = plt.subplots(figsize=(8,6))
nx.draw(G, pos, with_labels=True,
        node_size=[300+100*G.nodes[n]['triggered_count'] for n in G.nodes],
        node_color=['lightgreen' if G.nodes[n]['gender']=='Male' else 'lightblue' for n in G.nodes],
        edge_color='gray', linewidths=1.5,
        node_shape='o',
        nodelist=list(G.nodes),
        font_size=8,
        ax=ax)
st.pyplot(fig)
