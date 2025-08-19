import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

rss_url = "https://feeds.npr.org/510307/rss.xml"
podcast_items = get_podcasts_from_rss(rss_url)

# --- Step 4: Scrape ListenNotes Podcast Metadata ---
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
all_content = podcast_items

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
    G.nodes[i]['gifted'] = False

# --- Step 6: Feature Engineering ---
def calc_sentiment_trends():
    trends = []
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        if neighbors:
            trends.append(np.mean([1 if G.nodes[n]['sentiment'] == 'pro-health' else 0 for n in neighbors]))
        else:
            trends.append(0)
    return trends

sent_trends = calc_sentiment_trends()
centrality = nx.betweenness_centrality(G)

features, labels = [], []
for n in G.nodes:
    u = G.nodes[n]
    features.append([
        1 if u['gender'] == 'Female' else 0,
        1 if u['has_chronic_disease'] else 0,
        1 if u['ideology'] == 'pro-health' else 0,
        1 if u['ideology'] == 'anti-health' else 0,
        1 if u['ideology'] == 'neutral' else 0,
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

# --- Step 9: Contagion Simulation with Leadership Board Style ---
pos = nx.spring_layout(G, seed=42)

# Initialize gifted users
initial_gifted = random.sample(list(G.nodes), INIT_SHARED)
for node in initial_gifted:
    G.nodes[node]['shared'] = True
    G.nodes[node]['gifted'] = True

contagion_steps = [set(initial_gifted)]

def run_contagion_simulation():
    new_shared = set(initial_gifted)
    all_shared = set(initial_gifted)

    while new_shared:
        next_new_shared = set()
        for user in new_shared:
            for neighbor in G.neighbors(user):
                if not G.nodes[neighbor]['shared']:
                    prob = SHARE_PROB
                    if G.nodes[user]['gifted']:
                        prob += GIFT_BONUS / 100
                    if G.nodes[user]['ideology'] != G.nodes[neighbor]['ideology']:
                        prob += IDEOLOGY_CROSS_BONUS
                    if G.nodes[neighbor]['has_chronic_disease']:
                        prob = max(prob, CHRONIC_PROPENSITY)
                    if G.nodes[user]['gender'] == G.nodes[neighbor]['gender']:
                        prob += GENDER_HOMOPHILY_BONUS
                    prob = min(max(prob, 0), 1)
                    if random.random() < prob:
                        G.nodes[neighbor]['shared'] = True
                        G.nodes[neighbor]['triggered_count'] += 1
                        next_new_shared.add(neighbor)
                        all_shared.add(neighbor)
        if not next_new_shared:
            break
        contagion_steps.append(next_new_shared)
        new_shared = next_new_shared

run_contagion_simulation()

# Animation function
fig, ax = plt.subplots(figsize=(10, 7))

def animate(i):
    ax.clear()
    if i < len(contagion_steps):
        current_shared = contagion_steps[i]
    else:
        current_shared = contagion_steps[-1]

    # Node colors by gender
    male_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Male']
    female_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Female']

    node_sizes = [300 + 100 * G.nodes[n]['triggered_count'] for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, nodelist=male_nodes, node_color='lightgreen', node_size=node_sizes)
    nx.draw_networkx_nodes(G, pos, nodelist=female_nodes, node_color='lightblue', node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5, edge_color='gray')

    # Labels as ideology
    labels = {node: G.nodes[node]['ideology'] for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')

    ax.set_title(f"Step {i + 1}: Podcast Health Info Spread")
    ax.axis('off')

st.subheader("Podcast-Based Health Info Spread Simulation with Leadership Board")
ani = FuncAnimation(fig, animate, frames=len(contagion_steps), interval=1000, repeat=False)
st.pyplot(fig)
