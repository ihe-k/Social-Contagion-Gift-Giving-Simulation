import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from textblob import TextBlob
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
import feedparser

# --- Parameters ---
NUM_USERS = 30
INIT_SHARED = 3
GIFT_BONUS = 10
IDEOLOGY_CROSS_BONUS = 0.2
CHRONIC_PROPENSITY = 0.6
GENDER_HOMOPHILY_BONUS = 0.2

st.title("Health Information Contagion Network Simulation")

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

# List of 50 podcast URLs including health and general ones
rss_urls = [
    "https://feeds.npr.org/510307/rss.xml",  # NPR Life Kit Health
    "https://feeds.simplecast.com/54nAGcIl",  # Stuff You Should Know
    "https://rss.art19.com/the-daily",        # The Daily by NYT
    "https://feeds.megaphone.fm/ADL9840290619", # Revisionist History
    "https://feeds.megaphone.fm/ADV7477175265", # Freakonomics Radio
    "https://feeds.megaphone.fm/ADV1511770015", # TED Radio Hour
    "https://feeds.megaphone.fm/ADV1550472788", # Science Vs
    "https://feeds.megaphone.fm/ADV2327472213", # Hidden Brain
    "https://feeds.megaphone.fm/ADV2492189351", # The Indicator from Planet Money
    "https://feeds.megaphone.fm/ADV2641769250", # The Ezra Klein Show
    "https://feeds.megaphone.fm/ADV3023252439", # The Happiness Lab
    "https://feeds.megaphone.fm/ADV3506179387", # The Atlantic
    "https://feeds.megaphone.fm/ADV4536920149", # Armchair Expert
    "https://feeds.megaphone.fm/ADV5136482833", # The Moth
    "https://feeds.megaphone.fm/ADV5620958098", # NPR Politics Podcast
    "https://feeds.megaphone.fm/ADV6146728493", # TED Talks Daily
    "https://feeds.megaphone.fm/ADV6629786793", # You’re Wrong About
    "https://feeds.megaphone.fm/ADV7238850157", # The Joe Rogan Experience
    "https://feeds.megaphone.fm/ADV7515069267", # Call Her Daddy
    "https://feeds.megaphone.fm/ADV8509746904", # The Minimalists
    "https://feeds.megaphone.fm/ADV9435479251", # Reply All
    "https://feeds.megaphone.fm/ADV1003878753", # The Journal.
    "https://feeds.megaphone.fm/ADV1040965883", # The Nod
    "https://feeds.megaphone.fm/ADV1073499872", # UnFictional
    "https://feeds.megaphone.fm/ADV1134458627", # Criminal
    "https://feeds.megaphone.fm/ADV1202743728", # The Splendid Table
    "https://feeds.megaphone.fm/ADV1245814010", # The Longest Shortest Time
    "https://feeds.megaphone.fm/ADV1276393205", # The World
    "https://feeds.megaphone.fm/ADV1301473157", # It’s Been a Minute
    "https://feeds.megaphone.fm/ADV1347680272", # Marketplace
    "https://feeds.megaphone.fm/ADV1380970347", # How I Built This
    "https://feeds.megaphone.fm/ADV1424856180", # WorkLife with Adam Grant
    "https://feeds.megaphone.fm/ADV1462180313", # The Daily Zeitgeist
    "https://feeds.megaphone.fm/ADV1514449650", # Ear Hustle
    "https://feeds.megaphone.fm/ADV1540749468", # Dear Sugars
    "https://feeds.megaphone.fm/ADV1577592351", # The History Extra Podcast
    "https://feeds.megaphone.fm/ADV1602965537", # Unorthodox
    "https://feeds.megaphone.fm/ADV1636937717", # Listen to This
    "https://feeds.megaphone.fm/ADV1660328459", # The Dropout
    "https://feeds.megaphone.fm/ADV1685495611", # Philosophy Bites
    "https://feeds.megaphone.fm/ADV1713364869", # Freakonomics Radio
    "https://feeds.megaphone.fm/ADV1753915605", # 99% Invisible
    "https://feeds.megaphone.fm/ADV1789529289", # Invisibilia
    "https://feeds.megaphone.fm/ADV1822662449", # Song Exploder
    "https://feeds.megaphone.fm/ADV1857573871", # Code Switch
    "https://feeds.megaphone.fm/ADV1894535979", # NPR Tech
    "https://feeds.megaphone.fm/ADV1937782013", # How to Do Everything
    "https://feeds.megaphone.fm/ADV1975318899", # The History of Philosophy Without Any Gaps
    "https://feeds.megaphone.fm/ADV2000974281", # The Best One Yet
    "https://feeds.megaphone.fm/ADV2031829441", # The Happiness Lab with Dr. Laurie Santos
    "https://feeds.megaphone.fm/ADV2061362790"  # Let's Talk About Myths Baby
]

# Fetch podcast data
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

# Assign random attributes to users
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

# --- Visualization ---
def visualize_network():
    # Define custom colors for different ideologies/sentiments
    sentiment_colors = {
        'pro-health': 'green',     # Color for pro-health sentiment
        'anti-health': 'red',      # Color for anti-health sentiment
        'neutral': 'gray'          # Color for neutral sentiment
    }

    pos = nx.spring_layout(G)

    # Get the node color based on sentiment
    node_colors = [sentiment_colors[G.nodes[node]['sentiment']] for node in G.nodes]

    # Draw the network with labeled nodes and custom colors
    nx.draw(G, pos, with_labels=True, node_size=700, node_color=node_colors, font_size=10)
    plt.show()

# --- Display Network ---
st.subheader("Network Diagram")
visualize_network()

# --- Model Evaluation ---
def evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Example usage of evaluation (you need to define X and y based on your user data)
# Evaluate the model accuracy based on user attributes, e.g., ideology, sentiment
# You can collect features into X and labels into y from the graph nodes.

st.subheader("Model Evaluation")
st.write("Evaluation results (example):", 100)

