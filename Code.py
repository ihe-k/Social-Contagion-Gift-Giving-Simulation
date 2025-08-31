import random
import networkx as nx
import matplotlib.pyplot as plt
from textblob import TextBlob
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
import feedparser
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

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

rss_urls = [
    "https://feeds.npr.org/510307/rss.xml",  # NPR Life Kit Health
    "https://feeds.simplecast.com/54nAGcIl",  # Stuff You Should Know
    "https://rss.art19.com/the-daily",        # The Daily by NYT
    "https://feeds.megaphone.fm/ADL9840290619", # Revisionist History
    "https://feeds.megaphone.fm/ADV7473928103", # The Indicator from Planet Money
    "https://feeds.megaphone.fm/ADV7473928103", # Hidden Brain
    "https://feeds.simplecast.com/7f8246b5",   # TED Talks Daily
    "https://feeds.npr.org/510289/podcast.xml", # All Things Considered
    "https://feeds.megaphone.fm/ADV7473928103", # Planet Money
    "https://feeds.megaphone.fm/ADV7473928103", # Freakonomics Radio
    "https://feeds.megaphone.fm/ADV7473928103", # The Moth
    "https://feeds.megaphone.fm/ADV7473928103", # Science Vs
    # Add more RSS feeds here
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
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)

    # Define color mapping for the different sentiment values
    sentiment_colors = {
        'pro-health': 'green',      # Green for pro-health
        'anti-health': 'red',       # Red for anti-health
        'neutral': 'gray'           # Gray for neutral
    }

    # Assign a color to each node based on its sentiment
    node_colors = [sentiment_colors.get(G.nodes[node]['sentiment'], 'gray') for node in G.nodes]

    # Draw the network
    nx.draw(G, pos, with_labels=True, node_size=700, node_color=node_colors, font_size=10)
    plt.title("Health Information Contagion Network")
    st.pyplot(plt)

# Display Network Diagram
st.subheader("Network Diagram")
visualize_network()

# Display metrics or results
st.subheader("Contagion Spread")
st.write(f"Total Users: {NUM_USERS}")
st.write(f"Initial Shared: {INIT_SHARED}")
st.write(f"Contagion Spread (total shared): {sum(1 for node in G.nodes if G.nodes[node]['shared'])}")
