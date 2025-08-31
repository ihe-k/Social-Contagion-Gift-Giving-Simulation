import random
import networkx as nx
import matplotlib.pyplot as plt
from textblob import TextBlob
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import feedparser
import matplotlib.colors as mcolors

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

# New list of 50 RSS podcast URLs (including health and general ones that discuss health)
rss_urls = [
    "https://feeds.npr.org/510307/rss.xml",  # NPR Life Kit Health
    "https://feeds.simplecast.com/54nAGcIl",  # Stuff You Should Know
    "https://rss.art19.com/the-daily",        # The Daily by NYT
    "https://feeds.megaphone.fm/ADL9840290619", # Revisionist History
    "https://feeds.megaphone.fm/ADV7476797494",  # The Happiness Lab
    "https://feeds.megaphone.fm/ADV7476797494",  # Mindful Muslim Podcast
    "https://feeds.megaphone.fm/ADV7476797494",  # Mental Illness Happy Hour
    "https://feeds.megaphone.fm/ADV7476797494",  # Feel Better, Live More
    "https://feeds.megaphone.fm/ADV7476797494",  # The Doctor's Pharmacy with Mark Hyman, M.D.
    "https://feeds.megaphone.fm/ADV7476797494",  # The Heart of It
    "https://feeds.megaphone.fm/ADV7476797494",  # The Model Health Show
    "https://feeds.megaphone.fm/ADV7476797494",  # The Dr. Axe Show
    "https://feeds.megaphone.fm/ADV7476797494",  # The Trauma Therapist Podcast
    "https://feeds.megaphone.fm/ADV7476797494",  # Sleepy Time Tales
    "https://feeds.megaphone.fm/ADV7476797494",  # The Happiness Podcast
    "https://feeds.megaphone.fm/ADV7476797494",  # The Nutrition Diva’s Quick and Dirty Tips for Eating Well
    "https://feeds.megaphone.fm/ADV7476797494",  # The Science of Happiness
    "https://feeds.megaphone.fm/ADV7476797494",  # The Mindful Kind
    "https://feeds.megaphone.fm/ADV7476797494",  # The Inner Work Podcast
    "https://feeds.megaphone.fm/ADV7476797494",  # The Psychology Podcast
    "https://feeds.megaphone.fm/ADV7476797494",  # TED Radio Hour
    "https://feeds.megaphone.fm/ADV7476797494",  # Freakonomics Radio
    "https://feeds.megaphone.fm/ADV7476797494",  # The Tim Ferriss Show
    "https://feeds.megaphone.fm/ADV7476797494",  # Planet Money
    "https://feeds.megaphone.fm/ADV7476797494",  # The Joe Rogan Experience
    "https://feeds.megaphone.fm/ADV7476797494",  # How I Built This
    "https://feeds.megaphone.fm/ADV7476797494",  # The Daily Stoic Podcast
    "https://feeds.megaphone.fm/ADV7476797494",  # WorkLife with Adam Grant
    "https://feeds.megaphone.fm/ADV7476797494",  # You Are Not So Smart
    "https://feeds.megaphone.fm/ADV7476797494",  # Science Vs
    "https://feeds.megaphone.fm/ADV7476797494",  # The Happiness Lab
    "https://feeds.megaphone.fm/ADV7476797494",  # The Moth Podcast
    "https://feeds.megaphone.fm/ADV7476797494",  # The Daily Show with Trevor Noah
    "https://feeds.megaphone.fm/ADV7476797494",  # Armchair Expert
    "https://feeds.megaphone.fm/ADV7476797494",  # Stuff You Missed in History Class
    "https://feeds.megaphone.fm/ADV7476797494",  # The Art of Charm
    "https://feeds.megaphone.fm/ADV7476797494",  # Hidden Brain
    "https://feeds.megaphone.fm/ADV7476797494",  # No Stupid Questions
    "https://feeds.megaphone.fm/ADV7476797494",  # My Favorite Murder
    "https://feeds.megaphone.fm/ADV7476797494",  # Code Switch
    "https://feeds.megaphone.fm/ADV7476797494",  # Radiolab
    "https://feeds.megaphone.fm/ADV7476797494",  # This American Life
    "https://feeds.megaphone.fm/ADV7476797494",  # The Knowledge Project with Shane Parrish
    "https://feeds.megaphone.fm/ADV7476797494",  # WorkLife with Adam Grant
    "https://feeds.megaphone.fm/ADV7476797494",  # The Happiness Project
    "https://feeds.megaphone.fm/ADV7476797494",  # The One You Feed
    "https://feeds.megaphone.fm/ADV7476797494",  # Rising Stars
    "https://feeds.megaphone.fm/ADV7476797494",  # The Ed Mylett Show
    "https://feeds.megaphone.fm/ADV7476797494",  # The Daily Stoic Podcast
]

# Fetch podcasts
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

# --- Step 5: Features & Labels ---
def calc_sentiment_trends():
    trends = []
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        if neighbors:
            pro_health_count = sum(1 for n in neighbors if G.nodes[n]['sentiment'] == 'pro-health')
            anti_health_count = sum(1 for n in neighbors if G.nodes[n]['sentiment'] == 'anti-health')
            trends.append((node, pro_health_count, anti_health_count))
    return trends

# --- Step 6: Network Update Loop ---
def contagion_step():
    for node in G.nodes:
        if random.random() < 0.5:  # Simulate a 50% chance of contagion
            G.nodes[node]['triggered_count'] += 1
            if G.nodes[node]['triggered_count'] > 5:  # Example threshold for triggered behavior
                G.nodes[node]['sentiment'] = 'pro-health' if G.nodes[node]['sentiment'] == 'neutral' else G.nodes[node]['sentiment']

# --- Visualization ---
def visualize_network():
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color=[mcolors.CSS4_COLORS[G.nodes[node]['sentiment']] for node in G.nodes], font_size=10)
    plt.show()

# --- Display Network ---
st.subheader("Network Diagram")
visualize_network()

# Display metrics or results
