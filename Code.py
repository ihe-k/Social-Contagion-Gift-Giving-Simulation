# Streamlit + Network Simulation: Influencer-Focused Visualization
import random
import networkx as nx
import matplotlib.pyplot as plt
from textblob import TextBlob
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import feedparser
import matplotlib.patches as mpatches

# --- Parameters ---
NUM_USERS = 300
INIT_SHARED = 3
GIFT_BONUS = 10
IDEOLOGY_CROSS_BONUS = 0.2
CHRONIC_PROPENSITY = 0.6
GENDER_HOMOPHILY_BONUS = 0.2
MAX_ITER = 20

st.set_page_config(layout="wide")
st.title("🧠 Health Information Contagion Network Simulation")

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

# --- Step 2: Sentiment Analysis ---
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.5:
        return 'pro-health'
    elif polarity < -0.5:
        return 'anti-health'
    else:
        return 'neutral'

# --- Step 3: Fetch Podcasts ---
def get_podcasts_from_rss(feed_url, max_items=5):
    feed = feedparser.parse(feed_url)
    return [{"user": entry.get('author', 'podcaster'), "content": entry.title} for entry in feed.entries[:max_items]]

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
    except:
        pass

podcast_sentiments = [analyze_sentiment(p['content']) for p in podcast_items]
if not podcast_sentiments:
    podcast_sentiments = ['neutral'] * 10

# --- Step 4: Assign Attributes ---
counts = {s: podcast_sentiments.count(s) for s in ['pro-health', 'anti-health', 'neutral']}
total = sum(counts.values()) or 1
weights = {k: v / total for k, v in counts.items()}

for node in G.nodes:
    G.nodes[node]['gender'] = random.choice(['Male', 'Female'])
    G.nodes[node]['has_chronic_disease'] = random.choice([True, False])
    G.nodes[node]['ideology'] = random.choices(['pro-health', 'anti-health', 'neutral'],
        weights=[weights.get('pro-health', 0.33), weights.get('anti-health', 0.33), weights.get('neutral', 0.33)])[0]
    G.nodes[node]['sentiment'] = G.nodes[node]['ideology']

# --- Step 5: Features & Labels ---
def calc_sentiment_trends():
    trends = []
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        if neighbors:
            pro_count = sum(1 for n in neighbors if G.nodes[n]['sentiment'] == 'pro-health')
            trends.append(pro_count / len(neighbors))
        else:
            trends.append(0)
    return trends

sentiment_trends = calc_sentiment_trends()
betweenness = nx.betweenness_centrality(G)

features, labels = [], []
for node in G.nodes:
    n = G.nodes[node]
    features.append([
        1 if n['gender'] == 'Female' else 0,
        1 if n['has_chronic_disease'] else 0,
        sentiment_trends[node],
        betweenness[node]
    ])
    labels.append(n['ideology'])

le = LabelEncoder()
y_encoded = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(features, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# --- Step 6: Train Logistic Regression ---
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

st.subheader("📊 Model Evaluation")
st.write(f"**Test Accuracy:** {accuracy_score(y_test, y_pred):.2%}")
report_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)).transpose().round(2)
st.dataframe(report_df)

cv_scores = cross_val_score(logreg, X_train, y_train, cv=StratifiedKFold(n_splits=5))
st.write(f"**Cross-validated Accuracy:** {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

# --- Step 7: Contagion Simulation ---
st.subheader("🦠 Contagion Simulation")
SHARE_PROB = st.sidebar.slider("Base Share Probability", 0.0, 1.0, 0.3, 0.05)

# Reset contagion attributes
for node in G.nodes:
    G.nodes[node].update({'shared': False, 'gifted': False, 'triggered_count': 0, 'score': 0})

seed_nodes = random.sample(list(G.nodes), INIT_SHARED)
for node in seed_nodes:
    G.nodes[node]['shared'] = True
    G.nodes[node]['gifted'] = True

contagion, current = [set(seed_nodes)], set(seed_nodes)
iterations = 0

while current and iterations < MAX_ITER:
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
                    G.nodes[u]['triggered_count'] += 1
                    if G.nodes[u]['gender'] != G.nodes[v]['gender'] and G.nodes[u]['ideology'] != G.nodes[v]['ideology']:
                        G.nodes[u]['gifted'] = True
                    next_step.add(v)
    contagion.append(next_step)
    current = next_step
    iterations += 1

# --- Step 8: Metrics ---
gifted = [n for n in G.nodes if G.nodes[n]['gifted']]
others = [n for n in G.nodes if not G.nodes[n]['gifted']]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Users", len(G))
    st.metric("Gifted Bridgers", len(gifted))
with col2:
    st.metric("Avg Influence (Gifted)", f"{np.mean([G.nodes[n]['triggered_count'] for n in gifted]):.2f}")
    st.metric("Avg Influence (Others)", f"{np.mean([G.nodes[n]['triggered_count'] for n in others]):.2f}")
with col3:
    st.metric("Average Influence", f"{np.mean([G.nodes[n]['triggered_count'] for n in G.nodes]):.2f}")

# --- Step 9: Influencer Visualization ---
st.subheader("🧠 Influencer Network (By Ideology)")

influence_threshold = 2
H = G.subgraph([n for n in G.nodes if G.nodes[n]['triggered_count'] >= influence_threshold]).copy()

if len(H.nodes) < 2:
    st.warning("⚠️ Not enough influential nodes to display a network. Try increasing SHARE_PROB or changing simulation settings.")
else:
    ideology_colors = {
        'pro-health': '#2ca02c',
        'anti-health': '#d62728',
        'neutral': '#1f77b4'
    }
    node_colors = [ideology_colors[H.nodes[n]['ideology']] for n in H.nodes]
    node_sizes = [400 + 100 * H.nodes[n]['triggered_count'] for n in H.nodes]

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(H, seed=42, k=0.3)
    nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=node_sizes, ax=ax, edgecolors='black', alpha=0.85)
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.3)
    nx.draw_networkx_labels(H, pos, font_size=8, ax=ax)
    ax.set_title("Influencer Subgraph (Clustered by Ideology)")
    ax.axis('off')
    legend = [mpatches.Patch(color=color, label=label) for label, color in ideology_colors.items()]
    ax.legend(handles=legend)
    st.pyplot(fig)

# --- Step 10: Interpretation ---
with st.expander("ℹ️ How to Read the Network"):
    st.markdown("""
    - **Node Colors:** Ideology (Pro-health, Anti-health, Neutral)
    - **Node Size:** Number of users they influenced
    - **Edges:** User interactions during sharing
    - **Focus:** Only influential users (triggered ≥ 2) are shown
    """)
