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

# --- Constants ---
NUM_USERS = 30
INIT_SHARED = 3
GIFT_BONUS = 0.10
IDEOLOGY_CROSS_BONUS = 0.2
CHRONIC_PROPENSITY = 0.6
GENDER_HOMOPHILY_BONUS = 0.2
BASE_SHARE_PROB = 0.3
RSS_FEEDS = [
    "https://feeds.npr.org/510307/rss.xml",
    "https://feeds.simplecast.com/54nAGcIl",
    "https://rss.art19.com/the-daily",
    "https://feeds.megaphone.fm/ADL9840290619",
]

# --- Streamlit Setup ---
st.title("🧠 Health Information Contagion Network Simulation")

# --- Initialization Functions ---
def create_user_network(n_users):
    G = nx.erdos_renyi_graph(n_users, 0.1, seed=42)
    for node in G.nodes:
        G.nodes[node].update({
            'shared': False,
            'gifted': False,
            'triggered_count': 0,
            'gender': random.choice(['Male', 'Female']),
            'has_chronic_disease': random.choice([True, False]),
            'ideology': '',
            'sentiment': ''
        })
    return G

def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.5:
        return 'pro-health'
    elif polarity < -0.5:
        return 'anti-health'
    else:
        return 'neutral'

def fetch_and_analyze_rss(feeds):
    podcast_items = []
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                podcast_items.append({
                    "content": entry.title,
                    "sentiment": analyze_sentiment(entry.title)
                })
        except:
            continue
    return podcast_items

def assign_user_attributes(G, sentiments):
    sentiment_counts = pd.Series(sentiments).value_counts(normalize=True).to_dict()
    for node in G.nodes:
        ideology = random.choices(
            ['pro-health', 'anti-health', 'neutral'],
            weights=[
                sentiment_counts.get('pro-health', 0.33),
                sentiment_counts.get('anti-health', 0.33),
                sentiment_counts.get('neutral', 0.34)
            ]
        )[0]
        G.nodes[node]['ideology'] = ideology
        G.nodes[node]['sentiment'] = ideology

def extract_features(G):
    sentiment_trends = []
    bc = nx.betweenness_centrality(G)
    features, labels = [], []

    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        pro_health_ratio = sum(G.nodes[n]['sentiment'] == 'pro-health' for n in neighbors) / len(neighbors) if neighbors else 0
        sentiment_trends.append(pro_health_ratio)

    for i, node in enumerate(G.nodes):
        data = G.nodes[node]
        features.append([
            1 if data['gender'] == 'Female' else 0,
            1 if data['has_chronic_disease'] else 0,
            1 if data['ideology'] == 'pro-health' else 0,
            1 if data['ideology'] == 'anti-health' else 0,
            1 if data['ideology'] == 'neutral' else 0,
            sentiment_trends[i],
            bc[node]
        ])
        labels.append(data['ideology'])
    return features, labels, bc

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GridSearchCV(RandomForestClassifier(random_state=42),
                         {'n_estimators': [100], 'max_depth': [10]},
                         cv=2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T.round(2)
    return model.best_estimator_, acc, report

def simulate_contagion(G, share_prob, gifted_nodes):
    contagion = [set(gifted_nodes)]
    for node in G.nodes:
        G.nodes[node].update({'shared': False, 'gifted': False, 'triggered_count': 0})
    for node in gifted_nodes:
        G.nodes[node]['shared'] = True
        G.nodes[node]['gifted'] = True

    current = set(gifted_nodes)
    while current:
        next_wave = set()
        for u in current:
            for v in G.neighbors(u):
                if not G.nodes[v]['shared']:
                    prob = share_prob
                    if G.nodes[u]['gifted']:
                        prob += GIFT_BONUS
                    if G.nodes[u]['ideology'] != G.nodes[v]['ideology']:
                        prob += IDEOLOGY_CROSS_BONUS
                    if G.nodes[v]['has_chronic_disease']:
                        prob = max(prob, CHRONIC_PROPENSITY)
                    if G.nodes[u]['gender'] == G.nodes[v]['gender']:
                        prob += GENDER_HOMOPHILY_BONUS
                    if random.random() < min(prob, 1):
                        G.nodes[v]['shared'] = True
                        G.nodes[v]['triggered_count'] += 1
                        next_wave.add(v)
        if not next_wave:
            break
        contagion.append(next_wave)
        current = next_wave
    return contagion

def attribute_based_layout(G):
    spacing_x = 2
    spacing_y = 2
    gender_map = {'Male': 0, 'Female': 1}
    ideology_map = {'pro-health': 0, 'neutral': 1, 'anti-health': 2}

    pos = {}
    for node in G.nodes:
        gender = G.nodes[node]['gender']
        ideology = G.nodes[node]['ideology']
        row = gender_map[gender]
        col = ideology_map[ideology]
        pos[node] = (
            col * spacing_x + random.uniform(-0.3, 0.3),
            -row * spacing_y + random.uniform(-0.3, 0.3)
        )
    return pos

def draw_network(G, pos, bc):
    fig, ax = plt.subplots(figsize=(8, 6))
    node_colors = ['lightgreen' if G.nodes[n]['gender'] == 'Male' else 'lightblue' for n in G.nodes]
    node_sizes = [300 + 100 * G.nodes[n]['triggered_count'] for n in G.nodes]

    bc_array = np.array([bc[n] for n in G.nodes])
    if bc_array.max() > 0:
        node_borders = 1 + 5 * (bc_array - bc_array.min()) / (bc_array.max() - bc_array.min())
    else:
        node_borders = np.ones(len(G.nodes))

    edge_colors = []
    for u, v in G.edges:
        cu = 'lightgreen' if G.nodes[u]['gender'] == 'Male' else 'lightblue'
        cv = 'lightgreen' if G.nodes[v]['gender'] == 'Male' else 'lightblue'
        mix = tuple((a + b) / 2 for a, b in zip(mcolors.to_rgb(cu), mcolors.to_rgb(cv)))
        edge_colors.append(mcolors.to_hex(mix))

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           edgecolors='gray', linewidths=node_borders, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    legend = [
        mpatches.Patch(color='lightgreen', label='Male'),
        mpatches.Patch(color='lightblue', label='Female'),
        mpatches.Patch(color='white', label='🟢 Pro-Health (X=0)'),
        mpatches.Patch(color='white', label='🟡 Neutral (X=1)'),
        mpatches.Patch(color='white', label='🔴 Anti-Health (X=2)'),
    ]
    ax.legend(handles=legend)
    return fig

# --- Run Simulation ---
podcasts = fetch_and_analyze_rss(RSS_FEEDS)
sentiments = [p['sentiment'] for p in podcasts] or ['neutral'] * 10

G = create_user_network(NUM_USERS)
assign_user_attributes(G, sentiments)
features, labels, bc = extract_features(G)

model, acc, report = train_model(features, labels)
st.subheader("Model Evaluation")
st.write(f"**Accuracy:** {acc:.2%}")
st.dataframe(report)

SHARE_PROB = st.sidebar.slider("Base Share Probability", 0.0, 1.0, BASE_SHARE_PROB, 0.05)
seed_nodes = random.sample(list(G.nodes), INIT_SHARED)
contagion = simulate_contagion(G, SHARE_PROB, seed_nodes)

# --- Visualization ---
st.subheader("Network Contagion Visualization")
pos = attribute_based_layout(G)
fig = draw_network(G, pos, bc)
st.pyplot(fig)

# --- Metrics ---
total_shared = sum(1 for n in G.nodes if G.nodes[n]['shared'])
max_influencer = max(G.nodes, key=lambda n: G.nodes[n]['triggered_count'])
st.markdown(f"**Total Users Informed:** {total_shared}/{NUM_USERS}")
st.markdown(f"**Time Steps to Saturation:** {len(contagion)}")
st.markdown(f"**Most Influential User:** Node {max_influencer} with {G.nodes[max_influencer]['triggered_count']} triggers")

with st.expander("ℹ️ Interpretation Guide"):
    st.markdown("""
    - **Green Nodes** = Male users  
    - **Blue Nodes** = Female users  
    - **Larger nodes** = More influence (shared info more)
    - **Thicker borders** = More central in the network (bridge-like roles)
    - **Left to Right** = Pro → Neutral → Anti-health ideology
    - **Top to Bottom** = Male → Female
    """)
