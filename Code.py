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
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# --- Parameters ---
NUM_USERS = 300  # increased to 300
INIT_SHARED = 3
GIFT_BONUS = 10
IDEOLOGY_CROSS_BONUS = 0.2
CHRONIC_PROPENSITY = 0.6
GENDER_HOMOPHILY_BONUS = 0.2

st.title("Health Information Contagion Network Simulation")

# --- Step 1: Network Setup (Users Only) ---
G = nx.erdos_renyi_graph(NUM_USERS, 0.05, seed=42)  # sparser graph for better visualization
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
total = sum(counts.values()) or 1
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
        sentiment_trends[node],
        betweenness_centrality[node]
    ]
    user_features.append(features)
    user_labels.append(u['ideology'])

# --- Step 6: Logistic Regression Model Training & Evaluation ---

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(user_labels)

# Train/test split with stratify
X_train, X_test, y_train, y_test = train_test_split(
    user_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Initialize Logistic Regression model
logreg = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

logreg.fit(X_train, y_train)

# Predictions
y_pred = logreg.predict(X_test)

# Metrics
test_accuracy = accuracy_score(y_test, y_pred)
st.subheader("Model Evaluation (Logistic Regression)")
st.write(f"**Test Accuracy:** {test_accuracy:.2%}")

report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose().round(2)
st.dataframe(report_df)

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(logreg, X_train, y_train, cv=skf)
st.write(f"**Cross-validated Accuracy (train set):** {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

# --- Step 7: Contagion Simulation with Bridging Gifts ---
st.subheader("Contagion Simulation")

SHARE_PROB = st.sidebar.slider("Base Share Probability", 0.0, 1.0, 0.3, 0.05)

pos = nx.spring_layout(G, seed=42)
seed_nodes = random.sample(list(G.nodes), INIT_SHARED)

# Reset contagion-related attributes
for node in G.nodes:
    G.nodes[node]['shared'] = False
    G.nodes[node]['gifted'] = False
    G.nodes[node]['triggered_count'] = 0
    G.nodes[node]['score'] = 0

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
                    G.nodes[u]['triggered_count'] += 1
                    
                    # Reward users who bridge both gender and ideology boundaries
                    if (G.nodes[u]['gender'] != G.nodes[v]['gender']) and (G.nodes[u]['ideology'] != G.nodes[v]['ideology']):
                        G.nodes[u]['gifted'] = True
                    
                    next_step.add(v)
    if not next_step:
        break
    contagion.append(next_step)
    current = next_step

# --- Step 8: Dashboard: Show Reward & Influence Stats ---
gifted_nodes = [n for n in G.nodes if G.nodes[n]['gifted']]
gifted_influences = [G.nodes[n]['triggered_count'] for n in gifted_nodes]
other_nodes = [n for n in G.nodes if not G.nodes[n]['gifted']]
other_influences = [G.nodes[n]['triggered_count'] for n in other_nodes]

total_users = len(G.nodes)
num_gifted = len(gifted_nodes)
avg_influence_gifted = np.mean(gifted_influences) if gifted_influences else 0
avg_influence_others = np.mean(other_influences) if other_influences else 0
avg_score = np.mean([G.nodes[n]['score'] for n in G.nodes])
avg_influence = np.mean([G.nodes[n]['triggered_count'] for n in G.nodes])

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Users", total_users)
    st.metric("Gifted Bridgers", num_gifted)
with col2:
    st.metric("Avg Influence (Gifted)", f"{avg_influence_gifted:.2f}")
    st.metric("Avg Influence (Others)", f"{avg_influence_others:.2f}")
with col3:
    st.metric("Average User Score", f"{avg_score:.2f}")
    st.metric("Average Influence (Triggered Shares)", f"{avg_influence:.2f}")

# --- Step 9: Visualization ---
# --- Step 9: Simplified Influencer Visualization by Ideology ---
st.subheader("Influencer Network (Clustered by Ideology)")

influence_threshold = 2
influential_nodes = [n for n in G.nodes if G.nodes[n]['triggered_count'] >= influence_threshold]
H = G.subgraph(influential_nodes).copy()

if len(H.nodes) < 2:
    st.warning("Not enough influential nodes to display a network. Try increasing SHARE_PROB or adjusting parameters.")
else:
    # Positioning with cluster-aware layout
    pos = nx.spring_layout(H, seed=42, k=0.3)

    # Color by ideology
    ideology_colors = {
        'pro-health': '#2ca02c',
        'anti-health': '#d62728',
        'neutral': '#1f77b4'
    }
    node_colors = [ideology_colors.get(H.nodes[n]['ideology'], '#999999') for n in H.nodes]
    node_sizes = [400 + 100 * H.nodes[n]['triggered_count'] for n in H.nodes]

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=node_sizes, ax=ax, alpha=0.85, edgecolors='black')
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.3)
    nx.draw_networkx_labels(H, pos, font_size=8, ax=ax)

    # Legend
    patches = [mpatches.Patch(color=color, label=ideology) for ideology, color in ideology_colors.items()]
    ax.legend(handles=patches, loc='best')

    ax.set_title("Influencer Subgraph (Clustered by Ideology)")
    ax.axis('off')
    st.pyplot(fig)

# --- Step 10: Interpretation ---
with st.expander("ℹ️ Interpretation of the Network Diagram"):
    st.markdown("""
    ### **Network Diagram Interpretation**

    - **Node Colors** represent user gender (lightgreen = Male, lightblue = Female).
    - **Node Size** reflects the number of triggered shares (higher influence).
    - **Node Border Thickness** corresponds to betweenness centrality (network influence).
    - **Edges** are colored by the average gender of connected nodes.
    - **Gifted Bridgers** are users who helped spread information across different genders and ideologies.

    This visualization helps identify influential users and information pathways in the network.
    """)  # <--- Make sure to close triple quotes here
