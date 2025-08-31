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
CROSS_GENDER_BONUS = 0.3  # Increased probability for cross-gender connections
CROSS_IDEOLOGY_BONUS = 0.3  # Increased probability for cross-ideology connections

st.title("Health Information Contagion Network Simulation")

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
        ['pro-health', 'anti-health', 'neutral'],
        weights=[weights.get('pro-health', 0.33),
                 weights.get('anti-health', 0.33),
                 weights.get('neutral', 0.33)],
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

# --- Step 7: Model Evaluation ---
st.subheader("Model Evaluation")
accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(2)
st.write(f"**Accuracy:** {accuracy:.2%}")
st.dataframe(report_df)

# --- Step 8: Contagion Simulation ---
st.sidebar.header("Simulation Parameters")
SHARE_PROB = st.sidebar.slider("Base Share Probability", 0.0, 1.0, 0.3, 0.05)

# Move radio button to sidebar
view_option = st.sidebar.radio(
    "Select Network View",
    ("Male/Female Distribution", "Cross-Ideology & Gender Distribution")
)

pos = nx.spring_layout(G, seed=42)
seed_nodes = random.sample(list(G.nodes), INIT_SHARED)
for node in G.nodes:
    G.nodes[node]['shared'] = False
    G.nodes[node]['gifted'] = False
    G.nodes[node]['triggered_count'] = 0

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
                    prob += IDEOLOGY_CROSS_BONUS + CROSS_IDEOLOGY_BONUS
                if G.nodes[v]['has_chronic_disease']:
                    prob = max(prob, CHRONIC_PROPENSITY)
                if G.nodes[u]['gender'] != G.nodes[v]['gender']:  # Cross-gender connection boost
                    prob += CROSS_GENDER_BONUS
                prob = min(max(prob, 0), 1)
                if random.random() < prob:
                    G.nodes[v]['shared'] = True
                    G.nodes[v]['triggered_count'] += 1
                    next_step.add(v)
    if not next_step:
        break
    contagion.append(next_step)
    current = next_step

# --- Step 9: Visualization ---
st.subheader("User Network Contagion Simulation")

# Dashboard summary
triggered_shares = sum([G.nodes[node]['triggered_count'] for node in G.nodes])
avg_triggered_shares = triggered_shares / NUM_USERS
avg_ideology_influence = np.mean([sentiment_trends[node] for node in G.nodes])

st.write(f"**Total Shares Triggered**: {triggered_shares}")
st.write(f"**Average Shares Triggered per User**: {avg_triggered_shares:.2f}")
st.write(f"**Average Ideology Influence**: {avg_ideology_influence:.2f}")
st.write(f"**Average Betweenness Centrality**: {np.mean(list(betweenness_centrality.values())):.2f}")

fig_net, ax_net = plt.subplots(figsize=(8, 6))

# Define colors for ideologies
ideology_colors = {
    'pro-health': '#003A6B',  # Dark Blue
    'neutral': '#5293BB',      # Light Blue
    'anti-health': '#89CFF1'   # Light Blue (lighter for anti-health)
}

# --- Prepare node colors and sizes ---
node_colors = []
node_sizes = []
node_border_widths = []

# Normalize betweenness centrality for border widths
bc_values = np.array([betweenness_centrality[n] for n in G.nodes])
if bc_values.max() > 0:
    norm_bc = 1 + 5 * (bc_values - bc_values.min()) / (bc_values.max() - bc_values.min())
else:
    norm_bc = np.ones(len(G.nodes))

for idx, n in enumerate(G.nodes):
    # Assign color based on ideology
    color = ideology_colors[G.nodes[n]['ideology']]
    node_colors.append(color)
    node_sizes.append(300 + 100 * G.nodes[n]['triggered_count'])
    node_border_widths.append(norm_bc[idx])

# --- Prepare edge colors and widths ---
edge_colors = []
edge_widths = []

for u, v in G.edges:
    if G.nodes[u]['gender'] != G.nodes[v]['gender'] or G.nodes[u]['ideology'] != G.nodes[v]['ideology']:
        edge_colors.append('red')  # Cross-gender or Cross-ideology connections
        edge_widths.append(2)      # Thicker red edges for cross-gender or cross-ideology ties
    else:
        edge_colors.append('#AAAAAA')  # Grey edges for same-gender and same-ideology ties
        edge_widths.append(1)          # Normal width for regular edges

# --- Draw the Network ---
nx.draw_networkx(
    G,
    pos=pos,
    with_labels=True,
    labels={n: str(n) for n in G.nodes},
    node_size=node_sizes,
    node_color=node_colors,
    edge_color=edge_colors,
    width=edge_widths,  # Uniform edge widths
    style='solid',
    font_size=8,
    font_color='white',  # Make font color white
    edge_cmap=plt.cm.Blues,  # Color map for edge width
    ax=ax_net
)

# Legend for ideology
pro_health_patch = mpatches.Patch(color='#003A6B', label='Pro-Health')
neutral_patch = mpatches.Patch(color='#5293BB', label='Neutral')
anti_health_patch = mpatches.Patch(color='#89CFF1', label='Anti-Health')
ax_net.legend(handles=[pro_health_patch, neutral_patch, anti_health_patch], loc='best')

# Add title
ax_net.set_title("Health Information Contagion Network")

st.pyplot(fig_net)

# --- Step 11: Explanation ---
with st.expander("ℹ️ Interpretation of the Network Diagram"):
    st.markdown("""
    ### **Network Diagram Interpretation**
    - **Node Colors:**  
      - **Dark blue** represents **Pro-Health users**  
      - **Light blue** represents **Neutral users**  
      - **Lightest blue** represents **Anti-Health users**
    - **Node Size:**  
      Reflects how many other users this node has **influenced or triggered**.  
      Larger nodes = more shares triggered.
    - **Node Border Width:**  
      Indicates **betweenness centrality** — users with **thicker borders** serve as **important bridges** in the network, connecting different parts and enabling information spread. These are key nodes that facilitate the flow of information across ideologies.
    - **Edge Colors (Connections):**  
      - **Red edges** = **Cross-gender** or **Cross-ideology** ties.  
      - **Grey edges** = Connections between users of the same gender and same ideology.
""")

