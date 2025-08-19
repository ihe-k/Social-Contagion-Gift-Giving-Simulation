import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from textblob import TextBlob
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import numpy as np
import feedparser

# --- Parameters ---
NUM_USERS = 30
INIT_SHARED = 3
SHARE_PROB = 0.3
GIFT_BONUS = 10
IDEOLOGY_CROSS_BONUS = 0.2
CHRONIC_PROPENSITY = 0.6
GENDER_HOMOPHILY_BONUS = 0.2

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

# --- Step 3: Fetch Podcasts via RSS (Content Source Only) ---
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

# Example feeds (You can add/remove valid RSS feeds here)
rss_urls = [
    "https://feeds.npr.org/510307/rss.xml",  # NPR Life Kit Health
    "https://rss.art19.com/the-daily",      # The Daily - news podcast example
    "https://feeds.simplecast.com/54nAGcIl", # Reply All - non-health podcast
]

podcast_items = []
for url in rss_urls:
    try:
        podcast_items.extend(get_podcasts_from_rss(url))
    except Exception as e:
        st.warning(f"Failed to fetch or parse feed: {url}")

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

# --- Step 7: Evaluation ---
st.subheader("Model Evaluation")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
st.text(classification_report(y_test, y_pred))
fig_cm, ax_cm = plt.subplots()
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=ax_cm)
st.pyplot(fig_cm)

# --- Step 8: Contagion Simulation ---
pos = nx.spring_layout(G, seed=42)
seed_nodes = random.sample(list(G.nodes), INIT_SHARED)
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
                    G.nodes[v]['triggered_count'] += 1
                    next_step.add(v)
    if not next_step:
        break
    contagion.append(next_step)
    current = next_step

# --- Step 9: Visualization with improved clarity ---
st.subheader("User Network Contagion Simulation")

fig_net, ax_net = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)  # fixed layout for consistency

# Prepare node shapes/colors by gender
male_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Male']
female_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Female']

# Node sizes scaled by triggered_count
male_sizes = [300 + 100 * G.nodes[n]['triggered_count'] for n in male_nodes]
female_sizes = [300 + 100 * G.nodes[n]['triggered_count'] for n in female_nodes]

# Draw nodes by gender with different shapes
nx.draw_networkx_nodes(G, pos,
                       nodelist=male_nodes,
                       node_color='lightgreen',
                       node_size=male_sizes,
                       node_shape='o',
                       ax=ax_net,
                       label='Male')

nx.draw_networkx_nodes(G, pos,
                       nodelist=female_nodes,
                       node_color='lightblue',
                       node_size=female_sizes,
                       node_shape='s',
                       ax=ax_net,
                       label='Female')

# Draw edges with colors based on gender homophily
edge_colors = []
for u, v in G.edges():
    if G.nodes[u]['gender'] == 'Male' and G.nodes[v]['gender'] == 'Male':
        edge_colors.append('lightgreen')
    elif G.nodes[u]['gender'] == 'Female' and G.nodes[v]['gender'] == 'Female':
        edge_colors.append('lightblue')
    else:
        edge_colors.append('gray')

nx.draw_networkx_edges(G, pos, edge_color=edge_colors, ax=ax_net)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=8, ax=ax_net)

# Create legend manually
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Male',
           markerfacecolor='lightgreen', markersize=10),
    Line2D([0], [0], marker='s', color='w', label='Female',
           markerfacecolor='lightblue', markersize=10),
    Line2D([0], [0], color='lightgreen', lw=2, label='Male-Male Share'),
    Line2D([0], [0], color='lightblue', lw=2, label='Female-Female Share'),
    Line2D([0], [0], color='gray', lw=2, label='Male-Female Share'),
]

ax_net.legend(handles=legend_elements, loc='best')

st.pyplot(fig_net)

# --- Step 10: Explanatory Notes ---
st.markdown("""
### Interpretation of Network Contagion Results

- **Nodes represent users**, colored by gender:
    - 🟢 Green circles = Male users
    - 🔵 Blue squares = Female users

- **Node size reflects influence**: Larger nodes indicate users who shared content more often or influenced others more.

- **Edges represent sharing relationships**:
    - Light green edges connect male-to-male shares
    - Light blue edges connect female-to-female shares
    - Grey edges connect male-to-female shares

- **Gender homophily effect**: Users tend to share more within their own gender groups.

- **Ideology influence**: Users with similar health-related ideologies (pro-health, anti-health, neutral) are more likely to share content with each other.

- **Users with chronic diseases** are more likely to spread health information, acting as key amplifiers.

- **Trigger count** tracks how many times a user has been influenced or shared content, helping identify top influencers.

- This network simulation helps visualize how health-related information (and misinformation) spreads through social connections influenced by gender, ideology, and health status.

Use these insights to target interventions, optimize messaging, and understand community dynamics in health communication.
""")
