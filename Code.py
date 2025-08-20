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
    # Add your other podcast RSS URLs here as needed
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


# --- Enhanced Dashboard Summary for Gifted & Influential Users ---
st.subheader("🎁 Rewarded & Influential Users Overview")

# Collect data for gifted users
gifted_nodes = [n for n in G.nodes if G.nodes[n]['gifted']]
num_gifted = len(gifted_nodes)

# Average triggered shares for gifted vs non-gifted users
gifted_triggered = np.mean([G.nodes[n]['triggered_count'] for n in gifted_nodes]) if gifted_nodes else 0
non_gifted_nodes = [n for n in G.nodes if not G.nodes[n]['gifted']]
non_gifted_triggered = np.mean([G.nodes[n]['triggered_count'] for n in non_gifted_nodes]) if non_gifted_nodes else 0

# Scores distribution
scores = [G.nodes[n]['score'] for n in G.nodes]
avg_score = np.mean(scores)

# Influence = triggered_count (how many they influenced)
influence = [G.nodes[n]['triggered_count'] for n in G.nodes]
avg_influence = np.mean(influence)

# Gender and ideology counts
gender_counts = pd.Series([G.nodes[n]['gender'] for n in G.nodes]).value_counts()
ideology_counts = pd.Series([G.nodes[n]['ideology'] for n in G.nodes]).value_counts()
chronic_counts = pd.Series([G.nodes[n]['has_chronic_disease'] for n in G.nodes]).value_counts()

# Top gifted users table
top_gifted = sorted(gifted_nodes, key=lambda n: G.nodes[n]['triggered_count'], reverse=True)[:5]
top_gifted_data = [{
    'User': n,
    'Triggered Shares': G.nodes[n]['triggered_count'],
    'Score': G.nodes[n]['score'],
    'Gender': G.nodes[n]['gender'],
    'Ideology': G.nodes[n]['ideology']
} for n in top_gifted]

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Users", NUM_USERS)
col2.metric("Gifted Bridgers", num_gifted)
col3.metric("Avg Influence (Gifted)", f"{gifted_triggered:.2f}")
col4.metric("Avg Influence (Others)", f"{non_gifted_triggered:.2f}")

st.markdown(f"**Average User Score:** {avg_score:.2f}")
st.markdown(f"**Average Influence (Triggered Shares):** {avg_influence:.2f}")

# Plots row
colg, coli, colc = st.columns(3)

with colg:
    st.markdown("### Gender Distribution")
    st.bar_chart(gender_counts)

with coli:
    st.markdown("### Ideology Distribution")
    st.bar_chart(ideology_counts)

with colc:
    st.markdown("### Chronic Disease Status")
    st.bar_chart(chronic_counts)

# Gifted vs non-gifted users bar chart
counts = {'Gifted': num_gifted, 'Non-Gifted': NUM_USERS - num_gifted}
st.markdown("### Gifted vs Non-Gifted Users")
st.bar_chart(pd.DataFrame.from_dict(counts, orient='index', columns=['Count']))

# Top gifted users table
if top_gifted_data:
    st.markdown("#### Top Gifted Users by Influence")
    st.table(top_gifted_data)
else:
    st.write("No gifted users detected yet.")

# --- Step 8: Contagion Simulation ---

# First, identify and gift users who bridge ideology and gender boundaries
for node in G.nodes:
    neighbors = list(G.neighbors(node))
    cross_ideology_neighbors = [n for n in neighbors if G.nodes[n]['ideology'] != G.nodes[node]['ideology']]
    cross_gender_neighbors = [n for n in neighbors if G.nodes[n]['gender'] != G.nodes[node]['gender']]
    
    # Gift if user has neighbors with both different ideology and different gender
    if cross_ideology_neighbors and cross_gender_neighbors:
        G.nodes[node]['gifted'] = True
    else:
        G.nodes[node]['gifted'] = False

st.sidebar.header("Simulation Parameters")
SHARE_PROB = st.sidebar.slider("Base Share Probability", 0.0, 1.0, 0.3, 0.05)

pos = nx.spring_layout(G, seed=42)
seed_nodes = random.sample(list(G.nodes), INIT_SHARED)

# Reset contagion status for all nodes
for node in G.nodes:
    G.nodes[node]['shared'] = False
    G.nodes[node]['triggered_count'] = 0

# Seed initial spreaders
for node in seed_nodes:
    G.nodes[node]['shared'] = True

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
# --- Step 9: Visualization ---
st.subheader("User Network Contagion Simulation")
fig_net, ax_net = plt.subplots(figsize=(8, 6))

def darken_color(color, amount=0.6):
    c = mcolors.to_rgb(color)
    darkened = tuple(max(min(x * amount, 1), 0) for x in c)
    return darkened

node_colors = []
edge_colors = []
node_sizes = []
node_border_widths = []

# Normalize betweenness centrality for border widths (scale 1 to 6 for visibility)
bc_values = np.array([betweenness_centrality[n] for n in G.nodes])
if bc_values.max() > 0:
    norm_bc = 1 + 5 * (bc_values - bc_values.min()) / (bc_values.max() - bc_values.min())
else:
    norm_bc = np.ones(len(G.nodes))

for idx, n in enumerate(G.nodes):
    color = 'lightgreen' if G.nodes[n]['gender'] == 'Male' else 'lightblue'
    node_colors.append(color)
    node_sizes.append(300 + 100 * G.nodes[n]['triggered_count'])
    node_border_widths.append(norm_bc[idx])

for u, v in G.edges:
    color_u = 'lightgreen' if G.nodes[u]['gender'] == 'Male' else 'lightblue'
    color_v = 'lightgreen' if G.nodes[v]['gender'] == 'Male' else 'lightblue'
    rgb_u = mcolors.to_rgb(color_u)
    rgb_v = mcolors.to_rgb(color_v)
    mixed_rgb = tuple((x + y) / 2 for x, y in zip(rgb_u, rgb_v))
    dark_edge_color = darken_color(mcolors.to_hex(mixed_rgb), amount=0.6)
    edge_colors.append(dark_edge_color)

# Draw nodes
nx.draw_networkx_nodes(G, pos,
                       node_size=node_sizes,
                       node_color=node_colors,
                       linewidths=node_border_widths,
                       edgecolors='gray',
                       ax=ax_net)

# Draw edges
nx.draw_networkx_edges(G, pos,
                       edge_color=edge_colors,
                       ax=ax_net)

# Prepare label colors by gender
label_colors = {n: '#003A6B' if G.nodes[n]['gender'] == 'Female' else '#1B5886' for n in G.nodes}

# Draw labels with gender-specific colors
for node in G.nodes:
    nx.draw_networkx_labels(
        G, pos,
        labels={node: str(node)},
        font_color=label_colors[node],
        font_size=8,
        ax=ax_net
    )
    
# Legend for genders
male_patch = mpatches.Patch(color='lightgreen', label='Male')
female_patch = mpatches.Patch(color='lightblue', label='Female')
ax_net.legend(handles=[male_patch, female_patch], loc='best')

st.pyplot(fig_net)


# --- Step 10: Network Visualization ---


# --- Step 11: Explanation ---
with st.expander("ℹ️ Interpretation of the Network Diagram"):
    st.markdown("""
    ### **Network Diagram Interpretation**

    - **Node Colors:**  
      - **Green circles** represent **Male users**  
      - **Blue circles** represent **Female users**  

    - **Node Size:**  
      Reflects how many other users this node has **influenced or triggered**.  
      Larger nodes = more shares triggered.

    - **Node Border Width:**  
      Indicates **betweenness centrality** — users with thicker borders serve as **important bridges** in the network, connecting different parts and enabling information spread.

    - **Edge Colors (Connections):**  
      - **Light green edges** = Male-to-Male connections (**gender homophily**)  
      - **Light blue edges** = Female-to-Female connections (**gender homophily**)  
      - **Gray edges** = Male-to-Female or Female-to-Male (**cross-gender ties**)

    - **Clusters:**  
      The network shows **gender homophily** and **ideological alignment** influencing connections and information diffusion.

    - **Overall Insights:**  
      - Users with higher **centrality** act as **key influencers** or bridges.  
      - **Chronic disease status** and **ideological differences** impact sharing probabilities and contagion dynamics.
    """)

