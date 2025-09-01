import random
import networkx as nx
import matplotlib.pyplot as plt
from textblob import TextBlob
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerBase
import matplotlib.patches as mpatches

# --- Parameters ---
NUM_USERS = 300
INIT_SHARED = 3
GIFT_BONUS = 10
IDEOLOGY_CROSS_BONUS = 0.2
CHRONIC_PROPENSITY = 0.6
GENDER_HOMOPHILY_BONUS = 1.5
CROSS_GENDER_BONUS = 0.3
CROSS_IDEOLOGY_BONUS = 0.3
SHARE_PROB = 0.2
CROSS_IDEOLOGY_REDUCTION_FACTOR = 0.9
CROSS_GENDER_REDUCTION_FACTOR = 0.7
IDEOLOGY_HOMOPHILY_BONUS = 1.5
K_THRESHOLD = 3

# --- Network Setup ---
G = nx.erdos_renyi_graph(NUM_USERS, 0.05, seed=42)
nx.set_node_attributes(G, False, 'shared')
nx.set_node_attributes(G, 0, 'score')
nx.set_node_attributes(G, False, 'gifted')
nx.set_node_attributes(G, 0, 'triggered_count')
nx.set_node_attributes(G, '', 'gender')
nx.set_node_attributes(G, False, 'has_chronic_disease')
nx.set_node_attributes(G, '', 'ideology')
nx.set_node_attributes(G, '', 'sentiment')

###

class HandlerFixedSizeMarker(HandlerBase):
    def __init__(self, size_in_pixels=10, color='black'):
        self.size_in_pixels = size_in_pixels
        self.color = color
        super().__init__()

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Draw a circle with fixed pixel size
        radius = self.size_in_pixels / 2
        circle = mpatches.Circle(
            (width / 2, height / 2),
            radius=radius,
            facecolor=self.color,
            transform=trans
        )
        return [circle]


# Assign user attributes
for node in G.nodes:
    G.nodes[node]['gender'] = random.choice(['Male', 'Female'])
    G.nodes[node]['has_chronic_disease'] = random.choice([True, False])
    G.nodes[node]['ideology'] = random.choice(['pro-health', 'anti-health', 'neutral'])
    G.nodes[node]['sentiment'] = G.nodes[node]['ideology']
    G.nodes[node]['shared'] = False
    G.nodes[node]['score'] = 0
    G.nodes[node]['triggered_count'] = 0
    G.nodes[node]['gifted'] = False

# --- Features for ML ---
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

# --- Model training ---
param_grid = {'n_estimators': [100], 'max_depth': [10], 'min_samples_split': [2]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=2, n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# --- Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(2)

# --- Dashboard metrics ---
st.sidebar.header("Network Contagion & Settings")
network_view = st.sidebar.radio("Choose Network View", ("Gender View", "Ideology View"))

SHARE_PROB = st.sidebar.slider("Base Share Probability (Contagion Spread)", 0.0, 1.0, 0.3, 0.05)

zoom_level = st.sidebar.slider("Zoom In > Zoom Out", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

# --- Contagion setup ---
seed_nodes = random.sample(list(G.nodes), INIT_SHARED)
for node in G.nodes:
    G.nodes[node]['shared'] = False
    G.nodes[node]['gifted'] = False
    G.nodes[node]['triggered_count'] = 0

for node in seed_nodes:
    G.nodes[node]['shared'] = True
    G.nodes[node]['gifted'] = True

def get_share_probability(u, v):
    prob = SHARE_PROB
    if G.nodes[u]['gender'] == G.nodes[v]['gender']:
        prob *= GENDER_HOMOPHILY_BONUS
    else:
        prob *= (1 - CROSS_GENDER_REDUCTION_FACTOR)
    if G.nodes[u]['ideology'] == G.nodes[v]['ideology']:
        prob *= IDEOLOGY_HOMOPHILY_BONUS
    else:
        prob *= (1 - CROSS_IDEOLOGY_REDUCTION_FACTOR)
    return max(min(prob, 1), 0)

# Run contagion
contagion = [set(seed_nodes)]
current = set(seed_nodes)
while True:
    next_step = set()
    for u in G.nodes:
        if not G.nodes[u]['shared']:
            for v in G.neighbors(u):
                if G.nodes[v]['shared']:
                    share_prob = get_share_probability(v, u)
                    if random.random() < share_prob:
                        G.nodes[u]['shared'] = True
                        G.nodes[u]['triggered_count'] += 1
                        next_step.add(u)
                        break
    if not next_step:
        break
    contagion.append(next_step)
    current = next_step

# --- Dashboard metrics ---
total_shared = sum(1 for n in G.nodes if G.nodes[n]['shared'])
total_nodes = len(G.nodes)
total_edges = G.number_of_edges()

cross_gender_edges = sum(1 for u, v in G.edges if G.nodes[u]['gender'] != G.nodes[v]['gender'])
percent_cross_gender = (cross_gender_edges / total_edges) * 100 if total_edges > 0 else 0

cross_ideology_edges = sum(1 for u, v in G.edges if G.nodes[u]['ideology'] != G.nodes[v]['ideology'])
percent_cross_ideology = (cross_ideology_edges / total_edges) * 100 if total_edges > 0 else 0

bet_cen = nx.betweenness_centrality(G)
threshold_bet = np.percentile(list(bet_cen.values()), 80)
key_bridges = sum(1 for v in bet_cen.values() if v >= threshold_bet)

clinicians_engaged = sum(1 for n in G.nodes if G.nodes[n]['shared'] and random.random() < 0.3)

contagion_steps = len(contagion)
final_share_rate = (total_shared / total_nodes) * 100

# --- Visualization ---
st.markdown("## Network Contagion Visualisation")

# --- Colors ---
ideology_colors = {
    'pro-health': '#003A6B',
    'anti-health':  '#89CFF1',
    'neutral': '#5293BB'
}

# --- Calculate betweenness centrality for top 20% ---
bc = nx.betweenness_centrality(G)
threshold_bc = np.percentile(list(bc.values()), 80)

# --- Node border colors ---
node_border_colors = [
    'green' if bc[n] >= threshold_bc else 'none' for n in G.nodes
]

# --- Node colors and sizes ---
node_colors = []
node_sizes = []

for n in G.nodes:
    if network_view == "Gender View":
        color = '#003A6B' if G.nodes[n]['gender'] == 'Male' else '#5293BB'
    else:
        color = ideology_colors.get(G.nodes[n]['ideology'], '#000000')
    node_colors.append(color)
    node_sizes.append(300 + 100 * G.nodes[n]['triggered_count'])

# --- Edges coloring based on view ---
edge_colors = []
edge_widths = []

for u, v in G.edges:
    if network_view == "Gender View":
        if G.nodes[u]['gender'] != G.nodes[v]['gender']:
            edge_colors.append('red')
            edge_widths.append(2)
        else:
            edge_colors.append('#414141')
            edge_widths.append(1)
    else:
        u_ideo = G.nodes[u]['ideology']
        v_ideo = G.nodes[v]['ideology']
        if u_ideo != v_ideo:
            if 'neutral' in (u_ideo, v_ideo):
                edge_colors.append('red')
                edge_widths.append(2)
            else:
                edge_colors.append('#414141')
                edge_widths.append(1)
        else:
            edge_colors.append('#414141')
            edge_widths.append(1)

###

# Use a fixed DPI for consistent marker size
DPI = 100

# Create figure with fixed DPI
fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)


# --- Plot with zoom control ---

# Use the zoom_level from earlier
fig_size = (15 * zoom_level, 12 * zoom_level)
fig, ax = plt.subplots(figsize=fig_size)
pos = nx.spring_layout(G, seed=42, k=0.15)

nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, edge_color='gray')

# Labels for top 10% nodes by betweenness
labels = {node: str(node) for node in sorted(bc, key=bc.get, reverse=True)[:int(0.1*NUM_USERS)]}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='white')

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color=node_colors,
    linewidths=0.5,
    edgecolors='black'
)

# --- Legend setup ---
# Define fixed marker size in pixels
FIXED_MARKER_SIZE = 10

# Prepare legend labels and colors based on your view
if network_view == "Gender View":
    legend_handles = [
        ('Male', '#003A6B'),
        ('Female', '#5293BB')
    ]
else:
    legend_handles = [
        ('Pro-Health', '#003A6B'),
        ('Anti-Health', '#89CFF1'),
        ('Neutral', '#5293BB')
    ]

# Create dummy plot handles for the legend
fig, ax = plt.subplots()
for label, color in legend_handles:
    ax.plot([], [], marker='o', color=color, linestyle='None', label=label)

# Map labels to custom handlers
handler_map = {
    label: HandlerFixedSizeMarker(size_in_pixels=FIXED_MARKER_SIZE, color=color)
    for label, color in legend_handles
}

# Draw the legend with custom handlers
ax.legend(
    handles=[ax.lines[i] for i in range(len(legend_handles))],
    labels=[label for label, _ in legend_handles],
    handler_map=handler_map,
    loc='best'
)

ax.set_visible(False)  # hide the dummy plot
plt.close(fig)

# Now, when you call st.pyplot(fig), the legend will have fixed-size markers
st.pyplot(fig)
