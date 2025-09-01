import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerBase
import matplotlib.patches as mpatches
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# --- Custom Legend Handler for fixed marker size ---
class HandlerFixedSizeMarker(HandlerBase):
    def __init__(self, size_in_pixels=15, color='black'):
        self.size_in_pixels = size_in_pixels
        self.color = color
        super().__init__()

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        radius = self.size_in_pixels / 2
        circle = mpatches.Circle(
            (width / 2, height / 2),
            radius=radius,
            facecolor=self.color,
            transform=trans
        )
        return [circle]

# --- Streamlit Sidebar Controls ---
st.sidebar.header("Network Contagion & Settings")
network_view = st.sidebar.radio("Choose Network View", ("Gender View", "Ideology View"))
SHARE_PROB = st.sidebar.slider("Base Share Probability (Contagion Spread)", 0.0, 1.0, 0.3, 0.05)
zoom_level = st.sidebar.slider("Zoom In > Zoom Out", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

# --- Parameters ---
NUM_USERS = 300
INIT_SHARED = 3

# --- Generate Network ---
G = nx.erdos_renyi_graph(NUM_USERS, 0.05, seed=42)

# --- Initialize Node Attributes ---
for node in G.nodes:
    G.nodes[node]['gender'] = random.choice(['Male', 'Female'])
    G.nodes[node]['has_chronic_disease'] = random.choice([True, False])
    G.nodes[node]['ideology'] = random.choice(['pro-health', 'anti-health', 'neutral'])
    G.nodes[node]['sentiment'] = G.nodes[node]['ideology']
    G.nodes[node]['shared'] = False
    G.nodes[node]['triggered_count'] = 0

# --- Seed initial shared nodes ---
seed_nodes = random.sample(list(G.nodes), INIT_SHARED)
for node in seed_nodes:
    G.nodes[node]['shared'] = True
    G.nodes[node]['gifted'] = True
    G.nodes[node]['triggered_count'] = 1

# --- Contagion Logic ---
def get_share_probability(u, v):
    prob = SHARE_PROB
    if G.nodes[u]['gender'] == G.nodes[v]['gender']:
        prob *= 1.5  # GENDER_HOMOPHILY_BONUS
    else:
        prob *= (1 - 0.3)  # CROSS_GENDER_REDUCTION_FACTOR
    if G.nodes[u]['ideology'] == G.nodes[v]['ideology']:
        prob *= 1.5  # IDEOLOGY_HOMOPHILY_BONUS
    else:
        prob *= (1 - 0.9)  # CROSS_IDEOLOGY_REDUCTION_FACTOR
    return max(min(prob, 1), 0)

# Run contagion spread
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

# --- Metrics ---
total_shared = sum(1 for n in G.nodes if G.nodes[n]['shared'])
total_nodes = len(G.nodes)
total_edges = G.number_of_edges()
cross_gender_edges = sum(1 for u, v in G.edges if G.nodes[u]['gender'] != G.nodes[v]['gender'])
cross_ideology_edges = sum(1 for u, v in G.edges if G.nodes[u]['ideology'] != G.nodes[v]['ideology'])
percent_cross_gender = (cross_gender_edges / total_edges) * 100 if total_edges else 0
percent_cross_ideology = (cross_ideology_edges / total_edges) * 100 if total_edges else 0

# --- Betweenness Centrality ---
bc = nx.betweenness_centrality(G)
threshold_bc = np.percentile(list(bc.values()), 80)

# --- Visualization ---
st.markdown("## Network Contagion Visualisation")

# --- Node Colors ---
ideology_colors = {
    'pro-health': '#003A6B',
    'anti-health':  '#89CFF1',
    'neutral': '#5293BB'
}

node_colors = []
node_sizes = []

for n in G.nodes:
    if network_view == "Gender View":
        color = '#003A6B' if G.nodes[n]['gender'] == 'Male' else '#5293BB'
    else:
        color = ideology_colors.get(G.nodes[n]['ideology'], '#000000')
    node_colors.append(color)
    node_sizes.append(300 + 100 * G.nodes[n]['triggered_count'])

# --- Edges ---
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
        if G.nodes[u]['ideology'] != G.nodes[v]['ideology']:
            if 'neutral' in (G.nodes[u]['ideology'], G.nodes[v]['ideology']):
                edge_colors.append('red')
                edge_widths.append(2)
            else:
                edge_colors.append('#414141')
                edge_widths.append(1)
        else:
            edge_colors.append('#414141')
            edge_widths.append(1)

# --- Plot with zoom ---
DPI = 100
fig_size = (15 * zoom_level, 12 * zoom_level)
fig, ax = plt.subplots(figsize=fig_size, dpi=DPI)

pos = nx.spring_layout(G, seed=42, k=0.15)

nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, edge_color='gray')
nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in sorted(bc, key=bc.get, reverse=True)[:10]}, font_size=8, font_color='white')

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color=node_colors,
    linewidths=0.5,
    edgecolors='black'
)

# --- Legend with fixed marker size ---
# Define fixed marker size in pixels
FIXED_MARKER_SIZE = 15

if network_view == "Gender View":
    legend_data = [
        ('Male', '#003A6B'),
        ('Female', '#5293BB')
    ]
else:
    legend_data = [
        ('Pro-Health', '#003A6B'),
        ('Anti-Health', '#89CFF1'),
        ('Neutral', '#5293BB')
    ]

# Create dummy plot handles
for label, color in legend_data:
    ax.plot([], [], marker='o', color=color, linestyle='None', markersize=FIXED_MARKER_SIZE)

# Map labels to custom handlers
handler_map = {
    label: HandlerFixedSizeMarker(size_in_pixels=FIXED_MARKER_SIZE, color=color)
    for label, color in legend_data
}

# Create legend
ax.legend(
    handles=[ax.lines[i] for i in range(len(legend_data))],
    labels=[label for label, _ in legend_data],
    handler_map=handler_map,
    loc='best'
)

ax.set_axis_off()

# Show in Streamlit
st.pyplot(fig)
