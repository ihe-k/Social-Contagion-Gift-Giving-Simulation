import random
import networkx as nx
import matplotlib.pyplot as plt
from textblob import TextBlob
import streamlit as st
import numpy as np
import pandas as pd
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

# --- Step 3: Assign User Attributes ---
for node in G.nodes:
    G.nodes[node]['gender'] = random.choice(['Male', 'Female'])
    G.nodes[node]['has_chronic_disease'] = random.choice([True, False])
    G.nodes[node]['ideology'] = random.choice(['pro-health', 'anti-health', 'neutral'])
    G.nodes[node]['sentiment'] = G.nodes[node]['ideology']
    G.nodes[node]['shared'] = False
    G.nodes[node]['score'] = 0
    G.nodes[node]['triggered_count'] = 0
    G.nodes[node]['gifted'] = False

# --- Step 4: Visualization ---
st.sidebar.header("Network View")
view_option = st.sidebar.radio(
    "Select Network View:",
    ("Gender Distribution", "Ideology Distribution")
)

# Node color settings
gender_colors = {'Male': '#003A6B', 'Female': '#5293BB'}
ideology_colors = {'pro-health': '#003A6B', 'neutral': '#5293BB', 'anti-health': '#89CFF1'}

# --- Visualization Setup ---
fig_net, ax_net = plt.subplots(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)

# Prepare node and edge colors
node_colors = []
edge_colors = []
edge_widths = []

# Prepare node size and border widths based on betweenness centrality
bc_values = np.array(list(nx.betweenness_centrality(G).values()))
if bc_values.max() > 0:
    norm_bc = 1 + 5 * (bc_values - bc_values.min()) / (bc_values.max() - bc_values.min())
else:
    norm_bc = np.ones(len(G.nodes))

# Loop over nodes to assign colors and sizes
node_sizes = []
node_border_widths = []

for idx, n in enumerate(G.nodes):
    if view_option == "Gender Distribution":
        node_color = gender_colors[G.nodes[n]['gender']]
    elif view_option == "Ideology Distribution":
        node_color = ideology_colors[G.nodes[n]['ideology']]
    
    node_colors.append(node_color)
    node_sizes.append(300 + 100 * G.nodes[n]['triggered_count'])
    node_border_widths.append(norm_bc[idx])

    # Loop over edges to assign colors based on connection type (cross-gender or cross-ideology)
for u, v in G.edges:
    if (G.nodes[u]['gender'] != G.nodes[v]['gender']) or (G.nodes[u]['ideology'] != G.nodes[v]['ideology']):
        edge_colors.append('red')  # Red for cross-gender or cross-ideology connections
        edge_widths.append(2)      # Thicker red edges
    else:
        edge_colors.append('#AAAAAA')  # Grey for same-gender and same-ideology ties
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
    width=edge_widths,
    style='solid',
    font_size=8,
    font_color='white',
    edge_cmap=plt.cm.Blues, 
    ax=ax_net
)

# Legends for different views
if view_option == "Gender Distribution":
    male_patch = mpatches.Patch(color='#003A6B', label='Male')
    female_patch = mpatches.Patch(color='#5293BB', label='Female')
    ax_net.legend(handles=[male_patch, female_patch], loc='best')
elif view_option == "Ideology Distribution":
    pro_health_patch = mpatches.Patch(color='#003A6B', label='Pro-Health')
    neutral_patch = mpatches.Patch(color='#5293BB', label='Neutral')
    anti_health_patch = mpatches.Patch(color='#89CFF1', label='Anti-Health')
    ax_net.legend(handles=[pro_health_patch, neutral_patch, anti_health_patch], loc='best')

# Display the plot
ax_net.set_title(f"{view_option} Network Diagram")
st.pyplot(fig_net)

# --- Step 11: Explanation ---
with st.expander("ℹ️ Interpretation of the Network Diagram"):
    st.markdown("""
    ### **Network Diagram Interpretation**
    - **Node Colors:**  
      - **Dark blue** represents **Male users** (if Gender view selected) or **Pro-Health** ideology (if Ideology view selected).  
      - **Light blue** represents **Female users** (if Gender view selected) or **Neutral** ideology (if Ideology view selected).
      - **Lightest blue** represents **Anti-Health** ideology (if Ideology view selected).
    - **Node Size:**  
      Reflects how many other users this node has **influenced or triggered**.  
      Larger nodes = more shares triggered.
    - **Node Border Width:**  
      Indicates **betweenness centrality** — users with **thicker borders** serve as **important bridges** in the network, connecting different parts and enabling information spread. These are key nodes that facilitate the flow of information across ideologies.
    - **Edge Colors (Connections):**  
      - **Red edges** = **Cross-gender** or **Cross-ideology** ties.  
      - **Grey edges** = Connections between users of the same gender and same ideology.
""")
