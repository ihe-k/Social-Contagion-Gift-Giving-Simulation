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

st.title("Health Information Network Simulation")

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

# --- Assign User Attributes ---
for node in G.nodes:
    G.nodes[node]['gender'] = random.choice(['Male', 'Female'])
    G.nodes[node]['has_chronic_disease'] = random.choice([True, False])
    G.nodes[node]['ideology'] = random.choice(['pro-health', 'anti-health', 'neutral'])
    G.nodes[node]['sentiment'] = G.nodes[node]['ideology']
    G.nodes[node]['shared'] = False
    G.nodes[node]['score'] = 0
    G.nodes[node]['triggered_count'] = 0
    G.nodes[node]['gifted'] = False

# --- Assign resistance ---
for node in G.nodes:
    G.nodes[node]['resistance'] = np.random.uniform(0, 1)

# --- Seed initial nodes ---
num_seeds = max(1, int(0.08 * NUM_USERS))
initial_shared = random.sample(list(G.nodes), num_seeds)

# --- Initialize shared ---
for node in G.nodes:
    G.nodes[node]['shared'] = False
    G.nodes[node]['gifted'] = False
    G.nodes[node]['triggered_count'] = 0

# --- Set seed nodes ---
for node in initial_shared:
    G.nodes[node]['shared'] = True
    G.nodes[node]['gifted'] = True

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

X_train, X_test, y_train, y_test = train_test_split(user_features, user_labels, test_size=0.2, random_state=42)

# --- Model ---
param_grid = {'n_estimators': [100], 'max_depth': [10], 'min_samples_split': [2]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=2, n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# --- Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(2)

# --- Sidebar ---
st.sidebar.header("Network Contagion & Settings")
network_view = st.sidebar.radio("Choose Network View", ("Gender View", "Ideology View"))
SHARE_PROB = st.sidebar.slider("Base Share Probability (Contagion Spread)", 0.0, 1.0, 0.3, 0.05)

# --- Share probability ---
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

# --- Run contagion ---
contagion = [set(initial_shared)]
while True:
    next_step = set()
    for u in G.nodes:
        if not G.nodes[u]['shared']:
            for v in G.neighbors(u):
                if G.nodes[v]['shared']:
                    share_prob = get_share_probability(v, u)
                    if G.nodes[u]['resistance'] <= 0.3:
                        if random.random() < share_prob:
                            G.nodes[u]['shared'] = True
                            G.nodes[u]['triggered_count'] += 1
                            next_step.add(u)
                            break
    if not next_step:
        break
    contagion.append(next_step)

# --- Influence analysis ---
total_shared = sum(1 for n in G.nodes if G.nodes[n]['shared'])
total_nodes = len(G.nodes)
total_edges = G.number_of_edges()

cross_gender_edges = sum(1 for u, v in G.edges if G.nodes[u]['gender'] != G.nodes[v]['gender'])
percent_cross_gender = (cross_gender_edges / total_edges) * 100 if total_edges > 0 else 0

cross_ideology_edges = sum(1 for u, v in G.edges if G.nodes[u]['ideology'] != G.nodes[v]['ideology'])
percent_cross_ideology = (cross_ideology_edges / total_edges) * 100 if total_edges > 0 else 0

# Betweenness centrality for highlighting bridges
bc = nx.betweenness_centrality(G)
threshold_bet = np.percentile(list(bc.values()), 80)

# --- Determine top influencers ---
top_percent = 10
top_count = max(1, int(len(G) * top_percent / 100))
top_nodes_sorted_deg = sorted(bc, key=bc.get, reverse=True)
top_nodes_sorted_betw = sorted(bc, key=bc.get, reverse=True)

top_influencers_deg = top_nodes_sorted_deg[:top_count]
top_influencers_betw = top_nodes_sorted_betw[:top_count]

chronic_in_top_deg = sum(1 for n in top_influencers_deg if G.nodes[n]['has_chronic_disease'])
chronic_in_top_betw = sum(1 for n in top_influencers_betw if G.nodes[n]['has_chronic_disease'])

percent_chronic_in_top_deg = (chronic_in_top_deg / top_count) * 100
percent_chronic_in_top_betw = (chronic_in_top_betw / top_count) * 100

# --- Share involvement ---
chronic_sharers = [n for n in G.nodes if G.nodes[n]['shared'] and G.nodes[n]['has_chronic_disease']]
total_shares = sum(G.nodes[n]['triggered_count'] for n in G.nodes)
chronic_shares = sum(G.nodes[n]['triggered_count'] for n in chronic_users)
percent_chronic_shares = (chronic_shares / total_shares) * 100 if total_shares > 0 else 0

# --- Dashboard ---
st.markdown("## Dashboard Summary")
total_shared = sum(1 for n in G.nodes if G.nodes[n]['shared'])
total_nodes = len(G.nodes)
total_edges = G.number_of_edges()

cross_gender_edges = sum(1 for u, v in G.edges if G.nodes[u]['gender'] != G.nodes[v]['gender'])
percent_cross_gender = (cross_gender_edges / total_edges) * 100 if total_edges > 0 else 0

cross_ideology_edges = sum(1 for u, v in G.edges if G.nodes[u]['ideology'] != G.nodes[v]['ideology'])
percent_cross_ideology = (cross_ideology_edges / total_edges) * 100 if total_edges > 0 else 0

# Betweenness centrality for top 20%
threshold_bet = np.percentile(list(bc.values()), 80)

# --- Node border colors based on betweenness ---
node_border_colors = [
    'green' if bc[n] >= threshold_bet else 'black' for n in G.nodes
]

# --- Node colors and sizes ---
node_colors = []
node_sizes = []
for n in G.nodes:
    if network_view == "Gender View":
        color = '#003A6B' if G.nodes[n]['gender'] == 'Male' else '#5293BB'
    else:
        color = {'pro-health':'#003A6B','anti-health':'#89CFF1','neutral':'#5293BB'}.get(G.nodes[n]['ideology'],'#000000')
    node_colors.append(color)
    node_sizes.append(300 + 100 * G.nodes[n]['triggered_count'])

# --- Edges coloring ---
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

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 11), dpi=150)
pos = nx.spring_layout(G, seed=42, k=0.15)

nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, edge_color='gray')
labels = {node: str(node) for node in sorted(bc, key=bc.get, reverse=True)[:int(0.1*NUM_USERS)]}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='white')

# Draw nodes with border color indicating betweenness centrality
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color=node_colors,
    linewidths=0.5,
    edgecolors=node_border_colors  # <-- borders colored based on betweenness
)

if network_view == "Gender View":
    legend_handles = [
        mpatches.Patch(color='#003A6B', label='Male'),
        mpatches.Patch(color='#5293BB', label='Female')
    ]
else:
    legend_handles = [
        mpatches.Patch(color='#003A6B', label='Pro-Health'),
        mpatches.Patch(color='#89CFF1', label='Anti-Health'),
        mpatches.Patch(color='#5293BB', label='Neutral')
    ]
ax.legend(handles=legend_handles, loc='best')
ax.axis('off')
st.pyplot(fig)

# --- Interpretation ---
st.markdown("""
### **Network Diagram Interpretation**

- **Node Border Width:**  
  Indicates betweenness centrality — users with thicker borders serve as important bridges in the network, connecting different parts and enabling information spread.

- **Node Border Color:**  
  - **Green borders** highlight the top 20% of nodes with the highest betweenness centrality, marking them as key bridge nodes.  
  - Other nodes have black borders.

- **Edge Colors (Connections):**  
  - Red edges indicate cross-gender and ideology ties.

- **Clusters:**  
  The network shows that gender homophily and ideological alignment influence connections and information diffusion.

- **Overall Insights:**  
  - Users with higher centrality act as key influencers or bridges.  
  - Chronic disease status and ideological differences impact sharing probabilities and contagion dynamics.
""")
