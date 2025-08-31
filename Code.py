import random
import networkx as nx
import matplotlib.pyplot as plt
from textblob import TextBlob
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# --- Parameters ---
NUM_USERS = 30
INIT_SHARED = 3
GIFT_BONUS = 10
IDEOLOGY_CROSS_BONUS = 0.2
CHRONIC_PROPENSITY = 0.6
GENDER_HOMOPHILY_BONUS = 0.2
CROSS_GENDER_BONUS = 0.3  # Increased probability for cross-gender connections
CROSS_IDEOLOGY_BONUS = 0.3  # Increased probability for cross-ideology connections

# --- Network Setup ---
G = nx.erdos_renyi_graph(NUM_USERS, 0.1, seed=42)
nx.set_node_attributes(G, False, 'shared')
nx.set_node_attributes(G, 0, 'score')
nx.set_node_attributes(G, False, 'gifted')
nx.set_node_attributes(G, 0, 'triggered_count')
nx.set_node_attributes(G, '', 'gender')
nx.set_node_attributes(G, False, 'has_chronic_disease')
nx.set_node_attributes(G, '', 'ideology')
nx.set_node_attributes(G, '', 'sentiment')

# --- Sentiment Analysis ---
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.5:
        return 'pro-health'
    elif polarity < -0.5:
        return 'anti-health'
    else:
        return 'neutral'

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

# --- Features for Model ---
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

# --- Model Training ---
param_grid = {'n_estimators': [100], 'max_depth': [10], 'min_samples_split': [2]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=2, n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# --- Model Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(2)

# --- Sidebar Elements ---
st.sidebar.header("Network Contagion & Settings")

# Contagion Probability Slider
SHARE_PROB = st.sidebar.slider("Base Share Probability", 0.0, 1.0, 0.3, 0.05)

# Network View Toggle
network_view = st.sidebar.radio("Choose Network View", ("Gender View", "Ideology View"))

# --- Contagion Simulation ---
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

# --- Visualization ---
st.subheader("Network Contagion Visualization")

# Node Colors and Size
node_colors = []
node_sizes = []
node_border_widths = []
for idx, n in enumerate(G.nodes):
    if network_view == "Gender View":
        color = '#003A6B' if G.nodes[n]['gender'] == 'Male' else '#5293BB'
    else:  # Ideology View
        color = '#003A6B' if G.nodes[n]['ideology'] == 'pro-health' else '#89CFF1' if G.nodes[n]['ideology'] == 'anti-health' else '#5293BB'
    node_colors.append(color)
    node_sizes.append(300 + 100 * G.nodes[n]['triggered_count'])

    # Betweenness centrality is used to determine node border width
    centrality = betweenness_centrality[n]
    node_border_widths.append(centrality * 10)  # Adjust scaling factor for visual appeal

# Edge Colors and Widths
edge_colors = []
edge_widths = []
for u, v in G.edges:
    if G.nodes[u]['gender'] != G.nodes[v]['gender']:  # Cross-gender connection
        edge_colors.append('red')
        edge_widths.append(2)
    elif G.nodes[u]['ideology'] != G.nodes[v]['ideology']:  # Cross-ideology connection
        edge_colors.append('red')
        edge_widths.append(2)
    else:
        edge_colors.append('#AAAAAA')
        edge_widths.append(1)

# Draw the Network
fig_net, ax_net = plt.subplots(figsize=(8, 6))
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
    edgecolors='darkgrey',  # Node borders will be dark grey
    node_shape='o',  # Circular nodes
    linewidths=node_border_widths,  # Apply betweenness centrality to node borders
    ax=ax_net
)

# Legends for Gender and Ideology
if network_view == "Gender View":
    male_patch = mpatches.Patch(color='#003A6B', label='Male')
    female_patch = mpatches.Patch(color='#5293BB', label='Female')
    ax_net.legend(handles=[male_patch, female_patch], loc='best')
else:
    pro_health_patch = mpatches.Patch(color='#003A6B', label='Pro-Health')
    neutral_patch = mpatches.Patch(color='#5293BB', label='Neutral')
    anti_health_patch = mpatches.Patch(color='#89CFF1', label='Anti-Health')
    ax_net.legend(handles=[pro_health_patch, neutral_patch, anti_health_patch], loc='best')

# Display the plot
st.pyplot(fig_net)

# --- Model Evaluation ---
st.subheader("Model Evaluation")
st.write(f"**Accuracy:** {accuracy:.2%}")
st.dataframe(report_df)

with st.expander("ℹ️ Interpretation of the Network Diagram"):
    st.write("""
    **Node Border Width**: Users with thicker borders serve as **important bridges** in the network. These users help connect different parts of the network and enable information spread. They have high betweenness centrality.
    **Node Size**: The size of each node reflects how many other users that node has influenced or triggered.
    **Edge Colors**:  
    - **Red edges** represent **cross-gender** or **cross-ideology** connections.  
    - **Dark Grey edges** represent connections between users of the same gender and same ideology.
    """)
