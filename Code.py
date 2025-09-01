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

# --- Assign resistance attribute ---
for node in G.nodes:
    G.nodes[node]['resistance'] = np.random.uniform(0, 1)

# --- Seed initial nodes (e.g., 8%) ---
num_seeds = max(1, int(0.08 * NUM_USERS))
initial_shared = random.sample(list(G.nodes), num_seeds)

# --- Initialize shared status ---
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

# --- Sidebar ---
st.sidebar.header("Network Contagion & Settings")
network_view = st.sidebar.radio("Choose Network View", ("Gender View", "Ideology View"))
SHARE_PROB = st.sidebar.slider("Base Share Probability (Contagion Spread)", 0.0, 1.0, 0.3, 0.05)

# --- Share probability function ---
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

# --- Run contagion with resistance ---
contagion = [set(initial_shared)]
current = set(initial_shared)

RESISTANCE_THRESHOLD = 0.3

while True:
    next_step = set()
    for u in G.nodes:
        if not G.nodes[u]['shared']:
            for v in G.neighbors(u):
                if G.nodes[v]['shared']:
                    share_prob = get_share_probability(v, u)
                    if G.nodes[u]['resistance'] <= RESISTANCE_THRESHOLD:
                        if random.random() < share_prob:
                            G.nodes[u]['shared'] = True
                            G.nodes[u]['triggered_count'] += 1
                            next_step.add(u)
                            break
    if not next_step:
        break
    contagion.append(next_step)
    current = next_step

# --- Influence analysis of chronic users ---
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


chronic_users = [n for n in G.nodes if G.nodes[n]['has_chronic_disease']]
non_chronic_users = [n for n in G.nodes if not G.nodes[n]['has_chronic_disease']]

degree_cen = nx.degree_centrality(G)
betweenness_cen = nx.betweenness_centrality(G)

avg_deg_chronic = np.mean([degree_cen[n] for n in chronic_users]) if chronic_users else 0
avg_deg_non_chronic = np.mean([degree_cen[n] for n in non_chronic_users]) if non_chronic_users else 0

avg_betw_chronic = np.mean([betweenness_cen[n] for n in chronic_users]) if chronic_users else 0
avg_betw_non_chronic = np.mean([betweenness_cen[n] for n in non_chronic_users]) if non_chronic_users else 0

# --- Calculate Sharing Activity ---
total_shares = sum(G.nodes[n]['triggered_count'] for n in G.nodes)
chronic_shares = sum(G.nodes[n]['triggered_count'] for n in chronic_users)
percent_chronic_shares = (chronic_shares / total_shares) * 100 if total_shares > 0 else 0

# --- Dashboard Metrics ---
col1, col2, col3, col4 = st.columns(4)
col5, col6, col7, col8 = st.columns(4)

col1.metric("Total Users", total_nodes)
col2.metric("Key Bridges", key_bridges)
col3.metric("Final Share Rate (%)", f"{final_share_rate:.1f}%")
col4.metric("Cross-Gender Ties (%)", f"{percent_cross_gender:.1f}%")
col5.metric("Engaged Clinicians", clinicians_engaged)
col6.metric("Contagion Steps", contagion_steps)
col7.metric("Sharing Activity", f"{percent_chronic_shares:.2f}%")
col8.metric("Cross-Ideology Ties (%)", f"{percent_cross_ideology:.1f}%")

# --- Classification Results ---
st.subheader("Classification Report")
st.write(report_df)

# --- Original Network Visualization ---
fig, ax = plt.subplots(figsize=(10, 6))
node_color = ['blue' if G.nodes[n]['gender'] == 'Male' else 'pink' for n in G.nodes]
positions = nx.spring_layout(G, seed=42)
nx.draw(G, node_size=20, with_labels=False, node_color=node_color, ax=ax)
ax.set_title("Network Visualization")
st.pyplot(fig)
