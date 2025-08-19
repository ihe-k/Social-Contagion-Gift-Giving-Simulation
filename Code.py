import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from textblob import TextBlob
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import numpy as np

# --- Parameters ---
NUM_USERS = 30
INIT_SHARED = 3  # initial gifted users
SHARE_PROB = 0.3  # base probability of sharing health information
GIFT_BONUS = 10  # score bonus converted to probability scale inside simulation
IDEOLOGY_CROSS_BONUS = 0.2  # bonus for cross-ideology sharing
CHRONIC_PROPENSITY = 0.6  # propensity to share if chronic disease
GENDER_PROPENSITY = {"Male": 0.3, "Female": 0.5}  # gender-based propensity
GENDER_HOMOPHILY_BONUS = 0.2  # bonus for same-gender sharing

# --- Step 1: Create a social network ---
G = nx.erdos_renyi_graph(n=NUM_USERS, p=0.1, seed=42)
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
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.5:
        return 'pro-health'
    elif polarity < -0.5:
        return 'anti-health'
    else:
        return 'neutral'

# --- Step 3: Generate dummy user data ---
# Since scraping APIs may fail, we'll generate synthetic data for demo
user_data = []
for i in range(NUM_USERS):
    gender = random.choice(['Male', 'Female'])
    sentiment = random.choice(['pro-health', 'anti-health', 'neutral'])
    ideology = sentiment
    has_chronic_disease = random.choice([True, False])
    user_data.append({
        'user': i,
        'gender': gender,
        'sentiment': sentiment,
        'ideology': ideology,
        'has_chronic_disease': has_chronic_disease
    })

for user_info in user_data:
    node = user_info['user']
    G.nodes[node]['gender'] = user_info['gender']
    G.nodes[node]['sentiment'] = user_info['sentiment']
    G.nodes[node]['ideology'] = user_info['ideology']
    G.nodes[node]['has_chronic_disease'] = user_info['has_chronic_disease']
    G.nodes[node]['shared'] = False
    G.nodes[node]['score'] = 0
    G.nodes[node]['triggered_count'] = 0

# --- Step 4: Feature engineering ---
def calculate_sentiment_trends():
    sentiment_trends = []
    for node in G.nodes:
        scores = []
        for nbr in G.neighbors(node):
            scores.append(1 if G.nodes[nbr]['sentiment'] == 'pro-health' else 0)
        sentiment_trends.append(np.mean(scores) if scores else 0)
    return sentiment_trends

def calculate_betweenness_centrality():
    return nx.betweenness_centrality(G)

sentiment_trends = calculate_sentiment_trends()
betweenness_centrality = calculate_betweenness_centrality()

# --- Step 5: Prepare data for model ---
user_features = []
user_labels = []

for node in G.nodes:
    info = G.nodes[node]
    user_features.append([
        1 if info['gender'] == 'Female' else 0,
        1 if info['has_chronic_disease'] else 0,
        1 if info['ideology'] == 'pro-health' else 0,
        1 if info['ideology'] == 'anti-health' else 0,
        1 if info['ideology'] == 'neutral' else 0,
        sentiment_trends[node],
        betweenness_centrality[node]
    ])
    user_labels.append(info['ideology'])

X_train, X_test, y_train, y_test = train_test_split(user_features, user_labels, test_size=0.2, random_state=42)

# --- Step 6: Train Random Forest ---
param_grid = {
    'n_estimators': [100],
    'max_depth': [10],
    'min_samples_split': [2]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# --- Step 7: Initialize contagion ---
initial_gifted = random.sample(list(G.nodes), INIT_SHARED)
for node in initial_gifted:
    G.nodes[node]['shared'] = True
    G.nodes[node]['gifted'] = True

contagion_steps = [set(initial_gifted)]

# --- Step 8: Run contagion simulation with scoring ---
def run_contagion_simulation():
    new_shared = set(initial_gifted)
    all_shared = set(initial_gifted)

    while new_shared:
        next_new_shared = set()
        for user in new_shared:
            neighbors = list(G.neighbors(user))
            for neighbor in neighbors:
                if not G.nodes[neighbor]['shared']:
                    prob = SHARE_PROB
                    if G.nodes[user]['gifted']:
                        prob += GIFT_BONUS / 100
                    if G.nodes[user]['ideology'] != G.nodes[neighbor]['ideology']:
                        prob += IDEOLOGY_CROSS_BONUS
                    if G.nodes[neighbor]['has_chronic_disease']:
                        prob = max(prob, CHRONIC_PROPENSITY)
                    if G.nodes[user]['gender'] == G.nodes[neighbor]['gender']:
                        prob += GENDER_HOMOPHILY_BONUS
                    prob = min(max(prob, 0), 1)

                    if random.random() < prob:
                        G.nodes[neighbor]['shared'] = True
                        G.nodes[neighbor]['triggered_count'] += 1
                        G.nodes[user]['score'] += 10  # Increase score when triggering share
                        next_new_shared.add(neighbor)
                        all_shared.add(neighbor)
        if not next_new_shared:
            break
        contagion_steps.append(next_new_shared)
        new_shared = next_new_shared

run_contagion_simulation()

# --- Step 9: Streamlit UI ---
st.title("Health Information Spread Simulation")

st.subheader("Model Evaluation")
st.write(f"Accuracy: {accuracy:.2%}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Layout: left for graph, right for leaderboard
left_col, right_col = st.columns([2, 1])

with left_col:
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    male_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Male']
    female_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Female']
    shared_nodes = [n for n in G.nodes if G.nodes[n]['shared']]

    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=male_nodes, node_color='lightgreen', node_size=300, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=female_nodes, node_color='lightblue', node_size=300, ax=ax)
    nx.draw_networkx_nodes(
        G, pos, nodelist=shared_nodes,
        node_color='none', edgecolors='red', node_size=330, linewidths=2, ax=ax
    )
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes}, font_size=8, ax=ax)

    ax.set_title("Final Network State (Red outline = Shared)")
    ax.axis('off')
    st.pyplot(fig)

with right_col:
    st.markdown("### 🏆 Top Influencers")
    influencer_stats = []
    for node in G.nodes:
        influencer_stats.append({
            'user': node,
            'score': G.nodes[node]['score'],
            'triggered': G.nodes[node]['triggered_count'],
        })
    top_influencers = sorted(influencer_stats, key=lambda x: x['triggered'], reverse=True)[:5]

    for rank, inf in enumerate(top_influencers, 1):
        st.markdown(f"- **Rank {rank}**: User {inf['user']} — Score: {inf['score']}, Triggered: {inf['triggered']}")

    male_triggered = sum(1 for n in G.nodes if G.nodes[n]['shared'] and G.nodes[n]['gender'] == 'Male')
    female_triggered = sum(1 for n in G.nodes if G.nodes[n]['shared'] and G.nodes[n]['gender'] == 'Female')

    st.markdown(f"- **Male Users Triggered**: {male_triggered} shares")
    st.markdown(f"- **Female Users Triggered**: {female_triggered} shares")

    st.markdown("---")
    st.markdown("🕹️ *Simulation complete. Refresh page to re-run.*")
