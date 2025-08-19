import random
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.animation import FuncAnimation
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import numpy as np

# --- Parameters ---
NUM_USERS = 30
INIT_SHARED = 3
SHARE_PROB = 0.3
GIFT_BONUS = 10
IDEOLOGY_CROSS_BONUS = 0.2
CHRONIC_PROPENSITY = 0.6
GENDER_PROPENSITY = {"Male": 0.3, "Female": 0.5}
GENDER_HOMOPHILY_BONUS = 0.2

# --- Create Social Network ---
G = nx.erdos_renyi_graph(n=NUM_USERS, p=0.1, seed=42)
nx.set_node_attributes(G, False, 'shared')
nx.set_node_attributes(G, 0, 'score')
nx.set_node_attributes(G, False, 'gifted')
nx.set_node_attributes(G, 0, 'triggered_count')
nx.set_node_attributes(G, '', 'gender')
nx.set_node_attributes(G, False, 'has_chronic_disease')
nx.set_node_attributes(G, '', 'ideology')
nx.set_node_attributes(G, '', 'sentiment')

# --- Dummy Content for Sentiment ---
dummy_texts = [
    "Exercise is great for your heart!", "Vaccines are dangerous!",
    "Meditation helps with stress.", "Health tips are everywhere these days.",
    "Modern medicine is a scam!", "Drink water and sleep well."
]

# --- Sentiment Classification ---
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.5:
        return 'pro-health'
    elif polarity < -0.5:
        return 'anti-health'
    else:
        return 'neutral'

# --- Assign Attributes to Users ---
user_data = []
for i in range(NUM_USERS):
    text = random.choice(dummy_texts)
    gender = random.choice(["Male", "Female"])
    sentiment = analyze_sentiment(text)
    has_chronic_disease = random.choice([True, False])
    user_data.append({
        'user': f'user_{i}',
        'gender': gender,
        'sentiment': sentiment,
        'ideology': sentiment,
        'has_chronic_disease': has_chronic_disease
    })
    G.nodes[i]['gender'] = gender
    G.nodes[i]['sentiment'] = sentiment
    G.nodes[i]['ideology'] = sentiment
    G.nodes[i]['has_chronic_disease'] = has_chronic_disease

# --- Feature Engineering ---
def calculate_sentiment_trends():
    trends = []
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        scores = [1 if G.nodes[n]['sentiment'] == 'pro-health' else 0 for n in neighbors]
        trends.append(np.mean(scores) if scores else 0)
    return trends

sentiment_trends = calculate_sentiment_trends()
centrality = nx.betweenness_centrality(G)

# --- Prepare Training Data ---
features, labels = [], []
for node in G.nodes:
    info = G.nodes[node]
    features.append([
        1 if info['gender'] == 'Female' else 0,
        1 if info['has_chronic_disease'] else 0,
        1 if info['ideology'] == 'pro-health' else 0,
        1 if info['ideology'] == 'anti-health' else 0,
        1 if info['ideology'] == 'neutral' else 0,
        sentiment_trends[node],
        centrality[node]
    ])
    labels.append(info['ideology'])

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
clf = GridSearchCV(RandomForestClassifier(random_state=42), {
    'n_estimators': [100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}, cv=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# --- Model Results ---
st.title("Health Information Contagion Simulation")
st.subheader("Model Evaluation")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# --- Plot Confusion Matrix ---
fig_cm, ax_cm = plt.subplots()
ConfusionMatrixDisplay.from_estimator(clf.best_estimator_, X_test, y_test, ax=ax_cm)
st.pyplot(fig_cm)

# --- Contagion Spread Simulation ---
initial_gifted = random.sample(list(G.nodes), INIT_SHARED)
for node in initial_gifted:
    G.nodes[node]['shared'] = True
    G.nodes[node]['gifted'] = True

st.subheader("Contagion Simulation Log")
st.write(f"Initial gifted users: {initial_gifted}")
contagion_steps = [set(initial_gifted)]

def run_simulation():
    new_shared = set(initial_gifted)
    all_shared = set(initial_gifted)
    while new_shared:
        next_new_shared = set()
        for user in new_shared:
            for neighbor in G.neighbors(user):
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
                        next_new_shared.add(neighbor)
                        st.write(f"User {neighbor} shared info triggered by {user} (prob={prob:.2f})")
        if next_new_shared:
            contagion_steps.append(next_new_shared)
            new_shared = next_new_shared
        else:
            break

run_simulation()

# --- Final Network Visualization ---
st.subheader("Network Visualization (Final State)")
fig, ax = plt.subplots(figsize=(10, 7))
pos = nx.spring_layout(G, seed=42)

shared_nodes = [n for n in G.nodes if G.nodes[n]['shared']]
not_shared_nodes = [n for n in G.nodes if not G.nodes[n]['shared']]

nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=shared_nodes, node_color='red', node_size=300, label="Shared", ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=not_shared_nodes, node_color='black', node_size=300, label="Not Shared", ax=ax)
nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes}, font_size=8, ax=ax)

ax.set_title("Contagion Spread Network")
ax.axis('off')
st.pyplot(fig)

# --- Leaderboard ---
st.subheader("Trigger Leaderboard")
leaderboard = sorted(G.nodes(data=True), key=lambda x: x[1]['triggered_count'], reverse=True)
for rank, (node, data) in enumerate(leaderboard[:5], 1):
    st.write(f"{rank}. User {node} triggered {data['triggered_count']} shares")
