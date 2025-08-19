import random
import networkx as nx
import matplotlib.pyplot as plt
from textblob import TextBlob
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import numpy as np
import requests
from bs4 import BeautifulSoup
import feedparser

# --- Parameters ---
NUM_USERS = 30
INIT_SHARED = 3
SHARE_PROB = 0.3
GIFT_BONUS = 10
IDEOLOGY_CROSS_BONUS = 0.2
CHRONIC_PROPENSITY = 0.6
GENDER_HOMOPHILY_BONUS = 0.2

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

# --- Step 2: Sentiment Analyzer ---
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return 'pro-health' if polarity > 0.5 else ('anti-health' if polarity < -0.5 else 'neutral')

# --- Step 3: Podcast List (Hardcoded, no fetching) ---
podcast_items = [
    {"user": "MelRobbins", "content": "Motivational self-help and life coaching.", "platform": "Manual", "url": ""},
    {"user": "TheDaily", "content": "News and current events podcast by The New York Times.", "platform": "Manual", "url": ""},
    {"user": "HubermanLab", "content": "Science and neuroscience explained.", "platform": "Manual", "url": ""},
    {"user": "SmartLess", "content": "Humor and celebrity interviews.", "platform": "Manual", "url": ""},
    {"user": "DiaryOfCEO", "content": "Entrepreneurial journeys and mindset.", "platform": "Manual", "url": ""},
    {"user": "TuckerCarlsonShow", "content": "Political commentary.", "platform": "Manual", "url": ""},
    {"user": "NewHeightsJason", "content": "Inspirational and sports-related.", "platform": "Manual", "url": ""},
    {"user": "TravisKelce", "content": "Football and personal life stories.", "platform": "Manual", "url": ""},
]

# --- Step 4: Assign User Attributes ---
user_data = []
for content in podcast_items:
    sentiment = analyze_sentiment(content["content"])
    user_data.append({
        'user': content['user'],
        'gender': random.choice(['Male', 'Female']),
        'sentiment': sentiment,
        'ideology': sentiment,
        'has_chronic_disease': random.choice([True, False])
    })

# Assign attributes to graph nodes
for i in G.nodes:
    if i < len(user_data):
        u = user_data[i]
        G.nodes[i]['user'] = u['user']
        G.nodes[i]['gender'] = u['gender']
        G.nodes[i]['sentiment'] = u['sentiment']
        G.nodes[i]['ideology'] = u['ideology']
        G.nodes[i]['has_chronic_disease'] = u['has_chronic_disease']
    else:
        # Initialize missing attributes for nodes without user_data
        G.nodes[i]['user'] = f'User{i}'
        G.nodes[i]['gender'] = random.choice(['Male', 'Female'])
        G.nodes[i]['sentiment'] = random.choice(['pro-health', 'anti-health', 'neutral'])
        G.nodes[i]['ideology'] = G.nodes[i]['sentiment']
        G.nodes[i]['has_chronic_disease'] = random.choice([True, False])
    G.nodes[i]['score'] = 0
    G.nodes[i]['triggered_count'] = 0
    G.nodes[i]['shared'] = False
    G.nodes[i]['gifted'] = False

# --- Step 5: Features & Labels ---
def calculate_sentiment_trends():
    trends = []
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        if neighbors:
            pro_health_count = sum(1 for n in neighbors if G.nodes[n]['sentiment'] == 'pro-health')
            trends.append(pro_health_count / len(neighbors))
        else:
            trends.append(0)
    return trends

sent_trends = calculate_sentiment_trends()

def calculate_betweenness_centrality():
    return nx.betweenness_centrality(G)

centrality = calculate_betweenness_centrality()

user_features = []
user_labels = []

for n in G.nodes:
    u = G.nodes[n]
    ideology = u.get('ideology', 'neutral') or 'neutral'
    gender = u.get('gender', 'Male') or 'Male'
    chronic = u.get('has_chronic_disease', False)

    user_features.append([
        1 if gender == 'Female' else 0,
        1 if chronic else 0,
        1 if ideology == 'pro-health' else 0,
        1 if ideology == 'anti-health' else 0,
        1 if ideology == 'neutral' else 0,
        sent_trends[n],
        centrality[n]
    ])
    user_labels.append(ideology)

X_train, X_test, y_train, y_test = train_test_split(user_features, user_labels, test_size=0.2, random_state=42)

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
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=ax)
st.pyplot(fig)

# --- Step 8: Contagion Simulation ---
pos = nx.spring_layout(G, seed=42)
seed_nodes = random.sample(list(G.nodes), INIT_SHARED)
for node in seed_nodes:
    G.nodes[node]['shared'] = True
    G.nodes[node]['gifted'] = True

contagion = [set(seed_nodes)]
current = set(seed_nodes)

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

# --- Step 9: Visualize Network ---
st.subheader("Podcast-Based Information Spread Simulation")
fig, ax = plt.subplots(figsize=(8, 6))
nx.draw(
    G,
    pos,
    with_labels=True,
    labels={n: G.nodes[n]['user'] for n in G.nodes},
    node_size=[300 + 100 * G.nodes[n]['triggered_count'] for n in G.nodes],
    node_color=['lightgreen' if G.nodes[n]['gender'] == 'Male' else 'lightblue' for n in G.nodes],
    edge_color='gray',
    linewidths=1.5,
    node_shape='o',
    font_size=8,
    ax=ax,
)
st.pyplot(fig)
