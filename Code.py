import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from textblob import TextBlob
import streamlit as st
from TikTokApi import TikTokApi
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import numpy as np

# --- Parameters ---
NUM_USERS = 30
INIT_SHARED = 3  # initial gifted users
SHARE_PROB = 0.3  # base probability of sharing health information
GIFT_BONUS = 10  # score for triggering another user's share
IDEOLOGY_CROSS_BONUS = 0.2  # bonus for cross-ideology sharing
CHRONIC_PROPENSITY = 0.6  # probability of sharing health info for users with chronic disease
GENDER_HOMOPHILY_BONUS = 0.2  # Homophily bonus for same-gender sharing

# --- Step 1: Create a social network ---
G = nx.erdos_renyi_graph(n=NUM_USERS, p=0.1, seed=42)
nx.set_node_attributes(G, False, 'shared')
nx.set_node_attributes(G, 0, 'score')
nx.set_node_attributes(G, False, 'gifted')
nx.set_node_attributes(G, 0, 'triggered_count')
nx.set_node_attributes(G, '', 'gender')  # Assign empty gender initially
nx.set_node_attributes(G, False, 'has_chronic_disease')  # Chronic disease attribute
nx.set_node_attributes(G, '', 'ideology')  # Ideological stance (pro-health, anti-health, neutral)
nx.set_node_attributes(G, '', 'sentiment')  # Sentiment (pro-health, anti-health, neutral)

# --- Step 2: Sentiment Analysis and Ideological Mapping ---
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.5:
        return 'pro-health'
    elif polarity < -0.5:
        return 'anti-health'
    else:
        return 'neutral'

# --- Step 3: Stubs for content scraping ---
def get_health_videos_from_youtube(query="health"):
    # Stubbed list to avoid issues with external API in Streamlit
    return [
        {"user": f"YT_User_{i}", "content": f"Health video content {i}", "platform": "YouTube", "url": ""}
        for i in range(5)
    ]

def get_health_videos_from_tiktok(query="health"):
    # Stubbed list to avoid TikTok API issues
    return [
        {"user": f"TT_User_{i}", "content": f"Health TikTok content {i}", "platform": "TikTok", "url": ""}
        for i in range(5)
    ]

def get_twitter_data(query, limit=5):
    # Stub: empty list since snscrape was removed
    return []

youtube_videos = get_health_videos_from_youtube()
tiktok_videos = get_health_videos_from_tiktok()
twitter_tweets = get_twitter_data("health", limit=5)

all_videos_and_tweets = youtube_videos + tiktok_videos + twitter_tweets

# --- Step 4: Assign Gender, Ideology, Chronic Disease ---
user_data = []
for content in all_videos_and_tweets:
    user = content['user']
    text_content = content['content']

    gender = random.choice(['Male', 'Female'])
    sentiment = analyze_sentiment(text_content)
    has_chronic_disease = random.choice([True, False])

    user_data.append({
        'user': user,
        'gender': gender,
        'sentiment': sentiment,
        'ideology': sentiment,
        'has_chronic_disease': has_chronic_disease
    })

# Add users to network nodes
for i, user_info in enumerate(user_data):
    if i >= NUM_USERS:
        break
    G.nodes[i]['gender'] = user_info['gender']
    G.nodes[i]['sentiment'] = user_info['sentiment']
    G.nodes[i]['ideology'] = user_info['ideology']
    G.nodes[i]['has_chronic_disease'] = user_info['has_chronic_disease']
    G.nodes[i]['shared'] = False
    G.nodes[i]['score'] = 0
    G.nodes[i]['triggered_count'] = 0

# --- Step 5: Feature Engineering ---
def calculate_sentiment_trends():
    sentiment_trends = []
    for node in G.nodes:
        sentiment_scores = []
        for neighbor in G.neighbors(node):
            sentiment_scores.append(1 if G.nodes[neighbor]['sentiment'] == 'pro-health' else 0)
        sentiment_trends.append(np.mean(sentiment_scores) if sentiment_scores else 0)
    return sentiment_trends

def calculate_betweenness_centrality():
    return nx.betweenness_centrality(G)

sentiment_trends = calculate_sentiment_trends()
betweenness_centrality = calculate_betweenness_centrality()

# --- Step 6: Prepare data for model ---
user_features = []
user_labels = []

for node in G.nodes:
    user_info = G.nodes[node]
    user_features.append([
        1 if user_info['gender'] == 'Female' else 0,
        1 if user_info['has_chronic_disease'] else 0,
        1 if user_info['ideology'] == 'pro-health' else 0,
        1 if user_info['ideology'] == 'anti-health' else 0,
        1 if user_info['ideology'] == 'neutral' else 0,
        sentiment_trends[node],
        betweenness_centrality[node]
    ])
    user_labels.append(user_info['ideology'])

X_train, X_test, y_train, y_test = train_test_split(user_features, user_labels, test_size=0.2, random_state=42)

# --- Step 7: Train model ---
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# --- Step 8: Display model evaluation ---
st.subheader("Model Evaluation")
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.2%}")

report = classification_report(y_test, y_pred, output_dict=False)
st.text("Classification Report:")
st.text(report)

fig_cm, ax_cm = plt.subplots()
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=ax_cm)
st.pyplot(fig_cm)

# --- Step 9: Contagion Simulation Setup ---
pos = nx.spring_layout(G, seed=42)

# Initialize gifted users
initial_gifted = random.sample(list(G.nodes), INIT_SHARED)
for node in initial_gifted:
    G.nodes[node]['shared'] = True
    G.nodes[node]['gifted'] = True

st.write(f"Initial gifted users: {initial_gifted}")

contagion_steps = [set(initial_gifted)]

# --- Step 10: Run contagion simulation ---
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
                        next_new_shared.add(neighbor)
                        all_shared.add(neighbor)
                        st.write(f"User {neighbor} shared info triggered by {user} (prob={prob:.2f})")

        if not next_new_shared:
            break
        contagion_steps.append(next_new_shared)
        new_shared = next_new_shared

run_contagion_simulation()

st.write(f"Number of contagion steps: {len(contagion_steps)}")
st.write(f"Contagion steps detail: {contagion_steps}")

# --- Step 11: Animation ---
fig, ax = plt.subplots(figsize=(10, 7))

def animate(i):
    ax.clear()

    shared = contagion_steps[i] if i < len(contagion_steps) else contagion_steps[-1]

    node_colors = []
    node_borders = []
    for n in G.nodes:
        node_borders.append('red' if n in shared else 'black')
        node_colors.append('lightgreen' if G.nodes[n]['gender'] == 'Male' else 'lightblue')

    node_sizes = [300 + 100 * G.nodes[n]['triggered_count'] for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors=node_borders, linewidths=2)
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5, edge_color='gray')

    labels = {node: G.nodes[node]['ideology'] for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')

    ax.set_title(f"Step {i + 1}: Contagion Spread Simulation")
    ax.axis('off')

ani = FuncAnimation(fig, animate, frames=len(contagion_steps), interval=1000, repeat=False)
st.pyplot(fig)

# --- Step 12: Static plot sanity check ---
fig_static, ax_static = plt.subplots(figsize=(10, 7))
node_colors_static = ['red' if G.nodes[n]['shared'] else 'gray' for n in G.nodes]
nx.draw(G, pos, node_color=node_colors_static, with_labels=True, ax=ax_static)
ax_static.set_title("Static Network: Red = Shared Nodes")
st.pyplot(fig_static)
