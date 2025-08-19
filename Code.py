import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from textblob import TextBlob
import streamlit as st
# Commented out real YouTube search to avoid API issues; replaced with dummy data
# from youtubesearchpython import VideosSearch
from TikTokApi import TikTokApi
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- Parameters ---
NUM_USERS = 30
INIT_SHARED = 3  # initial gifted users
SHARE_PROB = 0.3  # base probability of sharing health information
GIFT_BONUS = 10  # score for triggering another user's share
IDEOLOGY_CROSS_BONUS = 0.2  # bonus for cross-ideology sharing
CHRONIC_PROPENSITY = 0.6  # probability of sharing health info for users with chronic disease
GENDER_PROPENSITY = {"Male": 0.3, "Female": 0.5}  # Gender-specific sharing propensity
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

# --- Step 3: Dummy Health-Related Content to Replace YouTube API ---
def get_health_videos_from_youtube():
    # Dummy videos so code runs without external calls
    return [
        {
            "user": "DummyYouTubeUser1",
            "content": "Health benefits of walking daily",
            "platform": "YouTube",
            "url": "https://www.youtube.com/watch?v=dummy1"
        },
        {
            "user": "WellnessChannel",
            "content": "Top 5 Healthy Eating Tips",
            "platform": "YouTube",
            "url": "https://www.youtube.com/watch?v=dummy2"
        },
    ]

# --- Step 3b: TikTok scraping with safe fallback ---
def get_health_videos_from_tiktok(query="health"):
    try:
        api = TikTokApi.get_instance()
        trending_videos = api.by_hashtag(query, count=5)
        videos = []
        for video in trending_videos:
            videos.append({
                "user": video['author']['uniqueId'],
                "content": video['desc'],
                "platform": "TikTok",
                "url": f"https://www.tiktok.com/@{video['author']['uniqueId']}/video/{video['id']}"
            })
        return videos
    except Exception as e:
        st.warning(f"Could not fetch TikTok data: {e}")
        return []

# --- Step 3c: Stub Twitter scraping ---
def get_twitter_data(query, limit=5):
    return []

# Gather data
youtube_videos = get_health_videos_from_youtube()
tiktok_videos = get_health_videos_from_tiktok()
twitter_tweets = get_twitter_data("health", limit=5)
all_videos_and_tweets = youtube_videos + tiktok_videos + twitter_tweets

# --- Step 4: Assign Gender, Ideology, and Chronic Disease to Users ---
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

# Add users to the network nodes (up to NUM_USERS)
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
        neighbor_sentiments = [1 if G.nodes[n]['sentiment'] == 'pro-health' else 0 for n in G.neighbors(node)]
        sentiment_trends.append(np.mean(neighbor_sentiments) if neighbor_sentiments else 0)
    return sentiment_trends

def calculate_betweenness_centrality():
    return nx.betweenness_centrality(G)

sentiment_trends = calculate_sentiment_trends()
betweenness_centrality = calculate_betweenness_centrality()

# --- Step 6: Prepare data for model ---
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

# --- Step 7: Hyperparameter tuning ---
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# --- Step 8: Model evaluation ---
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Model Evaluation")
st.write(f"Model Accuracy: {accuracy:.2%}")
report = classification_report(y_test, y_pred, output_dict=False)
st.text("Classification Report:")
st.text(report)

fig_cm, ax_cm = plt.subplots()
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=ax_cm)
st.pyplot(fig_cm)

# --- Step 9: Animation of contagion spread ---
fig, ax = plt.subplots(figsize=(10, 7))
pos = nx.spring_layout(G, seed=42)

def animate(i):
    ax.clear()
    shared = contagion_steps[i] if i < len(contagion_steps) else contagion_steps[-1]
    node_sizes = [300 + 100 * G.nodes[n]['triggered_count'] for n in G.nodes]

    male_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Male']
    female_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Female']

    nx.draw_networkx_nodes(G, pos, nodelist=male_nodes, node_size=node_sizes, node_color='lightgreen', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=female_nodes, node_size=node_sizes, node_color='lightblue', ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5, edge_color='gray', ax=ax)
    labels = {n: G.nodes[n]['ideology'] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black', ax=ax)

    ax.set_title(f"Step {i+1}: Contagion Spread Simulation")
    ax.axis('off')

# --- Step 10: Initialize contagion ---
initial_gifted = random.sample(list(G.nodes), INIT_SHARED)
for node in initial_gifted:
    G.nodes[node]['shared'] = True
    G.nodes[node]['gifted'] = True

contagion_steps = [set(initial_gifted)]

# --- Step 11: Contagion simulation ---
def run_contagion_simulation():
    new_shared = set(initial_gifted)
    all_shared = set(initial_gifted)

    while new_shared:
        next_shared = set()
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
                        next_shared.add(neighbor)
                        all_shared.add(neighbor)
        if not next_shared:
            break
        contagion_steps.append(next_shared)
        new_shared = next_shared

run_contagion_simulation()

# --- Step 12: Show animation in Streamlit ---
st.subheader("Health Information Spread Simulation")
ani = FuncAnimation(fig, animate, frames=len(contagion_steps), interval=1000, repeat=False)
st.pyplot(fig)
