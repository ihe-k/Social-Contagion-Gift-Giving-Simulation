import random
import networkx as nx
import matplotlib.pyplot as plt
from textblob import TextBlob
import streamlit as st
from youtubesearchpython import VideosSearch
from TikTokApi import TikTokApi
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import numpy as np
import requests
from bs4 import BeautifulSoup

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

# --- Step 3: Scrape podcasts from Reddit ---
def get_health_podcasts_from_reddit(subreddit="Health"):
    url = f"https://www.reddit.com/r/{subreddit}/search/?q=podcast&restrict_sr=1&sort=relevance&t=all"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all post titles and links related to health podcasts
    posts = soup.find_all('a', class_='SQnoC3ObvgnGjWf5p3r1')
    podcasts = []
    for post in posts:
        title = post.text
        link = "https://www.reddit.com" + post['href']
        podcasts.append({
            "user": post.text.split(' ')[0],  # Simplified: First word of the title as user
            "content": title,
            "platform": "Reddit",
            "url": link
        })
    return podcasts

# --- Step 4: Get videos and podcasts ---
youtube_videos = get_health_videos_from_youtube()
tiktok_videos = get_health_videos_from_tiktok()
reddit_podcasts = get_health_podcasts_from_reddit("Health")

all_content = youtube_videos + tiktok_videos + reddit_podcasts

# --- Step 5: Assign user attributes ---
user_data = []
for content in all_content:
    user = content['user']
    text = content['content']
    gender = random.choice(['Male', 'Female'])
    sentiment = analyze_sentiment(text)
    has_chronic = random.choice([True, False])
    user_data.append({
        'user': user,
        'gender': gender,
        'sentiment': sentiment,
        'ideology': sentiment,
        'has_chronic_disease': has_chronic
    })

# Add to graph nodes
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

# --- Step 6: Sentiment trend & centrality ---
def calculate_sentiment_trends():
    trends = []
    for node in G.nodes:
        scores = [1 if G.nodes[n]['sentiment']=='pro-health' else 0 for n in G.neighbors(node)]
        trends.append(np.mean(scores) if scores else 0)
    return trends

def calculate_betweenness_centrality():
    return nx.betweenness_centrality(G)

sentiment_trends = calculate_sentiment_trends()
betweenness_centrality = calculate_betweenness_centrality()

# --- Step 7: Prepare features and labels ---
features = []
labels = []

for node in G.nodes:
    u = G.nodes[node]
    features.append([
        1 if u['gender']=='Female' else 0,
        1 if u['has_chronic_disease'] else 0,
        1 if u['ideology']=='pro-health' else 0,
        1 if u['ideology']=='anti-health' else 0,
        1 if u['ideology']=='neutral' else 0,
        sentiment_trends[node],
        betweenness_centrality[node]
    ])
    labels.append(u['ideology'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# --- Step 8: Train Random Forest with GridSearchCV ---
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# --- Step 9: Evaluation ---
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Evaluation")
st.write(f"Accuracy: {accuracy:.2%}")

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

fig_cm, ax_cm = plt.subplots()
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=ax_cm)
st.pyplot(fig_cm)

# --- Step 10: Contagion Simulation ---
pos = nx.spring_layout(G, seed=42)

initial_gifted = random.sample(list(G.nodes), INIT_SHARED)
for node in initial_gifted:
    G.nodes[node]['shared'] = True
    G.nodes[node]['gifted'] = True

contagion_steps = []
contagion_steps.append(set(initial_gifted))

def run_contagion():
    new_shared = set(initial_gifted)
    all_shared = set(initial_gifted)

    while new_shared:
        next_new = set()
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
                        next_new.add(neighbor)
                        all_shared.add(neighbor)
        if not next_new:
            break
        contagion_steps.append(next_new)
        new_shared = next_new

run_contagion()

st.subheader("Health Information Spread Simulation (Static Graph)")

fig, ax = plt.subplots(figsize=(10,7))
ax.axis('off')

node_colors = ['lightgreen' if G.nodes[n]['gender']=='Male' else 'lightblue' for n in G.nodes]
node_sizes = [400 + 100 * G.nodes[n]['triggered_count'] for n in G.nodes]
node_borders = ['red' if G.nodes[n]['shared'] else 'black' for n in G.nodes]

nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, edge_color='gray')
nx.draw_networkx_nodes(G, pos, ax=ax,
                       node_size=node_sizes,
                       node_color=node_colors,
                       edgecolors=node_borders,
                       linewidths=1.5)
labels = {n: G.nodes[n]['ideology'] for n in G.nodes}
nx.draw_networkx_labels(G, pos, labels, font_size=9, font_color='black', ax=ax)

st.pyplot(fig)
