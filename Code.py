import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from textblob import TextBlob
import streamlit as st
import feedparser
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

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
    """
    Analyze sentiment of text and classify as pro-health, anti-health, or neutral.
    Pro-health: Positive sentiment (score > 0.5)
    Anti-health: Negative sentiment (score < -0.5)
    Neutral: Sentiment score between -0.5 and 0.5
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.5:
        return 'pro-health'
    elif polarity < -0.5:
        return 'anti-health'
    else:
        return 'neutral'

# --- Step 3: Podcast RSS Feed Parsing ---
def get_health_podcasts(feed_urls, keyword="health"):
    """
    Parse multiple podcast RSS feeds and return episodes with keyword in title or description.
    """
    episodes = []
    for feed_url in feed_urls:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            title = entry.get('title', '').lower()
            description = entry.get('description', '').lower()
            if keyword in title or keyword in description:
                episodes.append({
                    "title": entry.get('title', 'No Title'),
                    "author": feed.feed.get('title', 'Unknown Podcast'),
                    "link": entry.get('link', ''),
                    "published": entry.get('published', 'Unknown Date'),
                    "summary": entry.get('summary', '')
                })
    return episodes

# Podcast RSS feeds
daily_feed = ["https://rss.art19.com/the-daily"]  # NPR The Daily

other_feeds = [
    "https://feeds.megaphone.fm/ADV8924270618",   # Call Her Daddy
    "https://feeds.simplecast.com/5Z9BHkuQ",      # This Past Weekend w/ Theo Von
    "https://joeroganexp.joerogan.libsynpro.com/rss", # Joe Rogan Experience
    "https://feeds.npr.org/510289/podcast.xml",   # NPR Up First alternative
    "https://feeds.simplecast.com/tOjNXec5",      # Reply All
    "https://feeds.megaphone.fm/stuffyoushouldknow" # Stuff You Should Know
]

# --- Step 4: Collect podcast episodes with health keyword ---
st.header("Health-Related Podcast Episodes")

# The Daily episodes first
st.subheader("NPR The Daily")
daily_eps = get_health_podcasts(daily_feed)
if daily_eps:
    for i, ep in enumerate(daily_eps[:10]):
        st.markdown(f"**{i+1}. {ep['title']}**  ")
        st.markdown(f"*Podcast:* {ep['author']}  ")
        st.markdown(f"*Published:* {ep['published']}  ")
        st.markdown(f"[Listen here]({ep['link']})")
        st.markdown("---")
else:
    st.write("No health-related episodes found in The Daily.")

# Other podcasts
st.subheader("Other Popular Podcasts")
other_eps = get_health_podcasts(other_feeds)
if other_eps:
    for i, ep in enumerate(other_eps[:20]):
        st.markdown(f"**{i+1}. {ep['title']}**  ")
        st.markdown(f"*Podcast:* {ep['author']}  ")
        st.markdown(f"*Published:* {ep['published']}  ")
        st.markdown(f"[Listen here]({ep['link']})")
        st.markdown("---")
else:
    st.write("No health-related episodes found in other podcasts.")

# --- Step 5: Generate user data from podcast episodes ---
user_data = []
for i, ep in enumerate(daily_eps + other_eps):
    user = ep['author']
    text_content = ep['title'] + " " + ep['summary']

    # Assign gender randomly
    gender = random.choice(['Male', 'Female'])

    # Assign ideology based on sentiment
    sentiment = analyze_sentiment(text_content)

    # Assign chronic disease status randomly
    has_chronic_disease = random.choice([True, False])

    user_data.append({
        'user': user,
        'gender': gender,
        'sentiment': sentiment,
        'ideology': sentiment,  # Simplified
        'has_chronic_disease': has_chronic_disease
    })

# If there are fewer episodes than NUM_USERS, pad with random users
while len(user_data) < NUM_USERS:
    user_data.append({
        'user': f'User{len(user_data)}',
        'gender': random.choice(['Male', 'Female']),
        'sentiment': random.choice(['pro-health', 'anti-health', 'neutral']),
        'ideology': random.choice(['pro-health', 'anti-health', 'neutral']),
        'has_chronic_disease': random.choice([True, False])
    })

# Assign user data to graph nodes
for i in range(NUM_USERS):
    info = user_data[i]
    G.nodes[i]['gender'] = info['gender']
    G.nodes[i]['sentiment'] = info['sentiment']
    G.nodes[i]['ideology'] = info['ideology']
    G.nodes[i]['has_chronic_disease'] = info['has_chronic_disease']
    G.nodes[i]['shared'] = False
    G.nodes[i]['score'] = 0
    G.nodes[i]['triggered_count'] = 0

# --- Step 6: Feature Engineering ---
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

# --- Step 7: Prepare Data for Model Training ---
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

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(user_features, user_labels, test_size=0.2, random_state=42)

# --- Step 8: Hyperparameter Tuning ---
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

# --- Step 9: Evaluate Model ---
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Model Evaluation")
st.write(f"Model Accuracy: {accuracy:.2%}")

report = classification_report(y_test, y_pred)
st.text("Classification Report:")
st.text(report)

fig_cm, ax_cm = plt.subplots()
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=ax_cm)
st.pyplot(fig_cm)

# --- Step 10: Contagion Simulation ---

# Initialize contagion
initial_gifted = random.sample(list(G.nodes), INIT_SHARED)
for node in initial_gifted:
    G.nodes[node]['shared'] = True
    G.nodes[node]['gifted'] = True

contagion_steps = [set(initial_gifted)]

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

        if not next_new_shared:
            break
        contagion_steps.append(next_new_shared)
        new_shared = next_new_shared

run_contagion_simulation()

# --- Step 11: Static Contagion Graph ---
fig, ax = plt.subplots(figsize=(10, 7))
pos = nx.spring_layout(G, seed=42)

node_colors = ['lightgreen' if G.nodes[n]['gender'] == 'Male' else 'lightblue' for n in G.nodes]
node_sizes = [300 + 100 * G.nodes[n]['triggered_count'] for n in G.nodes]

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5, edge_color='gray', ax=ax)
labels = {node: G.nodes[node]['ideology'] for node in G.nodes}
nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black', ax=ax)

ax.set_title("Static Contagion Spread Simulation")
ax.axis('off')

st.subheader("Health Information Spread Simulation - Static Graph")
st.pyplot(fig)

# --- Step 12: Leaderboard ---
ideology_triggered = {"pro-health": 0, "anti-health": 0, "neutral": 0}
gender_triggered = {"Male": 0, "Female": 0}

for node in G.nodes:
    ideology = G.nodes[node]['ideology']
    gender = G.nodes[node]['gender']
    if ideology in ideology_triggered:
        ideology_triggered[ideology] += G.nodes[node]['triggered_count']
    if gender in gender_triggered:
        gender_triggered[gender] += G.nodes[node]['triggered_count']

st.subheader("Leaderboard")
st.write(f"Pro-health triggered: {ideology_triggered['pro-health']}")
st.write(f"Anti-health triggered: {ideology_triggered['anti-health']}")
st.write(f"Neutral triggered: {ideology_triggered['neutral']}")
st.write(f"Male triggered: {gender_triggered['Male']}")
st.write(f"Female triggered: {gender_triggered['Female']}")
