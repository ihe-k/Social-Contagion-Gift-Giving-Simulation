import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from textblob import TextBlob
import matplotlib.colors as mcolors
import feedparser
from sklearn.model_selection import train_test_split

# ---- Network Simulation ----
# Generate a random graph for users and their connections
def create_random_graph(num_users, prob_connection):
    G = nx.erdos_renyi_graph(num_users, prob_connection)
    for node in G.nodes:
        G.nodes[node]['gender'] = random.choice(['male', 'female'])
        G.nodes[node]['health_condition'] = random.choice(['healthy', 'chronic'])
        G.nodes[node]['ideology'] = random.choice(['pro-health', 'anti-health', 'neutral'])
        G.nodes[node]['sentiment'] = random.choice(['pro-health', 'anti-health', 'neutral'])
    return G

# ---- Sentiment Analysis ----
def classify_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return 'pro-health'
    elif polarity < -0.1:
        return 'anti-health'
    else:
        return 'neutral'

# ---- Machine Learning Classifier ----
def train_classifier(G):
    features = []
    labels = []
    for node in G.nodes:
        features.append([G.nodes[node]['gender'] == 'female', 
                         G.nodes[node]['health_condition'] == 'chronic',
                         G.nodes[node]['ideology'] == 'pro-health'])
        labels.append(G.nodes[node]['sentiment'] == 'pro-health')
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    return clf, classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred)

# ---- Visualization ----
def visualize_network(G):
    pos = nx.spring_layout(G)  # Use spring layout to organize nodes
    sentiment_map = {'pro-health': 'green', 'anti-health': 'red', 'neutral': 'gray'}
    node_colors = [sentiment_map[G.nodes[node]['sentiment']] for node in G.nodes]
    
    nx.draw(G, pos, with_labels=True, node_size=700, node_color=node_colors, font_size=10, edge_color='gray')
    plt.title('Social Network Graph')
    plt.show()

# ---- Streamlit Interface ----
st.title("Health Information Contagion Simulation")

st.sidebar.header("Network Parameters")
num_users = st.sidebar.slider("Number of Users", min_value=10, max_value=500, value=100)
prob_connection = st.sidebar.slider("Probability of Connection", min_value=0.01, max_value=0.5, value=0.1)

# ---- Generate Network ----
st.subheader("Network Visualization")
G = create_random_graph(num_users, prob_connection)

# Display Network
visualize_network(G)

# ---- Model Training and Evaluation ----
st.subheader("Model Training & Evaluation")
clf, class_report, conf_matrix = train_classifier(G)

# Display Classification Report
st.text("Classification Report")
st.text(class_report)

# Display Confusion Matrix
st.text("Confusion Matrix")
st.text(conf_matrix)

# ---- RSS Feed for Health Content ----
st.sidebar.header("RSS Feed Configuration")
rss_feed_url = st.sidebar.text_input("Enter RSS Feed URL")

def fetch_rss_feed(url):
    feed = feedparser.parse(url)
    return feed.entries

if rss_feed_url:
    st.subheader("Health Podcast RSS Feed")
    feed_entries = fetch_rss_feed(rss_feed_url)
    for entry in feed_entries[:5]:  # Display first 5 entries
        st.write(f"Title: {entry.title}")
        st.write(f"Link: {entry.link}")
        st.write(f"Summary: {entry.summary}")
        sentiment = classify_sentiment(entry.summary)
        st.write(f"Sentiment: {sentiment}")
        st.write("-" * 50)

# ---- Interactive Widgets for Customization ----
st.sidebar.subheader("User Customization")
gender_filter = st.sidebar.selectbox("Select Gender", ['All', 'Male', 'Female'])
health_condition_filter = st.sidebar.selectbox("Select Health Condition", ['All', 'Healthy', 'Chronic'])
ideology_filter = st.sidebar.selectbox("Select Ideology", ['All', 'Pro-Health', 'Anti-Health', 'Neutral'])

# ---- Filter Network by User Attributes ----
filtered_G = G.copy()

if gender_filter != 'All':
    filtered_G = nx.subgraph_view(filtered_G, filter_node=lambda n: G.nodes[n]['gender'] == gender_filter)

if health_condition_filter != 'All':
    filtered_G = nx.subgraph_view(filtered_G, filter_node=lambda n: G.nodes[n]['health_condition'] == health_condition_filter)

if ideology_filter != 'All':
    filtered_G = nx.subgraph_view(filtered_G, filter_node=lambda n: G.nodes[n]['ideology'] == ideology_filter)

st.subheader("Filtered Network Visualization")
visualize_network(filtered_G)

# ---- Show Metrics and Predictions ----
st.sidebar.header("Model Prediction")
user_id = st.sidebar.number_input("Enter User ID for Prediction", min_value=0, max_value=num_users-1, value=0)
user_data = G.nodes[user_id]
user_features = [[user_data['gender'] == 'female', 
                  user_data['health_condition'] == 'chronic', 
                  user_data['ideology'] == 'pro-health']]

prediction = clf.predict(user_features)[0]
predicted_sentiment = 'Pro-Health' if prediction else 'Anti-Health'
st.subheader(f"User {user_id} Sentiment Prediction")
st.write(f"Predicted Sentiment: {predicted_sentiment}")

