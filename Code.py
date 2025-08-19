import streamlit as st
import networkx as nx
import random
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- Scraping YouTube using BeautifulSoup ---
def scrape_youtube(query="health tips", max_results=5):
    query = query.replace(' ', '+')
    url = f"https://www.youtube.com/results?search_query={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return [], "Failed to retrieve YouTube data"
    
    soup = BeautifulSoup(response.text, "html.parser")
    video_links = []
    for a_tag in soup.find_all('a', href=True):
        if '/watch?v=' in a_tag['href']:
            video_url = 'https://www.youtube.com' + a_tag['href']
            video_links.append(video_url)
            if len(video_links) >= max_results:
                break
    
    if not video_links:
        return [], "No videos found. Check the search query or page structure."
    
    return video_links, "Success"

# --- Graph and Contagion Setup ---
G = nx.erdos_renyi_graph(30, 0.1)  # Example graph with 30 nodes
for node in G.nodes:
    G.nodes[node]['gender'] = 'Male' if node % 2 == 0 else 'Female'
    G.nodes[node]['score'] = random.randint(50, 100)
    G.nodes[node]['triggered_count'] = 0
    G.nodes[node]['shared'] = False

# --- Contagion Propagation ---
def propagate_contagion(G, initial_users, probability=0.5, max_steps=10):
    contagion_steps = []
    triggered = set(initial_users)
    
    for step in range(max_steps):
        new_triggered = set()
        
        for user in triggered:
            for neighbor in G.neighbors(user):
                if neighbor not in triggered and random.random() < probability:
                    new_triggered.add(neighbor)
        
        if not new_triggered:
            break
        
        triggered.update(new_triggered)
        contagion_steps.append(new_triggered)
    
    return contagion_steps

initial_users = [1, 2, 3]  # Start with a few users
contagion_steps = propagate_contagion(G, initial_users, probability=0.6, max_steps=15)

# --- Streamlit UI ---
st.title("Health Information Spread Simulation")
st.subheader("Model Evaluation")

# Simulate a confusion matrix (for illustration purposes)
y_test = [1, 0, 1, 1, 0]
y_pred = [1, 0, 1, 0, 1]

# Use sklearn's confusion_matrix function
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# Display the video results from YouTube scraping
videos, message = scrape_youtube(query="health tips", max_results=5)
if videos:
    st.write("Found videos:")
    for video in videos:
        st.markdown(f"[Watch video]({video})")
else:
    st.error(message)

# Display the contagion graph and leaderboard
left_col, right_col = st.columns([2, 1])

with left_col:
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes}, font_size=8, ax=ax)
    ax.set_title(f"Network at Contagion Step (Red outline = Shaped by triggers)")
    ax.axis('off')
    st.pyplot(fig)

# Rank the influencers
influencer_stats = []
for node in G.nodes:
    influencer_stats.append({
        'user': node,
        'score': G.nodes[node]['score'],
        'triggered': G.nodes[node]['triggered_count'],
    })
top_influencers = sorted(influencer_stats, key=lambda x: (x['triggered'], x['score']), reverse=True)[:5]

# Display top influencers
st.markdown("### 🏆 Top Influencers")
for rank, inf in enumerate(top_influencers, 1):
    st.markdown(f"- **Rank {rank}**: User {inf['user']} — Score: {inf['score']}, Triggered: {inf['triggered']}")

# Display triggered users
male_triggered = sum(1 for n in contagion_steps[-1] if G.nodes[n]['gender'] == 'Male')
female_triggered = sum(1 for n in contagion_steps[-1] if G.nodes[n]['gender'] == 'Female')

st.markdown(f"- **Male Users Triggered**: {male_triggered} shares")
st.markdown(f"- **Female Users Triggered**: {female_triggered} shares")

