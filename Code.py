import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

# --- Scraping YouTube using BeautifulSoup ---
def scrape_youtube(query="health tips"):
    search_url = f"https://www.youtube.com/results?search_query={query}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract video titles and URLs (you might need to adjust based on HTML structure)
    videos = []
    for video in soup.find_all('a', {'class': 'yt-uix-tile-link'}):
        video_title = video.get('title')
        video_url = 'https://www.youtube.com' + video.get('href')
        videos.append((video_title, video_url))
    return videos

# Get the top 5 videos based on the search query
videos = scrape_youtube("health tips")
st.write("### Top 5 YouTube Videos")
for i, (title, url) in enumerate(videos[:5]):
    st.write(f"{i+1}. [{title}]({url})")

# --- Sample Graph Generation for Contagion Simulation ---
G = nx.erdos_renyi_graph(30, 0.1)  # Example graph
for node in G.nodes():
    G.nodes[node]['triggered_count'] = 0
    G.nodes[node]['score'] = 0  # Initializing scores

# Sample nodes (randomly simulate contagion)
import random
for node in random.sample(G.nodes(), 5):
    G.nodes[node]['triggered_count'] = 1
    G.nodes[node]['score'] = random.randint(50, 100)

# --- Streamlit UI Layout ---
st.subheader("Contagion Spread Simulation")
left_col, right_col = st.columns([2, 1])  # Split into two columns for network graph and leaderboard

# --- LEFT COLUMN: Network Graph ---
with left_col:
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    # Draw the graph (nodes with different colors for visualization)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue', ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes()}, font_size=8, ax=ax)

    ax.set_title("Contagion Spread Simulation Network")
    ax.axis('off')
    st.pyplot(fig)

# --- RIGHT COLUMN: Leaderboard ---
with right_col:
    st.markdown("### 🏆 Top Influencers")

    influencer_stats = [{'user': node, 'score': G.nodes[node]['score'], 'triggered': G.nodes[node]['triggered_count']} for node in G.nodes()]
    top_influencers = sorted(influencer_stats, key=lambda x: (x['triggered'], x['score']), reverse=True)[:5]

    for rank, inf in enumerate(top_influencers, 1):
        st.markdown(f"- **Rank {rank}**: User {inf['user']} — Score: {inf['score']}, Triggered: {inf['triggered']}")

    st.markdown("---")
    st.markdown("🕹️ Use the slider to explore the contagion spread over time.")

# --- Confusion Matrix & Evaluation (Dummy Example) ---
from sklearn.metrics import confusion_matrix
import numpy as np

y_true = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1, 1, 1, 0, 0]

cm = confusion_matrix(y_true, y_pred)

# Display confusion matrix
st.write("### Confusion Matrix")
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(cm, cmap='Blues')
for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, val, ha='center', va='center', color='white')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
st.pyplot(fig)
