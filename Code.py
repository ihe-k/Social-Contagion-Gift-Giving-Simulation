import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns  # For Confusion Matrix visualization
import requests
from bs4 import BeautifulSoup  # BeautifulSoup4 for scraping

# --- Scraping YouTube using BeautifulSoup ---
def scrape_youtube(query="health tips"):
    search_url = f'https://www.youtube.com/results?search_query={query}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # Send HTTP request to YouTube search page
    response = requests.get(search_url, headers=headers)
    if response.status_code != 200:
        st.error(f"Failed to fetch YouTube page. Status code: {response.status_code}")
        return []

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all video links and titles
    video_data = []
    for video in soup.find_all('a', {'id': 'video-title'}):
        title = video.get('title')
        url = "https://www.youtube.com" + video.get('href')
        video_data.append({"title": title, "url": url})

    return video_data

# --- Sample Data Setup (adjust based on actual simulation logic) ---
# Example setup for a random graph (simulate users and contagion spread)
G = nx.erdos_renyi_graph(30, 0.2)  # 30 users, 20% chance of edge creation
for node in G.nodes:
    G.nodes[node]['score'] = random.randint(1, 100)  # Random score for each user
    G.nodes[node]['gender'] = 'Male' if node % 2 == 0 else 'Female'  # Male and Female alternated
    G.nodes[node]['triggered_count'] = 0  # Triggered count (for contagion)
    G.nodes[node]['shared'] = False  # Shared info status

# Example steps for contagion simulation
contagion_steps = [
    {1, 2, 3},  # Step 1: Users 1, 2, 3 share info
    {4, 5},     # Step 2: Users 4, 5 share info
    {6, 7, 8},  # Step 3: Users 6, 7, 8 share info
    # Add more steps for your logic
]

# --- Streamlit UI ---
st.title("Health Information Spread Simulation")

# Scrape YouTube for "health tips" as an example
st.markdown("### YouTube Scraped Data (Health Tips)")
videos = scrape_youtube("health tips")
if not videos:
    st.error("No videos found.")
else:
    for video in videos:
        st.markdown(f"[{video['title']}]({video['url']})")

# Slider to pick contagion step
max_step = len(contagion_steps)
step = st.slider("Select contagion step", 1, max_step, max_step)

# Collect nodes that shared up to current step
shared_up_to_step = set()
for i in range(step):
    shared_up_to_step.update(contagion_steps[i])

# Update trigger count based on shared nodes
for node in shared_up_to_step:
    G.nodes[node]['triggered_count'] += 1

# --- Display Network Diagram ---
left_col, right_col = st.columns([2, 1])  # Wider left column for network diagram

with left_col:
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    # Separate male and female nodes for color coding
    male_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Male']
    female_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Female']

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)

    # Draw nodes by gender
    nx.draw_networkx_nodes(G, pos, nodelist=male_nodes, node_color='#03396c', node_size=300, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=female_nodes, node_color='#6497b1', node_size=300, ax=ax)

    # Highlight nodes that shared up to current step (red outline)
    nx.draw_networkx_nodes(
        G, pos, nodelist=list(shared_up_to_step),
        node_color='none', edgecolors='red', node_size=330, linewidths=2, ax=ax
    )

    # Labels for nodes
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes}, font_size=8, ax=ax)

    ax.set_title(f"Network at Contagion Step {step} (Red outline = Shared)")
    ax.axis('off')
    st.pyplot(fig)

with right_col:
    st.markdown("### 🏆 Top Influencers")

    # Sort influencers by trigger count and score
    influencer_stats = []
    for node in G.nodes:
        influencer_stats.append({
            'user': node,
            'score': G.nodes[node]['score'],
            'triggered': G.nodes[node]['triggered_count'],
        })

    top_influencers = sorted(influencer_stats, key=lambda x: (x['triggered'], x['score']), reverse=True)[:5]

    for rank, inf in enumerate(top_influencers, 1):
        st.markdown(f"- **Rank {rank}**: User {inf['user']} — Score: {inf['score']}, Triggered: {inf['triggered']}")

    male_triggered = sum(1 for n in shared_up_to_step if G.nodes[n]['gender'] == 'Male')
    female_triggered = sum(1 for n in shared_up_to_step if G.nodes[n]['gender'] == 'Female')

    st.markdown(f"- **Male Users Triggered**: {male_triggered} shares")
    st.markdown(f"- **Female Users Triggered**: {female_triggered} shares")

    st.markdown("---")
    st.markdown("🕹️ Use the slider to explore the contagion spread over time.")

    # --- Move confusion matrix to the right of the model evaluation ---
    st.subheader("Confusion Matrix")
    cm = confusion_matrix([random.randint(0, 1) for _ in range(30)], [random.randint(0, 1) for _ in range(30)])  # Dummy confusion matrix for illustration
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

