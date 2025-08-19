import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import pandas as pd
import random

# --- Streamlit Layout ---
st.title("Health Information Spread Simulation")
st.subheader("Model Evaluation")

# Sample data setup
# Example G (Graph) initialization
G = nx.erdos_renyi_graph(30, 0.1)  # Random graph for demonstration
for node in G.nodes:
    G.nodes[node]['gender'] = 'Male' if node % 2 == 0 else 'Female'  # Random gender
    G.nodes[node]['score'] = random.randint(70, 100)  # Random score
    G.nodes[node]['triggered_count'] = 0  # Placeholder trigger count
    G.nodes[node]['shared'] = False  # Shared info status

# --- Scrape YouTube Data ---
def get_youtube_triggers(search_query="health tips"):
    # Setup Selenium WebDriver
    options = Options()
    options.headless = True  # Run in headless mode (no browser UI)
    driver = webdriver.Chrome(executable_path='/path/to/chromedriver', options=options)
    driver.get(f"https://www.youtube.com/results?search_query={search_query}")
    
    time.sleep(2)  # Wait for page to load
    
    # Scroll the page to load more results (simulate user scrolling)
    for _ in range(3):
        driver.execute_script("window.scrollBy(0, 1000);")
        time.sleep(2)
    
    # Get page source
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()
    
    # Extract video information (titles, views, etc.)
    video_data = []
    for video in soup.find_all('ytd-video-renderer', class_='style-scope ytd-item-section-renderer'):
        title = video.find('a', {'id': 'video-title'}).text
        views = video.find('span', {'class': 'style-scope ytd-video-meta-block'}).text
        
        # Extract number of views
        views = ''.join(filter(str.isdigit, views))  # Remove non-numeric characters
        video_data.append({'title': title, 'views': int(views)})
    
    # Return the video data
    return video_data

# Get YouTube data and simulate triggered users based on views
youtube_data = get_youtube_triggers()

# Add triggered count based on view count
for video in youtube_data:
    trigger_count = video['views'] // 100000  # Example: Every 100K views = 1 trigger
    # Assign a random user for each video and increase their triggered count
    for i in range(trigger_count):
        user = i % len(G.nodes)  # Assign users cyclically for simplicity
        G.nodes[user]['triggered_count'] += 1

# --- Contagion Step Simulation ---
contagion_steps = []
initial_gifted_users = [0, 1, 2]  # Example starting users

# Simulate contagion spread for X steps
for step in range(6):  # Assuming 6 steps
    new_sharers = set()
    for user in initial_gifted_users:
        if G.nodes[user]['triggered_count'] > 0:
            # Simulate sharing the info with others
            neighbors = list(G.neighbors(user))
            
            # Dynamically trigger a random number of neighbors based on the trigger count
            num_to_trigger = G.nodes[user]['triggered_count'] // 3  # Trigger based on the user's triggered count
            for _ in range(num_to_trigger):
                random_neighbor = random.choice(neighbors)  # Randomly pick a neighbor to trigger
                if G.nodes[random_neighbor]['triggered_count'] == 0:  # Only share if not triggered
                    new_sharers.add(random_neighbor)
    
    # Update nodes with new share information
    for user in new_sharers:
        G.nodes[user]['triggered_count'] += 1
    
    contagion_steps.append(new_sharers)
    initial_gifted_users = list(new_sharers)

# --- Network Graph ---
left_col, right_col = st.columns([2, 1])

with left_col:
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    male_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Male']
    female_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Female']
    shared_nodes = [n for n in G.nodes if G.nodes[n]['triggered_count'] > 0]

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)

    # Draw all nodes by gender
    nx.draw_networkx_nodes(G, pos, nodelist=male_nodes, node_color='#03396c', node_size=300, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=female_nodes, node_color='#6497b1', node_size=300, ax=ax)

    # Highlight nodes that have triggered (red outline)
    nx.draw_networkx_nodes(
        G, pos, nodelist=shared_nodes,
        node_color='none', edgecolors='red', node_size=330, linewidths=2, ax=ax
    )

    # Labels
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes}, font_size=8, ax=ax)

    ax.set_title("Network at Contagion Step (Red outline = Shared)")
    ax.axis('off')
    st.pyplot(fig)

# --- Leaderboard ---
with right_col:
    st.markdown("### 🏆 Top Influencers")

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

    st.markdown("---")
    st.markdown("🕹️ Use the slider to explore the contagion spread over time.")

    # Optional slider to explore contagion steps
    step = st.slider("Contagion Step", min_value=1, max_value=len(contagion_steps), value=len(contagion_steps))
    st.markdown(f"Showing step {step} of {len(contagion_steps)}")
