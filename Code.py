import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random

# --- Dummy Data: Use your actual graph here ---
NUM_USERS = 30
G = nx.erdos_renyi_graph(NUM_USERS, 0.1, seed=42)

# Assign random attributes for demo
for node in G.nodes:
    G.nodes[node]['gender'] = random.choice(['Male', 'Female'])
    G.nodes[node]['triggered_count'] = random.randint(0, 10)
    G.nodes[node]['score'] = random.randint(50, 150)

pos = nx.spring_layout(G, seed=42)

# --- Sidebar Control ---
top_n = st.sidebar.slider("Number of Top Influencers", 1, 10, 3)

# --- Leaderboard computation ---
sorted_nodes = sorted(G.nodes, key=lambda n: G.nodes[n]['score'], reverse=True)
top_nodes = sorted_nodes[:top_n]

male_triggered = sum(G.nodes[n]['triggered_count'] for n in G.nodes if G.nodes[n]['gender']=='Male')
female_triggered = sum(G.nodes[n]['triggered_count'] for n in G.nodes if G.nodes[n]['gender']=='Female')

# --- Layout ---
col1, col2 = st.columns([3,1])

with col1:
    st.subheader("Contagion Animation Graph")

    plt.figure(figsize=(8,6))
    ax = plt.gca()
    
    # Draw edges by gender pairs
    male_edges = [(u,v) for u,v in G.edges if G.nodes[u]['gender']=='Male' and G.nodes[v]['gender']=='Male']
    female_edges = [(u,v) for u,v in G.edges if G.nodes[u]['gender']=='Female' and G.nodes[v]['gender']=='Female']
    mixed_edges = [(u,v) for u,v in G.edges if G.nodes[u]['gender'] != G.nodes[v]['gender']]

    nx.draw_networkx_edges(G, pos, edgelist=male_edges, edge_color='lightgreen', alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=female_edges, edge_color='lightblue', alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=mixed_edges, edge_color='gray', alpha=0.4, ax=ax)

    # Draw nodes by gender and shape
    male_nodes = [n for n in G.nodes if G.nodes[n]['gender']=='Male']
    female_nodes = [n for n in G.nodes if G.nodes[n]['gender']=='Female']

    nx.draw_networkx_nodes(G, pos, nodelist=male_nodes,
                           node_color='green',
                           node_shape='o',
                           node_size=[300 + 50 * G.nodes[n]['triggered_count'] for n in male_nodes],
                           alpha=0.9, ax=ax)

    nx.draw_networkx_nodes(G, pos, nodelist=female_nodes,
                           node_color='blue',
                           node_shape='s',
                           node_size=[300 + 50 * G.nodes[n]['triggered_count'] for n in female_nodes],
                           alpha=0.9, ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    plt.axis('off')
    st.pyplot(plt.gcf())

with col2:
    st.subheader("🏆 Top Influencers")
    for rank, node in enumerate(top_nodes, 1):
        gender = G.nodes[node]['gender']
        score = G.nodes[node]['score']
        triggered = G.nodes[node]['triggered_count']
        st.markdown(f"**Rank {rank}: User {node} — Score: {score},** Triggered: {triggered}, Gender: {gender}")

    st.markdown("---")
    st.markdown(f"**Male Users Triggered Shares:** {male_triggered}")
    st.markdown(f"**Female Users Triggered Shares:** {female_triggered}")

    st.markdown("---")
    st.markdown("Use the slider to adjust the number of top influencers shown.")
