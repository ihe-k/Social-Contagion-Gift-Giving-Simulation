import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd

# --- Sample Data Setup (adjust based on actual simulation logic) ---
# Create a simple random graph with sample data
G = nx.erdos_renyi_graph(30, 0.2)  # Example graph with 30 nodes and 20% chance of edge creation
for node in G.nodes():
    # Assign genders based on node ID (even = Male, odd = Female)
    G.nodes[node]['gender'] = 'Male' if node % 2 == 0 else 'Female'
    # Random score between 1 and 100 for each node
    G.nodes[node]['score'] = np.random.randint(1, 100)
    # Initialize triggered count to 0
    G.nodes[node]['triggered_count'] = 0
    # Shared state based on a random chance (simulate contagion)
    G.nodes[node]['shared'] = np.random.choice([True, False])

# Dummy classification data for evaluation
y_test = np.random.randint(0, 2, size=30)  # Random binary classification labels (0 or 1)
y_pred = np.random.randint(0, 2, size=30)  # Random predictions (0 or 1)

# --- Streamlit Layout ---
st.title("Health Information Spread Simulation")

st.subheader("Model Evaluation")

# Model accuracy calculation and classification report
accuracy = (y_test == y_pred).mean()  # Dummy accuracy calculation for illustration
st.write(f"Accuracy: {accuracy:.2%}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots(figsize=(5, 4))  # Set smaller size for confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

# Divide into two columns: left for network graph and right for leaderboard
left_col, right_col = st.columns([2, 1])  # Adjust to make left (network) wider

# --- Left Column: Network Graph ---
with left_col:
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    male_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Male']
    female_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Female']

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)

    # Draw all nodes by gender (Blue for Male, Pink for Female)
    nx.draw_networkx_nodes(G, pos, nodelist=male_nodes, node_color='#03396c', node_size=300, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=female_nodes, node_color='#6497b1', node_size=300, ax=ax)

    # Labels for nodes
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes}, font_size=8, ax=ax)

    ax.set_title(f"Network at Contagion Step")
    ax.axis('off')  # Hide axis for clarity
    st.pyplot(fig)

# --- Right Column: Leaderboard ---
with right_col:
    st.markdown("### 🏆 Top Influencers")

    influencer_stats = []
    for node in G.nodes:
        influencer_stats.append({
            'user': node,
            'score': G.nodes[node]['score'],
            'triggered': G.nodes[node]['triggered_count'],
        })
    # Sort by triggered count first, then by score (both descending)
    top_influencers = sorted(influencer_stats, key=lambda x: (x['triggered'], x['score']), reverse=True)[:5]

    for rank, inf in enumerate(top_influencers, 1):
        st.markdown(
            f"- **Rank {rank}**: User {inf['user']} — Score: {inf['score']}, Triggered: {inf['triggered']}"
        )

    male_triggered = sum(1 for n in G.nodes if G.nodes[n]['triggered_count'] > 0 and G.nodes[n]['gender'] == 'Male')
    female_triggered = sum(1 for n in G.nodes if G.nodes[n]['triggered_count'] > 0 and G.nodes[n]['gender'] == 'Female')

    st.markdown(f"- **Male Users Triggered**: {male_triggered} shares")
    st.markdown(f"- **Female Users Triggered**: {female_triggered} shares")

    st.markdown("---")
    st.markdown("🕹️ Use the slider to explore the contagion spread over time.")

    # Slider for contagion step
    max_step = 10  # Update with actual number of steps you have
    step = st.slider("Select contagion step", 1, max_step, 1)

    # Logic for displaying network visualization based on the selected step can go here
