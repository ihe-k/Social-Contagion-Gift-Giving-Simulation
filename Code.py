import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# --- Sample Data Setup (adjust based on actual simulation logic) ---
# Assuming you already have G (Graph), y_test, and y_pred defined
# Example placeholders for the simulation
G = nx.erdos_renyi_graph(30, 0.2)  # Example graph (30 nodes, 20% chance of edge creation)
for node in G.nodes:
    G.nodes[node]['score'] = random.randint(1, 100)  # Random score
    G.nodes[node]['gender'] = 'Male' if node % 2 == 0 else 'Female'  # Alternating genders
    G.nodes[node]['triggered_count'] = 0  # Initialize trigger count
    G.nodes[node]['shared'] = False  # Shared info status (False by default)

# Example simulation of contagion spread (you would replace this with actual logic)
contagion_steps = [
    {1, 2, 3},  # Step 1: Users 1, 2, 3 share info
    {4, 5},     # Step 2: Users 4, 5 share info
    {6, 7, 8},  # Step 3: Users 6, 7, 8 share info
    # Add more steps based on the simulation logic
]

# --- Streamlit UI with interactive contagion step slider ---
st.title("Health Information Spread Simulation")

# Placeholder for Model Evaluation (dummy data for illustration)
accuracy = 0.75  # Example accuracy (should be calculated based on your model)
y_test = np.random.randint(0, 2, 30)  # Dummy ground truth
y_pred = np.random.randint(0, 2, 30)  # Dummy predictions

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

st.subheader("Model Evaluation")
st.write(f"Accuracy: {accuracy:.2%}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Slider to pick contagion step
max_step = len(contagion_steps)
step = st.slider("Select contagion step", 1, max_step, max_step)

# Collect nodes that shared up to current step
shared_up_to_step = set()
for i in range(step):
    shared_up_to_step.update(contagion_steps[i])

# Prepare graph visualization
left_col, right_col = st.columns([2, 1])  # Wider left for network diagram

with left_col:
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    male_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Male']
    female_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Female']

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)

    # Draw all nodes by gender with specified colors
    nx.draw_networkx_nodes(G, pos, nodelist=male_nodes, node_color='#03396c', node_size=300, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=female_nodes, node_color='#6497b1', node_size=300, ax=ax)

    # Highlight nodes that have shared *up to* current step (red outline)
    nx.draw_networkx_nodes(
        G, pos, nodelist=list(shared_up_to_step),
        node_color='none', edgecolors='red', node_size=330, linewidths=2, ax=ax
    )

    # Labels
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes}, font_size=8, ax=ax)

    ax.set_title(f"Network at Contagion Step {step} (Red outline = Shared)")
    ax.axis('off')
    st.pyplot(fig)

with right_col:
    st.markdown("### 🏆 Top Influencers")

    # Sort top influencers by triggered count first, then by score
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
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

