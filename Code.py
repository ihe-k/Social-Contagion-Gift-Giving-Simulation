import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# --- Sample Data Setup (adjust based on actual simulation logic) ---

# Create a sample graph G (for illustration purposes)
G = nx.erdos_renyi_graph(30, 0.1, seed=42)  # Graph with 30 nodes and 10% probability of edge creation

# Add sample attributes for nodes (gender and scores)
for node in G.nodes():
    G.nodes[node]['gender'] = 'Male' if node % 2 == 0 else 'Female'  # Alternating male/female
    G.nodes[node]['score'] = np.random.randint(50, 100)  # Random score between 50 and 100
    G.nodes[node]['triggered_count'] = 0  # Initialize triggered count to 0
    G.nodes[node]['shared'] = False  # Shared info initially set to False

# Simulating more contagion steps (extend as needed)
contagion_steps = []
for step in range(10):  # Increase this value for more steps
    users_triggered = set([i for i in range(30)])  # Example logic for contagion
    contagion_steps.append(users_triggered)

# Assuming y_test and y_pred are predefined for classification evaluation
y_test = np.random.choice([0, 1], size=30)  # Random ground truth (0 or 1 for simplicity)
y_pred = np.random.choice([0, 1], size=30)  # Random predictions (0 or 1)

# --- Streamlit UI with interactive contagion step slider ---
st.title("Health Information Spread Simulation")

st.subheader("Model Evaluation")
accuracy = np.mean(y_test == y_pred)  # Simple accuracy calculation
st.write(f"Accuracy: {accuracy:.2%}")  # Display accuracy
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)  # Assuming you have the confusion matrix
fig, ax = plt.subplots(figsize=(6, 4))  # Reduced size for confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
st.pyplot(fig)

# Slider to pick contagion step
max_step = len(contagion_steps)
step = st.slider("Select contagion step", 1, max_step, max_step)

# Collect nodes that shared up to current step
shared_up_to_step = set()
for i in range(step):
    shared_up_to_step.update(contagion_steps[i])

# Prepare graph visualization
left_col, right_col = st.columns([2, 1])  # Wider left for graph

with left_col:
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjusted size for network diagram
    pos = nx.spring_layout(G, seed=42)

    male_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Male']
    female_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Female']

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)

    # Draw all nodes by gender with new colors
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

    # Sort first by triggered count (descending), then by score (descending)
    influencer_stats = []
    for node in G.nodes:
        influencer_stats.append({
            'user': node,
            'score': G.nodes[node]['score'],
            'triggered': G.nodes[node]['triggered_count'],
        })
    top_influencers = sorted(influencer_stats, key=lambda x: (x['triggered'], x['score']), reverse=True)[:5]

    # Display top influencers
    for rank, inf in enumerate(top_influencers, 1):
        st.markdown(f"- **Rank {rank}**: User {inf['user']} — Score: {inf['score']}, Triggered: {inf['triggered']}")

    # Gender-triggered stats
    male_triggered = sum(1 for n in shared_up_to_step if G.nodes[n]['gender'] == 'Male')
    female_triggered = sum(1 for n in shared_up_to_step if G.nodes[n]['gender'] == 'Female')

    st.markdown(f"- **Male Users Triggered**: {male_triggered} shares")
    st.markdown(f"- **Female Users Triggered**: {female_triggered} shares")

    st.markdown("---")
    st.markdown("🕹️ Use the slider to explore the contagion spread over time.")

# Optional Play Animation for Contagion Steps (Auto advance)
step_container = st.empty()  # Creates an empty placeholder
current_step = 1

# Play button functionality to auto advance steps
if st.button("Play Animation"):
    while current_step <= max_step:
        step_container.slider("Select contagion step", 1, max_step, current_step, key="slider")
        current_step += 1
        st.time.sleep(1)  # Sleep for 1 second between steps
        st.experimental_rerun()  # Re-run to update the step
