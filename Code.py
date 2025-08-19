import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# --- Dummy Data Setup ---

# Create example graph with gender and sharing info
G = nx.erdos_renyi_graph(30, 0.1, seed=42)

for node in G.nodes:
    # Random gender
    G.nodes[node]['gender'] = 'Male' if node % 2 == 0 else 'Female'
    # Random score
    G.nodes[node]['score'] = np.random.randint(0, 101)
    # Random triggered count
    G.nodes[node]['triggered_count'] = np.random.randint(0, 6)
    # Shared flag false init
    G.nodes[node]['shared'] = False

# Simulated contagion steps as list of sets of node ids sharing info at each step
contagion_steps = [
    {20, 18, 27},
    {0, 3, 10, 12, 25},
    {8, 9, 22, 23, 29},
    {15, 21},
    {4, 26, 28}
]

# Mark nodes that shared info up to last step as shared=True
shared_nodes = set()
for step_nodes in contagion_steps:
    shared_nodes.update(step_nodes)
for n in G.nodes:
    G.nodes[n]['shared'] = (n in shared_nodes)

# Dummy classification results
y_test = [0, 1, 0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1, 0, 1]
accuracy = accuracy_score(y_test, y_pred)

# --- Streamlit App ---

st.title("Health Information Spread Simulation")

st.subheader("Model Evaluation")
st.write(f"Accuracy: {accuracy:.2%}")

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax_cm)
st.pyplot(fig_cm)

# Slider to select contagion step
max_step = len(contagion_steps)
step = st.slider("Select contagion step", 1, max_step, max_step)

# Nodes shared up to the current step
shared_up_to_step = set()
for i in range(step):
    shared_up_to_step.update(contagion_steps[i])

# Layout with two columns: Left = graph, Right = leaderboard
left_col, right_col = st.columns([2, 1])

with left_col:
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    male_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Male']
    female_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Female']

    # Draw edges lightly
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)

    # Draw male/female nodes
    nx.draw_networkx_nodes(G, pos, nodelist=male_nodes, node_color='lightgreen', node_size=300, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=female_nodes, node_color='lightblue', node_size=300, ax=ax)

    # Highlight shared nodes up to current step with red outline
    nx.draw_networkx_nodes(
        G, pos, nodelist=list(shared_up_to_step),
        node_color='none', edgecolors='red', node_size=330, linewidths=2, ax=ax
    )

    # Node labels
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes}, font_size=8, ax=ax)

    ax.set_title(f"Network at Contagion Step {step} (Red outline = Shared)")
    ax.axis('off')
    st.pyplot(fig)

with right_col:
    st.markdown("### 🏆 Top Influencers")

    influencer_stats = []
    for node in G.nodes:
        influencer_stats.append({
            'user': node,
            'score': G.nodes[node]['score'],
            'triggered': G.nodes[node]['triggered_count'],
        })

    top_influencers = sorted(influencer_stats, key=lambda x: x['triggered'], reverse=True)[:5]

    for rank, inf in enumerate(top_influencers, 1):
        st.markdown(f"- **Rank {rank}**: User {inf['user']} — Score: {inf['score']}, Triggered: {inf['triggered']}")

    male_triggered = sum(1 for n in shared_up_to_step if G.nodes[n]['gender'] == 'Male')
    female_triggered = sum(1 for n in shared_up_to_step if G.nodes[n]['gender'] == 'Female')

    st.markdown(f"- **Male Users Triggered**: {male_triggered} shares")
    st.markdown(f"- **Female Users Triggered**: {female_triggered} shares")

    st.markdown("---")
    st.markdown("🕹️ Use the slider to explore the contagion spread over time.")

