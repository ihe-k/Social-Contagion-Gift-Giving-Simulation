import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np

# Dummy example data: replace with your actual data & logic
G = nx.erdos_renyi_graph(30, 0.1, seed=42)
for node in G.nodes:
    G.nodes[node]['gender'] = 'Male' if node % 2 == 0 else 'Female'
    G.nodes[node]['shared'] = np.random.choice([True, False], p=[0.3, 0.7])
    G.nodes[node]['score'] = np.random.randint(0, 100)
    G.nodes[node]['triggered_count'] = np.random.randint(1, 6)

contagion_steps = [{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10}, {11, 12}, {13}]

# Dummy classification results
y_test = [0, 1, 0, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1, 1]

accuracy = np.mean(np.array(y_test) == np.array(y_pred))

st.title("Health Information Spread Simulation")

# --- Model Evaluation and Confusion Matrix side by side ---
eval_col, cm_col = st.columns([2, 1])

with eval_col:
    st.subheader("Model Evaluation")
    st.write(f"Accuracy: {accuracy:.2%}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

with cm_col:
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

# Slider for contagion step
max_step = len(contagion_steps)
step = st.slider("Select contagion step", 1, max_step, max_step)

# Collect nodes shared up to selected step
shared_up_to_step = set()
for i in range(step):
    shared_up_to_step.update(contagion_steps[i])

# Layout columns: Left = Network, Right = Leaderboard
left_col, right_col = st.columns([2, 1])

with left_col:
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    male_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Male']
    female_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Female']

    # Draw edges lightly
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)

    # Updated node colors: male=#03396c, female=#6497b1
    nx.draw_networkx_nodes(G, pos, nodelist=male_nodes, node_color='#03396c', node_size=300, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=female_nodes, node_color='#6497b1', node_size=300, ax=ax)

    # Highlight nodes that shared info up to current step (red outline)
    nx.draw_networkx_nodes(
        G, pos, nodelist=list(shared_up_to_step),
        node_color='none', edgecolors='red', node_size=330, linewidths=2, ax=ax
    )

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

    # Sort by triggered desc, then score desc
    top_influencers = sorted(
        influencer_stats,
        key=lambda x: (x['triggered'], x['score']),
        reverse=True
    )[:5]

    for rank, inf in enumerate(top_influencers, 1):
        st.markdown(f"- **Rank {rank}**: User {inf['user']} — Score: {inf['score']}, Triggered: {inf['triggered']}")

    male_triggered = sum(1 for n in shared_up_to_step if G.nodes[n]['gender'] == 'Male')
    female_triggered = sum(1 for n in shared_up_to_step if G.nodes[n]['gender'] == 'Female')

    st.markdown(f"- **Male Users Triggered**: {male_triggered} shares")
    st.markdown(f"- **Female Users Triggered**: {female_triggered} shares")

    st.markdown("---")
    st.markdown("🕹️ Use the slider to explore the contagion spread over time.")
