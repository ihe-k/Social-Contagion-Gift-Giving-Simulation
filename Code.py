import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

st.set_page_config(layout="wide")

# --- Setup graph and contagion simulation ---

# Create example graph (30 nodes, random edges)
G = nx.erdos_renyi_graph(30, 0.1, seed=42)

# Assign gender, score, and initialize sharing and triggered_count
for node in G.nodes():
    G.nodes[node]['gender'] = random.choice(['Male', 'Female'])
    G.nodes[node]['score'] = random.randint(10, 100)
    G.nodes[node]['triggered_count'] = 0
    G.nodes[node]['shared'] = False

# Simulated contagion steps as sets of nodes sharing info in that step
contagion_steps = [
    {18, 20, 27},
    {0, 3, 10, 12, 25},
    {8, 9, 22, 23, 29},
    {15, 21},
    {4, 28},
    {26}
]

# Mark nodes as shared and compute triggered counts
trigger_map = {}  # node -> set of triggered nodes

for node in contagion_steps[0]:
    G.nodes[node]['shared'] = True
    G.nodes[node]['triggered_count'] = 0
    trigger_map[node] = set()

for i in range(1, len(contagion_steps)):
    current_step_nodes = contagion_steps[i]
    prev_step_nodes = contagion_steps[i - 1]
    prev_list = list(prev_step_nodes)
    for j, node in enumerate(current_step_nodes):
        G.nodes[node]['shared'] = True
        triggered_by = prev_list[j % len(prev_list)]
        G.nodes[triggered_by]['triggered_count'] += 1
        trigger_map.setdefault(triggered_by, set()).add(node)

for node in G.nodes():
    if 'shared' not in G.nodes[node]:
        G.nodes[node]['shared'] = False

# --- Dummy model evaluation data for demonstration ---

# For simplicity, generate random true/pred labels for nodes shared or not
y_test = []
y_pred = []
for node in G.nodes():
    y_test.append(1 if G.nodes[node]['shared'] else 0)
    # Random predictions around actual with some noise
    y_pred.append(1 if random.random() < (0.7 if G.nodes[node]['shared'] else 0.3) else 0)

accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# --- Streamlit UI ---

st.title("Health Information Spread Simulation")

# Layout top: Model evaluation + confusion matrix side-by-side
eval_col, cm_col = st.columns([1, 1])

with eval_col:
    st.subheader("Model Evaluation")
    st.markdown(f"**Accuracy:** {accuracy:.2%}")
    st.text("Classification Report:")
    st.text(class_report)

with cm_col:
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

st.markdown("---")

# Contagion step slider and visualization
max_step = len(contagion_steps)
step = st.slider("Select contagion step", 1, max_step, max_step, key="contagion_step")

shared_up_to_step = set()
for i in range(step):
    shared_up_to_step.update(contagion_steps[i])

left_col, right_col = st.columns([1, 0.5])

with left_col:
    fig, ax = plt.subplots(figsize=(5, 10))
    pos = nx.spring_layout(G, seed=42)

    male_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Male']
    female_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Female']

    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)

    nx.draw_networkx_nodes(G, pos, nodelist=male_nodes, node_color='#03396c', node_size=300, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=female_nodes, node_color='#6497b1', node_size=300, ax=ax)

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
