import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# --- Setup ---

st.set_page_config(layout="wide")
st.title("Health Information Spread Simulation")

# --- Create Graph and Attributes ---

num_nodes = 30
G = nx.erdos_renyi_graph(num_nodes, 0.1, seed=42)

random.seed(42)
for node in G.nodes:
    G.nodes[node]['gender'] = random.choice(['Male', 'Female'])
    G.nodes[node]['score'] = random.randint(0, 100)
    G.nodes[node]['triggered_count'] = 0
    G.nodes[node]['shared'] = False

# --- Contagion Simulation ---

initial_gifted = random.sample(list(G.nodes), 3)
for node in initial_gifted:
    G.nodes[node]['shared'] = True

contagion_steps = [{node for node in initial_gifted}]
all_shared = set(initial_gifted)

prob_sharing_by_gender = {
    ('Male', 'Male'): 0.3,
    ('Male', 'Female'): 0.5,
    ('Female', 'Male'): 0.6,
    ('Female', 'Female'): 0.7
}

max_steps = 10
for _ in range(max_steps):
    new_shared = set()
    for user in contagion_steps[-1]:
        neighbors = list(G.neighbors(user))
        for neighbor in neighbors:
            if not G.nodes[neighbor]['shared']:
                gender_pair = (G.nodes[user]['gender'], G.nodes[neighbor]['gender'])
                prob = prob_sharing_by_gender.get(gender_pair, 0.4)
                if random.random() < prob:
                    G.nodes[neighbor]['shared'] = True
                    G.nodes[user]['triggered_count'] += 1
                    new_shared.add(neighbor)
                    all_shared.add(neighbor)
    if not new_shared:
        break
    contagion_steps.append(new_shared)

# --- Dummy Model Evaluation ---

accuracy = 0.82
y_test = [0, 1, 1, 0, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 0, 1, 1]

# --- Confusion Matrix ---

cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"])

# --- Streamlit UI ---

st.subheader("Model Evaluation")
col1, col2 = st.columns([1, 1])

with col1:
    st.write(f"Accuracy: {accuracy:.2%}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

with col2:
    fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
    cmd.plot(ax=ax_cm, cmap=plt.cm.Blues, colorbar=False)
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

# --- Contagion Animation Controls ---

max_step = len(contagion_steps)

play = st.checkbox("▶️ Play Animation")

if 'step' not in st.session_state:
    st.session_state.step = max_step

step_container = st.empty()

def plot_graph(step):
    shared_up_to_step = set()
    for i in range(step):
        shared_up_to_step.update(contagion_steps[i])

    # Two equal height columns for alignment
    left_col, right_col = st.columns([2, 1], gap="large")

    with left_col:
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)

        male_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Male']
        female_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Female']

        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=male_nodes, node_color='lightgreen', node_size=300, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=female_nodes, node_color='lightblue', node_size=300, ax=ax)

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
        top_influencers = sorted(influencer_stats, key=lambda x: x['triggered'], reverse=True)[:5]

        for rank, inf in enumerate(top_influencers, 1):
            st.markdown(f"- **Rank {rank}**: User {inf['user']} — Score: {inf['score']}, Triggered: {inf['triggered']}")

        male_triggered = sum(1 for n in shared_up_to_step if G.nodes[n]['gender'] == 'Male')
        female_triggered = sum(1 for n in shared_up_to_step if G.nodes[n]['gender'] == 'Female')

        st.markdown(f"- **Male Users Triggered**: {male_triggered} shares")
        st.markdown(f"- **Female Users Triggered**: {female_triggered} shares")

        st.markdown("---")
        st.markdown("🕹️ Use the slider or play animation to explore the contagion spread over time.")

if play:
    for current_step in range(1, max_step + 1):
        st.session_state.step = current_step
        step_container.slider("Select contagion step", 1, max_step, current_step, key="slider")
        plot_graph(current_step)
        time.sleep(1)
    st.session_state.step = max_step
else:
    step = st.slider("Select contagion step", 1, max_step, st.session_state.step, key="slider")
    st.session_state.step = step
    plot_graph(step)
