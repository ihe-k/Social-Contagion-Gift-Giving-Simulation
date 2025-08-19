import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from textblob import TextBlob
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import numpy as np
import random
from sklearn.metrics import classification_report, accuracy_score

# --- Setup and Data Preparation ---

st.set_page_config(layout="wide")

st.title("Health Information Spread Simulation")

# Create a random graph (Erdos-Renyi)
num_nodes = 30
G = nx.erdos_renyi_graph(num_nodes, 0.1, seed=42)

# Add node attributes: gender, score, triggered_count, shared flag
random.seed(42)
for node in G.nodes:
    G.nodes[node]['gender'] = random.choice(['Male', 'Female'])
    G.nodes[node]['score'] = random.randint(0, 100)
    G.nodes[node]['triggered_count'] = 0
    G.nodes[node]['shared'] = False

# --- Contagion Simulation ---

# Select initial seeded (gifted) users
initial_gifted = random.sample(list(G.nodes), 3)
for node in initial_gifted:
    G.nodes[node]['shared'] = True

contagion_steps = [{node for node in initial_gifted}]
all_shared = set(initial_gifted)

# Probability matrix for sharing by gender pairs
prob_sharing_by_gender = {
    ('Male', 'Male'): 0.3,
    ('Male', 'Female'): 0.5,
    ('Female', 'Male'): 0.6,
    ('Female', 'Female'): 0.7
}

max_steps = 10
for step_i in range(max_steps):
    new_shared = set()
    for user in contagion_steps[-1]:
        neighbors = list(G.neighbors(user))
        for neighbor in neighbors:
            if not G.nodes[neighbor]['shared']:
                gender_pair = (G.nodes[user]['gender'], G.nodes[neighbor]['gender'])
                prob = prob_sharing_by_gender.get(gender_pair, 0.4)
                if random.random() < prob:
                    G.nodes[neighbor]['shared'] = True
                    G.nodes[user]['triggered_count'] += 1  # Increase count for triggerer
                    new_shared.add(neighbor)
                    all_shared.add(neighbor)
    if not new_shared:
        break
    contagion_steps.append(new_shared)

# --- Dummy Model Metrics for Display ---
# Replace these with your actual model evaluation outputs
accuracy = 0.82
y_test = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

# --- Streamlit UI ---

st.subheader("Model Evaluation")
st.write(f"Accuracy: {accuracy:.2%}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Contagion step slider
max_step = len(contagion_steps)
step = st.slider("Select contagion step", 1, max_step, max_step)

# Collect users who shared up to the selected step
shared_up_to_step = set()
for i in range(step):
    shared_up_to_step.update(contagion_steps[i])

# Layout: Graph left, leaderboard right
left_col, right_col = st.columns([2, 1])

with left_col:
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    male_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Male']
    female_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Female']

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)

    # Draw nodes by gender
    nx.draw_networkx_nodes(G, pos, nodelist=male_nodes, node_color='lightgreen', node_size=300, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=female_nodes, node_color='lightblue', node_size=300, ax=ax)

    # Highlight shared nodes with red outline
    nx.draw_networkx_nodes(
        G, pos, nodelist=list(shared_up_to_step),
        node_color='none', edgecolors='red', node_size=330, linewidths=2, ax=ax
    )

    # Draw labels
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

