import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import numpy as np

# ---- Sample Data Setup (replace with your actual data) ----
# Create a sample graph G with node attributes
G = nx.karate_club_graph()
# Add gender and other attributes for demo
for n in G.nodes:
    G.nodes[n]['gender'] = 'Male' if n % 2 == 0 else 'Female'
    G.nodes[n]['shared'] = False
    G.nodes[n]['score'] = np.random.randint(0, 100)
    G.nodes[n]['triggered_count'] = np.random.randint(0, 10)

# Sample contagion_steps: list of sets of nodes that shared at each step
contagion_steps = [{0, 1}, {2, 3, 4}, {5, 6}, {7, 8}, {9}]

# Mark nodes as shared based on last step
for step_nodes in contagion_steps:
    for n in step_nodes:
        G.nodes[n]['shared'] = True

# Sample model evaluation data
y_test = np.random.randint(0, 2, size=50)
y_pred = np.random.randint(0, 2, size=50)
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# --- Streamlit App ---

st.title("Health Information Spread Simulation")

# Display model evaluation
st.subheader("Model Evaluation")
st.write(f"Accuracy: {accuracy:.2%}")
st.text("Classification Report:")
st.text(class_report)

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax_cm)
st.pyplot(fig_cm)

max_step = len(contagion_steps)

# Initialize session state for step and play if not exists
if 'step' not in st.session_state:
    st.session_state.step = max_step
if 'play' not in st.session_state:
    st.session_state.play = False

def plot_graph(step):
    # Collect all nodes shared up to selected step
    shared_up_to_step = set()
    for i in range(step):
        shared_up_to_step.update(contagion_steps[i])

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    male_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Male']
    female_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Female']

    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=male_nodes, node_color='lightgreen', node_size=300, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=female_nodes, node_color='lightblue', node_size=300, ax=ax)

    # Highlight shared nodes with red outline
    nx.draw_networkx_nodes(
        G, pos, nodelist=list(shared_up_to_step),
        node_color='none', edgecolors='red', node_size=330, linewidths=2, ax=ax
    )

    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes}, font_size=8, ax=ax)

    ax.set_title(f"Network at Contagion Step {step} (Red outline = Shared)")
    ax.axis('off')

    st.pyplot(fig)

left_col, right_col = st.columns([2, 1])

with left_col:
    # Container for step slider for animation
    step_container = st.empty()

    # Play/pause button
    play_button = st.button("▶ Play" if not st.session_state.play else "⏸ Pause")

    # Toggle play state
    if play_button:
        st.session_state.play = not st.session_state.play

    if st.session_state.play:
        # Animate contagion steps
        for current_step in range(st.session_state.step, max_step + 1):
            st.session_state.step = current_step
            # Use unique slider key to avoid duplication
            step_container.slider("Select contagion step", 1, max_step, current_step, key=f"slider_play_{current_step}")
            plot_graph(current_step)
            time.sleep(1)
            # Rerun to update UI
            st.experimental_rerun()
        st.session_state.play = False
    else:
        # Manual step selection slider
        step = step_container.slider("Select contagion step", 1, max_step, st.session_state.step, key="slider_manual")
        st.session_state.step = step
        plot_graph(step)

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

    # Compute gender-triggered shares up to current step
    shared_nodes = set()
    for i in range(st.session_state.step):
        shared_nodes.update(contagion_steps[i])

    male_triggered = sum(1 for n in shared_nodes if G.nodes[n]['gender'] == 'Male')
    female_triggered = sum(1 for n in shared_nodes if G.nodes[n]['gender'] == 'Female')

    st.markdown(f"- **Male Users Triggered**: {male_triggered} shares")
    st.markdown(f"- **Female Users Triggered**: {female_triggered} shares")

    st.markdown("---")
    st.markdown("🕹️ Use the slider or Play button to explore the contagion spread over time.")
