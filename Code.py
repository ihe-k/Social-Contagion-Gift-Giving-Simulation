import networkx as nx
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- Sample Data Setup (adjust based on actual simulation logic) ---
# Create a random graph with 30 nodes and 20% chance of edge creation
G = nx.erdos_renyi_graph(30, 0.2)  # Example graph with 30 nodes and 20% chance of edge creation

# Add node attributes for gender, score, triggered_count, and shared status
for node in G.nodes():
    # Assigning a gender based on even and odd nodes for simplicity
    G.nodes[node]['gender'] = 'Male' if node % 2 == 0 else 'Female'
    # Assigning a random score between 1 and 100 to simulate influencer power
    G.nodes[node]['score'] = np.random.randint(1, 100)
    # Triggered count is 0 initially
    G.nodes[node]['triggered_count'] = 0
    # Shared status is False initially
    G.nodes[node]['shared'] = False

# --- Streamlit Layout ---
st.title("Health Information Spread Simulation")

# Model evaluation using dummy values
st.subheader("Model Evaluation")

# --- Create Features for Model Evaluation ---
# Let's use 'score' as a feature to predict if a user shares info (triggered)
# We'll treat sharing info (triggered = 1) as a binary classification task
# Adding more features: score, gender (encoded), and degree centrality
X = []
y = []

for node in G.nodes():
    # Feature vector (score, gender, degree centrality)
    score = G.nodes[node]['score']
    gender = 1 if G.nodes[node]['gender'] == 'Male' else 0  # Male=1, Female=0
    degree_centrality = nx.degree_centrality(G).get(node, 0)
    
    # Simulate more realistic triggering based on probability (higher score, higher chance of triggering)
    triggered = 1 if np.random.rand() < (score / 100) else 0  # Probability based on score
    
    # Append features and target
    X.append([score, gender, degree_centrality])
    y.append(triggered)

# Convert to numpy arrays for model fitting
X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Scaling the features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# --- Model Evaluation ---
accuracy = (y_test == y_pred).mean()  # Dummy accuracy calculation for illustration
st.write(f"Accuracy: {accuracy:.2%}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots(figsize=(4, 3))  # Adjust size of confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

# --- Contagion Simulation ---
# Assuming you have steps of contagion in 'contagion_steps' as a list of sets of triggered users
# You can modify this part based on the actual contagion simulation data
contagion_steps = [
    {20, 18, 27},  # Example first contagion step
    {0, 3, 10, 12, 25},  # Second step
    {8, 9, 22, 23, 29},  # Third step
    {21, 15},  # Fourth step
    {4, 28},  # Fifth step
    {26},  # Sixth step
]

# --- Streamlit Layout: Left = Graph | Right = Leaderboard ---
left_col, right_col = st.columns([2, 1])  # Wider left for graph

# --- LEFT COLUMN: Contagion Step Slider ---
with left_col:
    # Slider to control contagion step
    max_step = len(contagion_steps)
    step = st.slider("Select contagion step", 1, max_step, max_step, key="step_slider")

    # Track which users are triggered up to the current step
    shared_up_to_step = set()
    for i in range(step):
        shared_up_to_step.update(contagion_steps[i])

    # --- Update Triggered Count for Each Contagion Step ---
    for user in shared_up_to_step:
        # Update triggered count and mark as shared
        if not G.nodes[user]['shared']:  # Only trigger once
            G.nodes[user]['triggered_count'] += 1
            G.nodes[user]['shared'] = True

    # --- Network Graph ---
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    male_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Male']
    female_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Female']

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)

    # Draw all nodes by gender
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

    # --- MOVE INFLUENCER TABLE BELOW NETWORK GRAPH ---
    st.markdown("### 🏆 Top Influencers")

    influencer_stats = []
    for node in G.nodes:
        influencer_stats.append({
            'user': node,
            'score': G.nodes[node]['score'],
            'triggered': G.nodes[node]['triggered_count'],
        })

    # Sort by triggered count (descending), then by score (descending)
    top_influencers = sorted(influencer_stats, key=lambda x: (x['triggered'], x['score']), reverse=True)[:5]

    for rank, inf in enumerate(top_influencers, 1):
        st.markdown(f"- **Rank {rank}**: User {inf['user']} — Score: {inf['score']}, Triggered: {inf['triggered']}")

    male_triggered = sum(1 for n in shared_up_to_step if G.nodes[n]['gender'] == 'Male')
    female_triggered = sum(1 for n in shared_up_to_step if G.nodes[n]['gender'] == 'Female')

    st.markdown(f"- **Male Users Triggered**: {male_triggered} shares")
    st.markdown(f"- **Female Users Triggered**: {female_triggered} shares")

