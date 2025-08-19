import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import praw

# --- Reddit Scraping using PRAW ---
def scrape_reddit(query="health"):
    # Reddit API credentials (replace with your own)
    reddit = praw.Reddit(client_id="YOUR_CLIENT_ID", 
                         client_secret="YOUR_CLIENT_SECRET", 
                         user_agent="YOUR_USER_AGENT")
    
    # Search for posts related to health in the health subreddit
    subreddit = reddit.subreddit("health")
    posts = subreddit.search(query, limit=5)  # Limiting to top 5 posts
    
    health_posts = []
    for post in posts:
        health_posts.append({
            'title': post.title,
            'url': post.url,
            'score': post.score,
            'comments': post.num_comments
        })
    
    return health_posts

# --- Model Evaluation Section (Dummy data for now) ---
y_test = [1, 0, 1, 1, 0]  # Actual values (dummy for illustration)
y_pred = [1, 0, 1, 0, 1]  # Predicted values (dummy for illustration)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# --- Streamlit Layout: Left = Graph | Right = Leaderboard ---
st.title("Health Information Spread Simulation")

st.subheader("Model Evaluation")
st.write(f"Classification Report:\n{classification_rep}")

# Display Confusion Matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(6, 4))
# Using matplotlib for the confusion matrix visualization
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(['0', '1']))
plt.xticks(tick_marks, ['0', '1'])
plt.yticks(tick_marks, ['0', '1'])

thresh = cm.max() / 2.
for i in range(len(cm)):
    for j in range(len(cm[i])):
        plt.text(j, i, format(cm[i][j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i][j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
st.pyplot(fig)

# --- Scrape Reddit for Health Posts ---
st.subheader("Health Information from Reddit")
posts = scrape_reddit(query="health")  # Search query for health-related posts

if posts:
    st.write("Found the following health-related posts:")
    for post in posts:
        st.markdown(f"**{post['title']}**: [Link]({post['url']}) | Score: {post['score']} | Comments: {post['comments']}")
else:
    st.write("No posts found. Try changing the query or check the subreddit.")

# --- Contagion Simulation ---
# Example Network (You can modify this part based on your real data)
G = nx.erdos_renyi_graph(30, 0.1)  # A random graph with 30 nodes and a 10% chance of edge creation
for node in G.nodes:
    G.nodes[node]['score'] = np.random.randint(50, 100)  # Random scores for nodes
    G.nodes[node]['triggered_count'] = np.random.randint(0, 5)  # Random trigger count for nodes

# Contagion Steps (Dummy example)
contagion_steps = [{0, 1}, {2, 3, 4}, {5, 6, 7, 8}]

# Slider for Contagion Step
max_step = len(contagion_steps)
step = st.slider("Select contagion step", 1, max_step, max_step)

# Collect nodes that shared up to current step
shared_up_to_step = set()
for i in range(step):
    shared_up_to_step.update(contagion_steps[i])

# Network Graph Visualization
left_col, right_col = st.columns([2, 1])  # Wider left for graph

with left_col:
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    male_nodes = [n for n in G.nodes if G.nodes[n]['score'] % 2 == 0]  # Random condition for males
    female_nodes = [n for n in G.nodes if G.nodes[n]['score'] % 2 != 0]  # Random condition for females

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)

    # Draw all nodes by gender (male: blue, female: pink)
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

# --- Leaderboard Panel: Influencers ---
with right_col:
    st.markdown("### 🏆 Top Influencers")

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

    # Gender-triggered stats
    male_triggered = sum(1 for n in shared_up_to_step if G.nodes[n]['score'] % 2 == 0)
    female_triggered = sum(1 for n in shared_up_to_step if G.nodes[n]['score'] % 2 != 0)

    st.markdown(f"- **Male Users Triggered**: {male_triggered} shares")
    st.markdown(f"- **Female Users Triggered**: {female_triggered} shares")

    st.markdown("---")
    st.markdown("🕹️ *Use the slider to explore the contagion spread over time*")

