# --- Streamlit Layout: Left = Graph | Right = Leaderboard ---
st.subheader("Contagion Spread Simulation")

# Divide into two columns
left_col, right_col = st.columns([2, 1])  # Wider left for graph

# --- LEFT COLUMN: Network Graph ---
with left_col:
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    male_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Male']
    female_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Female']
    shared_nodes = [n for n in G.nodes if G.nodes[n]['shared']]

    # Edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)

    # Male / Female Nodes
    nx.draw_networkx_nodes(G, pos, nodelist=male_nodes, node_color='lightgreen', node_size=300, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=female_nodes, node_color='lightblue', node_size=300, ax=ax)

    # Highlight shared nodes
    nx.draw_networkx_nodes(
        G, pos, nodelist=shared_nodes,
        node_color='none', edgecolors='red', node_size=330, linewidths=2, ax=ax
    )

    # Labels
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes}, font_size=8, ax=ax)

    ax.set_title("Final Network State (Red = Shared)")
    ax.axis('off')
    st.pyplot(fig)

# --- RIGHT COLUMN: Leaderboard & Stats ---
with right_col:
    st.markdown("### 🏆 Top Influencers")

    # Compute top influencers by triggered_count
    influencer_stats = []
    for node in G.nodes:
        influencer_stats.append({
            'user': node,
            'score': G.nodes[node]['score'],
            'triggered': G.nodes[node]['triggered_count'],
        })
    top_influencers = sorted(influencer_stats, key=lambda x: x['triggered'], reverse=True)[:5]

    for rank, inf in enumerate(top_influencers, 1):
        st.markdown(
            f"- **Rank {rank}**: User {inf['user']} — Score: {inf['score']}, Triggered: {inf['triggered']}"
        )

    # Gender-triggered stats
    male_triggered = sum(1 for n in G.nodes if G.nodes[n]['shared'] and G.nodes[n]['gender'] == 'Male')
    female_triggered = sum(1 for n in G.nodes if G.nodes[n]['shared'] and G.nodes[n]['gender'] == 'Female')

    st.markdown(f"- **Male Users Triggered**: {male_triggered} shares")
    st.markdown(f"- **Female Users Triggered**: {female_triggered} shares")

    st.markdown("---")
    st.markdown("🕹️ *Use slider or animation to explore contagion over time* (optional)")

    # Optional slider to explore contagion steps
    step = st.slider("Contagion Step", min_value=1, max_value=len(contagion_steps), value=len(contagion_steps))
    st.markdown(f"Showing step {step} of {len(contagion_steps)}")
