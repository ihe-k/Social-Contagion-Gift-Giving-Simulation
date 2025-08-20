import matplotlib.lines as mlines

# -- after contagion simulation and before visualization

st.subheader("User Network Contagion Simulation with Homophily Patterns")

fig_net, ax_net = plt.subplots(figsize=(10, 8))

# Define colors for ideology
ideology_colors = {
    'pro-health': '#2ca02c',  # green
    'anti-health': '#d62728', # red
    'neutral': '#7f7f7f'      # gray
}

# Separate nodes by gender for different shapes
male_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Male']
female_nodes = [n for n in G.nodes if G.nodes[n]['gender'] == 'Female']

pos = nx.spring_layout(G, seed=42)

# Node sizes (scaled by triggered_count)
node_sizes_male = [300 + 100 * G.nodes[n]['triggered_count'] for n in male_nodes]
node_sizes_female = [300 + 100 * G.nodes[n]['triggered_count'] for n in female_nodes]

# Node colors by ideology
node_colors_male = [ideology_colors[G.nodes[n]['ideology']] for n in male_nodes]
node_colors_female = [ideology_colors[G.nodes[n]['ideology']] for n in female_nodes]

# Node border widths (betweenness centrality)
bc_values = np.array([betweenness_centrality[n] for n in G.nodes])
if bc_values.max() > 0:
    norm_bc = 1 + 5 * (bc_values - bc_values.min()) / (bc_values.max() - bc_values.min())
else:
    norm_bc = np.ones(len(G.nodes))

node_border_widths_male = [norm_bc[n] for n in male_nodes]
node_border_widths_female = [norm_bc[n] for n in female_nodes]

# Draw edges in two groups:
same_ideo_edges = [(u, v) for u, v in G.edges if G.nodes[u]['ideology'] == G.nodes[v]['ideology']]
diff_ideo_edges = [(u, v) for u, v in G.edges if G.nodes[u]['ideology'] != G.nodes[v]['ideology']]

# Draw same ideology edges thicker and darker
nx.draw_networkx_edges(G, pos, edgelist=same_ideo_edges, ax=ax_net,
                       width=2, edge_color='black', alpha=0.6)

# Draw different ideology edges lighter and thinner
nx.draw_networkx_edges(G, pos, edgelist=diff_ideo_edges, ax=ax_net,
                       width=0.7, edge_color='lightgray', alpha=0.4)

# Draw male nodes as squares
nx.draw_networkx_nodes(G, pos, nodelist=male_nodes, node_color=node_colors_male,
                       node_size=node_sizes_male, node_shape='s',
                       edgecolors='black', linewidths=node_border_widths_male, ax=ax_net)

# Draw female nodes as circles
nx.draw_networkx_nodes(G, pos, nodelist=female_nodes, node_color=node_colors_female,
                       node_size=node_sizes_female, node_shape='o',
                       edgecolors='black', linewidths=node_border_widths_female, ax=ax_net)

# Draw user numbers as labels
nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes}, font_size=8, ax=ax_net)

# Legend for ideology colors
pro_patch = mlines.Line2D([], [], color=ideology_colors['pro-health'], marker='o', linestyle='None', markersize=8, label='Pro-health')
anti_patch = mlines.Line2D([], [], color=ideology_colors['anti-health'], marker='o', linestyle='None', markersize=8, label='Anti-health')
neutral_patch = mlines.Line2D([], [], color=ideology_colors['neutral'], marker='o', linestyle='None', markersize=8, label='Neutral')

# Legend for gender shapes
male_patch = mlines.Line2D([], [], color='black', marker='s', linestyle='None', markersize=8, label='Male')
female_patch = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label='Female')

ax_net.legend(handles=[pro_patch, anti_patch, neutral_patch, male_patch, female_patch], loc='best')

st.pyplot(fig_net)
