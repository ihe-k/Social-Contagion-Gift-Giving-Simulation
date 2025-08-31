import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import Normalize
from matplotlib import cm

# Create a random graph (example for 50 nodes)
G = nx.erdos_renyi_graph(50, 0.05)

# Assign random attributes to nodes
for node in G.nodes:
    G.nodes[node]['sentiment'] = np.random.choice(['pro-health', 'anti-health', 'neutral'])
    G.nodes[node]['trust'] = np.random.choice([0, 1])  # Trust in clinician (0 = low, 1 = high)
    G.nodes[node]['health_condition'] = np.random.choice(['chronic', 'healthy'])
    G.nodes[node]['ideology'] = np.random.choice(['pro-health', 'anti-health', 'neutral'])

# Visualization Function
def visualize_network():
    # Create a 1x2 subplot layout
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # First Plot: Misinformation Exposure (Sentiment-based color)
    ax1 = axes[0]
    pos = nx.spring_layout(G, seed=42)  # Fix layout for reproducibility
    node_color_1 = [1 if G.nodes[node]['sentiment'] == 'pro-health' 
                    else 0.5 if G.nodes[node]['sentiment'] == 'neutral' 
                    else 0 for node in G.nodes]
    nx.draw(G, pos, ax=ax1, with_labels=True, node_size=700, node_color=node_color_1, font_size=10, 
            edge_color='gray', alpha=0.6, width=0.7)
    
    # Add a horizontal color bar for Misinformation Exposure (based on sentiment)
    norm1 = Normalize(vmin=0, vmax=1)
    sm1 = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm1)
    sm1.set_array([])  # No data is actually assigned here, it's just for colorbar mapping
    cbar1 = fig.colorbar(sm1, ax=ax1, orientation='horizontal', fraction=0.02, pad=0.1)
    cbar1.set_label('Misinformation Exposure', fontsize=12)
    
    # Second Plot: Trust in Clinician (Trust-based color)
    ax2 = axes[1]
    node_color_2 = [G.nodes[node]['trust'] for node in G.nodes]  # 0 or 1 for trust in clinician
    nx.draw(G, pos, ax=ax2, with_labels=True, node_size=700, node_color=node_color_2, font_size=10, 
            edge_color='gray', alpha=0.6, width=0.7)
    
    # Add a horizontal color bar for Trust in Clinician
    norm2 = Normalize(vmin=0, vmax=1)
    sm2 = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm2)
    sm2.set_array([])  # No data is actually assigned here, it's just for colorbar mapping
    cbar2 = fig.colorbar(sm2, ax=ax2, orientation='horizontal', fraction=0.02, pad=0.1)
    cbar2.set_label('Trust in Clinician', fontsize=12)
    
    # Remove legends from within the charts
    ax1.get_legend().set_visible(False) if ax1.get_legend() else None
    ax2.get_legend().set_visible(False) if ax2.get_legend() else None
    
    plt.tight_layout()
    plt.show()

# --- Display Network ---
visualize_network()
