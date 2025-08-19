import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import feedparser

# ---------------------------
# -- Contagion Simulation Setup
# ---------------------------

# Create graph
G = nx.erdos_renyi_graph(30, 0.1, seed=42)

# Add node attributes: ideology, gender, triggered count
import random

ideologies = ["pro-health", "anti-health", "neutral"]
genders = ["Male", "Female"]

for node in G.nodes:
    G.nodes[node]['ideology'] = random.choice(ideologies)
    G.nodes[node]['gender'] = random.choice(genders)
    # Random triggered count (simulate contagion)
    G.nodes[node]['triggered_count'] = random.randint(0, 5)

# ---------------------------
# -- Static Contagion Graph Plot
# ---------------------------

st.title("Health Info Contagion Simulation")

pos = nx.spring_layout(G, seed=42)
node_colors = []

color_map = {
    "pro-health": "green",
    "anti-health": "red",
    "neutral": "gray"
}

for node in G.nodes:
    ideology = G.nodes[node]['ideology']
    triggered = G.nodes[node]['triggered_count']
    base_color = color_map.get(ideology, "black")
    # Darken color based on triggered_count (more triggered = darker)
    if triggered > 0:
        alpha = min(triggered / 5, 1)
        color = plt.cm.Reds(alpha) if ideology == "anti-health" else plt.cm.Greens(alpha)
        node_colors.append(color)
    else:
        node_colors.append(base_color)

fig, ax = plt.subplots(figsize=(8, 6))
nx.draw_networkx(G, pos=pos, node_color=node_colors, with_labels=True, ax=ax)
ax.set_title("Static Contagion Graph")
ax.axis('off')
st.pyplot(fig)

# ---------------------------
# -- Leaderboard of Triggered Counts
# ---------------------------

ideology_triggered = {"pro-health": 0, "anti-health": 0, "neutral": 0}
gender_triggered = {"Male": 0, "Female": 0}

for node in G.nodes:
    ideology = G.nodes[node]['ideology']
    gender = G.nodes[node]['gender']
    triggered_count = G.nodes[node]['triggered_count']

    # Defensive: handle empty or unexpected ideology/gender
    if ideology in ideology_triggered:
        ideology_triggered[ideology] += triggered_count
    if gender in gender_triggered:
        gender_triggered[gender] += triggered_count

st.write("### Leaderboard: Triggered Influence by Ideology")
for ideol, count in ideology_triggered.items():
    st.write(f"- **{ideol.capitalize()}** triggered: {count}")

st.write("### Leaderboard: Triggered Influence by Gender")
for gend, count in gender_triggered.items():
    st.write(f"- **{gend}** triggered: {count}")

# ---------------------------
# -- Podcast Health-Related Search from Multiple RSS Feeds
# ---------------------------

st.write("---")
st.header("Health-Related Podcast Episodes (from popular RSS feeds)")

health_keywords = [
    "health", "chronic", "disease", "wellness", "medicine",
    "doctor", "mental", "fitness", "nutrition", "covid", "symptom",
    "treatment", "therapy", "healthcare", "condition", "disorder",
    "diagnosis", "epidemic", "pandemic", "recovery", "exercise"
]

def is_health_related(text):
    text = text.lower()
    return any(keyword in text for keyword in health_keywords)

def get_health_related_podcasts(feeds, max_items=15):
    filtered_podcasts = []
    for feed_url in feeds:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:max_items]:
            content_text = (entry.get('title','') + " " + entry.get('description','')).lower()
            if is_health_related(content_text):
                filtered_podcasts.append({
                    "user": entry.get('author', 'podcaster'),
                    "content": entry.get('title','') + " - " + entry.get('description', ''),
                    "platform": "RSS",
                    "url": entry.get('link', '')
                })
    return filtered_podcasts

rss_feeds = [
    "https://rss.art19.com/the-daily",  # NPR Up First
    "https://feeds.megaphone.fm/ADV8924270618",  # Call Her Daddy
    "https://feeds.simplecast.com/5Z9BHkuQ",  # This Past Weekend w/ Theo Von
    "https://joeroganexp.joerogan.libsynpro.com/rss",  # Joe Rogan Experience (Unofficial)
    "https://feeds.npr.org/510289/podcast.xml",  # NPR Up First alternative
    "https://feeds.simplecast.com/tOjNXec5",  # Reply All
    "https://feeds.megaphone.fm/stuffyoushouldknow",  # Stuff You Should Know
]

with st.spinner("Fetching podcast episodes..."):
    podcast_items = get_health_related_podcasts(rss_feeds)

if podcast_items:
    for i, podcast in enumerate(podcast_items[:20]):  # limit to 20 for UI
        st.markdown(f"**{i+1}. {podcast['content']}**  ")
        st.markdown(f"[Listen here]({podcast['url']})")
        st.markdown("---")
else:
    st.write("No health-related podcast episodes found from the selected feeds.")

