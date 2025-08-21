import random
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

# --- Parameters ---
NUM_USERS = 300
EDGE_PROB = 0.05  # Adjust network density as you like

# --- Step 1: Create Graph ---
G = nx.erdos_renyi_graph(NUM_USERS, EDGE_PROB, seed=42)

# Assign node attributes
for node in G.nodes:
    G.nodes[node]['gender'] = random.choice(['Male', 'Female'])
    G.nodes[node]['has_chronic_disease'] = random.choice([True, False])
    G.nodes[node]['ideology'] = random.choice(['pro-health', 'anti-health', 'neutral'])
    G.nodes[node]['sentiment'] = G.nodes[node]['ideology']  # For simplicity, sentiment = ideology

# --- Calculate betweenness centrality and sentiment trends ---
betweenness_centrality = nx.betweenness_centrality(G)

def calc_sentiment_trends():
    trends = []
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        if neighbors:
            pro_health_count = sum(1 for n in neighbors if G.nodes[n]['sentiment'] == 'pro-health')
            trends.append(pro_health_count / len(neighbors))
        else:
            trends.append(0)
    return trends

sentiment_trends = calc_sentiment_trends()

# --- Extract Features and Labels ---
user_features = []
user_labels = []
for node in G.nodes:
    u = G.nodes[node]
    features = [
        1 if u['gender'] == 'Female' else 0,
        1 if u['has_chronic_disease'] else 0,
        sentiment_trends[node],
        betweenness_centrality[node]
    ]
    user_features.append(features)
    user_labels.append(u['ideology'])

X = np.array(user_features)
y = np.array(user_labels)

# Remove duplicates if any (optional)
df = pd.DataFrame(X, columns=['is_female', 'has_chronic_disease', 'sentiment_trend', 'betweenness_centrality'])
df['label'] = y
df = df.drop_duplicates()
X = df.drop('label', axis=1).values
y = df['label'].values

# --- Encode labels ---
le = LabelEncoder()
y_enc = le.fit_transform(y)

# --- Proper Train/Test Split with Stratification ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# --- Model Setup ---
model = XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    use_label_encoder=False,
    max_depth=3,
    n_estimators=100,
    random_state=42
)

# --- Cross-validation on Training Set Only ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

# --- Train on Full Training Set ---
model.fit(X_train, y_train)

# --- Evaluate on Test Set ---
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# --- Streamlit UI ---
st.title("Health Ideology Classification")

st.subheader("Model Evaluation (XGBoost)")
st.write(f"Test Accuracy: {test_accuracy:.2%}")
st.write(f"Cross-validated Accuracy (train set): {cv_mean:.2%} ± {cv_std:.2%}")

st.dataframe(report_df.style.format("{:.2f}"))

st.write("### Class distribution in dataset:")
st.json({k: int(v) for k, v in zip(le.classes_, np.bincount(y_enc))})

st.write("### Class distribution in training set:")
st.json({k: int(v) for k, v in zip(le.classes_, np.bincount(y_train))})

st.write("### Class distribution in test set:")
st.json({k: int(v) for k, v in zip(le.classes_, np.bincount(y_test))})
