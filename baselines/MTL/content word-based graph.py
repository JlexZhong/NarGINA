import networkx as nx
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Subset
from tqdm import tqdm
import jieba

# Step 1: Data Preparation
def preprocess_text(text):
    """Clean and tokenize Chinese text using jieba to extract content words."""
    if not text:
        return []
    words = jieba.lcut(text)  # Use jieba to segment Chinese text
    words = [word for word in words if len(word) > 1]  # Filter out single-character words
    return words

def build_graph(essay):
    """Construct a graph from essay content words."""
    words = preprocess_text(essay)
    graph = nx.DiGraph()

    # Add nodes and edges based on adjacent words
    for i in range(len(words) - 1):
        graph.add_edge(words[i], words[i + 1])

    return graph

# Step 2: Extract Graph Features
def extract_graph_features(graph):
    """Extract features from the graph."""
    features = {}
    try:
        # Node Degree Features
        degrees = [deg for _, deg in graph.degree()]
        features['degree_max'] = max(degrees) if degrees else 0
        features['degree_mean'] = np.mean(degrees) if degrees else 0
        features['degree_median'] = np.median(degrees) if degrees else 0
        
        # PageRank Features
        pagerank = nx.pagerank(graph) if len(graph) > 0 else {}
        pagerank_values = list(pagerank.values())
        features['pagerank_max'] = max(pagerank_values) if pagerank_values else 0
        features['pagerank_mean'] = np.mean(pagerank_values) if pagerank_values else 0
        features['pagerank_median'] = np.median(pagerank_values) if pagerank_values else 0
    except Exception as e:
        print(f"Error processing graph features: {e}")
        features = {key: 0 for key in ['degree_max', 'degree_mean', 'degree_median', 'pagerank_max', 'pagerank_mean', 'pagerank_median']}

    return features

# Step 3: Load Data from JSON Files
def load_data(json_path, split_path):
    """Load dataset and split indices from JSON files."""
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    with open(split_path, 'r', encoding='utf-8') as file:
        split_indices = json.load(file)

    train_data = Subset(dataset=data, indices=split_indices['train'])
    validate_data = Subset(dataset=data, indices=split_indices['valid'])
    test_data = Subset(dataset=data, indices=split_indices['test'])

    return train_data, validate_data, test_data

# Step 4: Main Pipeline
def main():
    # Load data
    json_path = '/disk/NarGINA/ChildText/raw_graph/updated_total_data_sum.json'
    split_path = '/disk/NarGINA/ChildText/raw_graph/split_index.json'
    train_data, validate_data, test_data = load_data(json_path, split_path)

    # Extract features and labels from train data
    train_features = []
    train_scores = {"macro_score": [], "micro_score": [], "psych_score": [], "total_score": []}
    for entry in tqdm(train_data, desc="Processing Training Data"):
        essay = entry.get('essay_text', "")
        scores = [entry.get(score_key, 0) for score_key in train_scores.keys()]  # Get all score dimensions
        graph = build_graph(essay)
        features = extract_graph_features(graph)
        train_features.append(features)
        for i, score_key in enumerate(train_scores.keys()):
            train_scores[score_key].append(scores[i])

    # Convert features to DataFrame
    features_df = pd.DataFrame(train_features).fillna(0)

    # Prepare data for modeling
    X_train = StandardScaler().fit_transform(features_df)

    # Train Elastic Net model for each score dimension
    models = {}
    for score_key in train_scores.keys():
        y_train = train_scores[score_key]
        model = ElasticNetCV(cv=5, random_state=42)
        model.fit(X_train, y_train)
        models[score_key] = model

    # Extract features from test data
    test_features = []
    test_scores = {"macro_score": [], "micro_score": [], "psych_score": [], "total_score": []}
    for entry in tqdm(test_data, desc="Processing Test Data"):
        essay = entry.get('essay_text', "")
        scores = [entry.get(score_key, 0) for score_key in test_scores.keys()]  # Get all score dimensions
        graph = build_graph(essay)
        features = extract_graph_features(graph)
        test_features.append(features)
        for i, score_key in enumerate(test_scores.keys()):
            test_scores[score_key].append(scores[i])

    X_test = StandardScaler().fit_transform(pd.DataFrame(test_features).fillna(0))

    # Predict and evaluate on test data for each score dimension
    for score_key in test_scores.keys():
        y_test = test_scores[score_key]
        y_pred = models[score_key].predict(X_test)
        qwk = cohen_kappa_score(y_test, np.round(y_pred), weights='quadratic')
        print(f"Test QWK for {score_key}: {qwk:.3f}")

"""
Test QWK for macro_score: 0.494
Test QWK for micro_score: 0.605
Test QWK for psych_score: 0.439
Test QWK for total_score: 0.537
"""
"""SUM_score
Test QWK for macro_score: 0.507
Test QWK for micro_score: 0.506
Test QWK for psych_score: 0.565
Test QWK for total_score: 0.546"""
if __name__ == "__main__":
    main()
