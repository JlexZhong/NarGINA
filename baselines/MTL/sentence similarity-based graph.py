import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import skew, kurtosis, hmean, gmean, iqr
import json
import thulac  # Using THULAC for Chinese POS tagging
from tqdm import tqdm  # Importing tqdm for progress bar

# Initialize THULAC
thu = thulac.thulac()

# Data Preprocessing for Chinese
def preprocess_text(text):
    # Tokenize using THULAC (word segmentation only)
    tokens = [word for word, _ in thu.cut(text)]
    # Do not remove stopwords for now as specified
    tokens = [word for word in tokens if word.strip()]
    return tokens

# Feature Extraction - Adjusted for Chinese
def extract_syntactic_features(text):
    sentences = [sentence for sentence in text.replace("。", ".").split(".") if sentence.strip()]
    tokens = preprocess_text(text)
    pos_tags = thu.cut(text)
    pos_counts = {}
    for word, tag in pos_tags:
        pos_counts[tag] = pos_counts.get(tag, 0) + 1

    features = {
        'num_sentences': len(sentences),
        'num_tokens': len(tokens),
        'num_unique_tokens': len(set(tokens)),
        'avg_sentence_length': len(tokens) / len(sentences) if len(sentences) > 0 else 0,
        'unique_pos_tags': len(pos_counts),
        'adjective_num': pos_counts.get('a', 0),
        'predeterminer': pos_counts.get('p', 0),
        'coordinating_conjunction': pos_counts.get('c', 0),
        'superlative_adjective': pos_counts.get('d', 0),
        'nouns': pos_counts.get('n', 0),
        'comparative_adj': pos_counts.get('m', 0),
        'verbs': pos_counts.get('v', 0),
        'length_of_essay': len(text)
    }
    return features

# Semantic Feature Extraction - Adjusted for Chinese
# Ensures the same feature count as described in the paper
def get_similarity_vector(text, prompt):
    sentences = [sentence for sentence in text.replace("。", ".").split(".") if sentence.strip()]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([prompt] + sentences)
    prompt_vector = tfidf_matrix[0]
    similarity_scores = []
    for i in range(1, len(sentences) + 1):
        similarity = (prompt_vector * tfidf_matrix[i].T).toarray()[0][0]
        similarity_scores.append(similarity)
    return similarity_scores

# Statistical Feature Extraction - Algorithm 3
# Ensures the same feature count as described in the paper
def construct_statistical_features(similarity_vector):
    try:
        mode_similarity = max(set(similarity_vector), key=lambda x: list(similarity_vector).count(x))
    except ValueError:
        mode_similarity = 0

    features = {
        'mean_similarity': np.mean(similarity_vector),
        'std_similarity': np.std(similarity_vector),
        'variance_similarity': np.var(similarity_vector),
        'skew_similarity': skew(similarity_vector),
        'kurtosis_similarity': kurtosis(similarity_vector),
        'hmean_similarity': hmean(similarity_vector) if all(x > 0 for x in similarity_vector) else 0,
        'gmean_similarity': gmean(similarity_vector) if all(x > 0 for x in similarity_vector) else 0,
        'median_similarity': np.median(similarity_vector),
        'mode_similarity': mode_similarity,
        'iqr_similarity': iqr(similarity_vector)
    }
    return features

# Full Feature Extraction Pipeline
def extract_features(text, prompt):
    syntactic_features = extract_syntactic_features(text)
    similarity_vector = get_similarity_vector(text, prompt)
    statistical_features = construct_statistical_features(similarity_vector)
    features = {**syntactic_features, **statistical_features}
    return features

# Dataset Loading
with open('/disk/NarGINA/ChildText/raw_graph/updated_total_data_sum.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
with open('/disk/NarGINA/ChildText/raw_graph/split_index.json', 'r', encoding='utf-8') as file:
    split_indices = json.load(file)

train_data = [data[i] for i in split_indices['train']]
validate_data = [data[i] for i in split_indices['valid']]
test_data = [data[i] for i in split_indices['test']]

# Define all target dimensions
targets = ["macro_score", "micro_score", "psych_score", "total_score"]

# Train and evaluate a model for each target dimension
models = {}
scores = {}

for target in targets:
    print(f"Training model for target: {target}")

    # Extract Features from Train Data with Progress Bar
    train_features = []
    train_scores = []
    for entry in tqdm(train_data, desc=f"Processing Training Data for {target}"):
        essay = entry['essay_text']
        prompt = entry.get('prompt', '')  # Ensure prompt exists or use default
        score = entry[target]
        features = extract_features(essay, prompt)
        train_features.append(features)
        train_scores.append(score)

    features_df = pd.DataFrame(train_features)

    # Check for NaN values in training data
    if features_df.isnull().values.any():
        print(f"NaN values found in training data for {target}:")
        print(features_df[features_df.isnull().any(axis=1)])

    # Model Training
    X_train = features_df.fillna(0)
    y_train = train_scores

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    models[target] = model

    # Predictions and Evaluation on Validation Set
    test_features = []
    test_scores = []
    for entry in tqdm(test_data, desc=f"Processing Validation Data for {target}"):
        essay = entry['essay_text']
        prompt = entry.get('prompt', '')
        score = entry[target]
        features = extract_features(essay, prompt)
        test_features.append(features)
        test_scores.append(score)

    X_test = pd.DataFrame(test_features)
    if X_test.isnull().values.any():
        print(f"NaN values found in validation data for {target}:")
        print(X_test[X_test.isnull().any(axis=1)])

    X_test = X_test.fillna(0)
    y_test = test_scores

    # Predictions and Kappa Score
    y_pred = model.predict(X_test)
    kappa_score = cohen_kappa_score(y_test, np.round(y_pred), weights='quadratic')
    scores[target] = kappa_score
    print(f"Quadratic Weighted Kappa Score for {target}: {kappa_score}")

# Print all scores
print("\nFinal QWK Scores for all targets:")
for target, score in scores.items():
    print(f"{target}: {score}")
