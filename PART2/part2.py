import numpy as np
import fasttext
import fasttext.util
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Step 1: Load Pre-Trained Embeddings
logging.info("Loading English Word2Vec embeddings...")
try:
    english_model = KeyedVectors.load_word2vec_format("../GoogleNews-vectors-negative300.bin", binary=True)
    logging.info("English Word2Vec embeddings loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load English Word2Vec embeddings: {e}")
    exit()

logging.info("Loading Hindi FastText embeddings...")
try:
    hindi_model = fasttext.load_model("../wiki.hi/wiki.hi.bin")  # Using FastText directly
    logging.info("Hindi FastText embeddings loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Hindi FastText embeddings: {e}")
    exit()

# Step 2: Load Bilingual Dictionary
logging.info("Loading bilingual dictionary...")

try:
    with open("../en-hi.txt", "r", encoding="utf-8") as f:
        lines = (line.strip().split() for line in f)
        hindi_vocab = set(hindi_model.get_words())  # Store Hindi words in a set for fast lookup
        
        # Dictionary comprehension for efficiency
        bilingual_dict = {
            en_word: hi_word
            for en_word, hi_word in lines
            if len(en_word) > 0 and len(hi_word) > 0 and en_word in english_model and hi_word in hindi_vocab
        }

    logging.info(f"Bilingual dictionary loaded with {len(bilingual_dict)} valid translation pairs.")
except Exception as e:
    logging.error(f"Failed to load bilingual dictionary: {e}")
    exit()


# Step 3: Extract Embeddings for Translation Pairs
logging.info("Extracting embeddings for translation pairs...")
try:
    english_embeddings = np.array([english_model[word] for word in bilingual_dict.keys() if word in english_model])
    hindi_embeddings = np.array([hindi_model.get_word_vector(word) for word in bilingual_dict.values()])
    logging.info(f"Extracted {len(english_embeddings)} English and {len(hindi_embeddings)} Hindi embeddings.")
except Exception as e:
    logging.error(f"Failed to extract embeddings: {e}")
    exit()

# Step 4: Align Hindi Embeddings Using Procrustes Analysis
logging.info("Aligning Hindi embeddings using Procrustes analysis...")
try:
    W, _ = orthogonal_procrustes(hindi_embeddings, english_embeddings)
    aligned_hindi_embeddings = hindi_embeddings @ W  # Using @ for matrix multiplication
    logging.info("Hindi embeddings aligned successfully.")
except Exception as e:
    logging.error(f"Failed to align Hindi embeddings: {e}")
    exit()

# Step 5: Evaluate Cross-Lingual Alignment
# (A) Word Similarity (Cosine Similarity)
def word_similarity(word1, word2, model1, model2, aligned_embeddings):
    # Handle Gensim KeyedVectors
    if isinstance(model1, KeyedVectors):
        words1 = set(model1.index_to_key)  # Gensim vocab list
    else:
        words1 = set(model1.get_words())  # FastText vocab

    if isinstance(model2, KeyedVectors):
        words2 = set(model2.index_to_key)
    else:
        words2 = set(model2.get_words())

    if word1 not in words1 or word2 not in words2:
        logging.warning(f"One of the words '{word1}' or '{word2}' is not in the vocabulary.")
        return None

    vec1 = model1[word1] if isinstance(model1, KeyedVectors) else model1.get_word_vector(word1)
    
    if word2 not in bilingual_dict.values():
        logging.warning(f"The word '{word2}' is not in the bilingual dictionary.")
        return None
    
    vec2 = aligned_embeddings[list(bilingual_dict.values()).index(word2)]
    
    return cosine_similarity([vec1], [vec2])[0][0]

# Example
logging.info("Calculating word similarity between 'dog' and 'कुत्ता'...")
similarity = word_similarity("dog", "कुत्ता", english_model, hindi_model, aligned_hindi_embeddings)
if similarity is not None:
    logging.info(f"Cosine Similarity: {similarity}")

# (B) Nearest Neighbor Retrieval
def nearest_neighbor(word, model, aligned_embeddings, bilingual_dict, top_k=5):
    if word not in model:
        logging.warning(f"Word '{word}' not found in the model.")
        return []
    vec = model[word]
    similarities = cosine_similarity([vec], aligned_embeddings)[0]
    nearest_indices = np.argsort(similarities)[-top_k:][::-1]
    return [list(bilingual_dict.values())[i] for i in nearest_indices if i < len(bilingual_dict.values())]

# Example
logging.info("Finding nearest Hindi words to 'dog'...")
nearest_hindi_words = nearest_neighbor("dog", english_model, aligned_hindi_embeddings, bilingual_dict)
logging.info(f"Nearest Hindi words to 'dog': {nearest_hindi_words}")

# (C) Cross-Lingual Word Clustering
logging.info("Performing cross-lingual word clustering...")
try:
    if english_embeddings.shape[0] == 0 or aligned_hindi_embeddings.shape[0] == 0:
        logging.warning("Skipping clustering: One or both embedding sets are empty.")
    else:
        combined_embeddings = np.vstack((english_embeddings, aligned_hindi_embeddings))

        # Use MiniBatchTSNE for faster processing on large data
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500, method='barnes_hut')
        reduced_embeddings = tsne.fit_transform(combined_embeddings)

        # Split the embeddings
        eng_emb, hi_emb = reduced_embeddings[: len(english_embeddings)], reduced_embeddings[len(english_embeddings):]

        # Plot with better visualization
        plt.figure(figsize=(10, 10))
        plt.scatter(eng_emb[:, 0], eng_emb[:, 1], c="r", label="English", alpha=0.7, edgecolors="k")
        plt.scatter(hi_emb[:, 0], hi_emb[:, 1], c="b", label="Hindi", alpha=0.7, edgecolors="k")
        
        plt.title("Cross-Lingual Word Clustering with t-SNE", fontsize=14)
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.show()

        logging.info("Cross-lingual word clustering completed and plot displayed.")
except Exception as e:
    logging.error(f"Failed to perform cross-lingual word clustering: {e}")