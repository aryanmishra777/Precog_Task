import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import logging
from pathlib import Path
import time
from typing import List, Tuple, Dict, Optional
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PretrainedModel:
    """Loads and handles comparisons with a pre-trained Word2Vec model."""
    
    def __init__(self, model_path: str = "GoogleNews-vectors-negative300.bin"):
        """
        Initialize the pretrained model.
        Args:
            model_path: Path to the Word2Vec binary file
        """
        logger.info(f"Loading pre-trained model from {model_path}...")
        try:
            self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
            logger.info("Pre-trained model loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Model file not found at {model_path}")
            logger.info("Please download the Google News vectors from:")
            logger.info("https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing")
            raise

    def get_similarity(self, word1: str, word2: str) -> Optional[float]:
        """Computes cosine similarity between two words using pretrained embeddings."""
        try:
            if word1 in self.model and word2 in self.model:
                return float(self.model.similarity(word1, word2))
            logger.warning(f"Words '{word1}' or '{word2}' not found in pretrained model")
            return None
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return None

    def get_nearest_neighbors(self, word: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Finds nearest words to a given word in pretrained embeddings."""
        try:
            if word in self.model:
                return self.model.most_similar(word, topn=top_n)
            logger.warning(f"Word '{word}' not found in pretrained model")
            return []
        except Exception as e:
            logger.error(f"Error finding nearest neighbors: {str(e)}")
            return []

class TrainedEmbeddings:
    """Loads and handles comparisons with trained co-occurrence + SVD embeddings."""
    
    def __init__(self, embeddings_path: str, vocab_path: str):
        """
        Initialize the trained embeddings model.
        Args:
            embeddings_path: Path to the numpy embeddings file
            vocab_path: Path to the vocabulary CSV file
        """
        logger.info(f"Loading trained embeddings from {embeddings_path}...")
        try:
            self.embeddings = np.load(embeddings_path)
            self.vocab = self._load_vocab(vocab_path)
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            logger.info("Trained embeddings loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading trained embeddings: {str(e)}")
            raise

    def _load_vocab(self, vocab_path: str) -> Dict[str, int]:
        """Loads vocabulary from CSV file and returns a dictionary."""
        vocab_dict = {}
        try:
            with open(vocab_path, "r", encoding="utf-8") as file:
                next(file)  # Skip header
                for line_num, line in enumerate(file, 2):
                    try:
                        word, index = line.strip().split(",")
                        vocab_dict[word] = int(index)
                    except ValueError:
                        logger.warning(f"Skipping malformed line {line_num}: {line.strip()}")
        except Exception as e:
            logger.error(f"Error loading vocabulary: {str(e)}")
            raise
        return vocab_dict

    def get_similarity(self, word1: str, word2: str) -> Optional[float]:
        """Computes cosine similarity between two words using trained embeddings."""
        try:
            if word1 in self.vocab and word2 in self.vocab:
                idx1, idx2 = self.vocab[word1], self.vocab[word2]
                return float(cosine_similarity([self.embeddings[idx1]], [self.embeddings[idx2]])[0][0])
            logger.warning(f"Words '{word1}' or '{word2}' not found in trained embeddings")
            return None
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return None

    def get_nearest_neighbors(self, word: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Finds nearest words to a given word in trained embeddings."""
        try:
            if word not in self.vocab:
                logger.warning(f"Word '{word}' not found in trained embeddings")
                return []
            
            word_vector = self.embeddings[self.vocab[word]]
            similarities = cosine_similarity([word_vector], self.embeddings)[0]
            
            # Get indices of top similar words (excluding the word itself)
            nearest_indices = np.argsort(similarities)[::-1][1:top_n+1]
            
            return [(self.reverse_vocab[idx], float(similarities[idx])) 
                   for idx in nearest_indices]
        except Exception as e:
            logger.error(f"Error finding nearest neighbors: {str(e)}")
            return []

def compare_embeddings(trained_model: TrainedEmbeddings, 
                      pretrained_model: PretrainedModel, 
                      test_words: List[str]) -> None:
    """Compares similarities and nearest neighbors between both models."""
    logger.info("\nWord Similarity Comparison:")
    word_pairs = [("king", "queen"), ("man", "woman"), ("apple", "orange")]
    
    for word1, word2 in word_pairs:
        sim_trained = trained_model.get_similarity(word1, word2)
        sim_pretrained = pretrained_model.get_similarity(word1, word2)
        trained_str = f"{sim_trained:.4f}" if sim_trained is not None else "N/A"
        pretrained_str = f"{sim_pretrained:.4f}" if sim_pretrained is not None else "N/A"
        logger.info(f"{word1} - {word2}: Trained = {trained_str}, Pretrained = {pretrained_str}")

    logger.info("\nNearest Neighbors Comparison:")
    for word in test_words:
        logger.info(f"\nWord: {word}")
        
        logger.info("  Trained Embeddings:")
        for neighbor, score in trained_model.get_nearest_neighbors(word):
            logger.info(f"    - {neighbor}: {score:.4f}")

        logger.info("  Pretrained Embeddings:")
        for neighbor, score in pretrained_model.get_nearest_neighbors(word):
            logger.info(f"    - {neighbor}: {score:.4f}")

def visualize_embeddings(trained_model: TrainedEmbeddings, 
                        pretrained_model: PretrainedModel, 
                        words: List[str]) -> None:
    """Visualizes word embeddings from both models using t-SNE."""
    logger.info("\nVisualizing Embeddings with t-SNE...")
    
    # Filter words present in both models
    common_words = [w for w in words 
                   if w in trained_model.vocab and w in pretrained_model.model]
    
    if not common_words:
        logger.warning("No common words found between models")
        return
    
    try:
        # Set matplotlib backend to 'Agg' to avoid GUI issues
        plt.switch_backend('Agg')
        
        trained_vectors = np.array([trained_model.embeddings[trained_model.vocab[w]] 
                                  for w in common_words])
        pretrained_vectors = np.array([pretrained_model.model[w] 
                                     for w in common_words])
        
        # Adjust perplexity based on number of samples
        n_samples = len(common_words)
        perplexity = min(30, n_samples - 1)  # perplexity must be less than n_samples
        
        tsne = TSNE(n_components=2, 
                    random_state=42, 
                    perplexity=perplexity,
                    n_iter=1000)
        
        plt.figure(figsize=(12, 6))
        
        # Plot trained embeddings
        plt.subplot(1, 2, 1)
        trained_tsne = tsne.fit_transform(trained_vectors)
        plt.scatter(trained_tsne[:, 0], trained_tsne[:, 1], color='blue')
        for i, word in enumerate(common_words):
            plt.annotate(word, (trained_tsne[i, 0], trained_tsne[i, 1]))
        plt.title("Trained Word Embeddings")
        
        # Plot pretrained embeddings
        plt.subplot(1, 2, 2)
        pretrained_tsne = tsne.fit_transform(pretrained_vectors)
        plt.scatter(pretrained_tsne[:, 0], pretrained_tsne[:, 1], color='red')
        for i, word in enumerate(common_words):
            plt.annotate(word, (pretrained_tsne[i, 0], pretrained_tsne[i, 1]))
        plt.title("Pretrained Word Embeddings (Word2Vec)")
        
        plt.tight_layout()
        plt.savefig('embedding_visualization.png')
        plt.close()
        logger.info("Visualization saved as 'embedding_visualization.png'")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")

def main():
    # Paths configuration
    trained_embeddings_path = "./output/embeddings_100.npy"
    trained_vocab_path = "./output/co_occurrence_w5.vocab.csv"
    pretrained_model_path = "../GoogleNews-vectors-negative300.bin"
    
    # Test words
    test_words = [
    "king", "queen", "man", "woman", "prince", "princess",
    "apple", "orange", "banana", "grape", "fruit", "vegetable",
    "car", "bus", "train", "truck", "vehicle", "motorcycle",
    "dog", "cat", "bird", "fish", "animal", "pet",
    "house", "building", "home", "apartment", "office", "store"]
    
    try:
        # Load models
        trained_model = TrainedEmbeddings(trained_embeddings_path, trained_vocab_path)
        pretrained_model = PretrainedModel(pretrained_model_path)
        
        # Compare embeddings
        compare_embeddings(trained_model, pretrained_model, test_words)
        
        # Visualize embeddings
        visualize_embeddings(trained_model, pretrained_model, test_words)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    start_time = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
    logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")