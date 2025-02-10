import pandas as pd
import numpy as np
import time
from collections import Counter, defaultdict
from scipy.sparse import coo_matrix, save_npz, lil_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import spacy
import re
from tqdm import tqdm
from typing import List, Dict, Tuple, Set
import logging
from pathlib import Path
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoOccurrenceMatrix:
    def __init__(self, window_size: int, min_count: int = 5, batch_size: int = 10000):
        self.window_size = window_size
        self.min_count = min_count
        self.batch_size = batch_size
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        
    def build_vocabulary(self, tokenized_sentences: List[List[str]]) -> int:
        """Build vocabulary efficiently using batched processing"""
        logger.info("Building vocabulary...")
        word_counts = Counter()
        
        # Count words in batches
        for i in range(0, len(tokenized_sentences), self.batch_size):
            batch = tokenized_sentences[i:min(i + self.batch_size, len(tokenized_sentences))]
            for sentence in batch:
                word_counts.update(sentence)
        
        # Filter and create vocabulary
        valid_words = {word for word, count in word_counts.items() 
                      if count >= self.min_count}
        
        self.vocab = {word: idx for idx, word in enumerate(sorted(valid_words))}
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        
        logger.info(f"Vocabulary size: {len(self.vocab)}")
        return len(self.vocab)

    def construct_matrix(self, tokenized_sentences: List[List[str]]) -> Tuple[coo_matrix, Dict[str, int]]:
        vocab_size = self.build_vocabulary(tokenized_sentences)
        
        # Initialize sparse matrix
        co_matrix = lil_matrix((vocab_size, vocab_size), dtype=np.float32)
        
        # Process sentences in batches
        for i in tqdm(range(0, len(tokenized_sentences), self.batch_size), desc="Processing batches"):
            batch = tokenized_sentences[i:min(i + self.batch_size, len(tokenized_sentences))]
            self._process_batch(batch, co_matrix)
            
        return co_matrix.tocoo(), self.vocab

    def _process_batch(self, batch: List[List[str]], co_matrix: lil_matrix) -> None:
        """Process a batch of sentences to update co-occurrence matrix"""
        for sentence in batch:
            valid_tokens = [token for token in sentence if token in self.vocab]
            
            for i, token in enumerate(valid_tokens):
                token_idx = self.vocab[token]
                window_start = max(0, i - self.window_size)
                window_end = min(len(valid_tokens), i + self.window_size + 1)
                
                for j in range(window_start, window_end):
                    if i != j:
                        context_token = valid_tokens[j]
                        context_idx = self.vocab[context_token]
                        co_matrix[token_idx, context_idx] += 1

    def save_matrix(self, matrix: coo_matrix, file_path: str) -> None:
        logger.info(f"Saving matrix to {file_path}")
        save_npz(file_path, matrix)
        
        vocab_df = pd.DataFrame(list(self.vocab.items()), columns=['word', 'index'])
        vocab_path = Path(file_path).with_suffix('.vocab.csv')
        vocab_df.to_csv(vocab_path, index=False)
        logger.info("Matrix and vocabulary saved successfully")

class TextProcessor:
    def __init__(self, batch_size: int = 1000):
        logger.info("Loading spaCy model...")
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
        self.nlp.max_length = 2000000
        self.batch_size = batch_size

    def load_dataset(self, file_path: str, max_lines: int = None) -> List[str]:
        logger.info(f"Loading dataset from {file_path}")
        sentences = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(tqdm(file, desc="Loading sentences")):
                if max_lines and i >= max_lines:
                    break
                    
                clean_line = re.sub(r'[\t\n\r]', ' ', line)
                clean_line = re.sub(r'\s+', ' ', clean_line).strip()
                
                if clean_line:
                    sentences.append(clean_line)
        
        logger.info(f"Loaded {len(sentences)} sentences")
        return sentences

    def clean_and_tokenize(self, sentences: List[str]) -> List[List[str]]:
        logger.info("Cleaning and tokenizing sentences...")
        tokenized_sentences = []
        
        # Process in batches
        for i in tqdm(range(0, len(sentences), self.batch_size), desc="Processing batches"):
            batch = sentences[i:min(i + self.batch_size, len(sentences))]
            
            # Disable unnecessary pipeline components
            docs = self.nlp.pipe(batch, disable=['ner', 'parser'])
            
            for doc in docs:
                tokens = [token.text.lower() for token in doc 
                         if not token.is_stop 
                         and not token.is_punct 
                         and not token.is_space
                         and not token.like_num
                         and len(token.text.strip()) > 1]
                
                if tokens:
                    tokenized_sentences.append(tokens)
            
            # Clear memory periodically
            if i % (self.batch_size * 10) == 0:
                gc.collect()
        
        return tokenized_sentences

class EmbeddingEvaluator:
    def __init__(self, matrix: coo_matrix, vocab: Dict[str, int]):
        self.matrix = matrix
        self.vocab = vocab
        self.reverse_vocab = {idx: word for word, idx in vocab.items()}

    def reduce_dimensions(self, d: int) -> np.ndarray:
        logger.info(f"Reducing dimensions to {d}")
        svd = TruncatedSVD(n_components=d, random_state=42)
        reduced_matrix = svd.fit_transform(self.matrix)
    
        # Save embeddings
        np.save(f"output/embeddings_{d}.npy", reduced_matrix)
        logger.info(f"Saved reduced embeddings to output/embeddings_{d}.npy")

        return reduced_matrix

    def plot_variance_explained(self, max_d: int = 300) -> None:
        n_components = min(max_d, self.matrix.shape[1])
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd.fit(self.matrix)
        
        explained_variance = np.cumsum(svd.explained_variance_ratio_)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(explained_variance) + 1), explained_variance)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Explained Variance vs. Number of Components')
        plt.grid(True)
        plt.savefig(f'variance_explained_{n_components}.png')
        plt.close()

    def evaluate_cosine_similarity(self, word_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str, float]]:
        logger.info("Evaluating cosine similarity...")
        results = []
        
        for word1, word2 in word_pairs:
            if word1 in self.vocab and word2 in self.vocab:
                idx1, idx2 = self.vocab[word1], self.vocab[word2]
                vec1 = self.matrix.getrow(idx1).toarray()
                vec2 = self.matrix.getrow(idx2).toarray()
                similarity = cosine_similarity(vec1, vec2)[0][0]
                results.append((word1, word2, similarity))
                logger.info(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
            else:
                logger.warning(f"Words '{word1}' or '{word2}' not in vocabulary")
        
        return results

def main():
    # Configuration
    config = {
        'file_path': 'eng_news_2024_300K-sentences.txt',
        'max_lines': 300000,
        'batch_size': 1000,
        'window_sizes': [2, 5, 10],
        'dimensions': [50, 100, 300],
        'word_pairs': [('king', 'queen'), ('man', 'woman'), ('apple', 'orange')],
        'min_count': 5
    }
    
    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Initialize processors
    text_processor = TextProcessor(batch_size=config['batch_size'])
    
    try:
        # Load and process data
        sentences = text_processor.load_dataset(config['file_path'], config['max_lines'])
        tokenized_sentences = text_processor.clean_and_tokenize(sentences)
        
        # Clear memory
        del sentences
        gc.collect()
        
        # Process different window sizes
        for window_size in config['window_sizes']:
            logger.info(f"\nProcessing window size {window_size}")
            
            try:
                co_matrix = CoOccurrenceMatrix(
                    window_size=window_size,
                    min_count=config['min_count'],
                    batch_size=config['batch_size']
                )
                
                matrix, vocab = co_matrix.construct_matrix(tokenized_sentences)
                co_matrix.save_matrix(matrix, output_dir / f'co_occurrence_w{window_size}.npz')
                
                evaluator = EmbeddingEvaluator(matrix, vocab)
                evaluator.plot_variance_explained()
                
                for d in config['dimensions']:
                    logger.info(f"\nAnalyzing {d} dimensions...")
                    reduced_matrix = evaluator.reduce_dimensions(d)
                    evaluator.evaluate_cosine_similarity(config['word_pairs'])
                    
                    # Clear memory
                    del reduced_matrix
                    gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing window size {window_size}: {str(e)}")
                continue
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")