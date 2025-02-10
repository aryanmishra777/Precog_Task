# PART2: Cross-Lingual Embedding Alignment

This part of the project focuses on aligning Hindi embeddings to English embeddings and evaluating the alignment.

## Dependencies and Libraries Used

- numpy
- fasttext
- gensim
- scikit-learn
- matplotlib
- logging

## Approach

### part2.py

1. **Load Pre-Trained Embeddings**: Load English Word2Vec embeddings and Hindi FastText embeddings.
2. **Load Bilingual Dictionary**: Load a bilingual dictionary from en-hi.txt
3. **Extract Embeddings**: Extract embeddings for translation pairs from the bilingual dictionary.
4. **Align Hindi Embeddings**: Align Hindi embeddings to English embeddings using Procrustes analysis.
5. **Evaluate Cross-Lingual Alignment**:
    - **Word Similarity**: Calculate cosine similarity between English and Hindi word pairs.
    - **Nearest Neighbor Retrieval**: Find nearest Hindi words to given English words.
    - **Cross-Lingual Word Clustering**: Perform cross-lingual word clustering using t-SNE and visualize the results.