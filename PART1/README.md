# PART1: Embedding Training and Visualization

This part of the project focuses on training word embeddings and visualizing them.

## Dependencies and Libraries Used

- numpy
- gensim
- scikit-learn
- matplotlib
- spacy
- tqdm
- pandas
- scipy

## Approach

### part1.py

1. **Data Loading**: Load English text data from `eng_news_2024_300K-sentences.txt`.
2. **Text Processing**: Clean and tokenize the text data using spaCy.
3. **Co-occurrence Matrix**: Generate co-occurrence matrices with different window sizes.
4. **Dimensionality Reduction**: Reduce the dimensions of the co-occurrence matrices using Truncated SVD.
5. **Evaluation**: Evaluate the embeddings using cosine similarity and plot the explained variance.

### compare_pretrained.py

1. **Load Models**: Load the trained embeddings and the pre-trained Word2Vec model.
2. **Compare Embeddings**: Compare similarities and nearest neighbors between the trained and pre-trained embeddings.
3. **Visualization**: Visualize the embeddings using t-SNE and save the visualization as `embedding_visualization.png`.