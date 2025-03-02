# Word2Vec Implementation

## Overview

This project implements a basic **Word2Vec** model using the **Skip-Gram** approach with **softmax function** and **cross-entropy loss**. The model trains word embeddings by maximizing the probability of context words given a center word. It employs **stochastic gradient descent (SGD)** for optimization.

## Dataset

I have taken a custom text dataset from my own source. The dataset was preprocessed as follows:

- Tokenization of text.
- Creation of a vocabulary and word-to-index mapping.
- Generation of training pairs using a fixed context window.

## Model Architecture

Each word in the vocabulary has two representations:

- **Center word representation (Uc)**
- **Context word representation (Vt)**

### Probability Computation

Given a center word `wc`, the probability of a context word `wt` is computed as:

$$
 P(w_t | w_c) = \frac{\exp(U_c^T V_t)}{\sum_{w \in V} \exp(U_c^T V_w)}
$$

### Loss Function

The model minimizes the **negative log-likelihood loss**:

$$
 L = - \sum_{(w_c, w_t) \in D} \log P(w_t | w_c)
$$

where `D` is the dataset containing center-context word pairs.

## Training Process

The model was trained with various hyperparameters to analyze its performance:

- **Embedding dimensions:** `10`, `50`, `100`
- **Learning rates:** `0.01`, `0.1`, `1.0`
- **Epochs:** Multiple runs were conducted to track loss reduction.

### Gradient Descent Implementation

- **Stochastic Gradient Descent (SGD)** was used for parameter updates.
- An efficient **softmax function** was implemented.
- The gradients were computed for both **Uc** and **Vt** to update embeddings iteratively.

## Results & Findings

1. **Loss Reduction Over Epochs**
   - The training loss decreased over epochs, confirming effective learning.
   - Higher embedding dimensions led to better representations but required more epochs.
2. **Word Similarity Analysis**
   - Words with semantic similarities had higher cosine similarity scores.
   - The model successfully captured word relationships based on context.
3. **Effect of Learning Rate**
   - A **high learning rate (1.0)** caused instability.
   - A **moderate learning rate (0.1)** achieved better convergence.
4. **Impact of Embedding Size**
   - **Lower dimensions (10)** led to poor word representations.
   - **Higher dimensions (100)** resulted in more meaningful embeddings.

## Limitations & Improvements

1. **Computational Inefficiency of Softmax**
   - The full softmax function is computationally expensive for large vocabularies.
   - **Improvement:** Use **Negative Sampling** or **Hierarchical Softmax** to speed up training.
2. **Handling Rare Words**
   - The model struggles with rare words due to limited training examples.
   - **Improvement:** Implement **subsampling** to balance word frequency.
3. **Training on a Larger Dataset**
   - The embeddings improve significantly when trained on larger corpora.
   - **Improvement:** Use a **pretrained word embedding model** or expand the dataset.

## References

- Mikolov et al. (2013a): **Efficient Estimation of Word Representations in Vector Space**
- Mikolov et al. (2013b): **Distributed Representations of Words and Phrases and their Compositionality**

## Running the Model

To train and evaluate the model:

```bash
python train_word2vec.py --embedding_size 50 --learning_rate 0.1 --epochs 10
```

This implementation provides a basic framework for learning word embeddings, with potential enhancements for efficiency and performance in real-world applications.

