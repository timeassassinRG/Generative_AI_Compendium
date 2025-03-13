# Generative AI Language Modeling with Transformers

This document provides a comprehensive exploration of the fundamental and advanced concepts of Language Modeling with Transformers. We will delve into the theoretical underpinnings and practical implementations of embedding techniques, tokenization, attention mechanisms, positional encoding, and transformer-based architectures. In addition, we will examine how these building blocks fit into broader tasks such as text classification, machine translation, and pretraining bidirectional transformers (like BERT). Throughout this document, we will explore not only how to set up these concepts in PyTorch, but also the reasons why they work as they do, grounded in the theory of modern natural language processing (NLP).

---

## Table of Contents
1. [Introduction to Word Embeddings and Token Indices](#introduction-to-word-embeddings-and-token-indices)  
2. [Working with Datasets, Tokenizers, and Vocabularies](#working-with-datasets-tokenizers-and-vocabularies)  
3. [Text Classification Using DataLoader](#text-classification-using-dataloader)  
4. [Positional Encoding in Transformers](#positional-encoding-in-transformers)  
5. [Attention Mechanisms](#attention-mechanisms)  
    - 5.1 [Self-Attention](#self-attention)  
    - 5.2 [Scaled Dot-Product Attention](#scaled-dot-product-attention)  
    - 5.3 [Multi-Head Attention](#multi-head-attention)  
6. [Transformer Architecture for Language Modeling](#transformer-architecture-for-language-modeling)  
    - 6.1 [Encoder Layer Implementation in PyTorch](#encoder-layer-implementation-in-pytorch)  
    - 6.2 [Text Classification with Transformers](#text-classification-with-transformers)  
7. [Decoders and GPT-like Models for Language Translation](#decoders-and-gpt-like-models-for-language-translation)  
8. [Encoder Models with BERT](#encoder-models-with-bert)  
    - 8.1 [Masked Language Modeling (MLM)](#masked-language-modeling-mlm)  
    - 8.2 [Next Sentence Prediction (NSP)](#next-sentence-prediction-nsp)  
    - 8.3 [Data Preparation for BERT in PyTorch](#data-preparation-for-bert-in-pytorch)  
9. [Building a Translation Model from Scratch](#building-a-translation-model-from-scratch)  
10. [Transformer Architecture for Translation](#transformer-architecture-for-translation)  
    - 10.1 [PyTorch Implementation of Transformers for Translation](#pytorch-implementation-of-transformers-for-translation)  
11. [Conclusions](#conclusions)  

---

## Introduction to Word Embeddings and Token Indices

In natural language processing, **word embeddings** play a crucial role by converting words into dense vector representations. These embeddings capture semantic and syntactic relationships, allowing machine learning models to perform language-related tasks more effectively.

- **Word2Vec, GloVe, and FastText**: Earlier popular methods for generating embeddings, utilizing different training techniques and objective functions.
- **Contextual Embeddings (e.g., BERT, GPT)**: Newer models produce embeddings that depend on the context in which the word appears.

Within PyTorch, we often use the `nn.Embedding` module to generate embeddings for token indices. A token index is simply an integer mapping to a particular token (a word, subword, or character). The embedding layer then converts these indices into trainable dense vectors.

### Why Embeddings?
- Capture semantic similarity in a continuous space (e.g., “king” is closer to “queen” than to “apple”).
- Reduce the dimensionality of sparse one-hot encodings.
- Allow models to learn representations directly from data.

### Implementation in PyTorch
```python
import torch
import torch.nn as nn

# Suppose we have a vocabulary of size 5000 and we want embedding dimension of 300
embedding = nn.Embedding(num_embeddings=5000, embedding_dim=300)

# Example token indices
token_indices = torch.tensor([2, 45, 134, 523], dtype=torch.long)

# Get their embeddings
embedded_tokens = embedding(token_indices)  # Shape: (4, 300)
```

## Working with Datasets, Tokenizers, and Vocabularies

Before building NLP models, we must convert raw text data into a format suitable for processing by neural networks. The following steps typically apply:

1. **Tokenization**: Splitting text into subunits, such as words or subwords.  
2. **Vocabulary Building**: Determining a mapping of tokens to unique integer IDs.  
3. **Encoding**: Converting text sequences to lists of integers using the vocabulary.  
4. **Data Batching**: Grouping examples into batches for training.

### Tokenization Techniques
- **Word-level tokenization**: Splits on whitespace and punctuation. Prone to large vocabulary sizes.
- **Subword tokenization (Byte-Pair Encoding, WordPiece)**: More effective at handling rare words and morphological variants.
- **Character-level tokenization**: Useful for languages with ambiguous token boundaries or for specialized tasks like speech recognition or OCR.

### Vocabulary and EmbeddingBag
In PyTorch, the **EmbeddingBag** layer is optimized for text or sequence embeddings, especially when dealing with bags of embeddings rather than a single continuous sequence. It computes a mean or sum of all embeddings in a “bag,” which is useful for tasks like text classification where word order can be partially or entirely disregarded (depending on the approach).

```python
import torch
import torch.nn as nn

# Suppose we have multiple "bags" of token indices
# Each example in the batch might be of variable length
input_offsets = torch.tensor([0, 3, 6], dtype=torch.long)
input_indices = torch.tensor([10, 14, 15, 0, 3, 21, 9, 17, 19], dtype=torch.long)

# Embedding Bag layer
emb_bag = nn.EmbeddingBag(num_embeddings=100, embedding_dim=16, mode='mean')

output = emb_bag(input_indices, input_offsets)
```
This mechanism handles the variable-length sequences internally by summing or averaging embeddings in the same “bag.”

---

## Text Classification Using DataLoader

Once token indices are prepared, you can leverage PyTorch’s `DataLoader` to:
1. Shuffle data.  
2. Batch examples.  
3. Potentially pad sequences to ensure uniform batch dimensions.

### Process
1. **Create a Dataset** object that returns `(input_sequence, label)`.  
2. **Initialize a DataLoader** with the created dataset.  
3. **Iterate** over the DataLoader to fetch batches for training or inference.

```python
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = self.texts[idx].split()
        token_indices = [self.vocab[token] for token in tokens if token in self.vocab]
        label = self.labels[idx]
        return torch.tensor(token_indices, dtype=torch.long), label

# Suppose we have a pretrained vocabulary 'vocab_dict'
dataset = TextDataset(["Hello world", "PyTorch is great"], [0, 1], vocab_dict)
loader = DataLoader(dataset, batch_size=2, shuffle=True)
```

With this setup, you can pass the batches to a neural network for **text classification**.

---

## Positional Encoding in Transformers

While embeddings encode the meaning of tokens, they do not carry information about *where* those tokens appear in the sequence. **Positional encoding** bridges this gap by injecting a representation of the token’s position into the embedding vector. In the original “Attention is All You Need” transformer paper, the authors proposed using sinusoidal positional embeddings with frequencies scaled by powers of 10000.

### Rationale
Transformers use *attention* mechanisms that are *order-invariant* by design. We need to provide positional context so that the model can differentiate “dog bites man” from “man bites dog.”

### Sinusoidal Form

TODO

### PyTorch Implementation Example
```python
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
    
    def forward(self, x):
        # x is expected to have shape (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x
```
In practice, you’ll add this encoding to the embedded input tokens before feeding them to the attention mechanism.

---

## Attention Mechanisms

### Concept
An **attention mechanism** allows a model to focus on different parts of the input sequence when processing a specific token. Rather than passing through an entire sequence uniformly, attention weighs the importance of each token context to the current token under consideration.

### Key, Query, Value
- **Query (Q)**: Representation of the current token.  
- **Key (K)**: Representation of each token in the sequence (including the current one).  
- **Value (V)**: Another representation used to construct the final output.

#### Computation
1. Compute the dot product \( QK^T \).  
2. Apply a scaling factor.  
3. Use softmax to derive the attention weights.  
4. Multiply attention weights by \( V \).

---

### Self-Attention

In **self-attention**, the same sequence is used to compute Q, K, and V. This technique is key to transformer-based models. Each token “attends” to others in the sequence to form a contextual representation, capturing long-range dependencies more efficiently than recurrent networks.

A single self-attention operation can be expressed as:

\[
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

where \( d_k \) is the dimensionality of keys.

---

### Scaled Dot-Product Attention

The scaled dot-product attention is a refinement of the self-attention mechanism described above. It simply divides the dot product by \( \sqrt{d_k} \), stabilizing gradients and preventing extremely large values when the dimensionality \( d_k \) is large.

\[
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

---

### Multi-Head Attention

Instead of performing a single attention function, **multi-head attention** runs multiple parallel attention layers (heads). Each head transforms Q, K, V into different subspaces, allowing the model to capture different types of relationships.

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
\]

- **Heads**: \(\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)\)

Each head can focus on different parts of the sequence. These outputs are then concatenated and projected back to the original embedding dimension.

---

## Transformer Architecture for Language Modeling

The **transformer** is composed of a stack of **encoder** layers (for tasks like machine translation encoders or BERT) and/or **decoder** layers (for machine translation decoders or GPT-like generative models). Each encoder layer has:
1. Multi-head self-attention sublayer.  
2. Feed-forward sublayer.  
3. Residual connections and layer normalization after each sublayer.

---

### Encoder Layer Implementation in PyTorch

A simplified version of an encoder layer in PyTorch:

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src shape: (src_len, batch_size, d_model)
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src
```

In practice, multiple layers of these encoders are stacked to form the full transformer encoder.

---

### Text Classification with Transformers

You can adapt the encoder architecture to classification tasks by:

1. Feeding token embeddings (plus positional encodings) into the transformer encoder.  
2. Aggregating the final hidden states (e.g., taking the hidden state of a special classification token `[CLS]` or doing a mean-pool).  
3. Passing the aggregate representation through a linear classifier to predict the label.

**Key Steps**:
- **Create a text pipeline** to handle tokenization, indexing, and batching.  
- **Initialize the transformer-based model** with positional encoding and multiple encoder layers.  
- **Train the model** end-to-end to minimize classification loss (e.g., cross-entropy).

---

## Decoders and GPT-like Models for Language Translation

While **encoders** read and understand an input sequence, **decoders** generate an output sequence one token at a time. In tasks like machine translation, the encoder processes the source language, and the decoder autoregressively generates the target language.

For **GPT-like** language models, the model is typically just a stack of transformer decoder blocks. These blocks use:
1. Self-attention over the previously generated tokens (masked attention to avoid looking ahead).  
2. (Optionally) cross-attention to an encoder output, e.g., in standard seq2seq translation tasks.  
3. Feed-forward layers plus residual connections.

**GPT** (Generative Pretrained Transformer) is pre-trained on large corpora to predict the next token, thus enabling powerful zero-shot or few-shot capabilities for tasks like translation.

---

## Encoder Models with BERT

**BERT (Bidirectional Encoder Representations from Transformers)** is an encoder-only architecture. It is pre-trained on large text corpora with two main tasks:

1. **Masked Language Modeling (MLM)**: Randomly mask some tokens and train the model to predict them.  
2. **Next Sentence Prediction (NSP)**: Train the model to predict if one sentence follows another.

### Masked Language Modeling (MLM)
During MLM pretraining, a portion of tokens (e.g., 15%) are replaced with `[MASK]`. The model must then predict the original tokens, encouraging bidirectional context usage.

### Next Sentence Prediction (NSP)
BERT also learns inter-sentence relationships by predicting if two sentences are *consecutive* or *random*. This helps capture discourse-level understanding.

---

### Data Preparation for BERT in PyTorch

To prepare data for BERT-like pretraining:

1. **Tokenization**: BERT uses WordPiece tokenization.  
2. **Generate input IDs** and **attention masks** (where 1 indicates real tokens and 0 indicates padded tokens).  
3. **Create masked labels** for MLM. For tokens chosen to be masked, the label is the original token ID.  
4. **Prepare sentence pairs** for NSP (50% of the time the second sentence is the actual next sentence, 50% random).  
5. **Batch the data** for efficient training.

---

## Building a Translation Model from Scratch

### Step-by-Step
1. **Data Collection**: Parallel corpus of source-target languages (e.g., German-English).  
2. **Tokenization and Vocabulary**: Build or use pre-built subword tokenizers on both source and target.  
3. **Encoder**: Consumes the source sequence and produces contextual embeddings.  
4. **Decoder**: Autoregressively decodes the target sequence, attending to both previously generated tokens and the encoder output.  
5. **Loss Function**: Cross-entropy with teacher forcing for training.

When constructing from scratch, you need an encoder-decoder transformer architecture. After training, you can feed an unseen German sentence into the encoder and let the decoder generate the English translation one token at a time until it emits a special end-of-sequence token.

---

## Transformer Architecture for Translation

A full transformer for translation comprises:

- **Encoder**: Stacked self-attention layers with feed-forward sublayers.  
- **Decoder**: Stacked self-attention layers that attend to the encoder’s output (cross-attention). A mask ensures the model does not peek at future tokens in the target sequence during training.

### PyTorch Implementation of Transformers for Translation

PyTorch provides a `nn.Transformer` module that wraps much of this functionality. Example usage:

```python
import torch
import torch.nn as nn

transformer_model = nn.Transformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048
)

src = torch.rand((10, 32, 512))  # (source sequence length, batch size, d_model)
tgt = torch.rand((20, 32, 512))  # (target sequence length, batch size, d_model)
src_mask = None  # specify masks if needed
tgt_mask = nn.Transformer.generate_square_subsequent_mask(20)

out = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
# out shape: (target sequence length, batch size, d_model)
```
You can also implement custom modules if you want greater control, but PyTorch’s native implementation provides a solid starting point.

---

## Conclusions

In this module, we covered:

- Fundamental steps of preparing text data: tokenization, vocabulary building, and data loaders.  
- The crucial role of embeddings in representing tokens as dense vectors and how PyTorch’s `nn.Embedding` or `nn.EmbeddingBag` can be utilized for language modeling and classification tasks.  
- How positional encoding helps transformers encode sequence order.  
- The mechanism of self-attention and scaled dot-product attention, serving as the backbone of transformer models.  
- Multi-head attention for capturing various contextual relationships in parallel.  
- Transformer encoder layers for language modeling and text classification in PyTorch.  
- Transformer decoder architectures, including GPT-like models, for tasks such as language translation.  
- BERT’s encoder-only approach leveraging masked language modeling (MLM) and next sentence prediction (NSP) for pretraining.  
- Steps to construct translation models from scratch, leveraging both encoder-decoder architectures and PyTorch’s built-in `nn.Transformer` for machine translation tasks.

By understanding these concepts and their PyTorch implementations, you will be well-positioned to build sophisticated NLP models that leverage the power of transformers, whether for text classification, translation, or broader language modeling tasks.
```