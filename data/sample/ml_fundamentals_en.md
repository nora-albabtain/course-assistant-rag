# Machine Learning Fundamentals — Course Notes

## What is Machine Learning?
Machine Learning (ML) is a branch of Artificial Intelligence in which a computer learns patterns from data instead of following hand-written rules. A model is trained on examples and then used to make predictions on new, unseen data.

## Types of Machine Learning
- **Supervised learning:** the training data has labels (the correct answers). Examples: predicting house prices (regression) or classifying emails as spam or not spam (classification).
- **Unsupervised learning:** the data has no labels, and the model finds structure on its own. Example: grouping customers with similar behavior (clustering).
- **Reinforcement learning:** an agent learns by trial and error, receiving rewards or penalties for its actions. Example: training an agent to play a game.

## Training, Validation and Test Sets
Data is usually split into three parts. The training set is used to fit the model. The validation set is used to tune hyperparameters and choose between models. The test set is used only once at the end to estimate how the model will perform in the real world.

## Overfitting and Underfitting
**Overfitting** happens when a model memorizes the training data, including its noise, so it performs very well on training data but poorly on new data. Signs: very high training accuracy with much lower validation accuracy.
Ways to reduce overfitting: collect more data, simplify the model, use regularization (L1/L2), apply dropout in neural networks, and use early stopping.

**Underfitting** happens when the model is too simple to capture the pattern, so it performs poorly on both training and validation data. Fix it by using a more powerful model or better features.

## Evaluation Metrics
- **Accuracy:** the share of correct predictions. Misleading when classes are imbalanced.
- **Precision:** of the items predicted positive, how many were truly positive.
- **Recall:** of the truly positive items, how many the model found.
- **F1-score:** the harmonic mean of precision and recall.

## Neural Networks
A neural network is made of layers of connected neurons. Each neuron computes a weighted sum of its inputs and applies an activation function such as ReLU or sigmoid. Training uses backpropagation and gradient descent to adjust the weights so the loss decreases. Popular frameworks are PyTorch and TensorFlow.

## Retrieval-Augmented Generation (RAG)
RAG combines a Large Language Model with a search step. Documents are split into chunks, converted into embeddings, and stored in a vector database such as FAISS, Chroma, Pinecone or Milvus. When a user asks a question, the most similar chunks are retrieved and given to the LLM as context, which reduces hallucinations and lets the model answer from private or up-to-date material.
