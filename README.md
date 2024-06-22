# Deep Learning in Signal Processing


This is repository is part of a special course at DTU, where the the objective is to get acquainted with some state of the art deep learning methods within signal processing and time series analysis. The study activities included study and discussion of the below papaers on a bi-weekly basis as well as reimplementation of selected methods.

Curriculum:
* Universal Time-Series Representation Learning: A Survey [1]
* Probabilistic Machine Learning: An Introduction (Chapter 15 - Sequences) [2]
* Attention is all you need [3]
* Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks [4]
* Supervised Sequence Labelling with Recurrent
Neural Networks (Chapter 7 -Connectionist Temporal Classification) [5]
* Wav2vec: Unsupervised Pre-training for Speech Recognition [6]
* Wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations [7]

## Summary

### Reccurent Neural Networks:

TODO: illustration of architectures

* **Vanilla Recurrent Neural Network (RNN)**: They are designed to process sequences of data by maintaining a hidden state that is updated at each time step. This hidden state captures information from previous inputs in the sequence, allowing the network to model temporal dependencies. RNNs can struggle with long-term dependencies due to issues like vanishing and exploding gradients, making it hard for the network to learn from data that is far back in the sequence.
* **Gated Recurrent Unit (GRU)**: This method introduces gating mechanisms to control the flow of information, which helps mitigate the vanishing gradient problem. GRUs have two main gates: the reset gate and the update gate. The reset gate determines how much of the past information to forget, while the update gate controls how much of the new information to incorporate.

* **Long Short-Term Memory (LSTM)**: They are designed to remember information for long periods, addressing the vanishing gradient problem more effectively than standard RNNs. LSTMs have three main gates: the input gate, the forget gate, and the output gate. These gates regulate the flow of information into and out of the cell state, which serves as the network’s memory. LSTMs are particularly good at capturing long-term dependencies, making them suitable for tasks like language modeling and time-series prediction.

* **Backpropagation through time**: Training reccurent type of architectures involve unfolding the RNN in time and applying the chain rule for derivatives to compute the gradients for each time step.

TODO: math from Murphy + derivation

### Transformers

* **Applications**: Transformers are effective for machine translation tasks, where the goal is to translate text from one language to another. Unlike RNNs, transformers do not require sequential processing, allowing them to process entire sentences simultaneously. This parallelism makes transformers faster and more efficient for training on large datasets.
* **Attention**: The attention mechanism enables the model to focus on relevant parts of the input sequence when generating each output element. It computes a set of attention weights that determine the importance of each input element relative to the current output. In transformers, self-attention allows the model to consider the relationships between all elements in the input sequence simultaneously. 
* **Architecture** : Transformers have two parts: The encoder processes the input sequence and generates a set of context vectors. It consists of multiple layers, each containing a self-attention mechanism followed by a feedforward neural network.
The decoder generates the output sequence by attending to the context vectors from the encoder and using its own self-attention mechanisms. It also consists of multiple layers, each containing self-attention, encoder-decoder attention, and feedforward neural networks. 
TODO: illustration
* **Positional Encoding**: Transformers use positional encoding to retain information about the order of the input sequence, as the self-attention mechanism itself does not inherently capture positional information. In practice this can sinusoidal encoding, so for example for odd words one would use:
$$PE_{(pos,2i+1)} =cos\left( \frac{pos}{10000\frac{2i}{d}}\right)$$
where d is the dimention of the embeddings (eg. 512)

### Connectionist Temporal Classification (CTC)
* **Applications**: CTC is used to train RNNs for labeling unsegmented sequence data, making it ideal for tasks like speech-to-text, where the alignment between the input audio and the output text is unknown.
* **Gradient Calculation**: Training with CTC involves calculating the gradient of the CTC loss function with respect to the network parameters. This process uses dynamic programming techniques to efficiently compute the probabilities of all possible label sequences and their alignments with the input sequence.
* **Forward-Backward Algorithm**: This algorithm is used to compute the CTC loss. It iterates through the input sequence, computing probabilities forward in time to capture prefix probabilities and backward in time to capture suffix probabilities. These probabilities are then combined to calculate the overall loss.
* **Decoding**: To find the post probable path through the label probabilities we can use dynamic programming. A variant of the forward-backward algorithm calculates the probability of each possible label sequence. The final output is thus the most likely sequence of labels by considering all possible alignments and their corresponding probabilities.

### Wav2vec
* **Application** Wav2vec models are designed for speech recognition tasks, learning to extract meaningful features from raw audio waveforms. These features are then used to improve the performance of automatic speech recognition (ASR) systems. By pre-training on large amounts of unlabeled audio data, wav2vec models can learn rich representations that capture the nuances of speech, improving the accuracy of downstream ASR tasks.
* **Contrastive learning** Contrastive learning is a self-supervised learning approach where the model is trained to differentiate between similar and dissimilar pairs of data points. In the context of wav2vec, this involves learning to distinguish between different segments of audio. The model is trained by maximizing the similarity between representations of nearby audio segments while minimizing the similarity between representations of non-neighboring segments. This process encourages the model to capture meaningful patterns in the audio data.

TODO: math, losses

* **Product quantization**: This is a technique used to reduce the dimensionality of the feature space by partitioning it into smaller subspaces and quantizing each one separately. This helps in efficiently encoding and compressing the learned representations. By using product quantization, wav2vec models can manage large amounts of audio data more effectively, making the representations more compact and easier to work with for downstream tasks like speech recognition.

## Miniproject

### Dataset
I utilized a dataset of biosignals that included sleep stage annotations: [8]. This dataset provided detailed recordings of various physiological signals, such as EEG, EOG, and EMG, along with corresponding sleep stage labels TODO: Hz, stats about patients

### Objective
The goal was to train a deep learning model that could accurately predict the sequence of sleep stages from the biosignal data. To achieve this, I used the CTC loss. This model can be used to analyze and quantify the number of transitions between different sleep stages, providing insights into sleep quality and patterns.
For example, frequent transitions from deep sleep to lighter sleep stages could indicate sleep disturbances.

### Model training
TODO: preprocessing
TODO: architectures
TODO: diagnostic plot

### Evaluation
TODO: decode and compare





[1]: https://arxiv.org/abs/2401.03717
[2]: https://probml.github.io/pml-book/book1.html
[3]: https://arxiv.org/abs/1706.03762
[4]: https://www.cs.toronto.edu/~graves/icml_2006.pdf
[5]: https://www.cs.toronto.edu/~graves/preprint.pdf
[6]: https://arxiv.org/pdf/1904.05862
[7]: https://arxiv.org/abs/2006.11477
[8]: https://www.physionet.org/content/sleep-edfx/1.0.0/

