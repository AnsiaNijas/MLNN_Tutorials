# MLNN_Tutorial
# Sentiment Analysis with GloVe Embeddings and BiLSTM

This tutorial demonstrates how to perform sentiment analysis on the IMDB dataset using pretrained GloVe embeddings and a Bidirectional LSTM model. The following instructions guide you through running the code and understanding the workflow.

## Prerequisites

### Dependencies

Ensure the following libraries are installed:

- `numpy`
- `tensorflow`
- `tensorflow_datasets`
- `matplotlib`

You can install them using:

```bash
pip install numpy tensorflow tensorflow-datasets matplotlib
```

### Dataset

- The IMDB dataset will be downloaded automatically via TensorFlow Datasets.

### GloVe Embeddings

- GloVe embeddings will be downloaded automatically if not present in the project directory.

---

## Steps in the Tutorial

### 1. Import Libraries

The notebook imports essential libraries, including TensorFlow, TensorFlow Datasets, and Matplotlib, to handle data processing, model training, and evaluation.

### 2. Set Random Seeds

Random seeds are set for `numpy`, `random`, and TensorFlow to ensure reproducibility of results.

### 3. Load the IMDB Dataset

- The IMDB dataset is loaded and split into training and testing sets using TensorFlow Datasets.
- Basic metadata and sample reviews are displayed to understand the dataset.

### 4. Preprocess the Data

- Extract sentences and labels for training and testing.
- Tokenize sentences into sequences of integers.
- Pad sequences to ensure uniform input length.

### 5. Load GloVe Embeddings

- Check if GloVe embeddings exist locally.
- Download and extract GloVe embeddings if needed.
- Load the embeddings into a dictionary mapping words to vectors.

### 6. Create the Embedding Matrix

- Build an embedding matrix that maps the vocabulary in the dataset to the GloVe vectors.
- Use this matrix in the model's embedding layer.

### 7. Build the BiLSTM Model

- The model consists of:
  - An embedding layer using GloVe embeddings.
  - A Bidirectional LSTM layer with 64 units.
  - Dense and Dropout layers for feature extraction and regularization.
  - An output layer with sigmoid activation for binary classification.

### 8. Compile the Model

- Loss function: `binary_crossentropy`
- Optimizer: `adam`
- Metric: `accuracy`

### 9. Add Callbacks

- **EarlyStopping**: Stops training if validation loss does not improve for 3 consecutive epochs.
- **ReduceLROnPlateau**: Reduces learning rate when validation loss stagnates.

### 10. Train the Model

- The model is trained with the training data, using 20% for validation.
- Training stops early based on the callbacks.

### 11. Evaluate the Model

- The model is evaluated on the test data, and accuracy is printed.

### 12. Save the Model

- The trained model is saved as `sentiment_analysis_glove_bilstm.h5` for future use.

### 13. Analyze Performance

- Plot training and validation accuracy and loss over epochs.
- Alt-text is provided for the plots to ensure accessibility.

### 14. Predict Sentiment for New Data

- Load the saved model.
- Predict sentiment for a sample review.

---

## Outputs

- **Model Metrics**:
  - Training and validation accuracy/loss.
  - Test accuracy for model evaluation.
- **Plots**:
  - Accuracy and loss trends over training epochs.
- **Predictions**:
  - Sentiment probability for new text inputs.

---

## How to Run the Notebook

1. **Open the Notebook**:

   - Use Google Colab or a local Jupyter Notebook environment.

2. **Install Dependencies**:

   - Ensure all required libraries are installed as mentioned above.

3. **Run the Cells**:

   - Execute each cell sequentially to:
     - Load data, preprocess, and prepare the embeddings.
     - Build, train, and evaluate the model.
     - Save and test predictions with the trained model.

4. **Access Results**:

   - View plots for training and validation performance.
   - Generate predictions for new text inputs.

---

## References

- TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- GloVe Embeddings: [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
- IMDB Dataset: [https://www.tensorflow.org/datasets/catalog/imdb\_reviews](https://www.tensorflow.org/datasets/catalog/imdb_reviews)

---

## License

This tutorial is released under the MIT License. Refer to the LICENSE file in the [GitHub Repository](https://github.com/AnsiaNijas/MLNN) for details.

---

This README provides a detailed guide to replicate and understand the sentiment analysis pipeline using GloVe embeddings and BiLSTM. Enjoy exploring sentiment analysis!





