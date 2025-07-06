# Deep Learning Audio Classification for Capuchin Bird Calls

## 1. Overview

This project demonstrates a complete workflow for building a deep learning model to classify audio signals. Specifically, it trains a Convolutional Neural Network (CNN) to detect the presence of a Capuchin bird call in audio clips. The entire process, from data loading and preprocessing to model training and prediction, is contained within the `AudioClassification.ipynb` Jupyter Notebook.

This system serves as a foundational example of keyword spotting or sound event detection, a technology that powers applications like voice assistants ("Hey Google", "Alexa") and environmental sound monitoring.

---

## 2. The Workflow

The notebook follows a structured, step-by-step process to tackle the audio classification problem.

1.  **Setup and Dependencies**: The environment is prepared by installing essential Python libraries:
    *   `tensorflow`: For building and training the deep learning model.
    *   `librosa`: A powerful library for audio analysis, used here for loading and resampling audio files.
    *   `numpy`: For numerical operations.
    *   `matplotlib`: For data visualization, including plotting waveforms and spectrograms.

2.  **Data Loading and Structuring**:
    *   Audio clips are organized into two categories: positive (`Parsed_Capuchinbird_Clips`) and negative (`Parsed_Not_Capuchinbird_Clips`).
    *   `tf.data.Dataset` is used to create highly efficient input pipelines that load the file paths for these clips.
    *   Labels are assigned: `1` for Capuchin clips (positive) and `0` for non-Capuchin clips (negative).
    *   These two datasets are then concatenated into a single, unified dataset.

3.  **Audio Preprocessing and Feature Extraction**: This is the most critical step in transforming raw audio into a format suitable for a CNN.
    *   **Standardization**: All audio files are loaded and resampled to a consistent `16kHz` mono format using `librosa`. This ensures uniformity for the model.
    *   **Padding/Trimming**: To ensure all input samples are of the same length, the audio waves are padded with zeros or trimmed to a fixed length of 48,000 samples (equivalent to 3 seconds at 16kHz).
    *   **Spectrogram Conversion**: The 1D audio waveform is converted into a 2D **spectrogram** using `tf.signal.stft` (Short-Time Fourier Transform). A spectrogram is a visual representation of the spectrum of frequencies of a signal as it varies with time. This effectively turns our audio problem into an image recognition problem, which CNNs excel at.

4.  **Building the `tf.data` Pipeline**:
    *   The preprocessing function is mapped across the entire dataset.
    *   The pipeline is optimized for performance by:
        *   `.cache()`: Caching the dataset in memory after the first epoch to speed up subsequent epochs.
        *   `.shuffle()`: Randomizing the order of the data to prevent the model from learning any spurious order.
        *   `.batch()`: Grouping individual samples into batches for efficient processing on the GPU.
        *   `.prefetch()`: Preloading the next batch of data while the current one is being processed, minimizing GPU idle time.
    *   The final dataset is split into a training set and a testing set.

5.  **Model Architecture (CNN)**: A `Sequential` model is built using `tensorflow.keras`.
    *   **Input Layer**: The model expects the shape of the spectrograms (`1491, 257, 1`).
    *   **Convolutional Layers (`Conv2D`)**: Two `Conv2D` layers with 16 filters and a (3,3) kernel size are used. These layers act as feature extractors, learning to identify low-level patterns (edges, textures) in the spectrograms in the first layer, and combining them into more complex patterns in the second layer. The `ReLU` activation function is used to introduce non-linearity.
    *   **Flatten Layer**: This layer converts the 2D feature maps from the convolutional layers into a single 1D vector. This is necessary to feed the data into the dense layers.
    *   **Dense Layers (`Dense`)**:
        *   A fully connected layer with 128 neurons (`ReLU` activation) acts as a classifier, learning to combine the features detected by the convolutional layers.
        *   The final output layer has a single neuron with a `sigmoid` activation function. The sigmoid function outputs a value between 0 and 1, which is interpreted as the probability that the audio clip contains a Capuchin call.

6.  **Training and Evaluation**:
    *   The model is compiled with the `Adam` optimizer, `BinaryCrossentropy` loss function (standard for binary classification), and `Recall` and `Precision` as performance metrics.
    *   The model is trained for 4 epochs using the `.fit()` method, with the test set used for validation.
    *   The training history (loss, precision, recall) is plotted to visualize the model's learning progress and check for overfitting.

7.  **Prediction**:
    *   The trained model is used to make predictions on a batch of unseen data from the test set to demonstrate its practical application.
    *   The raw probability outputs (logits) from the model are converted into definitive classes (0 or 1) by thresholding at 0.5.

---

## 3. How to Run This Project

1.  **Prerequisites**:
    *   Python 3.x
    *   Jupyter Notebook or JupyterLab

2.  **Clone the Repository (Optional)**:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

3.  **Set Up a Virtual Environment**: It is highly recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

4.  **Install Dependencies**: The notebook includes the necessary pip command. Open the notebook and run the first code cell to install all required libraries.
    ```python
    %pip install tensorflow librosa numpy matplotlib
    ```

5.  **Launch Jupyter**:
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```

6.  **Run the Notebook**: Open the `AudioClassification.ipynb` file and run the cells sequentially from top to bottom.

---

## 4. Dataset

*   **Source**: The audio data is located in the `/archive` directory.
*   **Positive Class**: Clips containing Capuchin bird calls are in `archive/Parsed_Capuchinbird_Clips/`.
*   **Negative Class**: Clips that do not contain Capuchin calls (background forest sounds, other birds, etc.) are in `archive/Parsed_Not_Capuchinbird_Clips/`.

---

## 5. Technical Deep Dive

### Why Spectrograms?
A raw audio waveform is a complex 1D signal. While models like RNNs can process 1D sequences, CNNs have proven to be exceptionally effective at finding spatial patterns in 2D data (images). A spectrogram represents the frequency content of audio over time, making it a 2D data structure. The characteristic call of a bird has a unique "shape" or "texture" on a spectrogram, which a CNN can learn to recognize, just as it would learn to recognize the shape of a cat or a dog in a regular photograph.

### The CNN Model Explained
*   `model.add(Conv2D(16, (3,3), activation='relu', input_shape=(1491, 257,1)))`: This first layer scans the spectrogram with 16 different 3x3 filters, looking for very basic patterns like vertical lines, horizontal lines, or simple textures.
*   `model.add(Conv2D(16, (3,3), activation='relu'))`: This second layer takes the patterns found by the first layer and combines them into more complex features. For example, it might learn that a specific horizontal line followed by a series of dots is a key part of the Capuchin call.
*   `model.add(Flatten())`: This unrolls the 2D grid of complex features into a long 1D vector.
*   `model.add(Dense(128, activation='relu'))`: This layer looks at the entire set of features at once and learns which combinations of features are most indicative of a Capuchin call.
*   `model.add(Dense(1, activation='sigmoid'))`: This final layer takes the high-level feature analysis from the previous layer and squashes it into a single probability score. A score close to 1.0 means "very likely a Capuchin," and a score close to 0.0 means "very unlikely a Capuchin."
