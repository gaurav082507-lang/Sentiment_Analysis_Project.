# Sentiment Analysis 💬

A deep learning web app that classifies the sentiment of any piece of text as **Positive**, **Negative**, **Neutral**, or **Irrelevant**, powered by a Bidirectional LSTM model and served through a dark-themed Streamlit interface.

![Deep Learning](https://img.shields.io/badge/Deep%20Learning-NLP-8b7bf0?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)

## 📖 About

This project trains a **Bidirectional LSTM** neural network to automatically detect the sentiment behind a piece of text. The model is wrapped in an interactive, dark-gradient Streamlit UI where users can type in any sentence and instantly see the predicted sentiment along with a confidence breakdown across all four classes.

## ✨ Features

- **4-class sentiment detection** — Positive, Negative, Neutral, Irrelevant
- **Bidirectional LSTM** architecture with a `TextVectorization` layer built into the model (no separate tokenizer needed)
- **Real-time predictions** with confidence percentages for every class
- **Dark, gradient-themed UI** with animated confidence bars and a clean card-based layout
- **Simple deployment** — single `app.py` file, minimal dependencies

## 🧠 Model Architecture

```
Input (raw text string)
    ↓
TextVectorization (max_tokens=10000, seq_length=100)
    ↓
Embedding (10000 × 128)
    ↓
Bidirectional LSTM (64 units)
    ↓
Dropout (0.3)
    ↓
Dense (64, ReLU)
    ↓
Dense (4, Softmax)
```

- **Loss:** Sparse Categorical Crossentropy
- **Optimizer:** Adam (lr = 1e-3)
- **Training data:** ~70,000 labeled text samples
- **Callback:** Early stopping on validation loss

## 📂 Project Structure

```
├── app.py                      # Streamlit web app
├── requirements.txt            # Python dependencies
├── sentiment-analysis.ipynb    # Model training notebook
├── sentiment_model.keras       # Trained model (generate via notebook)
└── README.md
```

## 🚀 Getting Started

### 1. Clone / download the project files

Make sure `app.py`, `requirements.txt`, and the trained model file are in the same folder.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train and export the model

Run through the training notebook, then add this line at the end to save the trained model:

```python
model.save("sentiment_model.keras")
```

Place the resulting `sentiment_model.keras` file in the same directory as `app.py`.

### 4. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 🖥️ Usage

1. Type or paste any sentence into the text box.
2. Click **Detect Sentiment**.
3. View the predicted sentiment along with a confidence score and a full breakdown across all four classes.

## 🛠️ Tech Stack

- **TensorFlow / Keras** — model building and training
- **Streamlit** — web app framework
- **NumPy / Pandas** — data handling
- **scikit-learn** — train/test split, evaluation metrics

## 👤 Author

**Built by Gaurav Gupta**
🔗 [LinkedIn Profile](https://www.linkedin.com/in/gaurav-gupta-79754a377)

## 📄 License

This project is open for educational and personal use. Feel free to fork, modify, and build on it.
