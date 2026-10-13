# Tiny English Chatbot — Build a Transformer from Scratch

A minimal, fully trainable Transformer-based chatbot built with Python and PyTorch to demonstrate how language models learn and generate text.

**No pretrained models. No external AI APIs.**

### Live Demo

**[Try the Chatbot Online](https://tiny-english-chatbot.onrender.com)**

Interact with a small Transformer language model trained from scratch on 174 tokens and 18,536 English question–answer pairs.

### Web Interface

This chatbot supports interactive conversations, vocabulary exploration, and training dataset inspection.

<img width="1024" height="480" alt="Chatbot Demo 1 - All" src="https://github.com/user-attachments/assets/b181941d-f067-4093-ba26-8cf5b8bdccc3" />

It can generate coherent and meaningful responses to familiar English questions.

<img width="1024" height="480" alt="Chatbot Demo 2 - good" src="https://github.com/user-attachments/assets/56a1e16e-457c-44ab-84a2-74b18700a0bc" />

However, it may produce incorrect or unexpected responses sometimes or when faced with unfamiliar questions.

<img width="1024" height="480" alt="Chatbot Demo 3 -  Bad" src="https://github.com/user-attachments/assets/5d6d279d-ff5a-4584-b636-adc98de0a1f0" />

Despite its limitations, this project demonstrates the complete process of building, training, and deploying a Transformer language model from scratch using a limited vocabulary and dataset.

*The web interface of the Tiny English Chatbot, deployed on Render.*

---


## 1. Why I Built This

When I first started learning about Large Language Models (LLMs), I found that understanding concepts like Transformers, Self-Attention, and Next-Token Prediction was very different from understanding how they actually work together.

I wanted to build a language model myself, starting from random weights, to understand the complete process rather than just studying the theory.

This project is also intended as a simple, reproducible learning resource for anyone starting their journey into LLMs.

**The goal is not to build a powerful chatbot, but to make the fundamental principles of language models easier to understand.**

## 2. How It Works

The project follows a complete language-model development pipeline:

**Dataset → Tokenization → Embedding → Transformer → Training → Text Generation → Web Deployment**

### Model Architecture

- **Architecture:** Small causal Transformer implemented using PyTorch
- **Transformer Layers:** 2
- **Attention Heads:** 4
- **Embedding Dimension:** 64
- **Vocabulary Size:** 174 tokens, including special tokens
- **Training Method:** Next-Token Prediction
- **Initialization:** Random weights, without pretrained parameters

The model uses Token Embeddings, Positional Embeddings, Multi-Head Self-Attention, Feed-Forward Networks, and a final Linear Layer to predict the next token.

During inference, the chatbot generates responses one token at a time based on the input question and previously generated tokens.

## 3. Dataset

The model is trained on **18,536 English question–answer pairs** from two datasets:

| Dataset | Question–Answer Pairs | Source |
|---|---:|---|
| `dataset.csv` | 4,964 | Generated using Python templates |
| `dataset_2.csv` | 13,572 | Generated with ChatGPT |
| **Total** | **18,536** | |

The training process uses an 80% / 10% / 10% split for training, validation, and testing.

The datasets use deliberately simple English vocabulary and sentence structures to keep the model small and the learning process accessible.

## 4. Training Results

The model is trained using PyTorch with the following settings:

- **Optimizer:** AdamW
- **Loss Function:** Cross-Entropy Loss
- **Batch Size:** 64
- **Learning Rate:** 0.001
- **Maximum Epochs:** 50
- **Early Stopping:** Enabled

### Training and Validation Loss

<img width="1200" height="750" alt="loss_curve" src="https://github.com/user-attachments/assets/22e2824e-4839-46da-a0bd-e9282f3a28ba" />

The training and validation losses decrease rapidly and converge to relatively low values, demonstrating that the model learns the patterns present in the training data.

However, because the datasets contain many related sentence templates, low validation loss does not necessarily indicate strong generalization to unfamiliar questions.

## 5. Model Visualization

To make the Transformer easier to understand, the project includes scripts for visualizing its internal components.

- `visualize_qkv.py` — Visualizes the learned Query, Key, and Value projection matrices.
- `visualize_embedding_attention.py` — Visualizes Token Embeddings and the Self-Attention Matrix.

**Token Embeddings:** Visualizes the learned 64-dimensional representations of different tokens.

<img width="1300" height="405" alt="trained_token_embedding_matrix" src="https://github.com/user-attachments/assets/da3a280c-8f76-4412-9a0e-f08a7c53491e" />

**QKV Matrices:** Shows the learned projection weights used in the Self-Attention mechanism.

<img width="1500" height="500" alt="qkv" src="https://github.com/user-attachments/assets/aa5839a5-30ec-4469-b5e2-ef20952b552b" />

**Attention Matrix:** Shows how tokens attend to previous tokens when processing the example sentence **"What do you want to do?"**, illustrating the Causal Self-Attention mechanism.

<img width="600" height="500" alt="attention_matrix" src="https://github.com/user-attachments/assets/0e3c5ec7-422b-4296-8776-924617a1a0fe" />

These visualizations help connect theoretical concepts such as QKV Self-Attention with the actual learned parameters and computations inside the model.

## 6. Run the Project

### Install Dependencies

Install the required Python packages:

```bash
pip install torch matplotlib pandas fastapi uvicorn PyQt6
```

### Generate the Dataset

```bash
python generate_dataset.py
```

This creates `dataset.csv`. The additional `dataset_2.csv` is also used during training.

### Train the Model

```bash
python train.py
```

The trained model is saved as `model.pt`, and the training loss curve is saved as `loss_curve.png`.

### Chat in the Terminal

```bash
python chat.py
```

### Launch the Web Application

```bash
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000` in your browser to interact with your trained model.

## 7. Web Application

The chatbot is integrated into a simple web application using:

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** FastAPI
- **Model Inference:** PyTorch
- **Version Control:** Git and GitHub
- **Deployment:** Render

User messages are sent from the web interface to the FastAPI backend, where the trained Transformer generates a response and returns it to the browser.

## 8. Limitations

This is an educational model rather than a production-level LLM.

Its limitations include:

- A small vocabulary and limited training data
- Restricted conversational capabilities
- Limited generalization to unfamiliar language patterns
- Occasional grammatically incorrect or inconsistent responses

These limitations are intentional trade-offs to keep the model compact, understandable, and trainable with modest computing resources.

## 9. Final Thoughts

The most valuable part of this project is not the chatbot's performance, but the opportunity to experience the full language-model development process.

By building and training a small Transformer from scratch, abstract concepts like Embeddings, Self-Attention, and Autoregressive Generation become much more concrete.

**You don't need billions of parameters to start understanding how a language model works.**

---

*Built as a hands-on educational project for learning the fundamentals of LLMs.*
