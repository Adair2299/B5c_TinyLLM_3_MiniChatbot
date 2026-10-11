# Tiny English Chatbot

A small Transformer trained **from scratch** to generate short English answers from questions. No pretrained model or API is used.

## Files

- `model.py`: causal Transformer with learned positional embeddings and multi-head QKV attention.
- `data_utils.py`: shared tokenizer and vocabulary utilities.
- `generate_dataset.py`: reproducible English question-answer dataset generator.
- `dataset.csv`: generated question-answer training data.
- `train.py`: masked next-token training, train/validation split, early stopping, best checkpoint, QtAgg loss visualization.
- `chat.py`: interactive answer generation from the saved checkpoint.
- `model.pt`: created after training; contains learned weights, model configuration, and vocabulary.
- `loss_curve.png`: created after training with the QtAgg plot window.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python generate_dataset.py
python train.py --epochs 30
python chat.py
```

The `dataset.csv` is already included, so generation is optional. Training automatically saves the best model to `model.pt`. The training script opens a **QtAgg** loss chart and saves `loss_curve.png`. Close the chart to exit training. To run without a graphical desktop, use `python train.py --no-plot`.

Quick smoke test:

```bash
python train.py --epochs 1 --limit 100 --no-plot
```

## Caveats

This is an educational, very small model. Generated answers may be incorrect, inconsistent, or nonsensical. Weather answers are fictional training examples, **not** real-time weather reports. Multiple answers per question are intentionally included, so the model learns a distribution rather than a single hard-coded mapping. A random holdout can still contain nearly identical templates; low validation loss does not establish robust generalization. For a rigorous test, hold out whole question templates.
