# pycb-ml 

## Machine Learning for py-clash-bot

Scripts used to preprocess data, train and use models for [py-clash-bot](https://github.com/matthewmiglio/py-clash-bot).

### Training an Image Classification Model

The module **`train_model.py`** creates a trained image classification model.

#### Run on Google Colab

Open **`pycb_ml.ipynb`** with on [Google Colab](https://colab.research.google.com/).

#### Run locally

Clone this repository

```bash
git clone https://github.com/marmig0404/pycb-ml
cd pycb-ml
```

Install dependencies

```bash
pip install -r requirements.txt
```

Train a model after loading images to **`data\train`**, separated into folder by class.

```bash
python train_model.py
```

Trained model will be in **`models`**.