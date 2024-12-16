# Deep Voice Detection with FedAvg

<br>

*2024년 2학기 이화여자대학교 도전학기제 프로젝트로 진행하였음.*


## Overview
I developed a new DeepVoice detection model and applied techniques such as Federated Learning (FL) and unstructured pruning. The model is based on a modified version of the Deep Fingerprinting (DF) model, which I convert the [official TensorFlow implementation](https://dl.acm.org/doi/10.1145/3243734.3243768) to PyTorch, based on [NetCLR](https://github.com/SPIN-UMass/Realistic-Website-Fingerprinting-By-Augmenting-Network-Traces/blob/main/artifacts/src/NetCLR/pre-training.ipynb) official implementation.

## Project Structure
The project is organized as follows:

```
DETECT
├── preprocess          # preprocess raw data
├── non_fl				# traditional classification with DF
├── fl					# federated learning with flwr
│   ├── dvd					# DeepVoice Detection
│   │	├── __init__.py
│   │	├── client_app.py   # Define ClientApp
│   │	├── server_app.py   # Define ServerApp
│   |	└── task.py         # Define model, training and data loading
│   └── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```


## Datasets
The [ASVspoof 2019](https://www.asvspoof.org/index2019.html) dataset, which includes Logical Access (LA) and Physical Access (PA) attacks, was used for this task. I focused on the LA dataset, which is more relevant for DeepVoice voice phishing detection. Additionally, the [Fake-or-Real (FoR)](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset/data) dataset was also used. The combined datasets contain 257,865 speech samples: 164,977 fake and 92,888 real samples. I conducted experiments using raw features (numpy arrays) and Wav2Vec features extracted using a `Wav2Vec` feature extractor.

## Non-FL Setting

Run the following Python script to train the model in the Non-FL setting:

```
$ python non_fl/main.py
```

Additionally, I applied unstructured pruning to the DF model, focusing on the 1D convolution layers. 
Run the following Python script to apply pruning and train the model:

```
$ python non_fl/main.py --apply_pruning
```

## FL Setting
For the FL setting, FedAvg algorithm with the [Flower](flower.ai) framework is used. I refered to Flower's official pytorch [guideline](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch). Here, the model and dataset parts have been completely changed.
Run the following Python script to train the model in the FL setting:

```
$ cd fl/dvd
$ flwr run . local-simulation-gpu
```

