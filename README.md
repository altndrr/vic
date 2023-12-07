<div align="center">

<a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/-Python_3.9-blue?logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch_2.0-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning_2.0-792ee5?logo=pytorchlightning&logoColor=white"></a>

[![](https://img.shields.io/badge/code-github.altndrr%2Fvic-blue.svg)](https://github.com/altndrr/vic)
[![](https://img.shields.io/badge/demo-hf.altndrr%2Fvic-yellow.svg)](https://huggingface.co/spaces/altndrr/vic)
[![](http://img.shields.io/badge/paper-arxiv.2306.00917-B31B1B.svg)](https://arxiv.org/abs/2306.00917)
[![](https://img.shields.io/badge/website-gh--pages.altndrr%2Fvic-success.svg)](https://alessandroconti.me/papers/2306.00917)

# Vocabulary-free Image Classification

[Alessandro Conti](https://scholar.google.com/citations?user=EPImyCcAAAAJ), [Enrico Fini](https://scholar.google.com/citations?user=OQMtSKIAAAAJ), [Massimiliano Mancini](https://scholar.google.com/citations?user=bqTPA8kAAAAJ), [Paolo Rota](https://scholar.google.com/citations?user=K1goGQ4AAAAJ), [Yiming Wang](https://scholar.google.com/citations?user=KBZ3zrEAAAAJ), [Elisa Ricci](https://scholar.google.com/citations?user=xf1T870AAAAJ)

</div>

Recent advances in large vision-language models have revolutionized the image classification paradigm. Despite showing impressive zero-shot capabilities, a pre-defined set of categories, a.k.a. the vocabulary, is assumed at test time for composing the textual prompts. However, such assumption can be impractical when the semantic context is unknown and evolving. We thus formalize a novel task, termed as Vocabulary-free Image Classification (VIC), where we aim to assign to an input image a class that resides in an unconstrained language-induced semantic space, without the prerequisite of a known vocabulary. VIC is a challenging task as the semantic space is extremely large, containing millions of concepts, with hard-to-discriminate fine-grained categories.

<div align="center">

|         <img src="assets/task_left.png">         |  <img src="assets/task_right.png">   |
| :----------------------------------------------: | :----------------------------------: |
| Vision Language Model (VLM)-based classification | Vocabulary-free Image Classification |

</div>

In this work, we first empirically verify that representing this semantic space by means of an external vision-language database is the most effective way to obtain semantically relevant content for classifying the image. We then propose Category Search from External Databases (CaSED), a method that exploits a pre-trained vision-language model and an external vision-language database to address VIC in a training-free manner. CaSED first extracts a set of candidate categories from captions retrieved from the database based on their semantic similarity to the image, and then assigns to the image the best matching candidate category according to the same vision-language model. Experiments on benchmark datasets validate that CaSED outperforms other complex vision-language frameworks, while being efficient with much fewer parameters, paving the way for future research in this direction.

<div align="center">

|                                                                                                                                 <img src="assets/method.png">                                                                                                                                  |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Overview of CaSED. Given an input image, CaSED retrieves the most relevant captions from an external database filtering them to extract candidate categories. We classify image-to-text and text-to-text, using the retrieved captions centroid as the textual counterpart of the input image. |

</div>

## Inference

Our model CaSED is available on HuggingFace Hub. You can try it directly from the [demo](https://altndrr-vic.hf.space/) or import it from the `transformers` library.

To use the model from the HuggingFace Hub, you can use the following snippet:

```python
import requests
from PIL import Image
from transformers import AutoModel, CLIPProcessor

# download an image from the internet
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# load the model and the processor
model = AutoModel.from_pretrained("altndrr/cased", trust_remote_code=True)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# get the model outputs
images = processor(images=[image], return_tensors="pt", padding=True)
outputs = model(images, alpha=0.7)
labels, scores = outputs["vocabularies"][0], outputs["scores"][0]

# print the top 5 most likely labels for the image
values, indices = scores.topk(3)
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{labels[index]:>16s}: {100 * value.item():.2f}%")
```

Note that our model depends on some libraries you have to install manually. Please refer to the [model card](https://huggingface.co/altndrr/cased) for further details.

## Setup

### Install dependencies

```bash
# clone project
git clone https://github.com/altndrr/vic
cd vic

# install requirements
# it will create a .venv folder in the project root
# and install all the dependencies using flit
make install

# activate virtual environment
source .venv/bin/activate
```

### Setup environment variables

```bash
# copy .env.example to .env
cp .env.example .env

# edit .env file
vim .env
```

## Usage

The two entry points are `train.py` and `eval.py`. Calling them without any argument will use the default configuration.

```bash
# train model
python src/train.py

# test model
python src/eval.py
```

### Configuration

The full list of parameters can be found under [configs](configs/), but the most important ones are:

- [data](configs/data/): dataset to use, default to `caltech101`.
- [experiment](configs/experiment/): experiment to run, default to `baseline/clip`.
- [logger](configs/logger/): logger to use, default to `null`.

Parameters can be overwritten by passing them as command line arguments. You can additionally override any parameter from the config file by using the `++` prefix.

```bash
# train model on ucf101 dataset
python src/train.py data=ucf101 experiment=baseline/clip

# train model on ucf101 dataset with RN50 backbone
python src/train.py data=ucf101 experiment=baseline/clip model=clip ++model.model_name=RN50
```

Note that since all our approaches are training-free, there is virtually no difference between `train.py` and `eval.py`. However, we still keep them separate for clarity.

## Development

### Install pre-commit hooks

```bash
# install pre-commit hooks
pre-commit install
```

### Run tests

```bash
# run fast tests
make test

# run all tests
make test-full
```

### Format code

```bash
# run linters
make format
```

### Clean repository

```bash
# remove autogenerated files
make clean

# remove logs
make clean-logs
```

## Citation

```latex
@misc{conti2023vocabularyfree,
      title={Vocabulary-free Image Classification},
      author={Alessandro Conti and Enrico Fini and Massimiliano Mancini and Paolo Rota and Yiming Wang and Elisa Ricci},
      year={2023},
      journal={"NeurIPS},
}
```
