# URGENT 2026 — Track 2 (Speech Quality Assessment)


Predict the Mean Opinion Score (MOS) of speech processed by **speech enhancement (SE)** systems.
This repo provides the official implementation/baseline derived from [Uni-VERSA-Ext](https://arxiv.org/abs/2506.12260) for URGENT 2026 Track 2.

## Table of Contents

* [Installation](#installation)
* [Quickstart (Inference)](#quickstart-inference)
* [Training](#training)
* [Batch Inference & Evaluation](#batch-inference--evaluation)
* [Data](#data)
* [Build Your Own multi-Metric Dataset](#build-your-own-multi-metric-dataset)
* [Citations](#citations)

---

## Installation

```bash
# Create and activate environment
conda create -n urgent2026_sqa python=3.11 -y
conda activate urgent2026_sqa

# Install (minimal deps for inference)
pip install -e .

# For training, install extra dependencies (adjust to your pyproject):
# pip install -e .[train]
```

---

## Quickstart (Inference)

### Colab

Play with the model in Colab:
[https://colab.research.google.com/drive/1Y2OkPE0hGSG4XRj\_b7RsmWMVSg4KkhM7](https://colab.research.google.com/drive/1Y2OkPE0hGSG4XRj_b7RsmWMVSg4KkhM7)

### Local

> **Security note**: We use **HyperPyYAML** for config loading. Treat configs as code: **do not** load model from untrusted sources.


```python
from urgent2026_sqa.infer import infer_single, load_model

model = load_model("vvwang/uni_versa_ext-wavlm_base_plus-5_metrics")

print(infer_single(model, "./assets/gt.wav"))
print(infer_single(model, "./assets/noisy.wav"))
```

---

## Training

### 1) Prepare datasets

```bash
bash scripts/prepare_data.sh
```

This script fetches/organizes all datasets listed in the [Data](#data) section.

### 2) Launch training

```bash
accelerate launch urgent2026/train.py \
  --config configs/uni_versa_ext.yaml \
  --exp exp/uni_versa_ext \
  --train-data "data/*/train/data.jsonl" \
  --cv-data "data/chime-7-udase-eval/test/data.jsonl"
```

Notes:

* `configs/uni_versa_ext.yaml` is a symlink/alias to `configs/uni_versa_ext-wavlm_base_plus-5_metrics.yaml`.
* Training **auto-resumes** from the latest checkpoint in `--exp` if present.

---

## Batch Inference & Evaluation

### Batch inference
For inference on single audio file, Follow the [Quickstart (Inference)](#quickstart-inference) 

```bash
dataset="chime-7-udase-eval" python urgent2026_sqa/infer.py \
  --ckpt "exp/uni_versa_ext/model_last.pt" \
  --data "data/${dataset}/test/data.jsonl" \
  --outdir "exp/uni_versa_ext/infer/${dataset}"
```


### Evaluation

```bash
dataset="chime-7-udase-eval" python urgent2026_sqa/eval.py \
  --pred "exp/uni_versa_ext/infer/${dataset}/results.jsonl" \
  --ref  "data/${dataset}/test/data.jsonl"
```

You may also want to evaluate metric by comparing annotated metrics (e.g. scoreq) vs mos

```bash
dataset="chime-7-udase-eval" python urgent2026_sqa/eval.py \
  --pred "data/${dataset}/test/data.jsonl" \
  --ref  "data/${dataset}/test/data.jsonl" \
  --pred-metric "scoreq" \
```

#### Metrics

| Category           | Metric                      | Range    | Opt. |
| ------------------ | --------------------------- | -------- | --------- |
| Error              | **MSE (system level)**      | \[0, ∞)  | ↓         |
| Error              | **MSE (utterance level)**   | \[0, ∞)  | ↓         |
| Linear correlation | **LCC (system level)**      | \[-1, 1] | ↑         |
| Linear correlation | **LCC (utterance level)**   | \[-1, 1] | ↑         |
| Rank correlation   | **SRCC (system level)**     | \[-1, 1] | ↑         |
| Rank correlation   | **SRCC (utterance level)**  | \[-1, 1] | ↑         |
| Rank correlation   | **Kendall’s τ (system)**    | \[-1, 1] | ↑         |
| Rank correlation   | **Kendall’s τ (utterance)** | \[-1, 1] | ↑         |

---

## Data

| Split       | Corpus                            | #Samples | #Systems | Dur. (h) | Links                                                                                                                                                                                                                                   | License         |
| ----------- | --------------------------------- | -------: | -------: | -------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| Training    | BC19                              |      136 |       21 |     0.32 | [Original](https://zenodo.org/records/6572573/files/main.tar.gz)                                                                                                                                                                        | Custom          |
| Training    | BVCC                              |    4,973 |      175 |     5.56 | [Original](https://zenodo.org/records/6572573/files/ood.tar.gz)                                                                                                                                                                         | Custom          |
| Training    | NISQA                             |   11,020 |      N/A |    27.21 | [Original](https://zenodo.org/records/4728081/files/NISQA_Corpus.zip) · [Info](https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus)                                                                                                | Mixed           |
| Training    | PSTN                              |   58,709 |      N/A |   163.08 | [Original](https://challenge.blob.core.windows.net/pstn/train.zip)                                                                                                                                                                      | Unknown         |
| Training    | SOMOS                             |   14,100 |      181 |    18.32 | [Original](https://zenodo.org/records/7378801/files/somos.zip) · [HF Mirror](https://huggingface.co/datasets/urgent-challenge/urgent26_track2_sqa/resolve/main/somos.zip)                                                               | CC BY-NC-SA 4.0 |
| Training    | TCD-VoIP                          |      384 |       24 |     0.87 | [Original](https://drive.usercontent.google.com/download?id=1rHJN34vP-W8SJtjpNUnx5RIks3o5L5he&export=download&authuser=0) · [HF Mirror](https://huggingface.co/datasets/urgent-challenge/urgent26_track2_sqa/resolve/main/TCD-VOIP.zip) | CC BY-NC-SA 4.0 |
| Training    | Tencent                           |   11,563 |      N/A |    23.51 | [Original](https://share.weiyun.com/B4IS0l3z) · [HF Mirror](https://huggingface.co/datasets/urgent-challenge/urgent26_track2_sqa/resolve/main/TencentCorpus.zip)                                                                        | Apache          |
| Training    | TMHINT-QI                         |   12,937 |       98 |    11.35 | [Original](https://drive.google.com/file/d/1TMDiz6dnS76hxyeAcCQxeSqqEOH4UDN0/view?usp=sharing) · [HF Mirror](https://huggingface.co/datasets/urgent-challenge/urgent26_track2_sqa/resolve/main/TMHINTQI.zip)                            | MIT             |
| Training    | TTSDS2                            |      460 |       80 |     0.96 | [Original](https://huggingface.co/datasets/ttsds/listening_test) · [HF Mirror](https://huggingface.co/datasets/urgent-challenge/urgent26_track2_sqa/resolve/main/ttsds2.zip)                                                            | MIT             |
| Training    | URGENT2024-SQA                    |      238 |  238,000 |   429.34 | [HF](https://huggingface.co/datasets/urgent-challenge/urgent2024-sqa)                                                                                                                                                                   | CC BY-NC-SA 4.0 |
| Training    | URGENT2025-SQA                    |  100,000 |      100 |   261.31 | [HF](https://huggingface.co/datasets/urgent-challenge/urgent2025-sqa)                                                                                                                                                                   | CC BY-NC-SA 4.0 |
| Development | CHiME-7 UDASE Eval                |      640 |        5 |     0.84 | [Original](https://zenodo.org/records/10418311/files/CHiME-7-UDASE-evaluation-data.zip) · [HF Mirror](https://huggingface.co/datasets/urgent-challenge/urgent26_track2_sqa/resolve/main/CHiME-7-UDASE-evaluation-data.zip)              | CC BY-SA 4.0    |
| Evaluation  | URGENT2024-SQA (blind\_test\_mos) |       23 |    6,900 |    13.80 | [HF](https://huggingface.co/datasets/urgent-challenge/urgent2024-sqa)                                                                                                                                                                   | CC BY-NC-SA 4.0 |

---


## Build Your Own Multi-Metric Dataset

```bash
pip install -e .[dev]
```

## Citations

If you use this code or datasets, please consider citing:

```bibtex
@article{UniVersaExt-Wang2025,
  title={Improving Speech Enhancement with Multi-Metric Supervision from Learned Quality Assessment},
  author={Wang, Wei and Zhang, Wangyou and Li, Chenda and Shi, Jiatong and Watanabe, Shinji and Qian, Yanmin},
  journal={arXiv preprint arXiv:2506.12260},
  year={2025}
}

@article{P808-Sach2025,
  title={P.808 Multilingual Speech Enhancement Testing: Approach and Results of {URGENT} 2025 Challenge},
  author={Sach, Marvin and Fu, Yihui and Saijo, Kohei and Zhang, Wangyou and Cornell, Samuele and Scheibler, Robin and Li, Chenda and ...},
  journal={arXiv preprint arXiv:25xx.xxxxx},
  year={2025}
}

@inproceedings{URGENT-Zhang2024,
  title={{URGENT} Challenge: Universality, Robustness, and Generalizability For Speech Enhancement},
  author={Zhang, Wangyou and Scheibler, Robin and Saijo, Kohei and Cornell, Samuele and Li, Chenda and Ni, Zhaoheng and Pirklbauer, Jan and Sach, Marvin and Watanabe, Shinji and Fingscheidt, Tim and Qian, Yanmin},
  booktitle={Proc. Interspeech},
  pages={4868--4872},
  year={2024}
}
```