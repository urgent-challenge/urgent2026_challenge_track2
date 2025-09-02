# URGENT 2026 — Track 2 (Speech Quality Assessment)


Predict the Mean Opinion Score (MOS) of speech processed by **speech enhancement (SE)** systems.
This repo provides the official implementation/baseline derived from [UniVERSA-Ext](https://arxiv.org/abs/2506.12260) for URGENT 2026 Track 2.

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
conda create -n urgent2026-sqa python=3.11 -y
conda activate urgent2026-sqa

# Install (minimal deps for inference)
pip install -e .

# For training, install extra dependencies
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

model = load_model("vvwangvv/universa-ext_wavlm-base_5metric")

# examples are from https://labsites.rochester.edu/air/projects/is2012/examples.html
print(infer_single(model, config, "./assets/sp03.wav"))
print(infer_single(model, config, "./assets/sp03_casino_sn5.wav"))
```

---

## Training

### 1) Prepare datasets

```bash
bash scripts/prepare_data.sh </path/to/db>
```

This script fetches/organizes all datasets listed in the [Data](#data) section.

> NOTE: bvcc and bc19 datasets requires manual process after downloading
> if you don't want to include them, comment the correspoding line in `scripts/data/prepare.sh`

### 2) Launch training

The following command train the UniVERSA-Ext with all prepared training datasets.
```bash
accelerate launch urgent2026/train.py \
  --config configs/universa-ext.yaml \
  --exp exp/universa-ext \
  --train-data "data/*/train/data.jsonl" \
  --cv-data "data/chime-7-udase-eval/test/data.jsonl"
```

Training will auto-resume from the latest checkpoint.

Distributed training and mixed precision is supported with `accelerate`, example:
```bash
accelerate launch --num_processes=<N> \
  --main_process_port <port> \
  --mixed_precision=bf16 \
  urgent2026/train.py \
  ...
```

---

## Batch Inference & Evaluation

### Batch inference
For inference on single audio file, follow [Quickstart (Inference)](#quickstart-inference)

For batch inference:
```bash
dataset="chime-7-udase-eval" python urgent2026_sqa/infer.py \
  --ckpt "exp/universa-ext/model_last.pt" \
  --data "data/${dataset}/test/data.jsonl" \
  --outdir "exp/universa_ext/infer/${dataset}"
```

This will genenerate a `results.jsonl` file and `{metric}.scp` files for all metrics under the `--outdir`


### Evaluation

```bash
dataset="chime-7-udase-eval" python urgent2026_sqa/eval.py \
  --pred "exp/universa_ext/infer/${dataset}/results.jsonl" \
  --ref  "data/${dataset}/test/data.jsonl"
```

You may also want to evaluate metric by comparing annotated metrics (e.g. scoreq) vs mos:
```bash
dataset="chime-7-udase-eval" python urgent2026_sqa/eval.py \
  --pred "data/${dataset}/test/data.jsonl" \
  --ref  "data/${dataset}/test/data.jsonl" \
  --pred-metric "scoreq"
```

#### Metrics

<table>
<thead>
<tr>
    <th>Category</th>
    <th>Metric</th>
    <th>Value Range</th>
    <th>Opt.</th>
</tr>
</thead>
<tbody>
<tr>
    <td rowspan="2">Error</td>
    <td>System level MSE</td>
    <td>[0, ∞)</td>
    <td>↓</td>
</tr>
<tr>
    <td>Utterance level MSE </td>
    <td>[0, ∞)</td>
    <td>↓</td>
</tr>
<tr>
    <td rowspan="2">Linear Correlation</td>
    <td> System level LCC</td>
    <td>[-1, 1]</td>
    <td>↑</td>
</tr>
<tr>
    <td>Utterance level LCC</td>
    <td>[-1, 1]</td>
    <td>↑</td>
</tr>
<tr>
    <td rowspan="4">Rank Correlation</td>
    <td>System level SRCC</td>
    <td>[-1, 1]</td>
    <td>↑</td>
</tr>
<tr>
    <td>Utterance level SRCC</td>
    <td>[-1, 1]</td>
    <td>↑</td>
</tr>
<tr>
    <td>System level KTAU</td>
    <td>[-1, 1]</td>
    <td>↑</td>
</tr>
<tr>
    <td>Utterance level KTAU</td>
    <td>[-1, 1]</td>
    <td>↑</td>
</tr>

</tbody>
</table><br/>

---

## Data

<table>
<colgroup>
<col>
<col>
<col>
<col>
<col>
<col>
</colgroup>
<thead>
  <tr>
    <th></th>
    <th>Corpus</th>
    <th>#Samples</th>
    <th>#Systems</th>
    <th>Duration (hours)</th>
    <th>Links</th>
    <th>License</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="11">Training</td>
    <td>BC19<d-cite key="BC19"/></td>
    <td>136</td>
    <td>21</td>
    <td>0.32</td>
    <td><a href="https://zenodo.org/records/6572573/files/main.tar.gz">[Original]</a></td>
    <td><a href="https://www.cstr.ed.ac.uk/projects/blizzard/data.html">Custom</a></td>
  </tr>
  <tr>
    <td>BVCC<d-cite key="BVCC"/></td>
    <td>4973</td>
    <td>175</td>
    <td>5.56</td>
    <td><a href="https://zenodo.org/records/6572573/files/ood.tar.gz">[Original]</a></td>
    <td><a href="https://www.cstr.ed.ac.uk/projects/blizzard/data.html">Custom</a></td>
  </tr>
  <tr>
    <td>NISQA<d-cite key="NISQA"/></td>
    <td>11020</td>
    <td>N/A</td>
    <td>27.21</td>
    <td><a href="https://zenodo.org/records/4728081/files/NISQA_Corpus.zip">[Original]</a></td>
    <td><a href="https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus">Mixed</a></td>
  </tr>
  <tr>
    <td>PSTN<d-cite key="PSTN"/></td>
    <td>58709</td>
    <td>N/A</td>
    <td>163.08</td>
    <td><a href="https://challenge.blob.core.windows.net/pstn/train.zip">[Original]</a></td>
    <td>Unknown</td>
  </tr>
  <tr>
    <td>SOMOS<d-cite key="SOMOS"/></td>
    <td>14100</td>
    <td>181</td>
    <td>18.32</td>
    <td>
      <a href="https://zenodo.org/records/7378801/files/somos.zip">[Original]</a>
      <a href="https://huggingface.co/datasets/urgent-challenge/urgent26_track2_sqa/resolve/main/somos.zip">[Huggingface]</a>
    </td>
    <td>CC BY-NC-SA 4.0</td>
  </tr>
  <tr>
    <td>TCD-VoIP<d-cite key="TCD-VoIP"/></td>
    <td>384</td>
    <td>24</td>
    <td>0.87</td>
    <td>
      <a href="https://drive.usercontent.google.com/download?id=1rHJN34vP-W8SJtjpNUnx5RIks3o5L5he&export=download&authuser=0">[Original]</a>
      <a href="https://huggingface.co/datasets/urgent-challenge/urgent26_track2_sqa/resolve/main/TCD-VOIP.zip">[Huggingface]</a>
    </td>
    <td>CC BY-NC-SA 4.0</td>
  </tr>
  <tr>
    <td>Tencent<d-cite key="Tencent"/></td>
    <td>11563</td>
    <td>N/A</td>
    <td>23.51</td>
    <td>
      <a href="https://share.weiyun.com/B4IS0l3z">[Original]</a>
      <a href="https://huggingface.co/datasets/urgent-challenge/urgent26_track2_sqa/resolve/main/TencentCorpus.zip">[Huggingface]</a>
    </td>
    <td>Apache</td>
  </tr>
  <tr>
    <td>TMHINT-QI<d-cite key="TMHINT-QI"/></td>
    <td>12937</td>
    <td>98</td>
    <td>11.35</td>
    <td>
      <a href="https://drive.google.com/file/d/1TMDiz6dnS76hxyeAcCQxeSqqEOH4UDN0/view?usp=sharing">[Original]</a>
      <a href="https://huggingface.co/datasets/urgent-challenge/urgent26_track2_sqa/resolve/main/TMHINTQI.zip">[Huggingface]</a>
    </td>
    <td>MIT</td>
  </tr>
  <tr>
    <td>TTSDS2<d-cite key="TTSDS2"/></td>
    <td>460</td>
    <td>80</td>
    <td>0.96</td>
    <td>
      <a href="https://huggingface.co/datasets/ttsds/listening_test">[Original]</a>
      <a href="https://huggingface.co/datasets/urgent-challenge/urgent26_track2_sqa/resolve/main/ttsds2.zip">[Huggingface]</a>
    </td>
    <td>MIT</td>
  </tr>
  <tr>
    <td>urgent2024-sqa<d-cite key="UniVERSAExt"/><d-cite key="P808-Sach2025"/><d-cite key="URGENT-Zhang2024"/></td>
    <td>238</td>
    <td>238000</td>
    <td>429.34</td>
    <td>
      <a href="https://huggingface.co/datasets/urgent-challenge/urgent2024-sqa">[Huggingface]</a>
    </td>
    <td>CC BY-NC-SA 4.0</td>
  </tr>
  <tr>
    <td>urgent2025-sqa<d-cite key="UniVERSAExt"/><d-cite key="P808-Sach2025"/><d-cite key="Interspeech2025-Saijo2025"/></td>
    <td>100000</td>
    <td>100</td>
    <td>261.31</td>
    <td>
      <a href="https://huggingface.co/datasets/urgent-challenge/urgent2025-sqa">[Huggingface]</a>
    </td>
    <td>CC BY-NC-SA 4.0</td>
  </tr>
  <tr>
    <td rowspan="1">Dev</td>
    <td>CHiME-7 UDASE Eval<d-cite key="CHiME-7-UDASE-Eval"/></td>
    <td>640</td>
    <td>5</td>
    <td>0.84</td>
    <td>
      <a href="https://zenodo.org/records/10418311/files/CHiME-7-UDASE-evaluation-data.zip">[Original]</a>
      <a href="https://huggingface.co/datasets/urgent-challenge/urgent26_track2_sqa/resolve/main/CHiME-7-UDASE-evaluation-data.zip">[Huggingface]</a>
    </td>
    <td>CC BY-SA 4.0</td>
  </tr>
  <tr>
    <td rowspan="1">Test</td>
    <td>urgent2024-sqa (blind_test_mos)<d-cite key="UniVERSAExt"/><d-cite key="P808-Sach2025"/><d-cite key="URGENT-Zhang2024"/></td>
    <td>6900</td>
    <td>23</td>
    <td>13.80</td>
    <td>
      <a href="https://huggingface.co/datasets/urgent-challenge/urgent2025-sqa">[Huggingface]</a>
    </td>
    <td>CC BY-NC-SA 4.0</td>
  </tr>
</tbody>
</table>

---


## Build Your Own Multi-Metric Dataset
> 🚧 **Under Construction**  


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

@inproceedings{Interspeech2025-Saijo2025,
  title={Interspeech 2025 {URGENT} Speech Enhancement Challenge},
  author={Saijo, Kohei and Zhang, Wangyou and Cornell, Samuele and Scheibler, Robin and Li, Chenda and Ni, Zhaoheng and Kumar, Anurag and Sach, Marvin and Fu, Yihui and Wang, Wei and Fingscheidt, Tim and Watanabe, Shinji},
  booktitle={Proc. Interspeech},
  pages={858--862},
  year={2025},
}

@inproceedings{URGENT-Zhang2024,
  title={{URGENT} Challenge: Universality, Robustness, and Generalizability For Speech Enhancement},
  author={Zhang, Wangyou and Scheibler, Robin and Saijo, Kohei and Cornell, Samuele and Li, Chenda and Ni, Zhaoheng and Pirklbauer, Jan and Sach, Marvin and Watanabe, Shinji and Fingscheidt, Tim and Qian, Yanmin},
  booktitle={Proc. Interspeech},
  pages={4868--4872},
  year={2024}
}

@article{P808-Sach2025,
  title={P.808 Multilingual Speech Enhancement Testing: Approach and Results of {URGENT} 2025 Challenge},
  author={Sach, Marvin and Fu, Yihui and Saijo, Kohei and Zhang, Wangyou and Cornell, Samuele and Scheibler, Robin and Li, Chenda and Kumar, Anurag and Wang, Wei and Qian, Yanmin and Watanabe, Shinji and Fingscheidt, Tim},
  journal={arXiv preprint arXiv:2507.11306},
  year={2025}
}
```