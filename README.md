
# URGENT 2026 challenge (Track 2)

This track focuses on predicting the Mean Opinion Score (MOS) of speech processed by **speech enhancement systems**.

- Table of Contents
  - [Inference using pretrained checkpoints](#inference)




## installation

```bash
conda create -n python=3.11 urgent26_sqa
conda activate urgent26_sqa
pip install -r requirements
```

## Inference with pretraiend checkpoints

```bash
hf download urgent/urgent26_track2
```

# 


### run data preparation




## Build Your Own Multi-Metric Dataset

```bash
cd && pip install -r requirements.txt
```

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
    <td>URGENT2024-SQA<d-cite key="UniVERSAExt"/><d-cite key="P808-Sach2025"/><d-cite key="URGENT-Zhang2024"/></td>
    <td>238</td>
    <td>238000</td>
    <td>429.34</td>
    <td>
      <a href="https://huggingface.co/datasets/urgent-challenge/urgent2024-sqa">[Huggingface]</a>
    </td>
    <td>CC BY-NC-SA 4.0</td>
  </tr>
  <tr>
    <td>URGENT2025-SQA<d-cite key="UniVERSAExt"/><d-cite key="P808-Sach2025"/><d-cite key="Interspeech2025-Saijo2025"/></td>
    <td>100000</td>
    <td>100</td>
    <td>261.31</td>
    <td>
      <a href="https://huggingface.co/datasets/urgent-challenge/urgent2025-sqa">[Huggingface]</a>
    </td>
    <td>CC BY-NC-SA 4.0</td>
  </tr>
  <tr>
    <td rowspan="1">Development</td>
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
    <td rowspan="1">Evaluation</td>
    <td>URGENT2024-SQA</td>
    <td>23</td>
    <td>6900</td>
    <td>13.80</td>
    <td>
      <a href="https://huggingface.co/datasets/urgent-challenge/urgent26_track2_sqa/resolve/main/CHiME-7-UDASE-evaluation-data.zip">[Huggingface]</a>
    </td>
    <td>CC BY-NC-SA 4.0</td>
  </tr>
</tbody>
</table>

## Evaluation

<table class="tg">
<thead>
<tr>
    <th class="tg-uzvj">Category</th>
    <th class="tg-g7sd">Metric</th>
    <th class="tg-uzvj">Value Range</th>
</tr>
</thead>
<tbody>
<tr>
    <td class="tg-r6l2" rowspan="2">Error</td>
    <td class="tg-rt8k">System level MSE ↓</td>
    <td class="tg-51oy">[0, ∞)</td>
</tr>
<tr>
    <td class="tg-rt8k">Utterance level MSE ↓</td>
    <td class="tg-51oy">[0, ∞)</td>
</tr>
<tr>
    <td class="tg-r6l2" rowspan="2">Linear Correlation</td>
    <td class="tg-rt8k"> System level LCC ↑</td>
    <td class="tg-51oy">[-1, 1]</td>
</tr>
<tr>
    <td class="tg-0a7q">Utterance level LCC ↑</td>
    <td class="tg-51oy">[-1, 1]</td>
</tr>
<tr>
    <td class="tg-r6l2" rowspan="4">Rank Correlation</td>
    <td class="tg-rt8k"> System level SRCC ↑</td>
    <td class="tg-51oy">[-1, 1]</td>
</tr>
<tr>
    <td class="tg-0a7q">Utterance level SRCC ↑</td>
    <td class="tg-51oy">[-1, 1]</td>
</tr>
<tr>
    <td class="tg-rt8k"> System level KTAU ↑</td>
    <td class="tg-51oy">[-1, 1]</td>
</tr>
<tr>
    <td class="tg-0a7q">Utterance level KTAU ↑</td>
    <td class="tg-51oy">[-1, 1]</td>
</tr>

</tbody>
</table>
