# AIC Bilinear CNN

FinalPolicy `ALIGN` 단계용 6D correction regression 실험 코드.

## 📦 Dataset

| Item | Value |
|---|---|
| Label | `[x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad]` |
| Frame | `base_link` |
| Views | `left`, `center`, `right` |
| Connectors | `SFP`, `SC` |
| Rule | connector를 섞지 않고 connector별로 별도 학습 |

<br>

학습 전 데이터셋을 먼저 다운로드:

```bash
python3 download_dataset.py \
  --dataset-root data/vision_offset_dataset \
  --dataset-hf-repo-id aic-sejong-team/aic-vision-offset-dataset \
  --connectors all
```

| `--connectors` | 동작 |
|---|---|
| `all` | `SFP`, `SC`를 각각 검증/학습 |
| `SFP` | `SFP`만 사용 |
| `SC` | `SC`만 사용 |

## 🧠 Models

Input/output: 3-view image -> [x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad]

### Architecture

| Model | Backbone | View fusion |
|---|---|---|
| `simple_cnn` | shared timm backbone | mean feature map -> global average pooling |
| `shared_bilinear` | shared timm backbone | mean feature map -> one bilinear descriptor |
| `multiview_bilinear` | shared by default, optional per-view backbone | per-view bilinear descriptors -> concat |

### Options

| Option | Default | Scope | Description |
|---|---:|---|---|
| `--model` | required | selection | `simple_cnn`, `shared_bilinear`, `multiview_bilinear`, `all` |
| `--backbone-name` | `efficientnetv2_rw_s` | all | timm backbone |
| `--pretrained` | `true` | all | `true`/`false`; timm pretrained weights |
| `--feature-dim` | `128` | all | projected feature channels |
| `--share-backbone-weights` | `true` | `multiview_bilinear` | `true`/`false`; view 간 backbone 공유 |
| `--image-size` | `224` | all | input resize size |
| `--connectors` | `all` | dataset | `all`, `SFP`, `SC` 중 하나 |

| Backbone mode | When to use |
|---|---|
| shared backbone | 데이터가 적거나 GPU memory를 아끼면서 view 공통 feature를 학습할 때 |
| independent backbone | view별 시점/왜곡/occlusion 차이가 크고 데이터와 memory가 충분할 때 |

Boolean 값은 `true`/`false`, `1`/`0`, `yes`/`no`로 입력.
예: `--pretrained=false --share-backbone-weights=false`.

## 🚀 Train

### Setup

```bash
git clone https://github.com/whyz-dev/structure-stability.git
cd bilinear-cnn
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf auth login
```

### Full Run

```bash
python3 train.py \
  --model all \
  --connectors all \
  --backbone-name efficientnetv2_rw_s \
  --dataset-root data/vision_offset_dataset \
  --output-dir checkpoints
```

`--model all --connectors all` 출력 구조:

```text
checkpoints/
  SFP/
    simple_cnn/
      simple_cnn_best.pt
      training_summary.json
      loss_history.csv
      loss_curve.png
    shared_bilinear/
    multiview_bilinear/
  SC/
    simple_cnn/
    shared_bilinear/
    multiview_bilinear/
  model_comparison.csv
  model_comparison.json
```

### Single Runs

```bash
python3 train.py \
  --model simple_cnn \
  --connectors SFP \
  --dataset-root data/vision_offset_dataset \
  --output-dir checkpoints_sfp

python3 train.py \
  --model shared_bilinear \
  --connectors SC \
  --dataset-root data/vision_offset_dataset \
  --output-dir checkpoints_sc
```

### Training Controls

| Topic | Option | Default | Effect |
|---|---|---:|---|
| Early stopping | `--early-stopping-patience` | `20` | 개선 없는 epoch 수가 patience에 도달하면 중단 |
| Disable early stopping | `--early-stopping-patience 0` | - | patience 기반 중단 끔 |
| LR | `--lr` | `1e-4` | AdamW learning rate |
| Batch size | `--batch-size` | `8` | mini-batch size |
| Epochs | `--epochs` | `200` | 최대 학습 epoch |
| Skip done runs | `--skip-existing` | `False` | checkpoint와 summary가 있으면 재학습 생략 |

### Loss

| Component | Formula | Default |
|---|---|---:|
| xyz loss | `SmoothL1(xyz_error / xyz_loss_scale)` | `10mm` |
| rpy loss | `SmoothL1(rpy_error / rpy_loss_scale)` | `1deg` |
| total loss | `xyz_loss_weight * xyz_loss + rpy_loss_weight * rpy_loss` | `2 * xyz + 1 * rpy` |

```bash
python3 train.py \
  --model all \
  --connectors all \
  --xyz-loss-scale-mm 10 \
  --rpy-loss-scale-deg 1 \
  --xyz-loss-weight 2 \
  --rpy-loss-weight 1 \
  --output-dir checkpoints
```

### Comparison

| File | Content |
|---|---|
| `model_comparison.csv` | connector/model별 MAE와 선택 점수 |
| `model_comparison.json` | sorted summary |
| `loss_history.csv` | epoch별 total/xyz/rpy train-val loss |
| `loss_curve.png` | total/xyz/rpy train-val loss curves |
| model `README.md` | connector, model config, best validation metrics |

```text
selection_score = mean_xyz_mae_mm + rpy_score_weight * mean_rpy_mae_deg
```

## 🤗 Hugging Face Upload

```bash
python3 train.py \
  --model all \
  --connectors all \
  --dataset-root data/vision_offset_dataset \
  --output-dir checkpoints \
  --push-to-hub \
  --hub-repo-id aic-sejong-team/aic-vision-offset-models
```

| Option | Default | Effect |
|---|---:|---|
| `--hub-repo-id` | `aic-sejong-team/aic-vision-offset-models` | upload target |
| `--hub-revision` | repo default | 보통 `main` branch에 새 commit |
| `--hub-path-in-repo` | `.` | repo root에 `--output-dir` 전체 업로드 |
| `--hub-private` | `False` | repo private 생성/유지 |

| Upload behavior | Effect |
|---|---|
| Same remote path | 새 파일로 갱신 |
| Remote-only file | 자동 삭제 없음 |

## ☁️ Kaggle

| Item | Value |
|---|---|
| Dataset root | `/kaggle/working/vision_offset_dataset` |
| Output dir | `/kaggle/working/aic_vision_offset_checkpoints` |
| Required internet | HF dataset, timm pretrained weights |
| HF token | Kaggle Secret `HF_TOKEN` |

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

```bash
python download_dataset.py \
  --dataset-root /kaggle/working/vision_offset_dataset \
  --dataset-hf-repo-id aic-sejong-team/aic-vision-offset-dataset \
  --connectors all

python train.py \
  --model all \
  --connectors all \
  --dataset-root /kaggle/working/vision_offset_dataset \
  --dataset-hf-repo-id aic-sejong-team/aic-vision-offset-dataset \
  --output-dir /kaggle/working/aic_vision_offset_checkpoints \
  --batch-size 8 \
  --epochs 50
```

## 🗺️ Diagrams

```text
diagrams/simple_cnn.drawio
diagrams/shared_bilinear_cnn.drawio
diagrams/multiview_bilinear_cnn.drawio
```
