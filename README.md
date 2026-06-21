# AIC Bilinear CNN

AI for Industry Challenge의 FinalPolicy 노드의 ALIGN 단계에서 사용할 6D correction 모델을 학습하는 실험 코드

## Dataset

label은 아래와 같으며, `base_link` 좌표계 기준으로 포트에 정렬되기 위한 offset을 의미합니다.

```python
[x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad]
```

## Models

| Name | File class | Input | Feature handling | Output |
|---|---|---|---|---|
| `simple_cnn` | `SimpleCNNRegressor` | left/center/right group | shared timm backbone, mean feature, global average pooling | 6D correction |
| `shared_bilinear` | `SharedBilinearCNNRegressor` | left/center/right group | shared timm backbone, mean feature, one bilinear outer product | 6D correction |
| `multiview_bilinear` | `MultiViewBilinearCNNRegressor` | left/center/right group | view-specific timm backbones, bilinear outer product per view, concat | 6D correction |

## 시작하기

### 1. 가상환경 설정
```bash
git clone https://github.com/whyz-dev/structure-stability.git
cd bilinear-cnn
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. HuggingFace 토큰 설정
```bash
hf auth login
hf auth whoami
```

### 3. Train
`--pretrained`가 기본값이며, 첫 실행 시 timm 백본 weight 다운로드가 필요할 수 있습니다.
완전 scratch timm 백본으로 학습하려면 `--no-pretrained`를 붙입니다.

```bash
cd bilinear-cnn

python3 train.py \
  --model all \
  --backbone-name efficientnetv2_rw_s \
  --dataset-root data/vision_offset_dataset \
  --output-dir checkpoints
```

`--model all`은 아래 3개 모델을 순서대로 학습하고 `checkpoints/` 아래에 모델별 폴더를 만듭니다.

```text
checkpoints/
  simple_cnn/
    simple_cnn_best.pt
    training_summary.json
  shared_bilinear/
    shared_bilinear_best.pt
    training_summary.json
  multiview_bilinear/
    multiview_bilinear_best.pt
    training_summary.json
  model_comparison.csv
  model_comparison.json
```

EarlyStopping은 기본으로 켜져 있습니다. validation loss가 `--early-stopping-patience`
epoch 동안 개선되지 않으면 해당 모델 학습을 중단합니다.

```bash
python3 train.py \
  --model all \
  --early-stopping-patience 20 \
  --early-stopping-min-delta 0.0 \
  --output-dir checkpoints
```

EarlyStopping을 끄려면 patience를 `0`으로 둡니다.

이미 학습된 모델을 건너뛰고 비교표만 다시 만들고 싶다면 `--skip-existing`을 추가합니다.

`model_comparison.csv`에는 축별 MAE와 함께 최종 선택용 컬럼이 같이 저장됩니다.

```text
selection_score = mean_xyz_mae_mm + rpy_score_weight * mean_rpy_mae_deg
```

기본 `rpy_score_weight`는 `1.0`입니다. 위치 오차를 더 우선하고 싶으면 작은 값을 줄 수 있습니다.

```bash
python3 train.py \
  --model all \
  --rpy-score-weight 0.5 \
  --output-dir checkpoints \
  --skip-existing
```

개별 모델만 학습할 수도 있습니다.

```bash
python3 train.py \
  --model simple_cnn \
  --backbone-name efficientnetv2_rw_s \
  --dataset-root data/vision_offset_dataset \
  --output-dir checkpoints

python3 train.py \
  --model shared_bilinear \
  --backbone-name efficientnetv2_rw_s \
  --dataset-root data/vision_offset_dataset \
  --output-dir checkpoints

python3 train.py \
  --model multiview_bilinear \
  --backbone-name efficientnetv2_rw_s \
  --dataset-root data/vision_offset_dataset \
  --output-dir checkpoints
```

### Upload Model To HuggingFace

학습 종료 후 best checkpoint, `training_summary.json`, model card를 HuggingFace model repo로
올리려면 `--push-to-hub`를 추가합니다.
기본 업로드 대상 repo는 `aic-sejong-team/aic-vision-offset-models`입니다.

```bash
python train.py \
  --model all \
  --dataset-root /kaggle/working/vision_offset_dataset \
  --output-dir /kaggle/working/aic_vision_offset_checkpoints \
  --push-to-hub \
  --hub-private
```

repo가 없으면 `create_repo(..., exist_ok=True)`로 생성합니다. private repo에 올리려면
토큰에 write 권한이 있어야 합니다.

### Kaggle

Kaggle Notebook에서도 실행 가능합니다. 단, 기본 경로는 로컬 PC 기준이므로
`--dataset-root`와 `--output-dir`는 Kaggle 경로로 지정하는 것을 권장합니다.
Internet이 켜져 있어야 HuggingFace dataset과 timm pretrained weight를 받을 수 있습니다.

Kaggle Secrets에 `HF_TOKEN`을 저장했다면 첫 셀에서 환경변수로 넘깁니다.

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

의존성 설치:

```bash
pip install -q -r /kaggle/working/bilinear-cnn/ais_bilinear-cnn/requirements-kaggle.txt
```

Kaggle에는 이미 `torch`, `torchvision`, `numpy`, `pillow`가 설치되어 있습니다.
전체 `requirements.txt`를 설치하면 Kaggle의 CUDA/RAPIDS 패키지와 충돌 경고가 날 수 있으므로
Kaggle에서는 `requirements-kaggle.txt`를 사용합니다.

학습 예시:

```bash
cd /kaggle/working/bilinear-cnn/ais_bilinear-cnn

python train.py \
  --model all \
  --dataset-root /kaggle/working/vision_offset_dataset \
  --dataset-hf-repo-id aic-sejong-team/aic-vision-offset-dataset \
  --output-dir /kaggle/working/aic_vision_offset_checkpoints \
  --batch-size 8 \
  --epochs 50
```

## Diagrams

```text
diagrams/simple_cnn.drawio
diagrams/shared_bilinear_cnn.drawio
diagrams/multiview_bilinear_cnn.drawio
```
