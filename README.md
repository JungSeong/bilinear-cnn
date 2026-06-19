# AIC Bilinear CNN

PortOffsetCollect가 생성한 `vision_offset_dataset`으로 FinalPolicy ALIGN 단계에서 사용할 6D correction 모델을 학습하는 실험 코드

## Dataset

기본 입력 데이터셋:

```text
/home/swlinux/Desktop/workspace/AIC_Sejong/data/vision_offset_dataset
```

현재 loader는 다음 구조를 읽습니다.

```text
images/<split>/<connector>/<camera>/*.jpg
metadata/<split>/<connector>/<camera>/*.json
```

로더는 먼저 `--dataset-root`의 로컬 데이터셋을 검증합니다. 요청한
`split/connectors/cameras`에 대한 metadata가 없으면 HuggingFace dataset repo에서
같은 위치로 내려받습니다.

```text
aic-sejong-team/aic-vision-offset-dataset
```

다른 repo나 revision을 쓰려면 학습 시 다음 옵션을 바꿉니다.

```bash
python3 train.py \
  --model simple_cnn \
  --dataset-hf-repo-id aic-sejong-team/aic-vision-offset-dataset \
  --dataset-hf-revision main
```

private repo라면 먼저 `hf auth login`이 필요합니다. `HF_TOKEN` 환경변수가 설정되어 있으면
CLI 로그인보다 우선 적용됩니다.

label 순서는 모든 모델에서 동일합니다.

```python
[x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad]
```

이 값은 `base_link` 기준 correction입니다.

## Models

| Name | File class | Input | Feature handling | Output |
|---|---|---|---|---|
| `simple_cnn` | `SimpleCNNRegressor` | left/center/right group | shared timm backbone, mean feature, global average pooling | 6D correction |
| `shared_bilinear` | `SharedBilinearCNNRegressor` | left/center/right group | shared timm backbone, mean feature, one bilinear outer product | 6D correction |
| `multiview_bilinear` | `MultiViewBilinearCNNRegressor` | left/center/right group | view-specific timm backbones, bilinear outer product per view, concat | 6D correction |

## Requirements

```bash
pip install -r /home/swlinux/Desktop/cool-library/bilinear-cnn/ais_bilinear-cnn/requirements.txt
```

`--pretrained`가 기본값이라 첫 실행 시 timm 백본 weight 다운로드가 필요할 수 있습니다.
네트워크 없이 구조만 검증하거나 완전 scratch timm 백본으로 학습하려면 `--no-pretrained`를 붙입니다.

## Train

```bash
cd /home/swlinux/Desktop/cool-library/bilinear-cnn/ais_bilinear-cnn

python3 train.py \
  --model simple_cnn \
  --backbone-name efficientnetv2_rw_s \
  --dataset-root /home/swlinux/Desktop/workspace/AIC_Sejong/data/vision_offset_dataset \
  --output-dir checkpoints

python3 train.py \
  --model shared_bilinear \
  --backbone-name efficientnetv2_rw_s \
  --dataset-root /home/swlinux/Desktop/workspace/AIC_Sejong/data/vision_offset_dataset \
  --output-dir checkpoints

python3 train.py \
  --model multiview_bilinear \
  --backbone-name efficientnetv2_rw_s \
  --dataset-root /home/swlinux/Desktop/workspace/AIC_Sejong/data/vision_offset_dataset \
  --output-dir checkpoints
```

## Kaggle

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
pip install -r /kaggle/working/bilinear-cnn/ais_bilinear-cnn/requirements.txt
```

학습 예시:

```bash
cd /kaggle/working/bilinear-cnn/ais_bilinear-cnn

python train.py \
  --model multiview_bilinear \
  --dataset-root /kaggle/working/vision_offset_dataset \
  --dataset-hf-repo-id aic-sejong-team/aic-vision-offset-dataset \
  --output-dir /kaggle/working/aic_vision_offset_checkpoints \
  --batch-size 8 \
  --epochs 50
```

## Upload Model To HuggingFace

학습 종료 후 best checkpoint, `training_summary.json`, model card를 HuggingFace model repo로
올리려면 `--push-to-hub`를 추가합니다.

```bash
python train.py \
  --model multiview_bilinear \
  --dataset-root /kaggle/working/vision_offset_dataset \
  --output-dir /kaggle/working/aic_vision_offset_checkpoints \
  --push-to-hub \
  --hub-repo-id aic-sejong-team/aic-vision-offset-models \
  --hub-private
```

repo가 없으면 `create_repo(..., exist_ok=True)`로 생성합니다. private repo에 올리려면
토큰에 write 권한이 있어야 합니다.

## Diagrams

```text
diagrams/simple_cnn.drawio
diagrams/shared_bilinear_cnn.drawio
diagrams/multiview_bilinear_cnn.drawio
```
