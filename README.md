# clip-data-curation

> **CLIP Score-based Automatic Quality Filtering and Regeneration for Synthetic Image Datasets**

생성된 Synthetic 이미지의 품질을 CLIP Score로 자동 평가하고, 낮은 품질의 이미지를 걸러내며 Poor class를 LLM 기반으로 자동 재생성하는 데이터 큐레이션 모듈입니다.

---

## Overview

생성 모델로 만든 Synthetic 이미지는 품질이 균일하지 않습니다. 특정 class는 생성 자체가 잘 되지 않아 학습에 오히려 방해가 되기도 합니다.

이 모듈은 두 가지 문제를 자동으로 해결합니다.

- **품질 낮은 이미지 자동 제거**: CLIP Score로 이미지-텍스트 유사도를 측정해 threshold 미만 이미지 제거
- **Poor class 자동 재생성**: class 평균 score가 낮은 Poor class를 LLM으로 다양한 관점의 프롬프트를 생성해 재생성

---

## Key Features

- **CLIP ViT-B/32** 기반 이미지-텍스트 유사도 자동 측정
- **threshold 탐색 실험**을 통해 최적값 채택 (0.28)
- **Poor class 자동 식별**: class 평균 score 기준으로 분류
- **LLM 프롬프트 다양화**: 스타일 / 배경 / 디테일 / 색상 / 상황 5가지 관점으로 자동 생성 (Ollama/qwen3)
- 시각화 결과를 통한 **직관적인 품질 분석**

---

## Results

ImageNet-100 기반 Synthetic 데이터셋 적용 결과

| | 수치 |
|---|---|
| 전체 이미지 | 130,000개 |
| 제거된 이미지 | 28,830개 (22%) |
| Poor class 식별 | 17개 |
| 재생성 후 Poor class | 12개 |

<img width="1724" height="846" alt="image" src="https://github.com/user-attachments/assets/3d29fc7d-bce2-4e5c-9da1-2fdf289f6363" />

<img width="1731" height="849" alt="image" src="https://github.com/user-attachments/assets/7ca8bbc5-91c8-48d3-8f9e-280fc7504fc6" />
<img width="1726" height="775" alt="image" src="https://github.com/user-attachments/assets/3738011e-3a75-42c0-b079-89e04d37c03b" />
<img width="1731" height="777" alt="image" src="https://github.com/user-attachments/assets/8a91823a-0a9d-4d5c-8af8-9b0dc4d822ff" />



---

## File Structure

```
clip-data-curation/
├── clip_filter.py              # CLIP Score 기반 품질 필터
├── visualize_clip_filter.py    # 필터링 결과 시각화
├── regenerate_poor_classes.py  # LLM 기반 Poor class 재생성
├── requirements.txt
└── README.md
```

---

## Requirements

```bash
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install diffusers transformers requests pillow matplotlib
```

Ollama 설치 및 모델 실행
```bash
ollama run qwen3:4b-instruct
```

---

## References

- Fan et al. *Scaling Laws of Synthetic Images for Model Training* (2024)
- Radford et al. *Learning Transferable Visual Models From Natural Language Supervision* (ICML 2021)
