"""
CLIP-based Filter Module
생성된 synthetic 이미지 중 quality 낮은 것들을 걸러내기

핵심 아이디어 (Fan et al. 2024 - Recognizability):
  - CLIP으로 이미지와 클래스 텍스트 간 similarity 계산
  - 점수가 낮은 이미지 = 해당 클래스처럼 보이지 않는 이미지 → 제거
  - class별로 poor class 판별 가능

"""

from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torchvision import transforms

logger = logging.getLogger(__name__)


@dataclass
class FilterStats:
    """Filter 결과 통계."""
    total_before: int = 0
    total_after: int = 0
    removed: int = 0
    removal_rate: float = 0.0
    # class별 평균 CLIP score
    class_avg_scores: Dict[str, float] = field(default_factory=dict)
    # CLIP score 낮은 poor class 목록
    poor_classes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"[CLIPFilter] {self.total_before:,} → {self.total_after:,} "
            f"(제거: {self.removed:,}, {self.removal_rate:.1f}%)",
            f"  Poor classes ({len(self.poor_classes)}개): "
            f"{', '.join(self.poor_classes[:10])}{'...' if len(self.poor_classes) > 10 else ''}",
        ]
        return "\n".join(lines)


class CLIPFilter:
    """
    CLIP score 기반 이미지 품질 필터.

    Args:
        device: 'cuda' or 'cpu'
        clip_model_name: 사용할 CLIP 모델
    """

    # CLIP 입력 전처리 (224x224, ImageNet normalize)
    _PREPROCESS = transforms.Compose([
        transforms.Resize(224, antialias=True),
        transforms.CenterCrop(224),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    def __init__(
        self,
        device: str = "cuda",
        clip_model_name: str = "ViT-B/32",
    ):
        self.device = device
        self._load_clip(clip_model_name)

    def _load_clip(self, model_name: str):
        try:
            import clip
            self.model, _ = clip.load(model_name, device=self.device)
            self.model.eval()
            self._tokenize = clip.tokenize
            logger.info("CLIP loaded: %s", model_name)
        except ImportError:
            raise ImportError(
                "CLIP이 설치되어 있지 않습니다.\n"
                "pip install git+https://github.com/openai/CLIP.git"
            )

    @torch.no_grad()
    def compute_scores(
        self,
        samples: List[Tuple[torch.Tensor, int]],
        class_names: List[str],
        batch_size: int = 256,
    ) -> torch.Tensor:
        """
        각 이미지에 대해 해당 클래스와의 CLIP similarity 계산.

        Returns:
            scores: [N] float tensor, range [-1, 1]
        """
        # 클래스 텍스트 임베딩 미리 계산
        prompts = [f"a photo of a {name}" for name in class_names]
        text_tokens = self._tokenize(prompts).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)  # [C, D]

        all_scores = []
        for start in range(0, len(samples), batch_size):
            batch = samples[start: start + batch_size]
            from PIL import Image as PILImage

            imgs = []
            for s in batch:
                item = s[0]
                if isinstance(item, str):
                    img = PILImage.open(item).convert("RGB")
                    img = transforms.ToTensor()(img)
                else:
                    img = item
                imgs.append(img)
            images = torch.stack(imgs).to(self.device)
            labels = torch.tensor([s[1] for s in batch], dtype=torch.long)

            # 이미지가 [0,1] range인지 확인 후 전처리
            if images.max() > 1.0:
                images = images / 255.0
            images = self._PREPROCESS(images)

            img_features = self.model.encode_image(images)
            img_features = F.normalize(img_features, dim=-1)  # [B, D]

            # 각 이미지와 해당 클래스 텍스트 similarity
            # text_features[labels] → [B, D]
            class_text = text_features[labels]
            scores = (img_features * class_text).sum(dim=-1)  # [B]
            all_scores.append(scores.cpu())

        return torch.cat(all_scores)

    def filter(
        self,
        samples: List[Tuple[torch.Tensor, int]],
        class_names: List[str],
        threshold: float = 0.2,
        poor_class_threshold: float = 0.25,
        batch_size: int = 256,
    ) -> Tuple[List[Tuple[torch.Tensor, int]], FilterStats]:
        """
        CLIP score 기반으로 quality 낮은 이미지 제거.

        Args:
            samples: List of (image_tensor, class_idx)
            class_names: 클래스 이름 목록
            threshold: 이미지 단위 필터 기준 (이 값 미만 제거)
            poor_class_threshold: class 평균 score가 이 값 미만이면 poor class로 분류
            batch_size: CLIP 추론 배치 크기

        Returns:
            filtered_samples: 필터링된 샘플
            stats: FilterStats (통계 및 poor class 목록)
        """
        logger.info(
            "CLIPFilter 시작: %s개 이미지, threshold=%.2f",
            f"{len(samples):,}", threshold
        )

        scores = self.compute_scores(samples, class_names, batch_size)

        # 이미지 단위 필터
        keep_mask = scores >= threshold
        filtered_samples = [s for s, keep in zip(samples, keep_mask) if keep.item()]

        # class별 통계
        class_scores: Dict[int, List[float]] = {}
        for (_, label), score in zip(samples, scores):
            class_scores.setdefault(label, []).append(score.item())

        class_avg: Dict[str, float] = {}
        poor_classes: List[str] = []
        for idx, score_list in class_scores.items():
            if 0 <= idx < len(class_names):
                name = class_names[idx]
                avg = sum(score_list) / len(score_list)
                class_avg[name] = round(avg, 4)
                if avg < poor_class_threshold:
                    poor_classes.append(name)

        # poor class를 평균 score 낮은 순으로 정렬
        poor_classes.sort(key=lambda n: class_avg.get(n, 0))

        stats = FilterStats(
            total_before=len(samples),
            total_after=len(filtered_samples),
            removed=len(samples) - len(filtered_samples),
            removal_rate=(len(samples) - len(filtered_samples)) / max(1, len(samples)) * 100,
            class_avg_scores=class_avg,
            poor_classes=poor_classes,
        )

        logger.info(stats.summary())
        return filtered_samples, stats

    def save_stats(self, stats: FilterStats, save_path: str):
        """Filter 통계를 JSON으로 저장."""
        import json
        data = {
            "total_before": stats.total_before,
            "total_after": stats.total_after,
            "removed": stats.removed,
            "removal_rate": stats.removal_rate,
            "poor_classes": stats.poor_classes,
            "class_avg_scores": stats.class_avg_scores,
        }
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Filter 통계 저장: %s", save_path)