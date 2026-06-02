"""
ResNet-50 학습 스크립트
========================
Real dataset과 Synthetic dataset 성능 비교용

사용 예시:
  # Real dataset으로 학습
  python train_resnet.py \
    --mode real \
    --data_root ./dataset/data \
    --output_dir ./outputs/exp_real_resnet

  # Synthetic dataset으로 학습
  python train_resnet.py \
    --mode synthetic \
    --synth_data_root ./generate_init_synthetic/data/synth_initial_v2 \
    --data_root ./dataset/data \
    --output_dir ./outputs/exp_synth_resnet

  # Synthetic + CLIP Filter 적용
  python train_resnet.py \
    --mode synthetic \
    --synth_data_root ./generate_init_synthetic/data/synth_initial_v2 \
    --data_root ./dataset/data \
    --output_dir ./outputs/exp_synth_filter_resnet \
    --use_clip_filter
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── 전처리 ────────────────────────────────────────────────────────

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Dataset ────────────────────────────────────────────────────────

class PathDataset(Dataset):
    """
    (image_path, label) 리스트를 받아서 디스크에서 읽는 Dataset.
    OOM 방지용 - 이미지를 미리 RAM에 올리지 않음.
    """

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


class HFParquetDataset(Dataset):
    """
    HF parquet 형식의 real dataset을 로드하는 Dataset.
    """

    def __init__(self, hf_split, transform=None):
        self.ds = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        image_obj = ex["image"]
        label = int(ex["label"])

        if isinstance(image_obj, dict):
            if image_obj.get("bytes") is not None:
                image = Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")
            elif image_obj.get("path") is not None:
                image = Image.open(image_obj["path"]).convert("RGB")
            else:
                raise ValueError("Unsupported image dict format.")
        else:
            image = image_obj.convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# ── 데이터 로드 함수 ──────────────────────────────────────────────

def load_real_dataset(data_root: str, batch_size: int = 64, num_workers: int = 4):
    """
    Real dataset (parquet) 로드.
    train + validation 둘 다 반환.
    """
    import datasets as hf_datasets
    import pyarrow.parquet as pq

    root = Path(data_root)

    # Train
    train_shards = sorted(root.glob("train-*.parquet"))
    if not train_shards:
        raise FileNotFoundError(f"train parquet 파일을 찾을 수 없습니다: {root}")

    splits = [
        hf_datasets.Dataset(pq.read_table(str(p), columns=["image", "label"]))
        for p in train_shards
    ]
    hf_train = hf_datasets.concatenate_datasets(splits)
    label_feature = hf_train.features.get("label", None)
    class_names = getattr(label_feature, "names", None)
    if class_names is None:
        n_cls = int(max(hf_train["label"])) + 1
        class_names = [str(i) for i in range(n_cls)]

    train_ds = HFParquetDataset(hf_train, transform=TRAIN_TRANSFORM)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    logger.info("Real train: %s samples, %s classes", f"{len(train_ds):,}", len(class_names))

    # Validation
    val_shards = sorted(root.glob("validation-*.parquet"))
    val_splits = [
        hf_datasets.Dataset(pq.read_table(str(p), columns=["image", "label"]))
        for p in val_shards
    ]
    hf_val = hf_datasets.concatenate_datasets(val_splits)
    val_ds = HFParquetDataset(hf_val, transform=VAL_TRANSFORM)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    logger.info("Real val: %s samples", f"{len(val_ds):,}")

    return class_names, train_loader, val_loader


def _normalize_class_key(name: str) -> str:
    text = name.strip().lower().replace("_", " ")
    for ch in [",", ";", "/", "(", ")", "[", "]", "{", "}"]:
        text = text.replace(ch, " ")
    return " ".join(text.split())


def load_synthetic_dataset(
    synth_data_root: str,
    class_names: List[str],
    batch_size: int = 64,
    num_workers: int = 4,
    use_clip_filter: bool = False,
    clip_threshold: float = 0.20,
    poor_class_threshold: float = 0.25,
    device: str = "cuda",
):
    """
    Synthetic dataset (ImageFolder) 로드.
    경로만 저장하고 디스크에서 읽는 방식 (OOM 방지).
    """
    root = Path(synth_data_root)
    train_root = root / "train" if (root / "train").is_dir() else root

    folder_ds = datasets.ImageFolder(root=str(train_root))
    folder_idx_to_class = {v: k for k, v in folder_ds.class_to_idx.items()}

    # class name 매핑
    alias_to_idx: Dict[str, int] = {}
    for idx, name in enumerate(class_names):
        raw = name.strip()
        for alias in [
            raw.lower(),
            raw.lower().replace("_", " "),
            _normalize_class_key(raw),
            raw.split(",")[0].strip().lower(),
            _normalize_class_key(raw.split(",")[0].strip()),
        ]:
            if alias:
                alias_to_idx[alias] = idx

    folder_to_val_idx: Dict[int, int] = {}
    for folder_idx, folder_name in folder_idx_to_class.items():
        mapped = None
        for cand in [
            folder_name.strip().lower(),
            folder_name.strip().lower().replace("_", " "),
            _normalize_class_key(folder_name),
            folder_name.split(",")[0].strip().lower(),
            _normalize_class_key(folder_name.split(",")[0].strip()),
        ]:
            mapped = alias_to_idx.get(cand)
            if mapped is not None:
                break
        if mapped is not None:
            folder_to_val_idx[folder_idx] = mapped

    # 경로만 수집 (이미지 로드 없음)
    samples: List[Tuple[str, int]] = []
    skipped = 0
    for img_path, folder_label in folder_ds.samples:
        val_label = folder_to_val_idx.get(folder_label)
        if val_label is None:
            skipped += 1
            continue
        samples.append((img_path, val_label))

    logger.info("Synthetic train: %s samples, skipped=%s", f"{len(samples):,}", skipped)

    # CLIP Filter 적용
    if use_clip_filter:
        from clip_filter import CLIPFilter
        logger.info("CLIPFilter 적용 중...")
        cf = CLIPFilter(device=device)
        samples, filter_stats = cf.filter(
            samples=samples,
            class_names=class_names,
            threshold=clip_threshold,
            poor_class_threshold=poor_class_threshold,
        )
        logger.info("Filter 완료: %s → %s", f"{filter_stats.total_before:,}", f"{filter_stats.total_after:,}")
        logger.info("Poor classes: %s", filter_stats.poor_classes)

    train_ds = PathDataset(samples, transform=TRAIN_TRANSFORM)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader


def load_val_loader(data_root: str, batch_size: int = 64, num_workers: int = 4):
    """Validation loader만 따로 로드."""
    import datasets as hf_datasets
    import pyarrow.parquet as pq

    root = Path(data_root)
    val_shards = sorted(root.glob("validation-*.parquet"))
    splits = [
        hf_datasets.Dataset(pq.read_table(str(p), columns=["image", "label"]))
        for p in val_shards
    ]
    hf_val = hf_datasets.concatenate_datasets(splits)
    label_feature = hf_val.features.get("label", None)
    class_names = getattr(label_feature, "names", None)
    if class_names is None:
        n_cls = int(max(hf_val["label"])) + 1
        class_names = [str(i) for i in range(n_cls)]

    val_ds = HFParquetDataset(hf_val, transform=VAL_TRANSFORM)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return list(class_names), val_loader


# ── 모델 ──────────────────────────────────────────────────────────

def build_resnet50(num_classes: int, device: str) -> nn.Module:
    """ResNet-50 생성 (pretrained=False, scratch 학습)."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


# ── Validation ────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, device) -> float:
    """Top-1 accuracy 계산."""
    model.eval()
    correct = total = 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        pred = model(images).argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return correct / total * 100


# ── 학습 루프 ─────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 90,
    base_lr: float = 0.1,
    device: str = "cuda",
    output_dir: str = "./outputs",
):
    """
    ResNet-50 학습.
    SGD + CosineAnnealingLR (ImageNet 표준 세팅)
    """
    os.makedirs(output_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=base_lr,
        momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    log = {"epoch": [], "loss": [], "val_top1": []}
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        scheduler.step()
        avg_loss = total_loss / total_samples
        val_acc = validate(model, val_loader, device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{output_dir}/best_model.pth")

        log["epoch"].append(epoch)
        log["loss"].append(round(avg_loss, 4))
        log["val_top1"].append(round(val_acc, 2))

        logger.info(
            "[Epoch %3d/%d] loss=%.4f | val=%.2f%% | best=%.2f%%",
            epoch, num_epochs, avg_loss, val_acc, best_acc
        )

    logger.info("Training complete. Best Val Top-1: %.2f%%", best_acc)

    with open(f"{output_dir}/train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    return model


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["real", "synthetic"],
        required=True,
        help="real: real dataset으로 학습 / synthetic: synthetic dataset으로 학습",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./dataset/data",
        help="Real dataset 경로 (validation에도 사용)",
    )
    parser.add_argument(
        "--synth_data_root",
        type=str,
        default=None,
        help="Synthetic dataset 경로 (mode=synthetic일 때 필요)",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs/exp_resnet")
    parser.add_argument("--num_epochs", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--base_lr", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    # CLIP Filter 관련 인자
    parser.add_argument(
        "--use_clip_filter",
        action="store_true",
        help="CLIP score 기반 Filter 적용 (mode=synthetic일 때만 유효)",
    )
    parser.add_argument(
        "--clip_threshold",
        type=float,
        default=0.20,
        help="이미지 단위 CLIP score 기준 (이 값 미만 제거)",
    )
    parser.add_argument(
        "--poor_class_threshold",
        type=float,
        default=0.25,
        help="class 평균 CLIP score 기준 (미만 = poor class로 분류)",
    )
    args = parser.parse_args()

    logger.info("Mode: %s | Device: %s", args.mode, args.device)

    # Validation loader (real data 공통 사용)
    class_names, val_loader = load_val_loader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = len(class_names)
    logger.info("Classes: %d", num_classes)

    # Train loader
    if args.mode == "real":
        _, train_loader, _ = load_real_dataset(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        if args.synth_data_root is None:
            raise ValueError("--synth_data_root 경로를 지정해주세요.")
        train_loader = load_synthetic_dataset(
            synth_data_root=args.synth_data_root,
            class_names=class_names,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_clip_filter=args.use_clip_filter,
            clip_threshold=args.clip_threshold,
            poor_class_threshold=args.poor_class_threshold,
            device=args.device,
        )

    # 모델 생성
    model = build_resnet50(num_classes=num_classes, device=args.device)
    logger.info("Model: ResNet-50 (scratch), classes=%d", num_classes)

    # 학습
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        base_lr=args.base_lr,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()