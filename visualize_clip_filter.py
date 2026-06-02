"""
CLIP Score 기반 Failure Case 시각화 스크립트

- CLIP Filter 적용 후 각 class별로
    - 점수 낮은 이미지 (필터링됨) vs 점수 높은 이미지 (통과)를 나란히 비교해서 matplotlib으로 저장.

"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # GUI 없이 저장
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from PIL import Image
from torchvision import datasets, transforms

import sys
sys.path.insert(0, os.path.dirname(__file__))


# 유틸 

def _normalize_class_key(name: str) -> str:
    text = name.strip().lower().replace("_", " ")
    for ch in [",", ";", "/", "(", ")", "[", "]", "{", "}"]:
        text = text.replace(ch, " ")
    return " ".join(text.split())


def load_class_names(data_root: str) -> List[str]:
    """parquet에서 class 이름 목록 로드."""
    import datasets as hf_datasets
    import pyarrow.parquet as pq

    root = Path(data_root)
    val_shards = sorted(root.glob("validation-*.parquet"))
    splits = [
        hf_datasets.Dataset(pq.read_table(str(p), columns=["label"]))
        for p in val_shards
    ]
    hf_val = hf_datasets.concatenate_datasets(splits)
    label_feature = hf_val.features.get("label", None)
    class_names = getattr(label_feature, "names", None)
    if class_names is None:
        n_cls = int(max(hf_val["label"])) + 1
        class_names = [str(i) for i in range(n_cls)]
    return list(class_names)


def load_synth_paths(
    synth_data_root: str,
    class_names: List[str],
) -> Dict[str, List[str]]:
    """
    class 이름 → 이미지 경로 목록 딕셔너리 반환.
    """
    root = Path(synth_data_root)
    train_root = root / "train" if (root / "train").is_dir() else root

    folder_ds = datasets.ImageFolder(root=str(train_root))
    folder_idx_to_class = {v: k for k, v in folder_ds.class_to_idx.items()}

    alias_to_name: Dict[str, str] = {}
    for name in class_names:
        raw = name.strip()
        for alias in [
            raw.lower(),
            raw.lower().replace("_", " "),
            _normalize_class_key(raw),
            raw.split(",")[0].strip().lower(),
            _normalize_class_key(raw.split(",")[0].strip()),
        ]:
            if alias:
                alias_to_name[alias] = name

    # folder_name → class_name 매핑
    folder_to_class_name: Dict[int, str] = {}
    for folder_idx, folder_name in folder_idx_to_class.items():
        mapped = None
        for cand in [
            folder_name.strip().lower(),
            folder_name.strip().lower().replace("_", " "),
            _normalize_class_key(folder_name),
            folder_name.split(",")[0].strip().lower(),
            _normalize_class_key(folder_name.split(",")[0].strip()),
        ]:
            mapped = alias_to_name.get(cand)
            if mapped is not None:
                break
        if mapped is not None:
            folder_to_class_name[folder_idx] = mapped

    # class_name → 경로 목록
    class_to_paths: Dict[str, List[str]] = {name: [] for name in class_names}
    for img_path, folder_label in folder_ds.samples:
        class_name = folder_to_class_name.get(folder_label)
        if class_name is not None:
            class_to_paths[class_name].append(img_path)

    return class_to_paths


def compute_clip_scores_for_paths(
    paths: List[str],
    class_name: str,
    clip_model,
    clip_preprocess,
    device: str,
    batch_size: int = 64,
) -> List[Tuple[str, float]]:
    """
    주어진 경로 목록에 대해 CLIP score 계산.
    Returns: List of (path, score)
    """
    import clip

    text = clip.tokenize([f"a photo of {class_name}"]).to(device)
    with torch.no_grad():
        text_feat = clip_model.encode_text(text)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    results = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i: i + batch_size]
        imgs = []
        valid_paths = []
        for p in batch_paths:
            try:
                img = clip_preprocess(Image.open(p).convert("RGB"))
                imgs.append(img)
                valid_paths.append(p)
            except Exception:
                continue

        if not imgs:
            continue

        imgs_tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            img_feat = clip_model.encode_image(imgs_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            scores = (img_feat @ text_feat.T).squeeze(-1).cpu().tolist()

        for path, score in zip(valid_paths, scores):
            results.append((path, score))

    return results


def visualize_class(
    class_name: str,
    scored_paths: List[Tuple[str, float]],
    threshold: float,
    n_samples: int,
    output_path: str,
):
    """
    한 class에 대해 필터링된 이미지(낮은 점수) vs 통과한 이미지(높은 점수) 시각화.
    """
    # 점수 기준으로 정렬
    sorted_by_score = sorted(scored_paths, key=lambda x: x[1])

    # 필터링된 것 (점수 낮은 것)
    filtered = [(p, s) for p, s in sorted_by_score if s < threshold]
    # 통과한 것 (점수 높은 것)
    passed = [(p, s) for p, s in sorted_by_score if s >= threshold]

    # 각 그룹에서 샘플 추출
    filtered_samples = filtered[:n_samples]
    passed_samples = passed[-n_samples:]  # 점수 높은 것부터

    n_cols = n_samples
    n_rows = 2  # 위: 필터링됨, 아래: 통과

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3 + 1))
    fig.suptitle(
        f'Class: "{class_name}"\n'
        f'Filtered: {len(filtered)} / Passed: {len(passed)} (threshold={threshold})',
        fontsize=12, fontweight="bold"
    )

    # Top row: filtered images (low score)
    for col in range(n_cols):
        ax = axes[0][col] if n_cols > 1 else axes[0]
        if col < len(filtered_samples):
            path, score = filtered_samples[col]
            img = Image.open(path).convert("RGB")
            ax.imshow(img)
            ax.set_title(f"score: {score:.3f}", fontsize=9, color="red")
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Filtered\n(low score)", fontsize=9, color="red")

    # Bottom row: passed images (high score)
    for col in range(n_cols):
        ax = axes[1][col] if n_cols > 1 else axes[1]
        if col < len(passed_samples):
            path, score = passed_samples[col]
            img = Image.open(path).convert("RGB")
            ax.imshow(img)
            ax.set_title(f"score: {score:.3f}", fontsize=9, color="green")
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Passed\n(high score)", fontsize=9, color="green")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_score_distribution(
    all_scored: Dict[str, List[Tuple[str, float]]],
    threshold: float,
    output_path: str,
):
    """
    전체 class에 대한 CLIP score 분포 히스토그램.
    """
    all_scores = [s for paths in all_scored.values() for _, s in paths]
    filtered_scores = [s for s in all_scores if s < threshold]
    passed_scores = [s for s in all_scores if s >= threshold]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(passed_scores, bins=50, color="green", alpha=0.6, label=f"Passed ({len(passed_scores):,})")
    ax.hist(filtered_scores, bins=50, color="red", alpha=0.6, label=f"Filtered ({len(filtered_scores):,})")
    ax.axvline(x=threshold, color="black", linestyle="--", linewidth=2, label=f"threshold={threshold}")
    ax.set_xlabel("CLIP Score", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("CLIP Score Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_poor_classes(
    all_scored: Dict[str, List[Tuple[str, float]]],
    threshold: float,
    poor_class_threshold: float,
    output_path: str,
    top_n: int = 20,
):
    """
    class별 평균 CLIP score 막대그래프. poor class 강조.
    """
    class_avg = {}
    for class_name, scored in all_scored.items():
        if scored:
            class_avg[class_name] = sum(s for _, s in scored) / len(scored)

    # 평균 점수 낮은 순으로 정렬
    sorted_classes = sorted(class_avg.items(), key=lambda x: x[1])[:top_n]
    names = [c[:20] for c, _ in sorted_classes]  # 이름 잘라서 표시
    scores = [s for _, s in sorted_classes]
    colors = ["red" if s < poor_class_threshold else "steelblue" for s in scores]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(names, scores, color=colors)
    ax.axvline(x=poor_class_threshold, color="black", linestyle="--",
               linewidth=2, label=f"poor class threshold={poor_class_threshold}")
    ax.set_xlabel("Avg CLIP Score", fontsize=12)
    ax.set_title(f"Avg CLIP Score per Class (bottom {top_n})", fontsize=14, fontweight="bold")

    poor_patch = mpatches.Patch(color="red", label="Poor class")
    normal_patch = mpatches.Patch(color="steelblue", label="Normal class")
    ax.legend(handles=[poor_patch, normal_patch, ax.axvline(x=poor_class_threshold,
              color="black", linestyle="--")], fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# Main 

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--synth_data_root", type=str,
                        default="./generate_init_synthetic/data/synth_initial_v2")
    parser.add_argument("--data_root", type=str, default="./dataset/data")
    parser.add_argument("--output_dir", type=str, default="./outputs/clip_filter_viz")
    parser.add_argument("--threshold", type=float, default=0.20,
                        help="CLIP score 필터링 기준")
    parser.add_argument("--poor_class_threshold", type=float, default=0.25,
                        help="poor class 판별 기준")
    parser.add_argument("--n_classes", type=int, default=10,
                        help="시각화할 class 수 (poor class 위주)")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="class당 보여줄 이미지 수")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    import clip
    print(f"Loading CLIP... (device={args.device})")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=args.device)
    clip_model.eval()

    print("Loading class names...")
    class_names = load_class_names(args.data_root)

    print("Loading synthetic image paths...")
    class_to_paths = load_synth_paths(args.synth_data_root, class_names)

    # CLIP Score 계산
    print(f"\nComputing CLIP scores... ({len(class_names)} classes)")
    all_scored: Dict[str, List[Tuple[str, float]]] = {}

    for i, class_name in enumerate(class_names):
        paths = class_to_paths.get(class_name, [])
        if not paths:
            continue
        scored = compute_clip_scores_for_paths(
            paths=paths,
            class_name=class_name,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            device=args.device,
        )
        all_scored[class_name] = scored
        avg = sum(s for _, s in scored) / len(scored) if scored else 0
        print(f"  [{i+1:3d}/{len(class_names)}] {class_name[:30]:30s} | avg={avg:.3f} | n={len(scored)}")

    # 전체 분포 시각화 
    print("\nVisualizing score distribution...")
    visualize_score_distribution(
        all_scored=all_scored,
        threshold=args.threshold,
        output_path=os.path.join(args.output_dir, "score_distribution.png"),
    )
    print("  -> score_distribution.png saved")

    # Poor class 막대그래프
    print("Visualizing poor class ranking...")
    visualize_poor_classes(
        all_scored=all_scored,
        threshold=args.threshold,
        poor_class_threshold=args.poor_class_threshold,
        output_path=os.path.join(args.output_dir, "poor_class_ranking.png"),
    )
    print("  -> poor_class_ranking.png saved")

    # Class별 필터링 전후 이미지 시각화 
    class_avg = {
        name: sum(s for _, s in scored) / len(scored)
        for name, scored in all_scored.items() if scored
    }
    target_classes = sorted(class_avg.items(), key=lambda x: x[1])[:args.n_classes]

    print(f"\nVisualizing {args.n_classes} classes...")
    viz_dir = os.path.join(args.output_dir, "per_class")
    os.makedirs(viz_dir, exist_ok=True)

    for class_name, avg_score in target_classes:
        scored = all_scored.get(class_name, [])
        if not scored:
            continue

        safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in class_name)
        output_path = os.path.join(viz_dir, f"{safe_name}.png")

        visualize_class(
            class_name=class_name,
            scored_paths=scored,
            threshold=args.threshold,
            n_samples=args.n_samples,
            output_path=output_path,
        )
        print(f"  -> {class_name} (avg={avg_score:.3f}) saved")

    # 통계 JSON 저장 
    stats = {
        "threshold": args.threshold,
        "poor_class_threshold": args.poor_class_threshold,
        "total_images": sum(len(v) for v in all_scored.values()),
        "filtered_images": sum(1 for v in all_scored.values() for _, s in v if s < args.threshold),
        "passed_images": sum(1 for v in all_scored.values() for _, s in v if s >= args.threshold),
        "poor_classes": [
            {"class_name": name, "avg_score": round(avg, 4)}
            for name, avg in sorted(class_avg.items(), key=lambda x: x[1])
            if avg < args.poor_class_threshold
        ],
        "class_avg_scores": {
            name: round(avg, 4)
            for name, avg in sorted(class_avg.items(), key=lambda x: x[1])
        }
    }
    with open(os.path.join(args.output_dir, "filter_analysis.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Results saved to: {args.output_dir}")
    print(f"  Total:    {stats['total_images']:,}")
    print(f"  Filtered: {stats['filtered_images']:,}")
    print(f"  Passed:   {stats['passed_images']:,}")
    print(f"  Poor classes: {len(stats['poor_classes'])}")


if __name__ == "__main__":
    main()