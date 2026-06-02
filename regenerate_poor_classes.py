"""
Poor Class 재생성 스크립트 (MetaSynth 아이디어 적용)
1. filter_analysis.json에서 poor class 목록 읽기
2. Ollama(qwen3)로 각 poor class에 대해 다양한 관점의 프롬프트 생성
3. SD 1.5로 이미지 생성
4. 기존 synthetic 데이터 경로에 추가 저장

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import List

import requests
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Ollama 프롬프트 생성 

def generate_prompts_with_ollama(
    class_name: str,
    n_prompts: int = 5,
    ollama_url: str = "http://localhost:11434",
    model: str = "qwen3:4b-instruct",
) -> List[str]:
    """
    MetaSynth 아이디어:
    LLM이 하나의 class에 대해 여러 관점의 프롬프트를 생성.
    - 스타일 관점
    - 배경/맥락 관점
    - 디테일 관점
    - 색상/텍스처 관점
    - 용도/상황 관점
    """
    system_prompt = (
        "You are an expert at creating diverse image generation prompts. "
        "Generate varied prompts that will produce high-quality, realistic images "
        "of the given object from different perspectives and contexts. "
        "Each prompt should start with 'a photo of' and be specific and detailed. "
        "Output ONLY a JSON array of strings, nothing else."
    )

    user_prompt = (
        f"Generate {n_prompts} diverse image generation prompts for: '{class_name}'\n\n"
        f"Create prompts from different perspectives:\n"
        f"1. Style/appearance perspective\n"
        f"2. Background/context perspective\n"
        f"3. Detail/close-up perspective\n"
        f"4. Color/texture perspective\n"
        f"5. Usage/situation perspective\n\n"
        f"Example for 'bonnet':\n"
        f'["a photo of a Victorian bonnet with wide brim and silk ribbon", '
        f'"a photo of a bonnet placed on a wooden dressing table", '
        f'"a close-up photo of a bonnet showing detailed lace trim", '
        f'"a photo of a white bonnet with floral pattern", '
        f'"a photo of a woman wearing a traditional bonnet outdoors"]\n\n'
        f"Now generate {n_prompts} prompts for '{class_name}'. "
        f"Output ONLY the JSON array."
    )

    try:
        response = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.8},
            },
            timeout=60,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"].strip()

        # thinking 태그 제거 (qwen3 특성)
        if "<think>" in content:
            content = content[content.rfind("</think>") + len("</think>"):].strip()

        # JSON 파싱
        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end > start:
            prompts = json.loads(content[start:end])
            if isinstance(prompts, list) and len(prompts) > 0:
                return [str(p) for p in prompts[:n_prompts]]

    except Exception as e:
        logger.warning("Ollama 프롬프트 생성 실패: %s", e)

    # Fallback: 기본 프롬프트 템플릿
    logger.info("Fallback 프롬프트 사용: %s", class_name)
    base = class_name.split(",")[0].strip()
    return [
        f"a photo of {base}, high quality, realistic",
        f"a photo of a {base}, detailed, clear background",
        f"a close-up photo of {base}, sharp focus",
        f"a photo of {base} in natural lighting",
        f"a professional photo of {base}, studio quality",
    ][:n_prompts]


#SD 1.5 파이프라인 로드
def load_pipeline(device: str = "cuda") -> StableDiffusionPipeline:
    """SD 1.5 파이프라인 로드."""
    logger.info("Loading SD 1.5...")
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        scheduler=scheduler,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    logger.info("SD 1.5 loaded!")
    return pipe


# 이미지 생성 

def generate_images(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    n_images_per_prompt: int,
    output_class_dir: Path,
    cfg_scale: float = 7.5,
    num_steps: int = 50,
    device: str = "cuda",
) -> int:
    """
    주어진 프롬프트 목록으로 이미지 생성 후 저장.
    Returns: 생성된 이미지 수
    """
    output_class_dir.mkdir(parents=True, exist_ok=True)

    # 기존 이미지 수 파악 (파일명 중복 방지)
    existing = list(output_class_dir.glob("*.jpg"))
    start_idx = len(existing)

    count = 0
    for prompt_idx, prompt in enumerate(prompts):
        logger.info("  Prompt [%d/%d]: %s", prompt_idx + 1, len(prompts), prompt[:60])
        for i in range(n_images_per_prompt):
            with torch.no_grad():
                result = pipe(
                    prompt,
                    guidance_scale=cfg_scale,
                    num_inference_steps=num_steps,
                    height=512,
                    width=512,
                )
            img = result.images[0]
            fname = f"regen_{start_idx + count:06d}.jpg"
            img.save(output_class_dir / fname, quality=95)
            count += 1

    return count


# Main

def _safe_dir_name(class_name: str) -> str:
    """class 이름을 폴더명으로 변환."""
    safe = class_name.split(",")[0].strip()
    safe = "".join(c if c.isalnum() or c in {" ", "_", "-"} else "_" for c in safe)
    return safe.strip().replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--filter_json",
        type=str,
        default="./outputs/clip_filter_viz/filter_analysis.json",
        help="filter_analysis.json 경로 (poor class 목록)",
    )
    parser.add_argument(
        "--synth_data_root",
        type=str,
        default="./generate_init_synthetic/data/synth_initial_v2",
        help="기존 synthetic 데이터 경로 (여기에 추가 저장)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generate_init_synthetic/data/synth_regenerated",
        help="재생성 이미지 저장 경로",
    )
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=5,
        help="class당 생성할 프롬프트 수 (MetaSynth: 다양한 관점)",
    )
    parser.add_argument(
        "--n_images_per_prompt",
        type=int,
        default=20,
        help="프롬프트당 생성할 이미지 수",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="DDIM sampling steps",
    )
    parser.add_argument(
        "--ollama_url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API URL",
    )
    parser.add_argument(
        "--ollama_model",
        type=str,
        default="qwen3:4b-instruct",
        help="Ollama 모델 이름",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Poor class 목록 로드
    with open(args.filter_json) as f:
        filter_data = json.load(f)

    poor_classes = [pc["class_name"] for pc in filter_data["poor_classes"]]
    logger.info("Poor classes: %d개", len(poor_classes))
    for pc in poor_classes:
        logger.info("  - %s", pc)

    # SD 1.5 로드 
    pipe = load_pipeline(device=args.device)

    # 각 poor class에 대해 재생성
    total_generated = 0
    regen_log = []

    for i, class_name in enumerate(poor_classes):
        logger.info("\n[%d/%d] Class: %s", i + 1, len(poor_classes), class_name)

        # 1. Ollama로 다양한 프롬프트 생성 (MetaSynth)
        logger.info("  Generating prompts with Ollama...")
        prompts = generate_prompts_with_ollama(
            class_name=class_name,
            n_prompts=args.n_prompts,
            ollama_url=args.ollama_url,
            model=args.ollama_model,
        )
        logger.info("  Prompts: %s", prompts)

        # 2. SD 1.5로 이미지 생성
        dir_name = _safe_dir_name(class_name)
        output_class_dir = Path(args.output_dir) / "train" / dir_name

        n_generated = generate_images(
            pipe=pipe,
            prompts=prompts,
            n_images_per_prompt=args.n_images_per_prompt,
            output_class_dir=output_class_dir,
            cfg_scale=args.cfg_scale,
            num_steps=args.num_steps,
            device=args.device,
        )
        total_generated += n_generated

        regen_log.append({
            "class_name": class_name,
            "prompts": prompts,
            "n_generated": n_generated,
            "output_dir": str(output_class_dir),
        })

        logger.info("  Generated: %d images → %s", n_generated, output_class_dir)

    # 로그 저장 
    log_path = os.path.join(args.output_dir, "regen_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(regen_log, f, indent=2, ensure_ascii=False)

    logger.info("\n===== 재생성 완료 =====")
    logger.info("총 생성 이미지: %d개", total_generated)
    logger.info("저장 경로: %s", args.output_dir)
    logger.info("로그: %s", log_path)
    logger.info("\n다음 단계: train_resnet.py로 재학습")
    logger.info(
        "python train_resnet.py \\\n"
        "  --mode synthetic \\\n"
        "  --synth_data_root %s \\\n"
        "  --data_root ./dataset/data \\\n"
        "  --output_dir ./outputs/exp_regen_resnet",
        args.output_dir,
    )


if __name__ == "__main__":
    main()