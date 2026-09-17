

import argparse
import os
import shutil
from pathlib import Path

import torch
from PIL import Image

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14" 


def build_prompt_pairs(keyword: str) -> list[tuple[str, str, list[str]]]:

    return [
        (
            "category",
            f"a photo of real, high-quality {keyword}",
            ["a photo of something else entirely"],
        ),
        (
            "photorealism",
            f"a photo of real, high-quality {keyword}",
            [
                f"a drawing of {keyword}",
                f"an illustration of {keyword}",
                f"a cartoon of {keyword}",
            ],
        ),
        (
            "freshness",
            f"a photo of fresh, good-quality {keyword}",
            [f"a photo of rotten, spoiled, or moldy {keyword}"],
        ),
    ]


def load_clip(model_id: str, device: str):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor


@torch.no_grad()
def check_image(model, processor, device, image: Image.Image, prompt_pairs: list,
                 margin: float = 0.0, min_tests_passed: int | None = None,
                 negative_agg: str = "max", skip_tests: set | None = None) -> dict:


    skip_tests = skip_tests or set()
    active_pairs = [p for p in prompt_pairs if p[0] not in skip_tests]

    if min_tests_passed is None:
        min_tests_passed = len(active_pairs)  # 통과의 기준이 되는 테스트 합격 개수, 기본값은 평가 대상 전부

    all_texts: list[str] = []
    spans: list[tuple[str, int, list[int]]] = []  # (테스트명, positive 인덱스, negative 인덱스들)

    for name, pos, negs in prompt_pairs:  # 최종적으로 참고하기 위해 skip된 것도 점수를 계산해서 기록
        pos_idx = len(all_texts)
        all_texts.append(pos)
        neg_indices = []
        for neg in negs:
            neg_indices.append(len(all_texts))
            all_texts.append(neg)
        spans.append((name, pos_idx, neg_indices))

    inputs = processor(text=all_texts, images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)

    # image-text 유사도
    logits_per_image = outputs.logits_per_image[0] 

    details = {}
    for name, pos_idx, neg_indices in spans:
        pos_score = float(logits_per_image[pos_idx])
        neg_scores = [float(logits_per_image[ni]) for ni in neg_indices]

        if name in skip_tests:
            details[name] = {"positive": pos_score, "negatives": neg_scores, "passed": None}
            continue

        # negative_agg="max": negative들 중 consine similarity 최댓값과 비교, 비교 대상은 positive consine similarity.
        neg_ref = max(neg_scores) if negative_agg == "max" else sum(neg_scores) / len(neg_scores)
        # neg_ref에 margin을 주어 positive가 얼마나 앞서는지 확인. margin=0이면 가장 엄격한 기준이고 기본 값 , margin>0이면 통과 기준에 여유를 줄 수 있음.
        passed = pos_score > (neg_ref - margin)
        details[name] = {"positive": pos_score, "negatives": neg_scores, "passed": passed}

    num_passed = sum(1 for d in details.values() if d["passed"])
    num_evaluated = len(active_pairs)
    # min_tests_passed(기본=평가 대상 전부)이면 3개(기본값, 전부)를 "전부 통과해야 함".
    # 낮추면 N개 중 M개만 통과해도 합격하는 식으로 느슨해짐.
    all_passed = num_passed >= min_tests_passed

    return {"passed": all_passed, "num_passed": num_passed, "num_evaluated": num_evaluated, "details": details}


# 테스트별 positive/negative 점수를 참고할 수 있도록 한줄씩 정리
def format_score_line(details: dict) -> str:
    """테스트별 positive/negative 점수를 사람이 읽기 좋은 한 줄씩으로 정리."""
    lines = []
    for name, d in details.items():
        pos = d["positive"]
        negs = d["negatives"]
        neg_str = ", ".join(f"{n:.2f}" for n in negs)
        mark = "?" if d["passed"] is None else ("O" if d["passed"] else "X")
        diff = pos - max(negs)
        lines.append(
            f"      [{mark}] {name:12s} positive={pos:.2f}  negatives=[{neg_str}]  (차이: {diff:+.2f})"
        )
    return "\n".join(lines)


def filter_folder(
    image_dir: str,
    keyword: str,
    clip_model: str = DEFAULT_CLIP_MODEL,
    device: str | None = None,
    move_to: str = "rejected",
    dry_run: bool = False,
    margin: float = 0.0,
    min_tests_passed: int | None = None,
    negative_agg: str = "max",
    skip_tests: set | None = None,
    verbose: bool = False,
    extensions: tuple = DEFAULT_EXTENSIONS,
):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    image_paths = sorted(
        p for p in Path(image_dir).iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )
    if not image_paths:
        print(f"[경고] '{image_dir}'에서 이미지 파일을 찾지 못했습니다.")
        return

    reject_dir = Path(image_dir) / move_to
    if not dry_run:
        reject_dir.mkdir(exist_ok=True)

    prompt_pairs = build_prompt_pairs(keyword)
    print(f"[로딩] CLIP 모델 로딩 중... (model={clip_model}, device={device})")
    model, processor = load_clip(clip_model, device)

    print(f"[시작] 이미지 {len(image_paths)}개 검사 (keyword='{keyword}')")
    kept, rejected = 0, 0

    for i, img_path in enumerate(image_paths, start=1):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [{i}/{len(image_paths)}] {img_path.name} -> 열기 실패, 스킵 ({e!r})")
            continue

        result = check_image(model, processor, device, image, prompt_pairs,
                              margin=margin, min_tests_passed=min_tests_passed,
                              negative_agg=negative_agg, skip_tests=skip_tests)

        if result["passed"]:
            kept += 1
            status = f"통과 ({result['num_passed']}/{result['num_evaluated']})"
        else:
            rejected += 1
            failed_names = [name for name, d in result["details"].items() if d["passed"] is False]
            status = f"탈락 ({result['num_passed']}/{result['num_evaluated']}, 실패: " + ", ".join(failed_names) + ")"
            if not dry_run:
                shutil.move(str(img_path), str(reject_dir / img_path.name))

        print(f"  [{i}/{len(image_paths)}] {img_path.name} -> {status}")
        if verbose:
            print(format_score_line(result["details"]))

    print(f"\n[완료] 통과 {kept}개, 탈락 {rejected}개")
    if dry_run:
        print("[참고] --dry_run 이라 실제로 파일을 옮기지 않았습니다.")
    else:
        print(f"[저장] 탈락한 이미지는 '{reject_dir}'로 이동됨 (삭제되지 않음)")


def main():
    parser = argparse.ArgumentParser(
    )
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument(
        "--keyword", type=str, required=True
    )
    parser.add_argument(
        "--clip_model", type=str, default=DEFAULT_CLIP_MODEL
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--move_to", type=str, default="rejected"
    )
    parser.add_argument(
        "--dry_run", action="store_true"
    )
    parser.add_argument(
        "--margin", type=float, default=0.0
    )
    parser.add_argument(
        "--min_tests_passed", type=int, default=None
    )
    parser.add_argument(
        "--negative_agg", type=str, default="max", choices=["max", "mean"]
    )
    parser.add_argument(
        "--skip_tests", type=str, nargs="*", default=None
    )
    parser.add_argument(
        "--verbose", action="store_true"
    )
    args = parser.parse_args()

    filter_folder(
        image_dir=args.image_dir,
        keyword=args.keyword,
        clip_model=args.clip_model,
        device=args.device,
        move_to=args.move_to,
        dry_run=args.dry_run,
        margin=args.margin,
        min_tests_passed=args.min_tests_passed,
        negative_agg=args.negative_agg,
        skip_tests=set(args.skip_tests) if args.skip_tests else None,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()