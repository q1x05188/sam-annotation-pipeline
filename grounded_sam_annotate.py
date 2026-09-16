"""
grounded_sam_annotate.py
--------------------------
텍스트로 원하는 물체(예: "pear")만 지정해서 자동으로 찾아 마스킹하는 스크립트.

동작 방식 (Grounded-SAM):
    1. GroundingDINO(HuggingFace transformers 내장 버전)로 이미지에서
       text_prompts에 해당하는 물체의 박스(bounding box)를 찾는다.
    2. 그 박스를 SAM에 "여기를 정밀하게 세그멘테이션 해줘"라고 프롬프트로 준다.
    3. SAM이 박스 안에서 정확한 픽셀 단위 마스크를 만든다.
    => sam_annotate.py(전체 자동 마스킹)와 달리, 지정한 물체만 마스킹되어
       labelme에서 지울 것이 훨씬 줄어든다.

설치:
    pip install torch torchvision
    pip install git+https://github.com/facebookresearch/segment-anything.git
    pip install transformers pillow opencv-python pycocotools numpy

체크포인트 (SAM용):
    vit_b (가장 작고 빠름, ~375MB): https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    vit_l (중간): https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
    vit_h (가장 정확, ~2.4GB): https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

사용 예시:
    python grounded_sam_annotate.py \
        --image_dir ./crawled/pear \
        --checkpoint sam_vit_b_01ec64.pth \
        --model_type vit_b \
        --text_prompts "pear" \
        --out_dir ./annotations_pear \
        --save_vis

GroundingDINO 모델은 처음 실행할 때 HuggingFace에서 자동으로 다운로드됩니다 (인터넷 필요, 약 170MB).
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# 이미지로 인식할 확장자 목록. 필요하면 여기 추가/수정하면 됨.
DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 기본 GroundingDINO 모델. HuggingFace Hub의 다른 zero-shot-object-detection
# 모델 id로 --dino_model 옵션을 통해 교체 가능 (예: "IDEA-Research/grounding-dino-base").
DEFAULT_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"


def load_detector(device: str, dino_model_id: str = DEFAULT_DINO_MODEL):
    """GroundingDINO 텍스트 기반 물체 검출기를 로드.

    Args:
        device: "cuda" 또는 "cpu"
        dino_model_id: HuggingFace Hub의 zero-shot-object-detection 모델 id
    """
    from transformers import pipeline

    detector = pipeline(
        model=dino_model_id,
        task="zero-shot-object-detection",
        device=0 if device == "cuda" else -1,
    )
    return detector


def load_sam_predictor(checkpoint: str, model_type: str, device: str):
    """box 프롬프트를 받아 정밀 마스크를 뽑아주는 SamPredictor를 로드.

    Args:
        checkpoint: SAM 체크포인트(.pth) 파일 경로
        model_type: 체크포인트와 맞는 모델 크기 ("vit_b" | "vit_l" | "vit_h")
        device: "cuda" 또는 "cpu"
    """
    from segment_anything import sam_model_registry, SamPredictor

    if model_type not in sam_model_registry:
        raise ValueError(f"model_type은 {list(sam_model_registry.keys())} 중 하나여야 합니다.")

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)


def detect_boxes(detector, image_pil: Image.Image, text_prompts: list, box_threshold: float):
    """GroundingDINO로 text_prompts에 해당하는 박스들을 찾는다.

    Args:
        detector: load_detector()로 만든 파이프라인
        image_pil: PIL Image (RGB)
        text_prompts: 찾을 물체 이름 목록 (예: ["pear", "apple"])
        box_threshold: 검출 확신도 임계값 (0~1). 낮을수록 더 많이(느슨하게) 검출됨.

    Returns:
        [{"label": str, "score": float, "bbox_xyxy": [x1,y1,x2,y2]}, ...]
    """
    # GroundingDINO는 각 문구가 마침표로 끝나는 걸 권장함 (예: "a pear.")
    candidate_labels = [p.strip().rstrip(".") + "." for p in text_prompts]

    results = detector(image_pil, candidate_labels=candidate_labels, threshold=box_threshold)

    boxes = []
    for r in results:
        box = r["box"]  # {"xmin", "ymin", "xmax", "ymax"}
        boxes.append({
            "label": r["label"].rstrip("."),
            "score": float(r["score"]),
            "bbox_xyxy": [box["xmin"], box["ymin"], box["xmax"], box["ymax"]],
        })
    return boxes


def mask_to_coco_annotation(binary_mask: np.ndarray, image_id: int, ann_id: int,
                             category_id: int, score: float) -> dict:
    """SAM이 반환한 binary mask(bool 2D array)를 COCO annotation 형식(RLE)으로 변환."""
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")  # json 직렬화를 위해 str로 변환
    area = float(mask_utils.area(rle))
    bbox = mask_utils.toBbox(rle).tolist()  # [x, y, w, h]

    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": rle,
        "area": area,
        "bbox": bbox,
        "iscrowd": 0,
        "detection_score": score,  # GroundingDINO 검출 확신도 (COCO 표준 필드는 아니지만 참고용으로 남김)
    }


def save_visualization(image_rgb: np.ndarray, masks_and_labels: list, save_path: str):
    """검출된 마스크 + 라벨(이름, 확신도)을 이미지 위에 색깔로 겹쳐 그려서 저장."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(image_rgb.shape[1] / 100, image_rgb.shape[0] / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image_rgb)
    ax.axis("off")

    overlay = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4))
    for mask, label, score in masks_and_labels:
        color = np.concatenate([np.random.random(3), [0.5]])
        overlay[mask] = color
        ys, xs = np.where(mask)
        if len(xs) > 0:
            ax.text(xs.min(), ys.min() - 5, f"{label} {score:.2f}",
                     color="white", fontsize=10,
                     bbox=dict(facecolor="black", alpha=0.6, pad=1))
    ax.imshow(overlay)

    fig.savefig(save_path)
    plt.close(fig)


def save_binary_mask(masks: list, height: int, width: int, save_path: str):
    """
    검출된 모든 마스크를 합쳐서 흑백(0/255) 이미지로 저장.
    물체 영역 = 0(검정), 배경 = 255(흰색).

    Args:
        masks: bool 2D array(H, W)들의 리스트. 같은 이미지의 물체가 여러 개면 전부 합쳐진다.
        height, width: 출력 마스크 크기 (원본 이미지와 동일해야 함)
        save_path: 저장할 .png 경로
    """
    combined = np.full((height, width), 255, dtype=np.uint8)
    for m in masks:
        combined[m] = 0
    cv2.imwrite(save_path, combined)


def annotate_folder(
    image_dir: str,
    out_dir: str,
    checkpoint: str,
    text_prompts: list,
    model_type: str = "vit_b",
    device: str = "cpu",
    box_threshold: float = 0.3,
    dino_model: str = DEFAULT_DINO_MODEL,
    save_vis: bool = False,
    save_binary: bool = False,
    extensions: tuple = DEFAULT_EXTENSIONS,
):
    """
    image_dir 안의 모든 이미지에서 text_prompts에 해당하는 물체를 찾아 마스킹하고,
    out_dir/annotations.json (COCO 형식)으로 저장한다.

    Args:
        image_dir: 원본 이미지가 들어있는 폴더
        out_dir: 결과(annotations.json, 시각화, 이진마스크)를 저장할 폴더
        checkpoint: SAM 체크포인트(.pth) 경로
        text_prompts: 찾을 물체 이름 목록 (예: ["pear"])
        model_type: SAM 모델 크기 ("vit_b" | "vit_l" | "vit_h")
        device: "cuda" 또는 "cpu"
        box_threshold: GroundingDINO 검출 확신도 임계값
        dino_model: 사용할 GroundingDINO(zero-shot-object-detection) 모델 id
        save_vis: True면 컬러 오버레이 시각화 이미지도 저장
        save_binary: True면 물체=검정/배경=흰색 이진 마스크 PNG도 저장
        extensions: 이미지로 인식할 파일 확장자
    """
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "visualizations")
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)
    binary_dir = os.path.join(out_dir, "binary_masks")
    if save_binary:
        os.makedirs(binary_dir, exist_ok=True)

    image_paths = sorted(
        p for p in Path(image_dir).iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )
    if not image_paths:
        print(f"[경고] '{image_dir}'에서 이미지 파일을 찾지 못했습니다 (확장자: {extensions}).")
        return

    print(f"[로딩] GroundingDINO 검출기 로딩 중... (model={dino_model}, device={device})")
    detector = load_detector(device, dino_model_id=dino_model)
    print(f"[로딩] SAM 로딩 중... (model_type={model_type})")
    predictor = load_sam_predictor(checkpoint, model_type, device)

    # 카테고리: text_prompts 순서대로 id 부여 (COCO categories 필드로 그대로 저장됨)
    categories = [{"id": i + 1, "name": p} for i, p in enumerate(text_prompts)]
    label_to_id = {c["name"]: c["id"] for c in categories}

    coco = {"images": [], "annotations": [], "categories": categories}
    ann_id = 1

    print(f"[시작] 이미지 {len(image_paths)}개, 검색어 {text_prompts}, box_threshold={box_threshold}")

    for image_id, img_path in enumerate(image_paths, start=1):
        image_pil = Image.open(img_path).convert("RGB")
        image_rgb = np.array(image_pil)
        h, w = image_rgb.shape[:2]

        boxes = detect_boxes(detector, image_pil, text_prompts, box_threshold)

        coco["images"].append({
            "id": image_id, "file_name": img_path.name, "width": w, "height": h,
        })

        if not boxes:
            print(f"  [{image_id}/{len(image_paths)}] {img_path.name} -> 검출 0개")
            if save_binary:
                bin_path = os.path.join(binary_dir, f"{img_path.stem}.png")
                save_binary_mask([], h, w, bin_path)
            continue

        predictor.set_image(image_rgb)
        masks_for_vis = []

        for b in boxes:
            box_xyxy = np.array(b["bbox_xyxy"])
            masks, scores, _ = predictor.predict(box=box_xyxy, multimask_output=False)
            binary_mask = masks[0]  # multimask_output=False라 1개만 나옴

            category_id = label_to_id.get(b["label"], 1)
            coco["annotations"].append(
                mask_to_coco_annotation(binary_mask, image_id, ann_id, category_id, b["score"])
            )
            ann_id += 1
            masks_for_vis.append((binary_mask, b["label"], b["score"]))

        print(f"  [{image_id}/{len(image_paths)}] {img_path.name} -> 검출/마스킹 {len(boxes)}개")

        if save_vis:
            vis_path = os.path.join(vis_dir, f"{img_path.stem}_annotated.jpg")
            save_visualization(image_rgb, masks_for_vis, vis_path)

        if save_binary:
            bin_path = os.path.join(binary_dir, f"{img_path.stem}.png")
            save_binary_mask([m for m, _, _ in masks_for_vis], h, w, bin_path)

    out_json = os.path.join(out_dir, "annotations.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    print(f"\n[완료] 이미지 {len(coco['images'])}개, 마스크 {len(coco['annotations'])}개")
    print(f"[저장] '{out_json}'")
    if save_vis:
        print(f"[저장] 시각화 이미지: '{vis_dir}'")
    if save_binary:
        print(f"[저장] 이진 마스크(물체=0/검정, 배경=255/흰색): '{binary_dir}'")


def main():
    parser = argparse.ArgumentParser(description="Grounded-SAM: 텍스트로 지정한 물체만 자동 마스킹")
    parser.add_argument("--image_dir", type=str, required=True, help="이미지가 들어있는 폴더")
    parser.add_argument("--out_dir", type=str, default="./annotations_grounded", help="결과 저장 폴더")
    parser.add_argument("--checkpoint", type=str, required=True, help="SAM 체크포인트(.pth) 경로")
    parser.add_argument(
        "--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"],
        help="체크포인트와 맞는 SAM 모델 크기"
    )
    parser.add_argument(
        "--text_prompts", type=str, nargs="+", required=True,
        help="찾고 싶은 물체 이름(들). 예: --text_prompts pear   또는  --text_prompts pear apple"
    )
    parser.add_argument("--device", type=str, default="cpu", help="'cuda' (GPU 있을 때) 또는 'cpu'")
    parser.add_argument(
        "--box_threshold", type=float, default=0.3,
        help="검출 확신도 임계값 (0~1). 배경이나 엉뚱한 게 잡히면 0.4~0.5로 올려보세요."
    )
    parser.add_argument(
        "--dino_model", type=str, default=DEFAULT_DINO_MODEL,
        help="사용할 GroundingDINO(zero-shot-object-detection) HuggingFace 모델 id. "
             f"기본값: {DEFAULT_DINO_MODEL} (더 정확한 'IDEA-Research/grounding-dino-base' 등으로 교체 가능)"
    )
    parser.add_argument("--save_vis", action="store_true", help="시각화 이미지도 같이 저장")
    parser.add_argument(
        "--save_binary", action="store_true",
        help="물체=검정(0), 배경=흰색(255)인 이진 마스크 PNG도 같이 저장"
    )
    args = parser.parse_args()

    annotate_folder(
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        text_prompts=args.text_prompts,
        model_type=args.model_type,
        device=args.device,
        box_threshold=args.box_threshold,
        dino_model=args.dino_model,
        save_vis=args.save_vis,
        save_binary=args.save_binary,
    )


if __name__ == "__main__":
    main()
