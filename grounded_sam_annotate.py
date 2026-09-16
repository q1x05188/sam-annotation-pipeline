"""
grounded_sam_annotate.py
--------------------------
텍스트로 원하는 물체(예: "pear")만 지정해서 자동으로 찾아 마스킹하는 스크립트.
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
    from segment_anything import sam_model_registry, SamPredictor

    if model_type not in sam_model_registry:
        raise ValueError(f"model_type은 {list(sam_model_registry.keys())} 중 하나여야 합니다.")

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)


def detect_boxes(detector, image_pil: Image.Image, text_prompts: list, box_threshold: float):

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
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./annotations_grounded")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"]
    )
    parser.add_argument(
        "--text_prompts", type=str, nargs="+", required=True
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--box_threshold", type=float, default=0.3
    )
    parser.add_argument(
        "--dino_model", type=str, default=DEFAULT_DINO_MODEL
    )
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument(
        "--save_binary", action="store_true"
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
