

import argparse
import json
import os

import cv2
import numpy as np


def rle_to_polygons(rle: dict, epsilon_ratio: float = 0.002, min_points: int = 3):
    from pycocotools import mask as mask_utils

    rle = dict(rle)  # 원본 훼손 방지
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")

    binary_mask = mask_utils.decode(rle)  # (H, W) uint8, 0 또는 1
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        if len(contour) < min_points:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)
        if len(approx) < min_points:
            approx = contour  # 너무 단순화되면 원본 윤곽선 사용
        points = approx.reshape(-1, 2).tolist()
        if len(points) >= min_points:
            polygons.append(points)
    return polygons


def convert(coco_json: str, image_dir: str, label_field: str = "category_id"):
    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)
    # category_id -> 이름 매핑
    cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}

    anns_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    converted = 0
    skipped_no_image = 0

    for img in coco["images"]:
        image_path = os.path.join(image_dir, img["file_name"])
        if not os.path.exists(image_path):
            print(f"[스킵] 이미지 파일이 없음: {image_path}")
            skipped_no_image += 1
            continue

        shapes = []
        for ann in anns_by_image.get(img["id"], []):
            label = cat_id_to_name.get(ann.get(label_field), "object")
            polygons = rle_to_polygons(ann["segmentation"])
            for points in polygons:
                shapes.append({
                    "label": label,
                    "points": points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {},
                })

        labelme_json = {
            "version": "5.4.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": img["file_name"],
            "imageData": None,
            "imageHeight": img["height"],
            "imageWidth": img["width"],
        }

        out_path = os.path.join(image_dir, os.path.splitext(img["file_name"])[0] + ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(labelme_json, f, ensure_ascii=False, indent=2)
        converted += 1

    print(f"[완료] {converted}개 이미지에 대해 labelme json 생성 완료 (out: '{image_dir}')")
    if skipped_no_image:
        print(f"[알림] 이미지 파일을 못 찾아서 건너뛴 항목: {skipped_no_image}개")


def main():
    parser = argparse.ArgumentParser(description="COCO annotations.json -> labelme json 변환")
    parser.add_argument("--coco_json", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    args = parser.parse_args()

    convert(args.coco_json, args.image_dir)


if __name__ == "__main__":
    main()
