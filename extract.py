

import argparse
import json

import cv2
import numpy as np


def rle_to_polygons(rle: dict, epsilon_ratio: float = 0.002, min_points: int = 3):
    from pycocotools import mask as mask_utils

    rle = dict(rle)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")

    binary_mask = mask_utils.decode(rle)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        if len(contour) < min_points:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)
        if len(approx) < min_points:
            approx = contour
        points = approx.reshape(-1, 2).tolist()
        if len(points) >= min_points:
            polygons.append(points)
    return polygons


def extract(coco_json_path: str, out_json_path: str):
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    result: dict[str, list] = {}
    total = len(coco["annotations"])

    for i, ann in enumerate(coco["annotations"], start=1):
        fname = id_to_filename.get(ann["image_id"], f"image_{ann['image_id']}")
        polygons = rle_to_polygons(ann["segmentation"])

        # 물체 하나가 여러 조각으로 나뉘어도, 각 조각을 그냥 별개 항목으로 추가
        result.setdefault(fname, []).extend(polygons)

        if i % 20 == 0 or i == total:
            print(f"  {i}/{total} 변환 중...")

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] 이미지 {len(result)}개, 물체 {total}개 좌표 추출")
    print(f"[저장] '{out_json_path}'")


def main():
    parser = argparse.ArgumentParser(description="COCO json -> 좌표(폴리곤 점)만 추출")
    parser.add_argument("--coco_json", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    args = parser.parse_args()

    extract(args.coco_json, args.out_json)


if __name__ == "__main__":
    main()
