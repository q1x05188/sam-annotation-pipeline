"""
labelme_to_coco.py
--------------------
labelme로 수정한 이미지별 {이름}.json 파일들을 모아서
다시 COCO 형식 annotations.json으로 합친다.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def polygon_to_rle(points: list, height: int, width: int):

    import cv2
    from pycocotools import mask as mask_utils

    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)

    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")  # json 저장을 위해 str로
    area = float(mask_utils.area(rle))
    bbox = mask_utils.toBbox(rle).tolist()  # [x, y, w, h]
    return rle, area, bbox


def convert(image_dir: str, out_json: str, default_category: str = "object"):

    json_files = sorted(Path(image_dir).glob("*.json"))
    if not json_files:
        print(f"[경고] '{image_dir}'에서 labelme json 파일을 찾지 못했습니다.")
        return

    categories: dict[str, int] = {}  # label 이름 -> category_id (처음 등장하는 순서대로 부여)
    coco = {"images": [], "annotations": [], "categories": []}

    image_id = 1
    ann_id = 1

    for jp in json_files:
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)

        # labelme 형식이 아닌 json(예: annotations.json 자체)은 건너뜀
        if "imageHeight" not in data or "imageWidth" not in data or "shapes" not in data:
            print(f"[스킵] labelme 형식이 아닌 json: {jp.name}")
            continue

        file_name = data.get("imagePath") or (jp.stem + ".jpg")
        height = data["imageHeight"]
        width = data["imageWidth"]

        coco["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        })

        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "polygon":
                print(f"[스킵] polygon이 아닌 shape은 지원 안 함: {jp.name} ({shape.get('shape_type')})")
                continue

            label = shape.get("label") or default_category
            if label not in categories:
                categories[label] = len(categories) + 1
            category_id = categories[label]

            points = shape["points"]
            if len(points) < 3:
                continue

            rle, area, bbox = polygon_to_rle(points, height, width)

            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            })
            ann_id += 1

        image_id += 1

    coco["categories"] = [{"id": cid, "name": name} for name, cid in categories.items()]
    if not coco["categories"]:
        coco["categories"] = [{"id": 1, "name": default_category}]

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    print(f"[완료] 이미지 {len(coco['images'])}개, 어노테이션 {len(coco['annotations'])}개")
    print(f"[카테고리] {[c['name'] for c in coco['categories']]}")
    print(f"[저장] '{out_json}'")


def main():
    parser = argparse.ArgumentParser(description="labelme json들 -> COCO annotations.json 변환")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument(
        "--default_category", type=str, default="object"
    )
    args = parser.parse_args()

    convert(args.image_dir, args.out_json, default_category=args.default_category)


if __name__ == "__main__":
    main()
