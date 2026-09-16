"""
labelme_to_coco.py
--------------------
labelme로 수정한 이미지별 {이름}.json 파일들을 모아서
다시 COCO 형식 annotations.json으로 합친다.

카테고리(라벨)는 labelme에서 각 폴리곤에 붙인 label 값을 그대로 사용하며,
등장하는 순서대로 category_id가 자동 부여된다. labelme에서 label을 하나도
안 남긴(빈) shape은 --default_category 값으로 대체된다.

설치:
    pip install opencv-python pycocotools numpy

사용 예시:
    python labelme_to_coco.py --image_dir ./crawled/pear --out_json ./annotations/annotations_edited.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def polygon_to_rle(points: list, height: int, width: int):
    """polygon 점 목록을 COCO RLE 마스크로 변환.

    Args:
        points: [[x1,y1],[x2,y2],...] 형태의 폴리곤 좌표
        height, width: 이미지 크기 (마스크 캔버스 크기와 동일해야 함)

    Returns:
        (rle, area, bbox): COCO 형식 RLE 딕셔너리, 넓이(px), bbox [x,y,w,h]
    """
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
    """
    image_dir 안의 모든 labelme json({이름}.json)을 모아 COCO 형식으로 합쳐 out_json에 저장한다.

    Args:
        image_dir: labelme json 파일들(및 원본 이미지)이 들어있는 폴더
        out_json: 저장할 COCO 형식 json 경로 (상위 폴더가 없으면 자동 생성)
        default_category: shape에 label이 비어있을 때 대신 쓸 카테고리 이름
    """
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
    parser.add_argument("--image_dir", type=str, required=True,
                         help="labelme json 파일들(및 이미지)이 들어있는 폴더")
    parser.add_argument("--out_json", type=str, required=True, help="저장할 COCO json 경로")
    parser.add_argument(
        "--default_category", type=str, default="object",
        help="labelme shape에 label이 비어있을 때 대신 사용할 카테고리 이름 (기본값: 'object')"
    )
    args = parser.parse_args()

    convert(args.image_dir, args.out_json, default_category=args.default_category)


if __name__ == "__main__":
    main()
