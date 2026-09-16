"""
coords_to_binary_mask.py
---------------------------
extract_coordinates.py로 뽑은 "좌표만 있는" json
({"000001.jpg": [[[x,y],[x,y],...], [[...]]], ...})
을 읽어서, 폴리곤 안쪽=검정(0), 바깥쪽=흰색(255)인 마스크 이미지를 만든다.

이 좌표 json에는 이미지 크기(width/height) 정보가 없기 때문에,
원본 이미지 폴더(--image_dir)에서 실제 이미지를 열어 크기를 읽어온다.

설치:
    pip install opencv-python numpy

사용 예시:
    python coords_to_binary_mask.py --coords_json ./coordinates.json --image_dir ./crawled/pear --out_dir ./masks
"""

import argparse
import json
import os

import cv2
import numpy as np


def make_mask_for_image(polygons: list, height: int, width: int) -> np.ndarray:
    """폴리곤 점 목록들로 흑백(0/255) 마스크를 만든다. 폴리곤 안쪽=255(흰색), 바깥=0(검정)."""
    mask = np.full((height, width), 0, dtype=np.uint8)
    for points in polygons:
        if len(points) < 3:
            continue
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    return mask


def convert(coords_json_path: str, image_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    with open(coords_json_path, "r", encoding="utf-8") as f:
        coords = json.load(f)

    converted = 0
    skipped = 0

    for file_name, polygons in coords.items():
        image_path = os.path.join(image_dir, file_name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"[스킵] 원본 이미지를 찾을 수 없음: {image_path}")
            skipped += 1
            continue

        h, w = img.shape[:2]
        mask = make_mask_for_image(polygons, h, w)

        out_path = os.path.join(out_dir, os.path.splitext(file_name)[0] + ".png")
        cv2.imwrite(out_path, mask)
        converted += 1
        print(f"  {file_name} -> {os.path.basename(out_path)} (폴리곤 {len(polygons)}개)")

    print(f"\n[완료] {converted}개 마스크 저장 -> '{out_dir}'")
    if skipped:
        print(f"[경고] 원본 이미지를 못 찾아서 건너뛴 항목: {skipped}개")


def main():
    parser = argparse.ArgumentParser(description="좌표 json -> 흑백(0/255) 마스크 이미지 변환")
    parser.add_argument("--coords_json", type=str, required=True,
                         help="extract_coordinates.py로 만든 좌표만 있는 json 경로")
    parser.add_argument("--image_dir", type=str, required=True,
                         help="원본 이미지가 들어있는 폴더 (이미지 크기를 읽기 위해 필요)")
    parser.add_argument("--out_dir", type=str, required=True, help="마스크 이미지를 저장할 폴더")
    args = parser.parse_args()

    convert(args.coords_json, args.image_dir, args.out_dir)


if __name__ == "__main__":
    main()
