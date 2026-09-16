

import argparse
import json
import os

import cv2
import numpy as np


def make_mask_for_image(polygons: list, height: int, width: int) -> np.ndarray:
    # 어노테이션 점들로 흑백(0/255) 마스크 생성  /  안쪽=255(흰색), 바깥=0(검정)
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
        print(f"[알림] 원본 이미지를 못 찾아서 건너뛴 항목: {skipped}개")


def main():
    parser = argparse.ArgumentParser(description="좌표 json -> 흑백(0/255) 마스크 이미지 변환")
    parser.add_argument("--coords_json", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    convert(args.coords_json, args.image_dir, args.out_dir)


if __name__ == "__main__":
    main()
