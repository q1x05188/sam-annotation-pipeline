# SAM 기반 이미지 자동 어노테이션 파이프라인

웹에서 이미지를 수집하고, CLIP으로 품질을 검증하고, Grounded-SAM(GroundingDINO + SAM)으로 자동 세그멘테이션 어노테이션을 만든 뒤,
labelme(GUI 툴)로 수동 보정하고, COCO 형식 데이터셋과 흑백(0,255) 마스크 이미지로 최종 변환하는 파이프라인입니다.

검색어 생성(0단계)과 CLIP 필터링(2단계)은 "See & Sniff" (ECCV 2026) 논문의 SmellNet-V 데이터 구축 방법론(LLM 기반 검색어 생성 + CLIP prompt-based verification)을 참고했습니다.


<img width="400" height="300" alt="Image" src="https://github.com/user-attachments/assets/578a6820-df21-4df3-befb-9b7fb36748ce" />

<img width="400" height="300" alt="Image" src="https://github.com/user-attachments/assets/c2c0fb28-a039-445a-9900-0ec532f09863" />



## 전체 파이프라인

```
0. (선택) 검색어 자동 생성    (generate_queries.py)        -> keywords.txt
        │
        ▼
1. 이미지 크롤링              (crawling_localization.py)   -> ./crawled/pear/*.jpg
        │
        ▼
2. CLIP 이미지 검증           (clip_filter.py)              -> 통과/탈락 (rejected/ 로 이동)
        │
        ▼
3. 자동 어노테이션 생성        (grounded_sam_annotate.py)    -> annotations.json (COCO 형식)
        │
        ▼
4. labelme 형식으로 변환       (coco_to_labelme.py)          -> 이미지별 {이름}.json
        │
        ▼
5. labelme GUI로 수동 보정      (직접 작업, 코드 없음)
        │
        ▼
6. 다시 COCO로 합치기          (labelme_to_coco.py)          -> annotations_edited.json (최종 데이터셋)
        │
        ├──▶ 7. 좌표만 추출          (extract.py)                -> coordinates.json
        │           │
        │           ▼
        └──▶ 8. 흑백 마스크 생성      (coords_to_binary_mask.py)  -> *.png

```

## 요구 사항

- Python 3.10 이상
- (optional) NVIDIA GPU + CUDA — 없어도 CPU로 동작하지만 2단계(자동 어노테이션)가 느립니다.

## 설치

```bash
pip install -r requirements.txt
```

`segment-anything`은 pip 공식 배포가 없어 GitHub에서 직접 설치합니다:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

SAM 체크포인트(모델 가중치)는 용량이 커서 저장소에 포함하지 않았습니다. 아래에서 받아 프로젝트 루트에 둡니다.

```
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -OutFile "sam_vit_b_01ec64.pth"
```

## 사용법

아래는 "pear"를 예시로 한 전체 실행 순서입니다. 원하는 키워드로 바꿔서 쓰면 됩니다.

### 1. 이미지 크롤링

```bash
python crawling_localization.py `
    --keyword "pear" `
    --out ./crawled `
    --engine bing `
    --max_num 100 `
    --exclude_urls_json train_urls.json test_urls.json `
    --record_urls_json crawled_urls.json
```

- `--exclude_urls_json`: 이미 보유한 URL은 다시 받지 않음
- `--record_urls_json`: 이번에 새로 받은 URL을 기록 (다음 실행 때 exclude 목록에 추가해서 누적 관리)
- 여러 검색어(맥락)를 한 번에 돌리려면 `--use_contexts --contexts "pear" "pear on a tree" "sliced pear"` 사용

여러 검색어를 한 번에 크롤링하려면 `--use_contexts`와 `--contexts`를 사용합니다:

```bash
python crawling_localization.py `
    --use_contexts `
    --contexts "pear" "pear with other fruits" `
    --out ./crawled `
    --engine bing `
    --max_num_per_context 100 `
    --exclude_urls_json train_urls.json test_urls.json `
    --record_urls_json crawled_urls.json
```
- `--exclude_urls_json train_urls.json test_urls.json`은 예시이므로 초기엔 지우고 사용하였다가
- 만들어진 url 링크 json 파일명으로 재사용하면서 누적하면 됩니다.
- `--contexts`에 적은 검색어마다 `./crawled/pear`, `./crawled/apple`, `./crawled/banana` 처럼 하위 폴더가 자동 생성됩니다.
- `--max_num_per_context`는 검색어 **하나당** 최대 다운로드 개수입니다 (전체 합계가 아님).

### 2. CLIP 이미지 검증

```bash
python clip_filter.py --image_dir ./crawled/pear --keyword "pear" --margin 2.0
```

See & Sniff 논문 Supplementary Figure 9의 프롬프트를 그대로 사용해서 category(카테고리
일치),photorealism(사실적인 사진인지),freshness(정상 상태인지) 3가지를 CLIP으로 검증합니다.
탈락한 이미지는 삭제되지 않고 `./crawled/pear/rejected/`로 이동되고 3단계(SAM)는 이 폴더를
자동으로 건너뜁니다.

논문 원문 그대로는 아래 옵션 없이 실행하면 되고 너무 많이 탈락하면 아래 옵션으로 강도를
조절할 수 있습니다:

```bash
# 먼저 점수 분포를 보고 margin을 얼마로 줄지 가늠
python clip_filter.py --image_dir ./crawled/pear --keyword "pear" --dry_run --verbose

# 판정 여유값을 줘서 느슨하게
python clip_filter.py --image_dir ./crawled/pear --keyword "pear" --margin 2.0

# 3개 중 2개만 통과해도 인정
python clip_filter.py --image_dir ./crawled/pear --keyword "pear" --min_tests_passed 2

# 특정 테스트를 아예 빼기
python clip_filter.py --image_dir ./crawled/pear --keyword "pear" --skip_tests freshness
```

### 3. 자동 어노테이션 생성 (Grounded-SAM)

```bash
python grounded_sam_annotate.py `
    --image_dir ./crawled/pear `
    --checkpoint sam_vit_b_01ec64.pth `
    --model_type vit_b `
    --text_prompts "pear" `
    --out_dir ./annotations_pear `
    --save_vis
```

- `--text_prompts`로 지정한 물체만 찾아서 마스킹합니다 (SAM 혼자서는 텍스트로 클래스를 지정할 수 없어서, GroundingDINO가 위치를 먼저 찾고 SAM이 정밀 분할하는 구조)
- `--save_vis`: 결과를 눈으로 확인할 수 있는 컬러 오버레이 이미지도 같이 저장
- 결과: `./annotations_pear/annotations.json` (COCO 형식)

### 4. labelme 형식으로 변환

```bash
python coco_to_labelme.py `
    --coco_json ./annotations_pear/annotations.json `
    --image_dir ./crawled/pear
```

`./crawled/pear` 폴더 안에 이미지별 `{이름}.json`(labelme 형식)이 생성됩니다.

### 5. labelme로 수동 보정

```bash
labelme ./crawled/pear
```

잘못 잡힌 마스크는 삭제하고, 놓친 물체는 AI-Assisted Annotation(Point 도구)으로 클릭 몇 번으로 추가합니다.

### 6. 최종 COCO 데이터셋으로 재변환

```bash
python labelme_to_coco.py `
    --image_dir ./crawled/pear `
    --out_json ./annotations_pear/annotations_edited.json
```

이 파일이 실제 학습에 사용할 최종 어노테이션입니다.

### 7. 좌표만 추출 (선택)

```bash
python extract.py `
    --coco_json ./annotations_pear/annotations_edited.json `
    --out_json ./coordinates.json
```

RLE로 압축된 segmentation을 읽어, 사람이 읽을 수 있는 좌표 부분만 추출합니다.

### 8. 흑백 마스크 이미지 생성 (선택)

```bash
python coords_to_binary_mask.py `
    --coords_json ./coordinates.json `
    --image_dir ./crawled/pear `
    --out_dir ./masks
```

물체 영역과 배경을 흑/백으로 구분한 `.png` 마스크를 이미지별로 생성합니다.

## 스크립트별 옵션 상세

### 1. `crawling_localization.py`

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--keyword` | - | 검색할 단일 키워드 |
| `--keywords_file` | - | 한 줄에 하나씩 키워드가 적힌 txt 파일 (일괄 처리) |
| `--use_contexts` / `--contexts` | - | 여러 검색어(맥락)를 직접 리스트로 지정해 각각 크롤링 |
| `--max_num_per_context` | `--max_num` 값 | `--use_contexts` 사용 시 맥락당 최대 개수 |
| `--out` | `./crawled_images` | 저장 상위 폴더 |
| `--max_num` | `100` | 검색어당 최대 다운로드 개수 |
| `--engine` | `google` | `google` / `bing` / `baidu` |
| `--threads` | `4` | 동시 다운로드 스레드 수 |
| `--min_w`, `--min_h` | `200`, `200` | 최소 이미지 크기(px), `0`이면 필터링 끔 |
| `--exclude_urls_json` | - | 이미 보유한 URL 목록 json (여러 개 지정 가능) |
| `--record_urls_json` | - | 새로 받은 URL을 기록할 json (누적 저장) |

### 2. `grounded_sam_annotate.py`

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--image_dir` | (필수) | 이미지가 들어있는 폴더 |
| `--out_dir` | `./annotations_grounded` | 결과 저장 폴더 |
| `--checkpoint` | (필수) | SAM 체크포인트(.pth) 경로 |
| `--model_type` | `vit_b` | `vit_b` / `vit_l` / `vit_h` (체크포인트와 일치해야 함) |
| `--text_prompts` | (필수) | 찾을 물체 이름(들), 공백으로 여러 개 구분 |
| `--device` | `cpu` | `cuda` 또는 `cpu` |
| `--box_threshold` | `0.3` | 검출 확신도 임계값(0~1). 엉뚱한 게 잡히면 0.4~0.5로 상향 |
| `--dino_model` | `IDEA-Research/grounding-dino-tiny` | 사용할 GroundingDINO HuggingFace 모델 id |
| `--save_vis` | 꺼짐 | 컬러 오버레이 시각화 이미지 저장 |
| `--save_binary` | 꺼짐 | 물체=검정/배경=흰색 이진 마스크 PNG 저장 |

### 3. `coco_to_labelme.py`

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--coco_json` | (필수) | 변환할 COCO 형식 json 경로 |
| `--image_dir` | (필수) | 이미지 폴더 (labelme json도 여기 같이 저장됨) |

### 5. `labelme_to_coco.py`

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--image_dir` | (필수) | labelme json + 이미지가 들어있는 폴더 |
| `--out_json` | (필수) | 저장할 COCO json 경로 |
| `--default_category` | `object` | label이 비어있는 shape에 대신 쓸 카테고리 이름 |

### 6. `extract.py`

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--coco_json` | (필수) | 좌표를 뽑아낼 원본 COCO json |
| `--out_json` | (필수) | 좌표만 저장할 json 경로 |

### 7. `coords_to_binary_mask.py`

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--coords_json` | (필수) | `extract.py`로 만든 좌표 json |
| `--image_dir` | (필수) | 원본 이미지 폴더 (마스크 크기를 읽어오는 용도) |
| `--out_dir` | (필수) | 마스크 PNG 저장 폴더 |

### 8. `clip_filter.py`

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--image_dir` | (필수) | 검사할 이미지 폴더 |
| `--keyword` | (필수) | 프롬프트의 `{category}` 자리에 들어갈 물체/재료 이름 |
| `--clip_model` | `openai/clip-vit-large-patch14` | 사용할 HuggingFace CLIP 모델 id |
| `--device` | 자동 감지 | `cuda` 또는 `cpu` |
| `--move_to` | `rejected` | 탈락한 이미지를 옮길 하위 폴더 이름 |
| `--dry_run` | 꺼짐 | 실제로 옮기지 않고 결과만 미리 확인 |
| `--margin` | `0.0` | 판정 여유값. `0`=논문 원문(엄격), 올릴수록 느슨해짐 (예: `2.0`) |
| `--min_tests_passed` | 전부(3개) | 3개(category/photorealism/freshness) 중 몇 개만 통과해도 인정할지 |
| `--negative_agg` | `max` | negative가 여러 개인 테스트에서 `max`(엄격, 논문 원문) 또는 `mean`(느슨) |
| `--skip_tests` | 없음 | 판정에서 아예 제외할 테스트 이름(들) |
| `--verbose` | 꺼짐 | 이미지마다 positive/negative 실제 점수를 전부 출력 (margin 정할 때 참고용) |

가장 효과적인 손잡이는 `--margin`이고, 몇으로 줄지는 `--dry_run --verbose`로 나온 점수 차이
분포를 보고 정하는 걸 권장합니다 (감으로 정하지 말 것). `check_logit_scale.py`로 사용 중인
CLIP 모델의 점수 스케일(이론적 최대/최소)도 미리 확인할 수 있습니다.

## 참고 문헌

- Kim, S., Lee, S., Ryu, H., Chung, J.S., Senocak, A. "See & Sniff: Learning Visuo-Olfactory
  Representations." ECCV 2026. [프로젝트 페이지](https://mm.kaist.ac.kr/projects/SeeandSniff)
  — 검색어 생성(Figure 8) 및 CLIP 기반 이미지 필터링(Figure 9) 방법론 참고
- Kirillov, A., et al. "Segment Anything." ICCV 2023.
- IDEA-Research. [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
  — GroundingDINO + SAM 조합 어노테이션 파이프라인 원조

## 폴더 구조 예시

```
.
├── generate_queries.py
├── crawling_localization.py
├── clip_filter.py
├── grounded_sam_annotate.py
├── coco_to_labelme.py
├── labelme_to_coco.py
├── extract.py
├── coords_to_binary_mask.py
├── check_logit_scale.py
├── requirements.txt
├── README.md
├── train_urls.json          # 기존 보유 URL 목록 (예시)
├── test_urls.json
├── sam_vit_b_01ec64.pth     # SAM 체크포인트 (git에는 올리지 않음, .gitignore 참고)
└── crawled/                 # 크롤링 결과 (git에는 올리지 않음)
    └── pear/
        ├── 000001.jpg
        ├── 000001.json      # labelme 어노테이션
        ├── rejected/        # CLIP 필터링 탈락 이미지 (삭제 아님, 여기로 이동됨)
        └── ...
```

