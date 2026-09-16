"""
crawling_localization.py
---------------------------
키워드(들)를 검색어로 웹 이미지를 크롤링하는 스크립트.
Google / Bing / Baidu 이미지 검색을 지원하며(icrawler 라이브러리 사용),
이미 보유한 URL은 건너뛰고, 새로 받은 URL은 기록해서 다음 실행 때 중복 없이 이어서 쓸 수 있다.

설치:
    pip install icrawler pillow

기본 사용 예시 (키워드 하나):
    python crawling_localization.py --keyword "pear" --out ./crawled --max_num 100

여러 검색어(맥락)를 각각 따로 크롤링:
    python crawling_localization.py --use_contexts \
        --contexts "pear" "pear on a tree" "sliced pear" \
        --max_num_per_context 20 --out ./crawled

중복 방지 + 기록 누적:
    python crawling_localization.py --keyword "pear" --out ./crawled \
        --exclude_urls_json train_urls.json test_urls.json crawled_urls.json \
        --record_urls_json crawled_urls.json
"""

import argparse
import json
import os
import threading
from pathlib import Path

from icrawler.builtin import GoogleImageCrawler, BingImageCrawler, BaiduImageCrawler
from icrawler.downloader import ImageDownloader

# 지원하는 검색엔진과 대응되는 icrawler 클래스
ENGINE_MAP = {
    "google": GoogleImageCrawler,
    "bing": BingImageCrawler,
    "baidu": BaiduImageCrawler,
}


def load_exclude_urls(json_paths: list[str]) -> set[str]:
    """
    이미 보유한 이미지 URL 목록을 json 파일(들)에서 읽어와 set으로 합친다.
    이 URL들은 크롤링 중 다시 다운로드되지 않고 건너뛰어진다.

    지원하는 json 포맷 (자동으로 셋 다 시도함):
      1) {"로컬/경로.jpg": "https://...", ...}   -> dict의 value가 URL
      2) [["로컬/경로.jpg", "https://..."], ...]  -> 리스트의 각 원소가 [경로, url]
      3) [{"path": "...", "url": "..."}, ...]     -> 리스트의 각 원소가 dict

    Args:
        json_paths: 제외 목록으로 쓸 json 파일 경로들 (여러 개 지정 가능, 전부 합쳐짐)
    """
    urls: set[str] = set()
    for jp in json_paths:
        if not jp:
            continue
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            urls.update(str(v) for v in data.values())
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    urls.add(str(item[1]))
                elif isinstance(item, dict):
                    # 흔히 쓰는 키 이름들 시도
                    for key in ("url", "file_url", "image_url", "src"):
                        if key in item:
                            urls.add(str(item[key]))
                            break
        else:
            raise ValueError(f"지원하지 않는 json 최상위 타입: {type(data)} ({jp})")

    print(f"[제외 목록] 총 {len(urls)}개의 기존 URL을 로드했습니다.")
    return urls


def save_recorded_urls(json_path: str, recorded_urls: dict[str, str]):
    """
    이번에 새로 다운로드에 성공한 {로컬경로: URL}을 json 파일에 누적 저장한다.
    파일이 이미 있으면 기존 내용을 불러와서 병합(새 항목 추가/갱신)한 뒤 다시 저장하므로,
    여러 번 실행해도 기록이 계속 쌓인다. 다음 실행 때 이 파일을 --exclude_urls_json 으로
    넣으면 방금 받은 이미지들을 다시 받지 않는다.

    Args:
        json_path: 기록을 저장할 json 파일 경로
        recorded_urls: 이번 실행에서 모은 {로컬경로: URL}
    """
    existing: dict[str, str] = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
        if isinstance(data, dict):
            existing = data
        else:
            print(
                f"[경고] '{json_path}'가 dict 형식이 아니라서(list 등) 자동 병합할 수 없습니다. "
                f"새 내용으로 덮어씁니다."
            )

    before = len(existing)
    existing.update(recorded_urls)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    added = len(existing) - before
    print(
        f"[기록] '{json_path}' 저장 완료 -> 이번에 크롤링한 {len(recorded_urls)}개 중 "
        f"신규 {added}개 추가 (파일 내 총 {len(existing)}개)"
    )


def make_downloader_cls(
    exclude_urls: set[str] | None,
    skipped_urls: list,
    recorded_urls: dict,
    lock: threading.Lock,
):
    """
    exclude_urls에 있는 URL은 건너뛰고, 새로 다운로드에 성공한 (경로, URL)은
    recorded_urls에 기록하는 커스텀 icrawler ImageDownloader 클래스를 만들어 반환한다.

    icrawler가 다운로드에 성공하면 task["success"]=True, task["filename"]=저장된 파일명을
    채워주고 그 직후 process_meta(task)를 호출하는 점을 이용해 기록 훅을 건다.
    """
    exclude_urls = exclude_urls or set()

    class TrackingImageDownloader(ImageDownloader):
        _exclude_urls = exclude_urls
        _skipped_urls = skipped_urls
        _recorded_urls = recorded_urls
        _lock = lock

        def download(self, task, default_ext, timeout=5, max_retry=3, overwrite=False, **kwargs):
            file_url = task.get("file_url")
            if file_url in self._exclude_urls:
                with self._lock:
                    self._skipped_urls.append(file_url)
                self.logger.info("이미 보유한 URL이라 스킵: %s", file_url)
                return
            return super().download(
                task, default_ext, timeout=timeout, max_retry=max_retry,
                overwrite=overwrite, **kwargs
            )

        def process_meta(self, task):
            super().process_meta(task)
            if task.get("success") and task.get("filename"):
                full_path = str(Path(self.storage.root_dir) / task["filename"])
                with self._lock:
                    self._recorded_urls[full_path] = task["file_url"]

    return TrackingImageDownloader


def crawl_one_keyword(
    keyword: str,
    out_dir: str,
    max_num: int = 100,
    engine: str = "google",
    min_size: tuple | None = (200, 200),
    thread_num: int = 4,
    filters: dict | None = None,
    exclude_urls: set[str] | None = None,
    record_urls: dict | None = None,
) -> list[str]:
    """
    검색어 하나로 이미지를 크롤링해서 out_dir에 저장한다.

    Args:
        keyword: 검색어 (예: "pear", "pear on a tree")
        out_dir: 이미지를 저장할 폴더 (없으면 자동 생성)
        max_num: 이 검색어로 받을 최대 이미지 수
        engine: "google" | "bing" | "baidu"
        min_size: (최소 너비, 최소 높이) px. 이보다 작은 이미지는 다운로드 후 삭제. None이면 필터링 안 함
        thread_num: 동시 다운로드 스레드 수
        filters: 검색엔진별 추가 필터 딕셔너리 (예: {"size": "large"})
        exclude_urls: 다시 받지 않을 URL 집합
        record_urls: 새로 받은 {경로: URL}을 채워 넣을 딕셔너리 (None이면 내부에서 새로 만듦)

    Returns:
        skipped_urls: 이미 보유해서 건너뛴 URL 목록
    """
    if engine not in ENGINE_MAP:
        raise ValueError(f"engine must be one of {list(ENGINE_MAP.keys())}, got: {engine}")

    os.makedirs(out_dir, exist_ok=True)

    if record_urls is None:
        record_urls = {}

    skipped_urls: list[str] = []
    lock = threading.Lock()

    crawler_cls = ENGINE_MAP[engine]
    crawler_kwargs = {
        "storage": {"root_dir": out_dir},
        "downloader_threads": thread_num,
        "log_level": "INFO",
    }
    # exclude_urls 필터링 + 다운로드 성공 URL 기록을 위해 항상 커스텀 다운로더 사용
    crawler_kwargs["downloader_cls"] = make_downloader_cls(exclude_urls, skipped_urls, record_urls, lock)

    crawler = crawler_cls(**crawler_kwargs)

    # file_idx_offset="auto": 같은 폴더에 여러 번(여러 검색어) 크롤링해도 파일명이 겹쳐 덮어써지지 않도록 함
    crawl_kwargs = {"keyword": keyword, "max_num": max_num, "file_idx_offset": "auto"}
    if filters:
        crawl_kwargs["filters"] = filters

    print(f"[시작] engine={engine}, keyword='{keyword}', max_num={max_num}, out_dir='{out_dir}'")
    try:
        crawler.crawl(**crawl_kwargs)
    except Exception as e:
        print(f"[에러] 크롤링 중 예외 발생: {e!r}")
        raise

    downloaded = [p for p in Path(out_dir).glob("*") if p.is_file()]
    print(f"'{keyword}' -> 총 {len(downloaded)}개 파일 다운로드됨 (필터링 전)")

    if len(downloaded) == 0:
        print(
            "다운로드된 파일이 0개입니다. 보통 원인: "
            "1) 검색엔진이 요청을 차단함(--engine 을 bing/baidu로 바꿔보세요) "
            "2) 검색어에 결과가 거의 없음 3) 네트워크/방화벽 문제"
        )

    # 너무 작은/깨진 이미지 후처리 필터링 (선택)
    if min_size is not None:
        _filter_small_or_broken_images(out_dir, min_size, record_urls)

    # 스킵된(이미 보유한) URL 목록을 콘솔에 출력하고 파일로도 저장
    if skipped_urls:
        print(f"'{keyword}' -> 이미 보유해서 스킵한 URL {len(skipped_urls)}개:")
        for u in skipped_urls:
            print(f"  - {u}")

        skip_log_path = os.path.join(out_dir, "skipped_urls.txt")
        with open(skip_log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(skipped_urls))
        print(f"  (목록이 '{skip_log_path}'에도 저장됨)")

    return skipped_urls


def _filter_small_or_broken_images(folder: str, min_size: tuple, record_urls: dict | None = None):
    """다운로드 후 min_size보다 작거나 열리지 않는(깨진) 이미지를 삭제한다."""
    try:
        from PIL import Image
    except ImportError:
        print("Pillow가 없어 이미지 필터링을 건너뜁니다. `pip install pillow`로 설치하세요.")
        return

    removed = 0
    for path in Path(folder).glob("*"):
        if not path.is_file():
            continue
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                w, h = img.size
                if w < min_size[0] or h < min_size[1]:
                    path.unlink()
                    removed += 1
                    if record_urls is not None:
                        # 삭제된 파일은 기록에서도 제거 (다음 실행 때 다시 시도할 수 있도록)
                        record_urls.pop(str(path), None)
        except Exception:
            # 깨진 이미지 예외처리
            path.unlink(missing_ok=True)
            removed += 1
            if record_urls is not None:
                record_urls.pop(str(path), None)

    if removed:
        print(f"[필터링] {folder}: 작거나 손상된 이미지 {removed}개 제거")


def crawl_multiple_keywords(
    keywords: list[str],
    base_out_dir: str,
    record_urls: dict | None = None,
    **kwargs,
):
    """
    여러 검색어를 순회하며 각각 base_out_dir 아래 하위 폴더(검색어명 기반)에 크롤링한다.
    같은 record_urls 딕셔너리를 계속 넘겨서 검색어 전체에 걸쳐 URL 기록이 누적되도록 한다.

    Args:
        keywords: 검색어 목록
        base_out_dir: 결과를 저장할 상위 폴더 (검색어별로 하위 폴더가 자동 생성됨)
        record_urls: 여러 검색어에 걸쳐 공유할 기록용 딕셔너리
        **kwargs: crawl_one_keyword에 그대로 전달되는 나머지 인자들 (max_num, engine 등)
    """
    if record_urls is None:
        record_urls = {}

    all_skipped: dict[str, list[str]] = {}
    for kw in keywords:
        safe_name = kw.strip().replace(" ", "_").replace("/", "_")
        out_dir = os.path.join(base_out_dir, safe_name)
        print(f"\n=== '{kw}' 크롤링 시작 -> {out_dir} ===")
        skipped = crawl_one_keyword(keyword=kw, out_dir=out_dir, record_urls=record_urls, **kwargs)
        if skipped:
            all_skipped[kw] = skipped

    total_skipped = sum(len(v) for v in all_skipped.values())
    if total_skipped:
        print(f"\n=== 전체 요약: 이미 보유해서 스킵한 URL 총 {total_skipped}개 ===")
        for kw, urls in all_skipped.items():
            print(f"[{kw}] {len(urls)}개")
            for u in urls:
                print(f"  - {u}")

    return record_urls


def main():
    parser = argparse.ArgumentParser(description="키워드 기반 웹 이미지 크롤러")
    parser.add_argument("--keyword", type=str, default=None,
                         help="검색할 단일 키워드 (예: 'pear'). --use_contexts/--keywords_file 와 함께 쓰지 않음")
    parser.add_argument(
        "--keywords_file", type=str, default=None,
        help="한 줄에 하나씩 키워드가 적힌 txt 파일 경로 (여러 키워드를 순서대로 일괄 처리)"
    )
    parser.add_argument("--out", type=str, default="./crawled_images", help="이미지를 저장할 상위 폴더")
    parser.add_argument("--max_num", type=int, default=100, help="키워드(검색어)당 최대 다운로드 개수")
    parser.add_argument("--engine", type=str, default="google", choices=list(ENGINE_MAP.keys()),
                         help="사용할 이미지 검색엔진")
    parser.add_argument("--threads", type=int, default=4, help="동시 다운로드 스레드 수")
    parser.add_argument("--min_w", type=int, default=200, help="최소 너비(px), 0이면 크기 필터링 끔")
    parser.add_argument("--min_h", type=int, default=200, help="최소 높이(px), 0이면 크기 필터링 끔")
    parser.add_argument(
        "--exclude_urls_json", type=str, nargs="*", default=None,
        help="이미 보유한 URL 목록 json 파일들 (예: train_urls.json test_urls.json). "
             "여기 포함된 URL은 다시 다운로드하지 않음. 여러 파일 지정 가능."
    )
    parser.add_argument(
        "--record_urls_json", type=str, default=None,
        help="이번에 새로 크롤링한 {경로: URL} 을 기록할 json 파일 경로. "
             "--exclude_urls_json 과 같은 형식(dict)으로 저장되며, 파일이 이미 있으면 "
             "기존 내용에 이어서 누적 저장됩니다. 다음 실행 때 이 파일을 "
             "--exclude_urls_json 으로 다시 넣으면 중복 다운로드를 막을 수 있습니다."
    )
    parser.add_argument(
        "--use_contexts", action="store_true",
        help="--contexts 로 넘긴 여러 검색어(맥락)를 각각 따로 크롤링. "
             "템플릿으로 자동 생성하는 게 아니라 직접 적어준 문구를 그대로 검색어로 사용함."
    )
    parser.add_argument(
        "--contexts", type=str, nargs="*", default=None,
        help="--use_contexts 사용 시, 각각 따로 크롤링할 검색어(맥락) 목록. "
             "예: --contexts \"pear\" \"pear on a tree\" \"sliced pear\""
    )
    parser.add_argument(
        "--max_num_per_context", type=int, default=None,
        help="--use_contexts 사용 시, 맥락 하나당 최대 다운로드 개수. "
             "지정 안 하면 --max_num 값을 그대로 씀."
    )
    args = parser.parse_args()

    min_size = (args.min_w, args.min_h) if args.min_w and args.min_h else None

    exclude_urls = None
    if args.exclude_urls_json:
        exclude_urls = load_exclude_urls(args.exclude_urls_json)

    # 이번 실행에서 새로 다운로드한 {경로: URL}을 여기에 누적
    recorded_urls: dict[str, str] = {}

    if args.use_contexts:
        if not args.contexts:
            parser.error("--use_contexts 사용 시 --contexts 를 하나 이상 지정해야 합니다.")
        max_num = args.max_num_per_context if args.max_num_per_context is not None else args.max_num
        print(f"[맥락 크롤링] {len(args.contexts)}개 맥락, 맥락당 최대 {max_num}장: {args.contexts}")
        crawl_multiple_keywords(
            keywords=args.contexts,
            base_out_dir=args.out,
            max_num=max_num,
            engine=args.engine,
            thread_num=args.threads,
            min_size=min_size,
            exclude_urls=exclude_urls,
            record_urls=recorded_urls,
        )
    elif args.keywords_file:
        with open(args.keywords_file, "r", encoding="utf-8") as f:
            keywords = [line.strip() for line in f if line.strip()]
        crawl_multiple_keywords(
            keywords=keywords,
            base_out_dir=args.out,
            max_num=args.max_num,
            engine=args.engine,
            thread_num=args.threads,
            min_size=min_size,
            exclude_urls=exclude_urls,
            record_urls=recorded_urls,
        )
    elif args.keyword:
        crawl_one_keyword(
            keyword=args.keyword,
            out_dir=args.out,
            max_num=args.max_num,
            engine=args.engine,
            thread_num=args.threads,
            min_size=min_size,
            exclude_urls=exclude_urls,
            record_urls=recorded_urls,
        )
    else:
        parser.error("--keyword, --keywords_file, --use_contexts 중 하나는 지정해야 합니다.")

    if args.record_urls_json:
        save_recorded_urls(args.record_urls_json, recorded_urls)


if __name__ == "__main__":
    main()
