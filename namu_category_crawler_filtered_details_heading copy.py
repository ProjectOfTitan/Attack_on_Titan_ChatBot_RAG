#!/usr/bin/env python3
import argparse
import json
import re
import time
from functools import lru_cache
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set
from urllib.parse import urldefrag, urljoin, urlparse

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

try:
    import tiktoken
except ImportError:
    tiktoken = None


DEFAULT_START_URLS = [
    "https://namu.wiki/w/%EB%B6%84%EB%A5%98:%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8/%EC%9E%91%EC%A4%91%20%EC%82%AC%EA%B1%B4%20%EC%82%AC%EA%B3%A0",
    "https://namu.wiki/w/%EB%B6%84%EB%A5%98:%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8/%EC%84%A4%EC%A0%95",
    "https://namu.wiki/w/%EB%B6%84%EB%A5%98:%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8/%EC%A7%80%EC%97%AD",
    "https://namu.wiki/w/%EB%B6%84%EB%A5%98:%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8/%EB%93%B1%EC%9E%A5%EC%9D%B8%EB%AC%BC?namespace=%EB%AC%B8%EC%84%9C&cuntil=%EC%B9%B4%EB%A6%AC%EB%82%98%20%EB%B8%8C%EB%9D%BC%EC%9A%B4",
    "https://namu.wiki/w/%EB%B6%84%EB%A5%98:%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8/%EB%93%B1%EC%9E%A5%EC%9D%B8%EB%AC%BC?namespace=%EB%AC%B8%EC%84%9C&cfrom=%EC%B9%B4%EC%95%BC%28%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8%29",
    "https://namu.wiki/w/%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8/%EA%B2%B0%EB%A7%90%20%EB%85%BC%EB%9E%80", # 결말 논란
    "https://namu.wiki/w/%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8/%EB%96%A1%EB%B0%A5",  # 떡밥
    "https://namu.wiki/w/%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8/%EB%8B%A8%ED%96%89%EB%B3%B8%20%EA%B0%81%ED%99%94%20%EB%B6%80%EC%A0%9C%EB%AA%A9", # 단행본 각화 부제목
    "https://namu.wiki/w/%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8/%EB%93%B1%EC%9E%A5%EC%9D%B8%EB%AC%BC", # 등장 인물
    "https://namu.wiki/w/%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8:%20%ED%9B%84%ED%9A%8C%EC%97%86%EB%8A%94%20%EC%84%A0%ED%83%9D", #후회없는선택
    "https://namu.wiki/w/%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8%20Lost%20girls", # lost girls
    "https://namu.wiki/w/%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8%20Before%20the%20fall", # before the fall
    "https://namu.wiki/w/%EB%B6%84%EB%A5%98:%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8/%EC%95%A0%EB%8B%88%EB%A9%94%EC%9D%B4%EC%85%98", # 애니메이션
    "https://namu.wiki/w/%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8/%EB%B9%84%ED%8C%90%20%EB%B0%8F%20%EB%85%BC%EB%9E%80", # 비판 및 논란
    "https://namu.wiki/w/%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8/%EB%B9%84%ED%8C%90%20%EB%B0%8F%20%EB%85%BC%EB%9E%80/%EC%84%A4%EC%A0%95", # 비판 및 논란/설정
    "https://namu.wiki/w/%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8/%EC%97%B0%ED%91%9C", # 설정
    "https://namu.wiki/w/%EC%9D%B4%EC%82%AC%EC%95%BC%EB%A7%88%20%ED%95%98%EC%A7%80%EB%A9%94", # 이사야마 하지메
    "https://namu.wiki/w/%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8/%EC%9D%B8%EA%B8%B0", # 인기
    "https://namu.wiki/w/%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8/%EC%9D%B8%EA%B8%B0%ED%88%AC%ED%91%9C", # 인기투표
    "https://namu.wiki/w/%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8/%ED%8F%89%EA%B0%80" #평가
    # "https://namu.wiki/w/%EC%95%A0%EB%8B%88%20%EB%A0%88%EC%98%A8%ED%95%98%ED%8A%B8#s-8", # 애니 레온하트(실험용)
    # "https://namu.wiki/w/%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8%20The%20Final%20Season/%EA%B3%B5%EA%B0%9C%20%EC%A0%95%EB%B3%B4",
]

CATEGORY_PATH_PREFIX = "/w/%EB%B6%84%EB%A5%98:"


@dataclass
class CrawlResult:
    url: str
    title: str
    blocks: List[Dict[str, str]]

@lru_cache(maxsize=1)
def _get_token_encoder():
    if tiktoken is None:
        return None
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    encoder = _get_token_encoder()
    if encoder is None:
        # Fallback: approximate tokens when tiktoken isn't available.
        return max(1, len(text) // 4)
    return len(encoder.encode(text))


def split_by_max_tokens(text: str, max_tokens: int, overlap_chars: int) -> List[str]:
    if not text or max_tokens <= 0:
        return [text]
    if count_tokens(text) <= max_tokens:
        return [text]

    encoder = _get_token_encoder()
    if encoder is None:
        max_chars = max(max_tokens * 4, 1)
        step = max_chars
        if overlap_chars > 0:
            step = max(max_chars - overlap_chars, 1)
        parts: List[str] = []
        for i in range(0, len(text), step):
            part = text[i : i + max_chars].strip()
            if part:
                parts.append(part)
        return parts

    tokens = encoder.encode(text)
    parts = []
    for i in range(0, len(tokens), max_tokens):
        part = encoder.decode(tokens[i : i + max_tokens]).strip()
        if part:
            parts.append(part)
    return parts


def build_prefix(title: str, section: str) -> str:
    title = (title or "").strip()
    section = (section or "").strip()
    parts: List[str] = []
    if title:
        parts.append(f"제목: {title}")
    if section:
        parts.append(f"섹션: {section}")
    if not parts:
        return ""
    return "\n".join(parts) + "\n\n"


def effective_max_tokens(max_tokens: int, prefix: str) -> int:
    if max_tokens <= 0 or not prefix:
        return max_tokens
    prefix_tokens = count_tokens(prefix)
    if prefix_tokens >= max_tokens:
        return max_tokens
    return max_tokens - prefix_tokens


def normalize_url(url: str) -> str:
    clean, _ = urldefrag(url)
    return clean


def is_same_domain(url: str, domains: Set[str]) -> bool:
    return urlparse(url).netloc in domains


def is_category_url(url: str) -> bool:
    parsed = urlparse(url)
    if CATEGORY_PATH_PREFIX in parsed.path:
        return True
    # Fallback for decoded URLs in some environments.
    return "/w/분류:" in parsed.path


def wait_for_ready(driver: webdriver.Chrome, timeout: int) -> None:
    def _ready(drv: webdriver.Chrome) -> bool:
        return drv.execute_script("return document.readyState") == "complete"

    WebDriverWait(driver, timeout).until(_ready)


def normalize_line(text: str) -> str:
    cleaned = " ".join(text.split()).strip()
    cleaned = cleaned.replace("[편집]", "").strip()
    cleaned = re.sub(r"\[\s*펼치기\s*·\s*접기\s*\]", "", cleaned).strip()
    cleaned = re.sub(r"펼치기\s*·\s*접기", "", cleaned).strip()
    cleaned = re.sub(r"<img[^>]*>", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"<[^>]+>", "", cleaned).strip()
    return cleaned


def extract_structured_items(driver: webdriver.Chrome) -> List[Dict[str, object]]:
    script = """
    try {
        const elements = Array.from(document.querySelectorAll(
            'h2, h3, h4, h5, h6, dt, .wiki-paragraph, div._4tRz4taR, blockquote, table, ul, ol, summary'
        ));

        const extractText = (el) => {
            const clone = el.cloneNode(true);
            clone.querySelectorAll('img').forEach(img => img.remove());
            clone.querySelectorAll('a[href]').forEach(a => {
                const text = (a.innerText || a.textContent || '').trim();
                a.replaceWith(document.createTextNode(text || ''));
            });
            return (clone.innerText || clone.textContent || '').trim();
        };
        const findDlLabel = (el) => {
            const dd = el.closest('dd');
            if (dd) {
                const prev = dd.previousElementSibling;
                if (prev && prev.tagName === 'DT') {
                    return extractText(prev);
                }
            }
            const dl = el.closest('dl');
            if (!dl) return '';
            const dt = dl.querySelector('dt');
            return dt ? extractText(dt) : '';
        };

        return elements.map(el => {
            const tag = el.tagName.toLowerCase();
            const dlLabel = findDlLabel(el);
            
            // 1. [핵심 수정] 인용구 중복만 '핀포인트'로 제거
            // 부모가 blockquote인 문단(div)만 건너뜁니다. (표나 리스트는 건드리지 않음)
            if ((tag === 'div' || el.classList.contains('wiki-paragraph')) && 
                el.parentElement && el.parentElement.tagName === 'BLOCKQUOTE') {
                return null;
            }

            // 2. [범위 복구] 제외 필터를 최소화하여 섹션 9, 10이 나오도록 합니다.
            // .wiki-category 등을 제외 목록에서 빼서 최대한 다 가져오게 합니다.
            if (el.closest('nav, header, footer, aside, .wiki-toc, .wiki-toc-wrapper')) {
                return null;
            }

            // 3. 테이블 내부의 div/문단은 테이블과 중복되므로 스킵
            if (tag !== 'table' && el.closest('table')) {
                return null;
            }

            const text = extractText(el);
            if (!text && tag !== 'table') return null;

            // --- 데이터 구조화 ---
            if (tag.startsWith('h')) return {type: 'heading', level: tag, text: text};
            if (tag === 'dt') return {type: 'subheading', level: tag, text: text};
            if (tag === 'summary') return {type: 'heading', level: 'h3', text: text};
            if (tag === 'blockquote') return {type: 'blockquote', text: text};
            
            if (tag === 'table') {
                const rows = Array.from(el.querySelectorAll('tr')).map(tr => {
                    return Array.from(tr.querySelectorAll('th, td')).map(td => extractText(td));
                });
                return {type: 'table', rows, subheading: dlLabel};
            }
            
            if (tag === 'ul' || tag === 'ol') {
                const items = Array.from(el.querySelectorAll(':scope > li')).map(li => {
                    const liClone = li.cloneNode(true);
                    liClone.querySelectorAll('ul, ol').forEach(child => child.remove());
                    return extractText(liClone);
                });
                return {type: 'list', items, subheading: dlLabel};
            }

            return {type: 'paragraph', text: text, subheading: dlLabel};
        }).filter(Boolean);
    } catch (e) {
        console.error(e);
        return [];
    }
    """
    results = driver.execute_script(script)
    # print(f"[디버그] 수집된 아이템 개수: {len(results)}개")
    return results or []


def build_blocks(items: List[Dict[str, object]], min_block_chars: int) -> List[Dict[str, str]]:
    blocks: List[Dict[str, str]] = []
    current_section: str = "제목 없음" # 초기값 설정
    current_subsection: str = ""

    # print("\n[구조화 시작] 섹션 감지 테스트...") # 디버깅용

    for item in items:
        item_type = item.get("type")
        raw_text = str(item.get("text") or "").strip()
        
        # 1. 헤딩(제목) 처리 - 이 부분이 섹션을 전환함
        if item_type == "heading":
            # [수정] 헤딩은 너무 엄격하게 normalize 하지 말고 공백만 제거합니다.
            clean_title = raw_text.replace("[편집]", "").strip()
            if clean_title:
                current_section = clean_title
                current_subsection = ""
                # print(f" -> 새 섹션 감지됨: {current_section}") # 터미널 출력으로 확인 가능
            continue
        if item_type == "subheading":
            clean_title = normalize_line(raw_text)
            if clean_title:
                current_subsection = clean_title
            continue

        section_label = current_section
        item_subheading = normalize_line(str(item.get("subheading") or "").strip())
        subsection_label = item_subheading or current_subsection

        def apply_subsection_prefix(value: str) -> str:
            if subsection_label and value:
                return f"{subsection_label}\n{value}"
            return value

        if item_type == "paragraph":
            text = normalize_line(raw_text)
            # 텍스트가 너무 짧으면 무시 (min_block_chars 기준)
            if len(text) < min_block_chars:
                continue
            blocks.append(
                {
                    "section": section_label,
                    "type": "paragraph",
                    "text": apply_subsection_prefix(text),
                }
            )
        
        elif item_type == "blockquote":
            text = normalize_line(raw_text)
            if len(text) < min_block_chars:
                continue
            blocks.append(
                {
                    "section": section_label,
                    "type": "blockquote",
                    "text": apply_subsection_prefix(f"[QUOTE]\n{text}\n[/QUOTE]"),
                }
            )
            
        elif item_type == "table":
            rows = item.get("rows") or []
            rendered_rows: List[str] = []
            for row in rows:
                row_cells = [normalize_line(str(c)) for c in row if c is not None]
                row_cells = [c for c in row_cells if c]
                if not row_cells:
                    continue
                rendered_rows.append("| " + " | ".join(row_cells) + " |")
            if rendered_rows:
                blocks.append(
                    {
                        "section": section_label,
                        "type": "table",
                        "text": apply_subsection_prefix(
                            "[TABLE]\n" + "\n".join(rendered_rows) + "\n[/TABLE]"
                        ),
                    }
                )
                
        elif item_type == "list":
            items_list = item.get("items") or []
            list_items = [normalize_line(str(i)) for i in items_list if str(i).strip()]
            if list_items:
                rendered = "\n".join([f"- {i}" for i in list_items])
                blocks.append(
                    {
                        "section": section_label,
                        "type": "list",
                        "text": apply_subsection_prefix(rendered),
                    }
                )

    # print(f"[구조화 완료] 총 {len(blocks)}개의 블록 생성됨\n")
    return blocks


def chunk_blocks(
    blocks: List[Dict[str, str]],
    chunk_size: int,
    chunk_overlap: int,
    max_tokens: int,
    page_title: str,
) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []
    current: List[str] = []
    current_len = 0
    current_section = ""
    chunk_index = 0

    def append_chunk(text: str, section: str) -> None:
        nonlocal chunk_index
        prefix = build_prefix(page_title, section)
        limit = effective_max_tokens(max_tokens, prefix)
        for part in split_by_max_tokens(text, limit, chunk_overlap):
            chunks.append(
                {
                    "section": section,
                    "chunk_index": str(chunk_index),
                    "text": part,
                }
            )
            chunk_index += 1

    def flush() -> None:
        nonlocal current, current_len, chunk_index
        if not current:
            return
        text = "\n".join(current).strip()
        if text:
            append_chunk(text, current_section)
        
        # 중첩(overlap) 처리
        if chunk_overlap > 0 and text:
            tail = text[-chunk_overlap:]
            current = [tail]
            current_len = len(tail)
        else:
            current = []
            current_len = 0

    def emit_single(text: str, section: str) -> None:
        nonlocal current, current_len, chunk_index, current_section
        if current:
            # 기존 버퍼가 있다면 비우고 새로운 섹션 적용
            flush()
        
        append_chunk(text, section)
        current = []
        current_len = 0
        current_section = section

    for block in blocks:
        section = block.get("section", "")
        text = block.get("text", "").strip()
        block_type = block.get("type", "")
        
        if not text:
            continue

        # [수정 포인트] 섹션이 바뀌었는지 확인 (버퍼가 비어있어도 섹션 이름은 업데이트해야 함)
        if section and section != current_section:
            if current:
                # 버퍼에 내용이 있다면 이전 섹션 이름으로 내보냄
                # (단, overlap 데이터가 다음 섹션으로 넘어가지 않게 flush 로직 확인 필요)
                # 여기서는 섹션이 바뀌면 overlap을 적용하지 않고 깔끔하게 비웁니다.
                temp_overlap = chunk_overlap
                chunk_overlap = 0 # 섹션 경계에선 overlap 방지
                flush()
                chunk_overlap = temp_overlap
            
            current_section = section
            current = []
            current_len = 0

        # 표나 인용구는 덩어리째로 저장
        if block_type in {"table", "blockquote"}:
            emit_single(text, current_section)
            continue

        # 너무 긴 텍스트는 강제로 자름
        if len(text) > chunk_size:
            if current:
                flush()
            for i in range(0, len(text), max(chunk_size - chunk_overlap, 1)):
                part = text[i : i + chunk_size]
                if part.strip():
                    append_chunk(part.strip(), current_section)
            continue

        # 일반 텍스트 합치기
        if current_len + len(text) + 1 <= chunk_size:
            current.append(text)
            current_len += len(text) + 1
        else:
            flush()
            current.append(text)
            current_len = len(text)

    flush()
    return chunks


def get_category_links(driver: webdriver.Chrome) -> List[str]:
    selector = "._06t1dW0R#category-문서 a[href], ._06t1dW0R#category-분류 a[href]"
    containers = driver.find_elements(
        By.CSS_SELECTOR, "._06t1dW0R#category-문서, ._06t1dW0R#category-분류"
    )
    if not containers:
        return []
    links: List[str] = []
    for el in driver.find_elements(By.CSS_SELECTOR, selector):
        href = el.get_attribute("href")
        if href:
            links.append(href)
    return links


def crawl_page(
    driver: webdriver.Chrome,
    url: str,
    timeout: int,
    render_wait: float,
    min_block_chars: int,
) -> CrawlResult:
    driver.get(url)
    try:
        wait_for_ready(driver, timeout)
    except TimeoutException:
        pass
    if render_wait > 0:
        time.sleep(render_wait)

    items = extract_structured_items(driver)
    blocks = build_blocks(items, min_block_chars)
    return CrawlResult(
        url=url,
        title=driver.title or "",
        blocks=blocks,
    )


def iter_follow_links(
    links: Iterable[str],
    base_url: str,
    base_domains: Set[str],
    restrict_path_prefix: Optional[str],
) -> Iterable[str]:
    for href in links:
        abs_url = urljoin(base_url, href)
        abs_url = normalize_url(abs_url)
        if not is_same_domain(abs_url, base_domains):
            continue
        if restrict_path_prefix:
            if not urlparse(abs_url).path.startswith(restrict_path_prefix):
                continue
        yield abs_url


def should_write_page(
    result: CrawlResult,
    include_category: bool,
    min_page_chars: int,
    min_paragraphs: int,
) -> bool:
    if not include_category and is_category_url(result.url):
        return False
    total_chars = sum(len(b.get("text", "")) for b in result.blocks)
    if total_chars < min_page_chars:
        return False
    if min_paragraphs > 0:
        para_count = sum(1 for b in result.blocks if b.get("type") == "paragraph")
        if para_count < min_paragraphs:
            return False
    return True


def crawl(
    start_urls: List[str],
    max_pages: int,
    timeout: int,
    render_wait: float,
    restrict_path_prefix: Optional[str],
    headless: bool,
    min_block_chars: int,
) -> List[CrawlResult]:
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1200,900")

    driver = webdriver.Chrome(options=options)
    base_domains = {urlparse(u).netloc for u in start_urls if u}

    results: List[CrawlResult] = []
    seen: Set[str] = set()
    queue: deque[str] = deque([normalize_url(u) for u in start_urls if u])

    try:
        while queue and len(seen) < max_pages:
            url = queue.popleft()
            if url in seen:
                continue
            seen.add(url)

            print(f"[{len(seen)}/{max_pages}] crawling: {url}")
            result = crawl_page(driver, url, timeout, render_wait, min_block_chars)
            results.append(result)

            category_links = get_category_links(driver)
            if category_links:
                for next_url in iter_follow_links(
                    category_links,
                    url,
                    base_domains,
                    restrict_path_prefix,
                ):
                    if next_url not in seen:
                        queue.append(next_url)
            print(f"[{len(seen)}/{max_pages}] queued: {len(queue)}")
    finally:
        driver.quit()

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-url",
        action="append",
        default=[],
        help="Repeat this flag to pass multiple start URLs.",
    )
    parser.add_argument("--max-pages", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--render-wait", type=float, default=1.5)
    parser.add_argument(
        "--restrict-path-prefix",
        default="/w/",
        help="Limit crawled URLs to this path prefix. Use empty string to disable.",
    )
    parser.add_argument("--output", default="data/attack_on_Titan_Namu_new.jsonl")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=90)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1800,
        help="Max tokens per chunk. Set 0 to disable.",
    )
    parser.add_argument("--min-block-chars", type=int, default=30)
    parser.add_argument("--min-page-chars", type=int, default=300)
    parser.add_argument(
        "--min-paragraphs",
        type=int,
        default=1,
        help="Minimum paragraph blocks to keep a page. Set 0 to disable.",
    )
    parser.add_argument(
        "--include-category",
        action="store_true",
        help="Include category pages in output if set.",
    )
    parser.add_argument("--no-headless", action="store_true")
    return parser.parse_args()


def write_outputs(
    results: List[CrawlResult],
    output_path: str,
    chunk_size: int,
    chunk_overlap: int,
    max_tokens: int,
    include_category: bool,
    min_page_chars: int,
    min_paragraphs: int,
) -> None:
    base, ext = output_path.rsplit(".", 1)
    part_index = 1
    page_index = 0
    f = None
    try:
        for item in results:
            if not should_write_page(item, include_category, min_page_chars, min_paragraphs):
                continue
            if page_index % 500 == 0:
                if f:
                    f.close()
                part_path = f"{base}_part{part_index}.{ext}"
                f = open(part_path, "w", encoding="utf-8")
                part_index += 1
            page_index += 1
            chunks = chunk_blocks(
                item.blocks, chunk_size, chunk_overlap, max_tokens, item.title
            )
            for chunk in chunks:
                payload = {
                    "title": item.title,
                    "section": chunk.get("section", ""),
                    "chunk_index": chunk.get("chunk_index", ""),
                    "text": chunk.get("text", ""),
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    finally:
        if f:
            f.close()

    print(f"Wrote {page_index} pages to {base}_part*.{ext}")


def main() -> None:
    args = parse_args()
    start_urls = args.start_url or DEFAULT_START_URLS
    restrict_prefix = args.restrict_path_prefix or None
    results = crawl(
        start_urls=start_urls,
        max_pages=args.max_pages,
        timeout=args.timeout,
        render_wait=args.render_wait,
        restrict_path_prefix=restrict_prefix,
        headless=not args.no_headless,
        min_block_chars=args.min_block_chars,
    )
    write_outputs(
        results=results,
        output_path=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_tokens=args.max_tokens,
        include_category=args.include_category,
        min_page_chars=args.min_page_chars,
        min_paragraphs=args.min_paragraphs,
    )


if __name__ == "__main__":
    main()
