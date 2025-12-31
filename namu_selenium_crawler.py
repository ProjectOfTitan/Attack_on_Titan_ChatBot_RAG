#!/usr/bin/env python3
import argparse
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urldefrag, urljoin, urlparse

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


@dataclass
class CrawlResult:
    url: str
    content_text: str
    structured_text: str
    title: str
    blocks: List[Dict[str, str]]
    targeted_links: List[str]
    all_links: List[str]


def normalize_url(url: str) -> str:
    clean, _ = urldefrag(url)
    return clean


def is_same_domain(url: str, base_domain: str) -> bool:
    return urlparse(url).netloc == base_domain


def wait_for_ready(driver: webdriver.Chrome, timeout: int) -> None:
    def _ready(drv: webdriver.Chrome) -> bool:
        return drv.execute_script("return document.readyState") == "complete"

    WebDriverWait(driver, timeout).until(_ready)


def collect_links(driver: webdriver.Chrome, selector: str) -> List[str]:
    links: List[str] = []
    for el in driver.find_elements(By.CSS_SELECTOR, selector):
        href = el.get_attribute("href")
        if href:
            links.append(href)
    return links


def collect_content_text(driver: webdriver.Chrome, selector: str) -> str:
    blocks = driver.find_elements(By.CSS_SELECTOR, selector)
    texts = [b.text.strip() for b in blocks if b.text and b.text.strip()]
    return "\n".join(texts)


def extract_structured_items(driver: webdriver.Chrome) -> List[Dict[str, object]]:
    script = """
    const cssEscape = (s) => s.replace(/([ !"#$%&'()*+,.\\/\\/:;<=>?@[\\\\\\]^`{|}~])/g, '\\\\$1');
    const elements = Array.from(document.querySelectorAll(
      'h2,h3,h4,h5,h6,.wiki-paragraph,blockquote,table,img'
    ));
    return elements.map(el => {
      const escaped = (window.CSS && CSS.escape) ? CSS.escape('+0aMAP74') : cssEscape('+0aMAP74');
      if (el.closest('div.' + escaped + '.coLEPeSv')) return null;
      if (el.matches('table.XrDIkehY._99ba4330d69377b55a0ae92ad4998c5e')) return null;
      if (el.closest('nav,header,footer,aside')) return null;
      const tag = el.tagName.toLowerCase();
      if (tag.startsWith('h')) {
        return {type: 'heading', level: tag, text: el.innerText.trim()};
      }
      if (tag === 'img') {
        return {
          type: 'image',
          alt: el.getAttribute('alt') || '',
          src: el.getAttribute('src') || '',
          title: el.getAttribute('title') || ''
        };
      }
      if (tag === 'blockquote') {
        return {type: 'blockquote', text: el.innerText.trim()};
      }
      if (tag === 'table') {
        const rows = Array.from(el.querySelectorAll('tr')).map(tr => {
          return Array.from(tr.querySelectorAll('th,td')).map(td => td.innerText.trim());
        });
        return {type: 'table', rows};
      }
      return {type: 'paragraph', text: el.innerText.trim()};
    }).filter(Boolean);
    """
    items: List[Dict[str, object]] = driver.execute_script(script) or []
    return items


def normalize_line(text: str) -> str:
    return " ".join(text.split()).strip()


def build_blocks(items: List[Dict[str, object]]) -> Tuple[str, List[Dict[str, str]]]:
    blocks: List[Dict[str, str]] = []
    lines: List[str] = []
    current_section: Optional[str] = None

    for item in items:
        item_type = item.get("type")
        if item_type == "heading":
            text = (item.get("text") or "").strip()
            if text:
                current_section = text
                lines.append(f"[SECTION] {current_section}")
            continue

        if current_section:
            # Keep section context for all non-heading blocks.
            lines.append(f"[SECTION] {current_section}")

        if item_type == "paragraph":
            text = normalize_line(str(item.get("text") or ""))
            if text and text not in {"편집", "더 보기", "분류"}:
                lines.append(text)
                blocks.append(
                    {"section": current_section or "", "type": "paragraph", "text": text}
                )
        elif item_type == "blockquote":
            text = normalize_line(str(item.get("text") or ""))
            if text:
                if current_section:
                    lines.append(f'[QUOTE section="{current_section}"]')
                else:
                    lines.append("[QUOTE]")
                lines.append(text)
                lines.append("[/QUOTE]")
                blocks.append(
                    {
                        "section": current_section or "",
                        "type": "blockquote",
                        "text": text,
                    }
                )
        elif item_type == "table":
            rows = item.get("rows") or []
            if rows:
                if current_section:
                    lines.append(f'[TABLE section="{current_section}"]')
                else:
                    lines.append("[TABLE]")
                # Render a simple pipe table for RAG-friendly structure.
                rendered_rows: List[str] = []
                for row in rows:
                    row_cells = [str(c).replace("\n", " ").strip() for c in row if c is not None]
                    lines.append("| " + " | ".join(row_cells) + " |")
                    rendered_rows.append("| " + " | ".join(row_cells) + " |")
                lines.append("[/TABLE]")
                blocks.append(
                    {
                        "section": current_section or "",
                        "type": "table",
                        "text": "\n".join(rendered_rows),
                    }
                )
        elif item_type == "image":
            alt = normalize_line(str(item.get("alt") or ""))
            title = normalize_line(str(item.get("title") or ""))
            src = normalize_line(str(item.get("src") or ""))
            if alt or title or src:
                if current_section:
                    lines.append(
                        f'[IMAGE section="{current_section}"] alt="{alt}" title="{title}" src="{src}"'
                    )
                else:
                    lines.append(f'[IMAGE] alt="{alt}" title="{title}" src="{src}"')
                blocks.append(
                    {
                        "section": current_section or "",
                        "type": "image",
                        "text": f'alt="{alt}" title="{title}" src="{src}"',
                    }
                )

    # Remove immediate duplicate section markers.
    compact: List[str] = []
    for line in lines:
        if compact and line.startswith("[SECTION]") and compact[-1] == line:
            continue
        compact.append(line)

    return "\n".join(compact).strip(), blocks


def chunk_blocks(
    blocks: List[Dict[str, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []
    current: List[str] = []
    current_len = 0
    current_section = ""
    chunk_index = 0

    def flush() -> None:
        nonlocal current, current_len, chunk_index
        if not current:
            return
        text = "\n".join(current).strip()
        if text:
            chunks.append(
                {
                    "section": current_section,
                    "chunk_index": str(chunk_index),
                    "text": text,
                }
            )
            chunk_index += 1
        if chunk_overlap > 0 and text:
            tail = text[-chunk_overlap:]
            current = [tail]
            current_len = len(tail)
        else:
            current = []
            current_len = 0

    for block in blocks:
        section = block.get("section", "")
        text = block.get("text", "").strip()
        if not text:
            continue
        if section and section != current_section and current:
            flush()
            current_section = section
        elif section and not current_section:
            current_section = section

        if len(text) > chunk_size:
            if current:
                flush()
            for i in range(0, len(text), max(chunk_size - chunk_overlap, 1)):
                part = text[i : i + chunk_size]
                if part.strip():
                    chunks.append(
                        {
                            "section": current_section,
                            "chunk_index": str(chunk_index),
                            "text": part.strip(),
                        }
                    )
                    chunk_index += 1
            continue

        if current_len + len(text) + 1 <= chunk_size:
            current.append(text)
            current_len += len(text) + 1
        else:
            flush()
            current.append(text)
            current_len = len(text)

    flush()
    return chunks


def crawl_page(
    driver: webdriver.Chrome,
    url: str,
    timeout: int,
    render_wait: float,
) -> CrawlResult:
    driver.get(url)
    try:
        wait_for_ready(driver, timeout)
    except TimeoutException:
        pass
    if render_wait > 0:
        time.sleep(render_wait)

    content_text = collect_content_text(driver, ".fXfZ728v.Z4-o6fJU")
    items = extract_structured_items(driver)
    structured_text, blocks = build_blocks(items)
    targeted_links = collect_links(driver, ".dzF-Ff79 a.wGjUIbJ4")
    all_links = collect_links(driver, "a[href]")

    return CrawlResult(
        url=url,
        content_text=content_text,
        structured_text=structured_text,
        title=driver.title or "",
        blocks=blocks,
        targeted_links=targeted_links,
        all_links=all_links,
    )


def iter_follow_links(
    links: Iterable[str],
    base_url: str,
    base_domain: str,
    restrict_path_prefix: Optional[str],
) -> Iterable[str]:
    for href in links:
        abs_url = urljoin(base_url, href)
        abs_url = normalize_url(abs_url)
        if not is_same_domain(abs_url, base_domain):
            continue
        if restrict_path_prefix:
            if not urlparse(abs_url).path.startswith(restrict_path_prefix):
                continue
        yield abs_url


def crawl(
    start_url: str,
    max_pages: int,
    timeout: int,
    render_wait: float,
    restrict_path_prefix: Optional[str],
    headless: bool,
) -> List[CrawlResult]:
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1200,900")

    driver = webdriver.Chrome(options=options)
    base_domain = urlparse(start_url).netloc

    results: List[CrawlResult] = []
    seen: Set[str] = set()
    queue: deque[str] = deque([normalize_url(start_url)])

    try:
        while queue and len(seen) < max_pages:
            url = queue.popleft()
            if url in seen:
                continue
            seen.add(url)

            print(f"[{len(seen)}/{max_pages}] crawling: {url}")
            result = crawl_page(driver, url, timeout, render_wait)
            results.append(result)

            for next_url in iter_follow_links(
                result.targeted_links,
                url,
                base_domain,
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
        default="https://namu.wiki/w/%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8",
    )
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--render-wait", type=float, default=1.5)
    parser.add_argument(
        "--restrict-path-prefix",
        default="/w/",
        help="Limit crawled URLs to this path prefix. Use empty string to disable.",
    )
    parser.add_argument("--output", default="data/namu_crawl.jsonl")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--no-headless", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    restrict_prefix = args.restrict_path_prefix or None
    results = crawl(
        start_url=args.start_url,
        max_pages=args.max_pages,
        timeout=args.timeout,
        render_wait=args.render_wait,
        restrict_path_prefix=restrict_prefix,
        headless=not args.no_headless,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        for item in results:
            chunks = chunk_blocks(item.blocks, args.chunk_size, args.chunk_overlap)
            for chunk in chunks:
                payload = {
                    "url": item.url,
                    "title": item.title,
                    "section": chunk.get("section", ""),
                    "chunk_index": chunk.get("chunk_index", ""),
                    "text": chunk.get("text", ""),
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"Wrote {len(results)} pages to {args.output}")


if __name__ == "__main__":
    main()
