#!/usr/bin/env python3
import argparse
import json
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urldefrag, urljoin, urlparse

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


DEFAULT_START_URLS = [
    "https://namu.wiki/w/%EB%B6%84%EB%A5%98:%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8/%EC%9E%91%EC%A4%91%20%EC%82%AC%EA%B1%B4%20%EC%82%AC%EA%B3%A0",
    "https://namu.wiki/w/%EB%B6%84%EB%A5%98:%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8/%EC%84%A4%EC%A0%95",
    "https://namu.wiki/w/%EB%B6%84%EB%A5%98:%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8/%EC%A7%80%EC%97%AD",
    "https://namu.wiki/w/%EB%B6%84%EB%A5%98:%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8/%EB%93%B1%EC%9E%A5%EC%9D%B8%EB%AC%BC?namespace=%EB%AC%B8%EC%84%9C&cuntil=%EC%B9%B4%EB%A6%AC%EB%82%98%20%EB%B8%8C%EB%9D%BC%EC%9A%B4",
    "https://namu.wiki/w/%EB%B6%84%EB%A5%98:%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8/%EB%93%B1%EC%9E%A5%EC%9D%B8%EB%AC%BC?namespace=%EB%AC%B8%EC%84%9C&cfrom=%EC%B9%B4%EC%95%BC%28%EC%A7%84%EA%B2%A9%EC%9D%98%20%EA%B1%B0%EC%9D%B8%29",
]


@dataclass
class CrawlResult:
    url: str
    title: str
    blocks: List[Dict[str, str]]


def normalize_url(url: str) -> str:
    clean, _ = urldefrag(url)
    return clean


def is_same_domain(url: str, domains: Set[str]) -> bool:
    return urlparse(url).netloc in domains


def wait_for_ready(driver: webdriver.Chrome, timeout: int) -> None:
    def _ready(drv: webdriver.Chrome) -> bool:
        return drv.execute_script("return document.readyState") == "complete"

    WebDriverWait(driver, timeout).until(_ready)


def normalize_line(text: str) -> str:
    cleaned = " ".join(text.split()).strip()
    cleaned = cleaned.replace("[편집]", "").strip()
    cleaned = re.sub(r"<img[^>]*>", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"<[^>]+>", "", cleaned).strip()
    return cleaned


def extract_structured_items(driver: webdriver.Chrome) -> List[Dict[str, object]]:
    script = """
    const cssEscape = (s) => s.replace(/([ !"#$%&'()*+,.\\/\\/:;<=>?@[\\\\\\]^`{|}~])/g, '\\\\$1');
    const elements = Array.from(document.querySelectorAll(
      'h2,h3,h4,h5,h6,.wiki-paragraph,div._4tRz4taR,blockquote,table,ul,ol,details'
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
    return elements.map(el => {
      const escaped = (window.CSS && CSS.escape) ? CSS.escape('+0aMAP74') : cssEscape('+0aMAP74');
      if (el.closest('div.' + escaped + '.coLEPeSv')) return null;
      if (el.matches('table.XrDIkehY._99ba4330d69377b55a0ae92ad4998c5e')) return null;
      if (el.closest('nav,header,footer,aside')) return null;
      if (el.closest('.wiki-toc,.wiki-toc-wrapper,.toc,.toc-wrapper')) return null;
      if (el.closest('.wiki-category,.category')) return null;
      if (el.closest('.gyYVEIst._13s6c0Md')) return null;
      if (el.closest('.F83V3QG7')) return null;
      if (el.closest('.rqSF8he0')) return null;
      if (el.closest('.WMQhH6LY._0ahwpliD')) return null;
      const tag = el.tagName.toLowerCase();
      if (tag.startsWith('h')) {
        return {type: 'heading', level: tag, text: extractText(el)};
      }
      if (tag === 'blockquote') {
        return {type: 'blockquote', text: extractText(el)};
      }
      if (tag === 'table') {
        const rows = Array.from(el.querySelectorAll('tr')).map(tr => {
          return Array.from(tr.querySelectorAll('th,td')).map(td => extractText(td));
        });
        return {type: 'table', rows};
      }
      if (tag === 'ul' || tag === 'ol') {
        const items = Array.from(el.querySelectorAll(':scope > li')).map(li => extractText(li));
        return {type: 'list', items};
      }
      if (tag === 'details') {
        const summaryEl = el.querySelector('summary');
        const summary = summaryEl ? extractText(summaryEl) : '';
        const bodyClone = el.cloneNode(true);
        const summaryClone = bodyClone.querySelector('summary');
        if (summaryClone) summaryClone.remove();
        const body = extractText(bodyClone);
        return {type: 'toggle', summary, text: body};
      }
      return {type: 'paragraph', text: extractText(el)};
    }).filter(Boolean);
    """
    return driver.execute_script(script) or []


def build_blocks(items: List[Dict[str, object]]) -> List[Dict[str, str]]:
    blocks: List[Dict[str, str]] = []
    current_section: Optional[str] = None

    for item in items:
        item_type = item.get("type")
        if item_type == "heading":
            text = normalize_line(str(item.get("text") or ""))
            if text:
                current_section = text
            continue

        if item_type == "paragraph":
            text = normalize_line(str(item.get("text") or ""))
            if text:
                blocks.append(
                    {"section": current_section or "", "type": "paragraph", "text": text}
                )
        elif item_type == "blockquote":
            text = normalize_line(str(item.get("text") or ""))
            if text:
                blocks.append(
                    {"section": current_section or "", "type": "blockquote", "text": text}
                )
        elif item_type == "table":
            rows = item.get("rows") or []
            rendered_rows: List[str] = []
            for row in rows:
                row_cells = [
                    normalize_line(str(c).replace("\n", " "))
                    for c in row
                    if c is not None
                ]
                row_cells = [c for c in row_cells if c]
                if not row_cells:
                    continue
                rendered_rows.append("| " + " | ".join(row_cells) + " |")
            if rendered_rows:
                blocks.append(
                    {
                        "section": current_section or "",
                        "type": "table",
                        "text": "\n".join(rendered_rows),
                    }
                )
        elif item_type == "list":
            items_list = item.get("items") or []
            list_items = [normalize_line(str(i)) for i in items_list if str(i).strip()]
            if list_items:
                rendered = "\n".join([f"- {i}" for i in list_items])
                blocks.append(
                    {"section": current_section or "", "type": "list", "text": rendered}
                )
        elif item_type == "toggle":
            summary = normalize_line(str(item.get("summary") or ""))
            text = normalize_line(str(item.get("text") or ""))
            merged = "\n".join([s for s in [summary, text] if s]).strip()
            if merged:
                blocks.append(
                    {"section": current_section or "", "type": "toggle", "text": merged}
                )

    return blocks


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


def get_category_links(driver: webdriver.Chrome) -> List[str]:
    selector = "._06t1dW0R#category-문서 a[href], ._06t1dW0R#category-분류 a[href]"
    containers = driver.find_elements(By.CSS_SELECTOR, "._06t1dW0R#category-문서, ._06t1dW0R#category-분류")
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
) -> CrawlResult:
    driver.get(url)
    try:
        wait_for_ready(driver, timeout)
    except TimeoutException:
        pass
    if render_wait > 0:
        time.sleep(render_wait)

    items = extract_structured_items(driver)
    blocks = build_blocks(items)
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


def crawl(
    start_urls: List[str],
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
            result = crawl_page(driver, url, timeout, render_wait)
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
    parser.add_argument("--max-pages", type=int, default=3000)
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
    start_urls = args.start_url or DEFAULT_START_URLS
    restrict_prefix = args.restrict_path_prefix or None
    results = crawl(
        start_urls=start_urls,
        max_pages=args.max_pages,
        timeout=args.timeout,
        render_wait=args.render_wait,
        restrict_path_prefix=restrict_prefix,
        headless=not args.no_headless,
    )

    base, ext = args.output.rsplit(".", 1)
    part_index = 1
    page_index = 0
    f = None
    try:
        for item in results:
            if page_index % 500 == 0:
                if f:
                    f.close()
                part_path = f"{base}_part{part_index}.{ext}"
                f = open(part_path, "w", encoding="utf-8")
                part_index += 1
            page_index += 1
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
    finally:
        if f:
            f.close()

    print(f"Wrote {len(results)} pages to {base}_part*.{ext}")


if __name__ == "__main__":
    main()
