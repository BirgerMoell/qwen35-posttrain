#!/usr/bin/env python3
"""Discover official Högskoleprovet exam pages and PDF links from Studera.nu.

This script creates a source manifest only. It does not download or redistribute
exam text.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

INDEX_URL = "https://www.studera.nu/hogskoleprov/om/forbereda/tidigare/"
DEFAULT_OUT = Path("data/exam_mcq/oellm-eu-exam-mcq-v1/source_manifests/hogskoleprovet_sources.json")


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attrs_dict = dict(attrs)
        self._href = attrs_dict.get("href")
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._href:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href:
            text = re.sub(r"\s+", " ", "".join(self._text)).strip()
            self.links.append({"href": self._href, "text": text})
            self._href = None
            self._text = []


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "OpenEuroLLM exam source discovery"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def absolute(base: str, href: str) -> str:
    return urllib.parse.urljoin(base, href)


def parse_links(url: str) -> list[dict]:
    parser = LinkParser()
    parser.feed(fetch(url))
    return [{"url": absolute(url, link["href"]), "text": link["text"]} for link in parser.links if link["href"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--index-url", default=INDEX_URL)
    parser.add_argument("--limit-pages", type=int, default=0)
    args = parser.parse_args()

    index_links = parse_links(args.index_url)
    seen_urls = set()
    exam_pages = []
    for link in index_links:
        if "Högskoleprovet" not in link["text"] or "/hogskoleprov/" not in link["url"]:
            continue
        if link["url"].rstrip("/") == args.index_url.rstrip("/"):
            continue
        if link["url"] in seen_urls:
            continue
        seen_urls.add(link["url"])
        exam_pages.append(link)
    if args.limit_pages:
        exam_pages = exam_pages[: args.limit_pages]

    pages = []
    for page in exam_pages:
        links = parse_links(page["url"])
        pdfs = [link for link in links if ".pdf" in link["url"].lower() or "(pdf" in link["text"].lower()]
        pages.append(
            {
                "title": page["text"],
                "url": page["url"],
                "pdf_links": pdfs,
                "redistribution_status": "link_only_review_required",
                "notes": "Official Studera/UHR source. Do not redistribute raw question text until reuse terms are cleared.",
            }
        )

    manifest = {
        "source_id": "studera_hogskoleprovet_archive",
        "index_url": args.index_url,
        "discovered_pages": len(pages),
        "pages": pages,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "pages": len(pages), "pdfs": sum(len(p["pdf_links"]) for p in pages)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
