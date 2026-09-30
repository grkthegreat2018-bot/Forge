"""Shared web fetch/parse primitives (critique F20).

Extracted from the duplicated logic in:
- ``forge_gui/api/web_tools.py`` (agent harness)
- ``forge/self_play/discovery/discovery_tools.py`` (self-play discovery)

Stdlib-only (urllib, re, html.parser) — no third-party deps, no imports from
forge_gui or forge.self_play. Both subsystems import from here to avoid
duplicating URL safety checks, HTTP fetching, DuckDuckGo HTML parsing, and
HTML-to-text stripping.

Fetch model (R51): ``fetch_url`` reads ANY http(s) page — scheme-less
domains are upgraded to https, redirects are followed, non-HTML bodies
(JSON/text) pass through raw, and HTML is converted to markdown-ish text
with inline ``[label](url)`` links plus a separate deduped ``links`` list
so the model can see link names and follow them. ``offset`` pages through
long documents. Search has a multi-engine chain (DDG HTML → DDG Lite →
Bing → Google News RSS) so a single dead/bot-walled endpoint can't sink a
query.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, quote_plus, urljoin, urlparse
from urllib.request import Request, urlopen

# ── constants ───────────────────────────────────────────────────────────
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
# Wikimedia robot policy (w.wiki/4wJS) 403s browser-spoofed UAs on its
# API hosts; it requires a descriptive UA naming the client.
BOT_UA = ("ForgeAI/1.0 (self-play research; python-urllib; "
          "contact: local-dev)")
TIMEOUT_S = 12
MAX_FETCH_CHARS = 4000
MAX_SNIPPET = 400
MAX_TITLE = 200
MAX_LINKS = 60
DEFAULT_N = 5

# DuckDuckGo HTML result-block regex.
RE_RESULT = re.compile(
    r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
RE_TAG = re.compile(r"<[^>]+>")

# URL schemes permitted for web_fetch (block javascript:/file:/data:/...).
ALLOWED_SCHEMES = {"http", "https"}

# A leading "word:" prefix means the token is a real scheme
# (javascript:/file:/data:) — reject it rather than upgrading to https.
_RE_SCHEME_PREFIX = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")

_RE_WIKIMEDIA_HOST = re.compile(
    r"(^|\.)(wikimedia|wikipedia|wiktionary|wikidata|wikibooks|"
    r"wikiquote|wikisource|wikinews|wikiversity|wikivoyage|mediawiki|"
    r"wikifunctions)\.org$", re.IGNORECASE)


def _ua_for(url: str) -> str:
    """Pick the UA a host expects: descriptive bot UA for Wikimedia
    (their robot policy blocks browser UAs), browser UA elsewhere."""
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return UA
    return BOT_UA if _RE_WIKIMEDIA_HOST.search(host) else UA


def is_safe_url(url: str) -> bool:
    """True iff the URL has an http/https scheme."""
    try:
        return urlparse(url).scheme.lower() in ALLOWED_SCHEMES
    except Exception:
        return False


def normalize_url(url: str) -> str:
    """Upgrade a bare domain/host to https so the model can fetch
    'example.com' or 'docs.rs/...' directly. Anything with an explicit
    non-http scheme (javascript:, file:, data:, mailto:) is left alone for
    ``is_safe_url`` to reject."""
    url = url.strip()
    if url and not _RE_SCHEME_PREFIX.match(url):
        return "https://" + url
    return url


def strip_ddg_redirect(href: str) -> str:
    """DuckDuckGo wraps result URLs in a redirect; unwrap ``uddg=``."""
    if "uddg=" in href:
        qs = parse_qs(urlparse(href).query)
        return qs.get("uddg", [href])[0]
    return href


def html_to_text(html: str) -> str:
    """Strip HTML tags and collapse whitespace to a single line."""
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html,
                  flags=re.DOTALL | re.IGNORECASE)
    text = RE_TAG.sub(" ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def http_get(url: str, timeout: float = TIMEOUT_S,
             accept_language: str = "en") -> str:
    """GET a URL and return decoded text. Raises on network error."""
    text, _final, _ctype = http_get_full(
        url, timeout=timeout, accept_language=accept_language)
    return text


def http_get_full(url: str, timeout: float = TIMEOUT_S,
                  accept_language: str = "en") -> tuple[str, str, str]:
    """GET a URL → ``(decoded_text, final_url, content_type)``.

    ``final_url`` is the post-redirect location (news/short links resolve
    to the real article); ``content_type`` is the bare MIME type or ""
    when the response doesn't declare one.
    """
    req = Request(url, headers={"User-Agent": _ua_for(url),
                                "Accept-Language": accept_language,
                                "Accept": ("text/html,application/xhtml+xml,"
                                           "application/json;q=0.9,"
                                           "text/plain;q=0.9,*/*;q=0.5")})
    with urlopen(req, timeout=timeout) as r:
        raw = r.read()
        headers = getattr(r, "headers", None)
        ctype = ""
        charset = "utf-8"
        if headers is not None:
            try:
                ctype = (headers.get("Content-Type") or "")
            except Exception:
                ctype = ""
            m = re.search(r"charset=([\w.-]+)", ctype)
            if m:
                charset = m.group(1)
            ctype = ctype.split(";")[0].strip().lower()
        final = getattr(r, "geturl", None)
        try:
            final_url = final() if callable(final) else url
        except Exception:
            final_url = url
        return raw.decode(charset, "ignore"), (final_url or url), ctype


def parse_ddg_html(html: str, n: int = DEFAULT_N) -> list[dict[str, str]]:
    """Parse DuckDuckGo HTML results into a list of {url, title, snippet}."""
    results: list[dict[str, str]] = []
    for m in RE_RESULT.finditer(html):
        if len(results) >= n:
            break
        href = unescape(m.group(1))
        href = strip_ddg_redirect(href)
        # Skip ad/tracker links DDG injects as results (y.js redirectors
        # carry the destination base64 in u3=, not uddg=, so they cannot
        # be unwrapped) and anything still pointing at duckduckgo.com.
        if ("duckduckgo.com" in href or "ad_domain=" in href
                or not is_safe_url(href)):
            continue
        title = unescape(RE_TAG.sub("", m.group(2))).strip()
        snippet = unescape(RE_TAG.sub("", m.group(3))).strip()
        results.append({
            "url": href,
            "title": title[:MAX_TITLE],
            "snippet": snippet[:MAX_SNIPPET],
        })
    return results


# ── fallback search engines (keyless) ───────────────────────────────────
# DDG Lite: single-table layout — result anchors carry class=result-link,
# snippets sit in the following <td class="result-snippet"> row.
_RE_LITE_LINK = re.compile(
    r"<a\b[^>]*class=[\"']result-link[\"'][^>]*>(.*?)</a>",
    re.DOTALL | re.IGNORECASE)
_RE_LITE_SNIP = re.compile(
    r"<td[^>]*class=[\"']result-snippet[\"'][^>]*>(.*?)</td>",
    re.DOTALL | re.IGNORECASE)
_RE_HREF = re.compile(r"href\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)

# Bing: results are <li class="b_algo"> blocks with <h2><a> + <p>.
_RE_BING_BLOCK = re.compile(
    r'<li class="b_algo"[^>]*>(.*?)</li>', re.DOTALL | re.IGNORECASE)
_RE_BING_A = re.compile(
    r"<h2[^>]*>.*?<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
    re.DOTALL | re.IGNORECASE)
_RE_BING_P = re.compile(r"<p[^>]*>(.*?)</p>", re.DOTALL | re.IGNORECASE)


def _href_of(tag: str) -> str:
    m = _RE_HREF.search(tag)
    return unescape(m.group(1)) if m else ""


def ddg_lite_search(query: str, n: int = DEFAULT_N) -> dict[str, Any]:
    """DuckDuckGo Lite endpoint — survives when html.duckduckgo.com is
    bot-walled or changes markup."""
    try:
        html = http_get(
            f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}")
    except Exception as e:
        return {"results": [], "error": f"fetch failed: {e}"}
    anchors = list(_RE_LITE_LINK.finditer(html))
    links = [_href_of(m.group(0)) for m in anchors]
    titles = [unescape(RE_TAG.sub("", m.group(1))).strip()
              for m in anchors]
    snips = [unescape(RE_TAG.sub("", m.group(1))).strip()
             for m in _RE_LITE_SNIP.finditer(html)]
    results = []
    for i, href in enumerate(links):
        if len(results) >= n:
            break
        href = strip_ddg_redirect(href)
        if "duckduckgo.com" in href or not is_safe_url(href):
            continue
        results.append({
            "url": href,
            "title": (titles[i] if i < len(titles) else "")[:MAX_TITLE],
            "snippet": (snips[i] if i < len(snips) else "")[:MAX_SNIPPET],
        })
    return {"results": results,
            "error": None if results else "no results"}


def bing_search(query: str, n: int = DEFAULT_N) -> dict[str, Any]:
    """Bing HTML search — second fallback. Keyless; bot-walls some
    regions, in which case it parses to 0 results and the chain moves on."""
    try:
        html = http_get(
            f"https://www.bing.com/search?q={quote_plus(query)}"
            f"&count={max(n, 10)}&setlang=en")
    except Exception as e:
        return {"results": [], "error": f"fetch failed: {e}"}
    results = []
    for block in _RE_BING_BLOCK.findall(html):
        if len(results) >= n:
            break
        m = _RE_BING_A.search(block)
        if not m:
            continue
        href = unescape(m.group(1))
        if "bing.com" in href or "microsoft.com" in href \
                or not is_safe_url(href):
            continue
        pm = _RE_BING_P.search(block)
        results.append({
            "url": href,
            "title": unescape(RE_TAG.sub("", m.group(2))).strip()[:MAX_TITLE],
            "snippet": unescape(RE_TAG.sub(
                "", pm.group(1) if pm else "")).strip()[:MAX_SNIPPET],
        })
    return {"results": results,
            "error": None if results else "no results"}


def ddg_search(query: str, n: int = DEFAULT_N) -> dict[str, Any]:
    """Web search (no API key). Returns {results, error}.

    Engine chain: DDG HTML → DDG Lite → Bing → Google News RSS. A network
    error on the primary endpoint is reported as 'fetch failed' only when
    every fallback also fails.
    """
    if not query.strip():
        return {"results": [], "error": "empty query"}
    first_err = None
    try:
        html = http_get(
            f"https://html.duckduckgo.com/html/?q={quote_plus(query)}")
        results = parse_ddg_html(html, n=n)
        if results:
            return {"results": results, "error": None}
    except Exception as e:
        first_err = f"fetch failed: {e}"
    # fallbacks: a DDG parse that returns nothing parseable (portal
    # homepages, markup changes, bot walls) cascades to the next engine
    for fb in (ddg_lite_search, bing_search):
        res = fb(query, n)
        if res.get("results"):
            return res
    # last resort: Google News RSS — real dated items instead of an error
    news = google_news_search(query, n=n)
    if news.get("results"):
        return news
    return {"results": [],
            "error": first_err or news.get("error") or "no results"}


def google_news_search(query: str, n: int = DEFAULT_N) -> dict[str, Any]:
    """Google News RSS search — real headlines, no API key.

    DDG's HTML endpoint returns portal homepages and ad redirectors for
    news-style queries; the Google News RSS feed returns actual articles
    with title/link/pubDate/source. An empty query returns today's top
    headlines feed.
    """
    try:
        base = ("https://news.google.com/rss/search?q="
                f"{quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
                if query.strip() else
                "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en")
        root = ET.fromstring(http_get(base, timeout=15))
        results = []
        for item in root.iter("item"):
            if len(results) >= n:
                break
            src = item.find("source")
            results.append({
                "title": (item.findtext("title") or "").strip()[:MAX_TITLE],
                "url": (item.findtext("link") or "").strip(),
                "published": (item.findtext("pubDate") or "").strip(),
                "source": (src.text or "").strip() if src is not None else "",
                "snippet": html_to_text(
                    item.findtext("description") or "")[:MAX_SNIPPET],
            })
        return {"results": results, "error": None if results else "no results"}
    except Exception as e:
        return {"results": [], "error": str(e)}


def wikipedia_search(query: str, n: int = 3) -> dict[str, Any]:
    """Search Wikipedia via the REST API. Returns summaries."""
    try:
        search_url = (
            f"https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&format=json&srlimit={n}&srsearch={quote_plus(query)}"
        )
        data = json.loads(http_get(search_url, accept_language=""))
        items = data.get("query", {}).get("search", [])
        results = []
        for item in items[:n]:
            title = item.get("title", "")
            snippet = RE_TAG.sub("", item.get("snippet", "")).strip()
            results.append({
                "title": title,
                "snippet": snippet[:MAX_SNIPPET],
                "url": f"https://en.wikipedia.org/wiki/{quote_plus(title)}",
            })
        return {"results": results, "error": None if results else "no results"}
    except Exception as e:
        return {"results": [], "error": str(e)}


def arxiv_search(query: str, n: int = 3) -> dict[str, Any]:
    """Search arXiv for academic papers via the Atom API."""
    try:
        url = (f"http://export.arxiv.org/api/query?search_query=all:{quote_plus(query)}"
               f"&start=0&max_results={n}")
        xml = http_get(url, timeout=15)
        entries = re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL)
        results = []
        for entry in entries[:n]:
            title = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            summary = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
            link = re.search(r'<id>(.*?)</id>', entry, re.DOTALL)
            published = re.search(r"<published>(.*?)</published>", entry, re.DOTALL)
            authors = re.findall(r"<author>\s*<name>(.*?)</name>",
                                 entry, re.DOTALL)
            if title:
                results.append({
                    "title": RE_TAG.sub("", title.group(1)).strip()[:MAX_TITLE],
                    "summary": summary.group(1).strip()[:MAX_SNIPPET] if summary else "",
                    "url": link.group(1).strip() if link else "",
                    "published": published.group(1)[:10] if published else "",
                    "authors": [RE_TAG.sub("", a).strip()
                                for a in authors[:8]],
                })
        return {"results": results, "error": None if results else "no results"}
    except Exception as e:
        return {"results": [], "error": str(e)}


# ── HTML → markdown-ish page extraction ─────────────────────────────────
_RE_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)
_RE_META_DESC = re.compile(
    r"<meta[^>]+name=[\"']description[\"'][^>]+content=[\"']([^\"']+)[\"']",
    re.IGNORECASE)
# Cheap "is this HTML" sniff for bodies without a Content-Type.
_RE_HTML_SNIFF = re.compile(
    r"<(?:html|head|body|div|p|a|table|span|article|section|h[1-6])\b",
    re.IGNORECASE)


class _PageExtract(HTMLParser):
    """Streaming HTML → markdown-ish text extractor.

    - Drops script/style/nav-chrome entirely.
    - Block tags become newlines; headings get ``#``/``##``/``###``;
      list items get ``- ``.
    - ``<a href>`` renders inline as ``[label](absolute-url)`` and is
      also collected into ``links`` so the model can see link names and
      follow any of them.
    - ``<img alt>`` keeps its alt text as ``[image: alt]``.
    """

    BLOCK = frozenset({
        "p", "div", "br", "hr", "li", "ul", "ol", "tr", "td", "th",
        "table", "section", "article", "header", "footer", "nav", "aside",
        "main", "blockquote", "pre", "figure", "figcaption", "form",
        "fieldset", "dl", "dt", "dd", "address", "hgroup",
    })
    SKIP = frozenset({
        "script", "style", "noscript", "template", "iframe", "svg",
        "select", "option", "head", "button", "object", "embed", "canvas",
    })
    HEADING = {"h1": "# ", "h2": "## ", "h3": "### ",
               "h4": "#### ", "h5": "##### ", "h6": "###### "}

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base = base_url
        self.out: list[str] = []
        self.links: list[dict[str, str]] = []
        self._skip = 0
        self._a_href: str | None = None
        self._a_text: list[str] = []
        self._seen_links: set[str] = set()

    # ── helpers ──
    def _nl(self, n: int = 1) -> None:
        self.out.append("\n" * n)

    def _flush_link(self) -> None:
        text = " ".join("".join(self._a_text).split())[:120]
        href = self._a_href or ""
        self._a_href = None
        self._a_text = []
        if not text:
            return
        if not href or href.startswith(("#", "javascript:", "mailto:",
                                        "tel:", "data:")):
            self.out.append(text)
            return
        href = urljoin(self.base, href)
        if not is_safe_url(href):
            self.out.append(text)
            return
        self.out.append(f"[{text}]({href})")
        key = f"{text}|{href}"
        if key not in self._seen_links and len(self.links) < MAX_LINKS:
            self._seen_links.add(key)
            self.links.append({"text": text[:80], "url": href})

    # ── parser hooks ──
    def handle_starttag(self, tag, attrs):
        if self._skip:
            if tag in self.SKIP:
                self._skip += 1
            return
        if tag in self.SKIP:
            self._skip = 1
            return
        if tag == "a":
            if self._a_href is not None:
                self._flush_link()          # nested <a> — close outer
            self._a_href = dict(attrs).get("href") or ""
            self._a_text = []
            return
        if tag == "img":
            alt = dict(attrs).get("alt", "").strip()
            if alt:
                self.out.append(f"[image: {alt[:120]}]")
            return
        if tag in self.HEADING:
            self._nl(2)
            self.out.append(self.HEADING[tag])
        elif tag == "li":
            self._nl()
            self.out.append("- ")
        elif tag in self.BLOCK:
            self._nl()

    def handle_endtag(self, tag):
        if self._skip:
            if tag in self.SKIP:
                self._skip -= 1
            return
        if tag == "a":
            self._flush_link()
        elif tag in self.BLOCK or tag in self.HEADING:
            self._nl()

    def handle_data(self, data):
        if self._skip:
            return
        if self._a_href is not None:
            self._a_text.append(data)
        else:
            self.out.append(data)

    def result(self) -> tuple[str, list[dict[str, str]]]:
        if self._a_href is not None:
            self._flush_link()
        text = "".join(self.out)
        # collapse intra-line whitespace, then >2 newlines
        lines = [re.sub(r"[ \t\xa0]+", " ", ln).strip() for ln in
                 text.split("\n")]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text, self.links


def extract_page(html: str, base_url: str) -> dict[str, Any]:
    """HTML → {title, description, text, links} for LLM consumption."""
    title_m = _RE_TITLE.search(html)
    title = unescape(RE_TAG.sub("", title_m.group(1))).strip() \
        if title_m else ""
    desc_m = _RE_META_DESC.search(html)
    desc = unescape(desc_m.group(1)).strip() if desc_m else ""
    parser = _PageExtract(base_url)
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        pass                       # malformed HTML — use what parsed
    text, links = parser.result()
    return {"title": title[:MAX_TITLE], "description": desc[:MAX_SNIPPET],
            "text": text, "links": links}


def fetch_url(url: str, max_chars: int = MAX_FETCH_CHARS,
              offset: int = 0) -> dict[str, Any]:
    """Fetch any http(s) URL → readable page content for the model.

    - Bare domains are upgraded to https (``example.com`` works).
    - Redirects are followed; ``final_url`` reports where it landed
      (matters for news/short links).
    - HTML becomes markdown-ish text with inline ``[label](url)`` links;
      ``links`` lists the page's anchors so the model can follow them.
    - Non-HTML bodies (JSON, plain text, XML) pass through raw.
    - ``offset`` + ``next_offset`` page through documents longer than
      ``max_chars``.
    """
    url = normalize_url(url)
    if not url:
        return {"text": "", "url": url, "error": "empty url"}
    if not is_safe_url(url):
        return {"text": "", "url": url,
                "error": f"unsupported URL scheme (only {sorted(ALLOWED_SCHEMES)})"}
    try:
        body, final_url, ctype = http_get_full(url)
    except Exception as e:
        return {"text": "", "url": url, "error": f"fetch failed: {e}"}
    is_html = ("html" in ctype or
               (not ctype and bool(_RE_HTML_SNIFF.search(body[:4000]))))
    if is_html:
        page = extract_page(body, final_url)
        text, links = page["text"], page["links"]
        title, desc = page["title"], page["description"]
    else:
        text, links, title, desc = body, [], "", ""
    offset = max(0, int(offset))
    window = text[offset:offset + max_chars]
    end = offset + len(window)
    res = {"text": window, "url": url, "final_url": final_url,
           "title": title, "chars": len(text),
           "truncated": end < len(text), "error": None}
    if end < len(text):
        res["next_offset"] = end
    if desc:
        res["description"] = desc
    if links:
        res["links"] = links
    if not is_html:
        res["content_type"] = ctype or "text"
    return res
