#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apogee Retailer Monitor - Enhanced Anti-Bot Version
"""

import os
import re
import json
import time
import random
import logging
import sys
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import gspread
from gspread_dataframe import set_with_dataframe, get_as_dataframe
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    HAVE_PW = True
except Exception:
    HAVE_PW = False

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("apogee-monitor")

def _normalize(s: str) -> str:
    return (s or "").strip().lower()

def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        s = str(x).replace(",", "").strip()
        return int(float(s))
    except Exception:
        return None

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).replace(",", "").strip()
        val = float(s)
        if val > 5.0:
            return None
        return val
    except Exception:
        return None

def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-") or "item"

def human_delay(base: float = 3.0, jitter: float = 0.8) -> float:
    return max(0.5, base + random.uniform(-jitter, jitter))

def filter_by_retailer(df: pd.DataFrame, token: str) -> pd.DataFrame:
    if not token:
        return df
    
    tnorm = _normalize(token)
    ALIASES = {
        "gc": ["gc", "guitar center", "guitarcenter"],
        "bh": ["bh", "b&h", "b&h photo", "bhphoto", "bhphotovideo"],
        "sweetwater": ["sweetwater"],
        "vintage king": ["vintage king", "vintageking"],
        "thomann": ["thomann"],
        "apple": ["apple"],
        "yelp": ["yelp"],
    }
    
    candidates = ALIASES.get(tnorm, [tnorm])
    
    def matches(row):
        r = _normalize(str(row.get("retailer", "")))
        return any(c in r for c in candidates)
    
    return df[df.apply(matches, axis=1)].reset_index(drop=True)

def load_config() -> Dict[str, Any]:
    return {
        "USER_AGENT": os.getenv("USER_AGENT", 
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "SCRAPE_TIMEOUT": float(os.getenv("SCRAPE_TIMEOUT", "60")),
        "REQUEST_DELAY": float(os.getenv("REQUEST_DELAY", "3.0")),
        "JITTER": float(os.getenv("JITTER", "0.8")),
        "USE_PLAYWRIGHT": int(os.getenv("USE_PLAYWRIGHT", "0")),
        "DEBUG_ARTIFACTS": int(os.getenv("DEBUG_ARTIFACTS", "0")),
        "RETAILER_FILTER": os.getenv("RETAILER_FILTER", ""),
    }

def create_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none"
    })
    
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    return session

def fetch_with_requests(url: str, config: Dict) -> tuple[str, int]:
    try:
        session = create_session(config["USER_AGENT"])
        response = session.get(url, timeout=config["SCRAPE_TIMEOUT"], allow_redirects=True)
        
        if response.status_code >= 400:
            return "", response.status_code
        
        html = response.text
        if len(html) < 500 or any(x in html.lower() for x in ["cloudflare", "captcha", "access denied", "checking your browser", "press & hold"]):
            log.warning("Bot detection detected in requests response")
            return "", response.status_code
        
        return html, response.status_code
    except Exception as e:
        log.warning(f"Requests fetch failed: {e}")
        return "", 0

def fetch_with_playwright(url: str, config: Dict):
    """Enhanced Playwright with aggressive anti-bot tactics"""
    if not HAVE_PW:
        return "", 0, None, None
    
    try:
        p = sync_playwright().start()
        
        # Launch with more stealth args
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-setuid-sandbox"
            ]
        )
        
        # Create context with realistic settings
        context = browser.new_context(
            user_agent=config["USER_AGENT"],
            viewport={"width": 1920, "height": 1080},
            java_script_enabled=True,
            locale="en-US",
            timezone_id="America/New_York",
            permissions=["geolocation"],
            geolocation={"latitude": 40.7128, "longitude": -74.0060},  # NYC
            color_scheme="light",
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Windows"',
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none"
            }
        )
        
        page = context.new_page()
        
        # Enhanced stealth script
        page.add_init_script("""
            // Remove webdriver property
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            
            // Mock plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
            
            // Mock languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });
            
            // Mock hardware concurrency
            Object.defineProperty(navigator, 'hardwareConcurrency', {
                get: () => 8
            });
            
            // Mock device memory
            Object.defineProperty(navigator, 'deviceMemory', {
                get: () => 8
            });
            
            // Override permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
            
            // Mock Chrome runtime
            window.chrome = {
                runtime: {}
            };
        """)
        
        # Don't block as many resources - might look suspicious
        page.route("**/*", lambda route, request:
            route.abort() if request.resource_type in ("image", "media", "font", "stylesheet")
            else route.continue_()
        )
        
        log.info("  Navigating to URL...")
        timeout_ms = int(config["SCRAPE_TIMEOUT"] * 1000)
        
        # Navigate
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        
        # Wait longer for bot checks
        log.info("  Waiting for page to load (checking for bot detection)...")
        page.wait_for_timeout(5000)  # 5 seconds
        
        # Simulate human behavior
        try:
            # Random scroll
            for _ in range(3):
                scroll_y = random.randint(100, 500)
                page.evaluate(f"window.scrollBy(0, {scroll_y})")
                page.wait_for_timeout(random.randint(500, 1500))
            
            # Scroll back up
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(1000)
            
            # Move mouse randomly (helps with some bot detectors)
            page.mouse.move(random.randint(100, 500), random.randint(100, 500))
            page.wait_for_timeout(500)
            
        except Exception as e:
            log.debug(f"Human simulation error: {e}")
        
        # Wait a bit more
        page.wait_for_timeout(2000)
        
        html = page.content()
        
        # Check if we're still on a bot check page
        if any(x in html.lower() for x in ["press & hold", "verifying you are human", "checking your browser", "challenge"]):
            log.warning("  ⚠️  Bot detection still present after waits")
        
        return html, 200, p, (browser, context, page)
        
    except PWTimeout:
        log.warning(f"Playwright timeout on {url}")
        return "", 0, None, None
    except Exception as e:
        log.warning(f"Playwright error: {e}")
        return "", 0, None, None

def close_playwright(p, handles):
    try:
        if handles:
            browser, context, page = handles
            try:
                context.close()
            except:
                pass
            try:
                browser.close()
            except:
                pass
        if p:
            p.stop()
    except:
        pass

def extract_rating_and_count(text: str) -> Dict[str, Any]:
    result = {"avg_rating": None, "total_ratings": None, "review_count": None}
    
    rating_match = re.search(r'\b(\d+(?:\.\d+)?)\s+out\s+of\s+5', text, re.I)
    if rating_match:
        rating = _safe_float(rating_match.group(1))
        if rating and rating <= 5.0:
            result["avg_rating"] = rating
            log.debug(f"Found rating via 'X out of 5': {rating}")
    
    if not result["avg_rating"]:
        rating_match = re.search(r'\b(\d+(?:\.\d+)?)\s*/\s*5\b', text, re.I)
        if rating_match:
            rating = _safe_float(rating_match.group(1))
            if rating and rating <= 5.0:
                result["avg_rating"] = rating
                log.debug(f"Found rating via 'X/5': {rating}")
    
    if not result["avg_rating"]:
        rating_match = re.search(r'\b(?:rating|rated|score):\s*(\d+(?:\.\d+)?)', text, re.I)
        if rating_match:
            rating = _safe_float(rating_match.group(1))
            if rating and rating <= 5.0:
                result["avg_rating"] = rating
                log.debug(f"Found rating via 'Rating: X': {rating}")
    
    count_match = re.search(r'\((\d+(?:,\d+)?)\s+(?:review|rating)s?\)', text, re.I)
    if count_match:
        count = _safe_int(count_match.group(1))
        if count:
            result["total_ratings"] = count
            result["review_count"] = count
            log.debug(f"Found count via '(X reviews)': {count}")
    
    if not result["total_ratings"]:
        count_match = re.search(r'\b(\d+(?:,\d+)?)\s+(?:review|rating)s?\b', text, re.I)
        if count_match:
            count = _safe_int(count_match.group(1))
            if count:
                result["total_ratings"] = count
                result["review_count"] = count
                log.debug(f"Found count via 'X reviews': {count}")
    
    return result

def extract_jsonld(soup: BeautifulSoup) -> Dict[str, Any]:
    result = {
        "avg_rating": None,
        "total_ratings": None,
        "review_count": None,
    }
    
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or "{}")
            items = data if isinstance(data, list) else [data]
            
            for item in items:
                if not isinstance(item, dict):
                    continue
                
                agg = item.get("aggregateRating", {})
                if isinstance(agg, dict):
                    if agg.get("ratingValue"):
                        rating = _safe_float(agg.get("ratingValue"))
                        if rating and rating <= 5.0:
                            result["avg_rating"] = rating
                    
                    count = agg.get("ratingCount") or agg.get("reviewCount")
                    if count:
                        result["total_ratings"] = _safe_int(count)
                        if not result["review_count"]:
                            result["review_count"] = result["total_ratings"]
                
                if item.get("reviewCount"):
                    result["review_count"] = _safe_int(item.get("reviewCount"))
                    
        except Exception as e:
            log.debug(f"JSON-LD parse error: {e}")
            continue
    
    if result["avg_rating"] or result["total_ratings"]:
        log.debug(f"JSON-LD found: rating={result['avg_rating']}, count={result['total_ratings']}")
    
    return result

def wait_for_element(page, selectors: List[str], max_attempts: int = 20):
    for attempt in range(max_attempts):
        for sel in selectors:
            try:
                el = page.locator(sel).first
                if el.count() > 0 and el.is_visible():
                    log.debug(f"Found element with selector: {sel}")
                    return el
            except:
                pass
        page.wait_for_timeout(500)
    return None

def parse_sweetwater(page_or_html) -> Dict[str, Any]:
    if hasattr(page_or_html, 'content'):
        page = page_or_html
        
        try:
            tabs = page.locator("button, a").filter(has_text=re.compile(r"reviews?", re.I))
            if tabs.count() > 0:
                tabs.first.click()
                page.wait_for_timeout(2000)
                log.debug("Clicked reviews tab")
        except:
            pass
        
        el = wait_for_element(page, [
            "[data-qa='rating-summary']",
            "[class*='rating-summary']",
            "[class*='review-summary']",
            ".sw-rating-stars",
            "[aria-label*='rating']"
        ])
        
        if el:
            html = el.inner_html()
            result = extract_rating_and_count(html)
            if result["avg_rating"] or result["total_ratings"]:
                return result
        
        html = page.content()
    else:
        html = page_or_html
    
    soup = BeautifulSoup(html, "lxml")
    result = extract_jsonld(soup)
    
    if not (result["avg_rating"] or result["total_ratings"]):
        result.update(extract_rating_and_count(html))
    
    return result

def parse_guitarcenter(page_or_html) -> Dict[str, Any]:
    if hasattr(page_or_html, 'content'):
        page = page_or_html
        
        try:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
            page.wait_for_timeout(3000)
        except:
            pass
        
        el = wait_for_element(page, [
            "[class*='BVRRRatingSummary']",
            "[id*='BVRRSummaryContainer']",
            "[class*='bv-rating-summary']",
            "[class*='pr-snippet']",
            ".bv_avgRating_component_container",
            "[data-bv-show='rating_summary']"
        ])
        
        if el:
            html = el.inner_html()
            result = extract_rating_and_count(html)
            if result["avg_rating"] or result["total_ratings"]:
                return result
        
        html = page.content()
    else:
        html = page_or_html
    
    soup = BeautifulSoup(html, "lxml")
    result = extract_jsonld(soup)
    
    if not (result["avg_rating"] or result["total_ratings"]):
        result.update(extract_rating_and_count(html))
    
    return result

def parse_bh(page_or_html) -> Dict[str, Any]:
    if hasattr(page_or_html, 'content'):
        page = page_or_html
        
        try:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
            page.wait_for_timeout(2000)
        except:
            pass
        
        el = wait_for_element(page, [
            "[data-selenium='reviewSummary']",
            "[data-selenium='ratingSummary']",
            "[class*='review-summary']",
            ".bv_avgRating_component_container",
            "[class*='pr-snippet']"
        ])
        
        if el:
            html = el.inner_html()
            result = extract_rating_and_count(html)
            if result["avg_rating"] or result["total_ratings"]:
                return result
        
        html = page.content()
    else:
        html = page_or_html
    
    soup = BeautifulSoup(html, "lxml")
    result = extract_jsonld(soup)
    
    if not (result["avg_rating"] or result["total_ratings"]):
        result.update(extract_rating_and_count(html))
    
    return result

def parse_vintageking(page_or_html) -> Dict[str, Any]:
    if hasattr(page_or_html, 'content'):
        page = page_or_html
        html = page.content()
    else:
        html = page_or_html
    
    soup = BeautifulSoup(html, "lxml")
    result = extract_jsonld(soup)
    
    if not (result["avg_rating"] or result["total_ratings"]):
        rating_divs = soup.find_all(['div', 'span'], class_=re.compile(r'rating|review', re.I))
        for div in rating_divs[:5]:
            text = div.get_text()
            parsed = extract_rating_and_count(text)
            if parsed["avg_rating"] or parsed["total_ratings"]:
                result.update(parsed)
                break
    
    return result

def parse_thomann(page_or_html) -> Dict[str, Any]:
    if hasattr(page_or_html, 'content'):
        page = page_or_html
        
        el = wait_for_element(page, [
            "[class*='rating']",
            "[class*='review']",
            "a[href*='reviews']",
            "[data-testing-id*='rating']"
        ])
        
        if el:
            html = el.inner_html()
            result = extract_rating_and_count(html)
            if result["avg_rating"] or result["total_ratings"]:
                return result
        
        html = page.content()
    else:
        html = page_or_html
    
    soup = BeautifulSoup(html, "lxml")
    result = extract_jsonld(soup)
    
    if not (result["avg_rating"] or result["total_ratings"]):
        result.update(extract_rating_and_count(html))
    
    return result

def parse_apple(page_or_html) -> Dict[str, Any]:
    if hasattr(page_or_html, 'content'):
        html = page_or_html.content()
    else:
        html = page_or_html
    
    soup = BeautifulSoup(html, "lxml")
    result = extract_jsonld(soup)
    
    if not (result["avg_rating"] or result["total_ratings"]):
        result.update(extract_rating_and_count(html))
    
    return result

def parse_yelp(page_or_html) -> Dict[str, Any]:
    if hasattr(page_or_html, 'content'):
        html = page_or_html.content()
    else:
        html = page_or_html
    
    soup = BeautifulSoup(html, "lxml")
    result = extract_jsonld(soup)
    
    if not (result["avg_rating"] or result["total_ratings"]):
        result.update(extract_rating_and_count(html))
    
    return result

def get_domain(url: str) -> str:
    try:
        domain = urlparse(url).netloc.lower()
        return domain[4:] if domain.startswith("www.") else domain
    except:
        return ""

def get_parser(url: str):
    domain = get_domain(url)
    
    if "sweetwater.com" in domain:
        return parse_sweetwater
    elif "guitarcenter.com" in domain:
        return parse_guitarcenter
    elif "bhphotovideo.com" in domain or "bhphoto.com" in domain:
        return parse_bh
    elif "vintageking.com" in domain:
        return parse_vintageking
    elif "thomann" in domain:
        return parse_thomann
    elif "apple.com" in domain:
        return parse_apple
    elif "yelp.com" in domain:
        return parse_yelp
    
    return None

def scrape_product(url: str, config: Dict, retailer: str, product: str) -> Dict[str, Any]:
    parser = get_parser(url)
    use_pw = config["USE_PLAYWRIGHT"] == 1 and HAVE_PW
    
    log.info(f"Scraping: {retailer} / {product}")
    log.debug(f"  URL: {url}")
    log.debug(f"  Mode: {'Playwright' if use_pw else 'Requests'}")
    
    if use_pw and parser:
        html, code, p, handles = fetch_with_playwright(url, config)
        result = {}
        try:
            if handles:
                _, _, page = handles
                result = parser(page)
                
                if result.get("avg_rating") or result.get("total_ratings"):
                    log.info(f"  ✓ Found: rating={result.get('avg_rating')}, reviews={result.get('total_ratings')}")
                    return result
            
            if html:
                soup = BeautifulSoup(html, "lxml")
                result = extract_jsonld(soup)
                if result.get("avg_rating") or result.get("total_ratings"):
                    log.info(f"  ✓ Found via JSON-LD: {result.get('avg_rating')} stars")
                    return result
        finally:
            if config["DEBUG_ARTIFACTS"] and html:
                if not (result.get("avg_rating") or result.get("total_ratings")):
                    save_artifact(html, retailer, product)
            close_playwright(p, handles)
    
    else:
        html, code = fetch_with_requests(url, config)
        
        if html:
            soup = BeautifulSoup(html, "lxml")
            result = extract_jsonld(soup)
            
            if result.get("avg_rating") or result.get("total_ratings"):
                log.info(f"  ✓ Found: {result.get('avg_rating')} stars, {result.get('total_ratings')} reviews")
                return result
            
            if parser:
                result = parser(html)
                if result.get("avg_rating") or result.get("total_ratings"):
                    log.info(f"  ✓ Found: {result.get('avg_rating')} stars")
                    return result
        
        if HAVE_PW:
            log.debug("  Falling back to Playwright...")
            html, code, p, handles = fetch_with_playwright(url, config)
            result = {}
            try:
                if html:
                    if parser and handles:
                        _, _, page = handles
                        result = parser(page)
                    else:
                        soup = BeautifulSoup(html, "lxml")
                        result = extract_jsonld(soup)
                    
                    if result.get("avg_rating") or result.get("total_ratings"):
                        log.info(f"  ✓ Found: {result.get('avg_rating')} stars")
                        return result
            finally:
                if config["DEBUG_ARTIFACTS"] and html:
                    save_artifact(html, retailer, product)
                close_playwright(p, handles)
    
    log.warning(f"  ✗ No ratings found")
    return {}

def save_artifact(html: str, retailer: str, product: str):
    try:
        artifact_dir = Path("/tmp/artifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{slugify(retailer)}__{slugify(product)}__{timestamp}.html"
        filepath = artifact_dir / filename
        
        filepath.write_text(html, encoding="utf-8", errors="ignore")
        log.info(f"  💾 Saved artifact: {filename}")
    except Exception as e:
        log.warning(f"Failed to save artifact: {e}")

def connect_sheets() -> tuple[gspread.Client, gspread.Spreadsheet]:
    log.info("Connecting to Google Sheets...")
    
    sa_json = os.getenv("GOOGLE_SA_JSON", "")
    if not sa_json:
        log.error("❌ GOOGLE_SA_JSON not set!")
        sys.exit(1)
    
    try:
        creds = json.loads(sa_json)
        if creds.get("type") != "service_account":
            log.error("❌ Invalid service account")
            sys.exit(1)
        log.info(f"✓ Service account: {creds.get('client_email')}")
    except json.JSONDecodeError as e:
        log.error(f"❌ Invalid JSON: {e}")
        sys.exit(1)
    
    try:
        gc = gspread.service_account_from_dict(creds)
        log.info("✓ Authenticated")
    except Exception as e:
        log.error(f"❌ Auth failed: {e}")
        sys.exit(1)
    
    sheet_id = os.getenv("SHEET_ID", "")
    if not sheet_id:
        log.error("❌ SHEET_ID not set")
        sys.exit(1)
    
    try:
        ss = gc.open_by_key(sheet_id)
        log.info(f"✓ Opened sheet: {ss.title}")
        return gc, ss
    except Exception as e:
        log.error(f"❌ Failed to open sheet: {e}")
        sys.exit(1)

def read_input(ss: gspread.Spreadsheet) -> pd.DataFrame:
    input_tab = os.getenv("INPUT_SHEET_NAME", "Input")
    
    try:
        ws = ss.worksheet(input_tab)
    except Exception as e:
        log.error(f"❌ Could not find '{input_tab}' tab")
        sys.exit(1)
    
    df = get_as_dataframe(ws, evaluate_formulas=True, header=0)
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    
    required = ["retailer", "product_name", "url"]
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        log.error(f"❌ Missing columns: {missing}")
        sys.exit(1)
    
    return df

def write_results(ss: gspread.Spreadsheet, results: List[Dict]):
    output_tab = os.getenv("OUTPUT_SHEET_NAME", "Retailer Results")
    
    try:
        ws = ss.worksheet(output_tab)
    except:
        log.info(f"Creating sheet: {output_tab}")
        ws = ss.add_worksheet(output_tab, rows=1000, cols=20)
    
    existing = get_as_dataframe(ws, evaluate_formulas=False, header=0)
    existing = existing.dropna(how="all", axis=0).dropna(how="all", axis=1)
    
    new_df = pd.DataFrame(results)
    
    cols = [
        "retailer", "product_name", "url", "status", "notes",
        "avg_rating", "total_ratings", "review_count",
        "qa_count", "answers_count", "rating_breakdown_json", "last_review_date"
    ]
    
    for col in cols:
        if col not in new_df.columns:
            new_df[col] = None
    new_df = new_df[cols]
    
    if existing.empty:
        final_df = new_df
    else:
        for col in cols:
            if col not in existing.columns:
                existing[col] = None
        existing = existing[cols]
        final_df = pd.concat([existing, new_df], ignore_index=True)
    
    ws.clear()
    set_with_dataframe(ws, final_df, include_index=False, include_column_header=True)
    log.info(f"✓ Wrote {len(results)} results")

def main():
    log.info("=" * 70)
    log.info("Apogee Retailer Monitor - ANTI-BOT VERSION")
    log.info("=" * 70)
    
    config = load_config()
    log.info(f"Config: USE_PLAYWRIGHT={config['USE_PLAYWRIGHT']}, DELAY={config['REQUEST_DELAY']}")
    
    gc, ss = connect_sheets()
    
    log.info("Reading input...")
    df = read_input(ss)
    log.info(f"Total products: {len(df)}")
    
    retailer_filter = config["RETAILER_FILTER"]
    if retailer_filter:
        df = filter_by_retailer(df, retailer_filter)
        log.info(f"Filtered to '{retailer_filter}': {len(df)} products")
    
    if df.empty:
        log.warning("No products to process")
        return
    
    results = []
    
    for idx, row in df.iterrows():
        retailer = str(row.get("retailer", "")).strip()
        product = str(row.get("product_name", "")).strip()
        url = str(row.get("url", "")).strip()
        
        if not url:
            results.append({
                "retailer": retailer,
                "product_name": product,
                "url": url,
                "status": "error",
                "notes": "Missing URL"
            })
            continue
        
        if idx > 0:
            delay = human_delay(config["REQUEST_DELAY"], config["JITTER"])
            log.info(f"Waiting {delay:.1f}s...")
            time.sleep(delay)
        
        try:
            data = scrape_product(url, config, retailer, product)
            
            if data and (data.get("avg_rating") or data.get("total_ratings")):
                results.append({
                    "retailer": retailer,
                    "product_name": product,
                    "url": url,
                    "status": "ok",
                    "notes": "Success",
                    "avg_rating": data.get("avg_rating"),
                    "total_ratings": data.get("total_ratings"),
                    "review_count": data.get("review_count"),
                    "qa_count": data.get("qa_count"),
                    "answers_count": data.get("answers_count"),
                    "rating_breakdown_json": data.get("rating_breakdown_json"),
                    "last_review_date": data.get("last_review_date"),
                })
            else:
                results.append({
                    "retailer": retailer,
                    "product_name": product,
                    "url": url,
                    "status": "error",
                    "notes": "No rating data - likely bot detected"
                })
        
        except Exception as e:
            log.exception(f"Error: {retailer}/{product}")
            results.append({
                "retailer": retailer,
                "product_name": product,
                "url": url,
                "status": "error",
                "notes": f"Exception: {str(e)[:200]}"
            })
    
    if results:
        log.info("Writing results...")
        write_results(ss, results)
    
    success = sum(1 for r in results if r["status"] == "ok")
    log.info("=" * 70)
    log.info(f"Complete! {success}/{len(results)} successful")
    log.info("=" * 70)

if __name__ == "__main__":
    main()
