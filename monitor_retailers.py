#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apogee Retailer Monitor - Enhanced Version
Scrapes reviews/ratings from multiple audio retailers
"""

import os
import re
import json
import time
import random
import logging
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
from pathlib import Path

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

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("apogee-monitor")

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

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
        return float(s)
    except Exception:
        return None

def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-") or "item"

def human_delay(base: float = 3.0, jitter: float = 0.8) -> float:
    return max(0.5, base + random.uniform(-jitter, jitter))

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

def filter_by_retailer(df: pd.DataFrame, token: str) -> pd.DataFrame:
    """Filter dataframe by retailer token with aliases"""
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
    """Load runtime configuration from environment"""
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

# --------------------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------------------

def create_session(user_agent: str) -> requests.Session:
    """Create configured requests session with retries"""
    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
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
    """Fetch page with requests library"""
    try:
        session = create_session(config["USER_AGENT"])
        response = session.get(url, timeout=config["SCRAPE_TIMEOUT"], allow_redirects=True)
        
        if response.status_code >= 400:
            return "", response.status_code
        
        # Check for bot detection
        html = response.text
        if len(html) < 500 or any(x in html.lower() for x in ["cloudflare", "captcha", "access denied", "checking your browser"]):
            return "", response.status_code
        
        return html, response.status_code
    except Exception as e:
        log.warning(f"Requests fetch failed: {e}")
        return "", 0

def fetch_with_playwright(url: str, config: Dict):
    """Fetch page with Playwright browser - returns (html, status, playwright_obj, page_obj)"""
    if not HAVE_PW:
        return "", 0, None, None
    
    try:
        p = sync_playwright().start()
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage"
            ]
        )
        
        context = browser.new_context(
            user_agent=config["USER_AGENT"],
            viewport={"width": 1920, "height": 1080},
            java_script_enabled=True,
        )
        
        page = context.new_page()
        
        # Stealth
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => false });
            Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3] });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US','en'] });
        """)
        
        # Block unnecessary resources
        page.route("**/*", lambda route, request:
            route.abort() if request.resource_type in ("image", "media", "font")
            else route.continue_()
        )
        
        # Navigate
        timeout_ms = int(config["SCRAPE_TIMEOUT"] * 1000)
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(1500)  # Let JS render
        
        html = page.content()
        return html, 200, p, (browser, context, page)
        
    except PWTimeout:
        log.warning(f"Playwright timeout on {url}")
        return "", 0, None, None
    except Exception as e:
        log.warning(f"Playwright error: {e}")
        return "", 0, None, None

def close_playwright(p, handles):
    """Safely close Playwright resources"""
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

# --------------------------------------------------------------------------------------
# Parsing - JSON-LD
# --------------------------------------------------------------------------------------

def extract_jsonld(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract ratings from JSON-LD structured data"""
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
                
                # Check for aggregateRating
                agg = item.get("aggregateRating", {})
                if isinstance(agg, dict):
                    if agg.get("ratingValue"):
                        result["avg_rating"] = _safe_float(agg.get("ratingValue"))
                    
                    count = agg.get("ratingCount") or agg.get("reviewCount")
                    if count:
                        result["total_ratings"] = _safe_int(count)
                        if not result["review_count"]:
                            result["review_count"] = result["total_ratings"]
                
                # Sometimes reviewCount is at top level
                if item.get("reviewCount"):
                    result["review_count"] = _safe_int(item.get("reviewCount"))
                    
        except Exception as e:
            log.debug(f"JSON-LD parse error: {e}")
            continue
    
    return result

# --------------------------------------------------------------------------------------
# Parsing - Retailer-specific
# --------------------------------------------------------------------------------------

def parse_rating_text(text: str) -> Dict[str, Any]:
    """Extract rating and count from text using regex"""
    result = {"avg_rating": None, "total_ratings": None, "review_count": None}
    
    # Match patterns like "4.5 out of 5" or "4.5/5"
    rating_match = re.search(r"(\d+\.?\d*)\s*(?:out of|\/)\s*5", text, re.I)
    if rating_match:
        result["avg_rating"] = _safe_float(rating_match.group(1))
    
    # Match patterns like "(123 reviews)" or "123 ratings"
    count_match = re.search(r"(\d+(?:,\d+)?)\s*(?:review|rating)", text, re.I)
    if count_match:
        count = _safe_int(count_match.group(1))
        result["total_ratings"] = count
        result["review_count"] = count
    
    return result

def wait_for_element(page, selectors: List[str], max_attempts: int = 10):
    """Wait for any of the given selectors to appear"""
    for _ in range(max_attempts):
        for sel in selectors:
            try:
                el = page.locator(sel).first
                if el.count() > 0 and el.is_visible():
                    return el
            except:
                pass
        page.wait_for_timeout(500)
    return None

def parse_sweetwater(page_or_html) -> Dict[str, Any]:
    """Parse Sweetwater product page"""
    if hasattr(page_or_html, 'content'):  # Playwright page
        page = page_or_html
        
        # Try clicking reviews tab
        try:
            tab = page.locator("text=/reviews/i").first
            if tab.count() > 0:
                tab.click()
                page.wait_for_timeout(1000)
        except:
            pass
        
        # Look for rating elements
        el = wait_for_element(page, [
            "[data-qa='rating-summary']",
            ".review-summary",
            ".rating-summary",
            "[class*='review'][class*='summary']"
        ])
        
        if el:
            html = el.inner_html()
            result = parse_rating_text(html)
            if result["avg_rating"] or result["total_ratings"]:
                return result
        
        # Fallback to full page
        html = page.content()
    else:
        html = page_or_html
    
    soup = BeautifulSoup(html, "lxml")
    result = extract_jsonld(soup)
    
    if not (result["avg_rating"] or result["total_ratings"]):
        # Try finding in page text
        text_result = parse_rating_text(html)
        result.update(text_result)
    
    return result

def parse_guitarcenter(page_or_html) -> Dict[str, Any]:
    """Parse Guitar Center product page"""
    if hasattr(page_or_html, 'content'):  # Playwright page
        page = page_or_html
        
        el = wait_for_element(page, [
            "[class*='BVRRRatingSummary']",
            "[id*='BVRRSummaryContainer']",
            ".pr-snippet",
            "[class*='review'][class*='summary']"
        ])
        
        if el:
            html = el.inner_html()
            result = parse_rating_text(html)
            if result["avg_rating"] or result["total_ratings"]:
                return result
        
        html = page.content()
    else:
        html = page_or_html
    
    soup = BeautifulSoup(html, "lxml")
    result = extract_jsonld(soup)
    
    if not (result["avg_rating"] or result["total_ratings"]):
        result.update(parse_rating_text(html))
    
    return result

def parse_bh(page_or_html) -> Dict[str, Any]:
    """Parse B&H Photo product page"""
    if hasattr(page_or_html, 'content'):
        page = page_or_html
        
        el = wait_for_element(page, [
            "[data-selenium='reviewSummary']",
            "[data-selenium='ratingSummary']",
            ".bv_avgRating_component_container",
            "[class*='review']"
        ])
        
        if el:
            result = parse_rating_text(el.inner_html())
            if result["avg_rating"] or result["total_ratings"]:
                return result
        
        html = page.content()
    else:
        html = page_or_html
    
    soup = BeautifulSoup(html, "lxml")
    result = extract_jsonld(soup)
    
    if not (result["avg_rating"] or result["total_ratings"]):
        result.update(parse_rating_text(html))
    
    return result

def parse_vintageking(page_or_html) -> Dict[str, Any]:
    """Parse Vintage King product page"""
    if hasattr(page_or_html, 'content'):
        page = page_or_html
        html = page.content()
    else:
        html = page_or_html
    
    soup = BeautifulSoup(html, "lxml")
    result = extract_jsonld(soup)
    
    if not (result["avg_rating"] or result["total_ratings"]):
        result.update(parse_rating_text(html))
    
    return result

def parse_thomann(page_or_html) -> Dict[str, Any]:
    """Parse Thomann product page"""
    if hasattr(page_or_html, 'content'):
        page = page_or_html
        
        el = wait_for_element(page, [
            "[class*='rating']",
            "[class*='review']",
            "a[href*='reviews']"
        ])
        
        if el:
            result = parse_rating_text(el.inner_html())
            if result["avg_rating"] or result["total_ratings"]:
                return result
        
        html = page.content()
    else:
        html = page_or_html
    
    soup = BeautifulSoup(html, "lxml")
    result = extract_jsonld(soup)
    
    if not (result["avg_rating"] or result["total_ratings"]):
        result.update(parse_rating_text(html))
    
    return result

def parse_apple(page_or_html) -> Dict[str, Any]:
    """Parse Apple App Store page"""
    if hasattr(page_or_html, 'content'):
        html = page_or_html.content()
    else:
        html = page_or_html
    
    soup = BeautifulSoup(html, "lxml")
    result = extract_jsonld(soup)
    
    if not (result["avg_rating"] or result["total_ratings"]):
        result.update(parse_rating_text(html))
    
    return result

def parse_yelp(page_or_html) -> Dict[str, Any]:
    """Parse Yelp business page"""
    if hasattr(page_or_html, 'content'):
        html = page_or_html.content()
    else:
        html = page_or_html
    
    soup = BeautifulSoup(html, "lxml")
    result = extract_jsonld(soup)
    
    if not (result["avg_rating"] or result["total_ratings"]):
        result.update(parse_rating_text(html))
    
    return result

# --------------------------------------------------------------------------------------
# Main scraper dispatcher
# --------------------------------------------------------------------------------------

def get_domain(url: str) -> str:
    """Extract domain from URL"""
    try:
        domain = urlparse(url).netloc.lower()
        return domain[4:] if domain.startswith("www.") else domain
    except:
        return ""

def get_parser(url: str):
    """Return appropriate parser function for URL"""
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
    """Main scraping function with fallback logic"""
    parser = get_parser(url)
    use_pw = config["USE_PLAYWRIGHT"] == 1 and HAVE_PW
    
    log.info(f"Scraping: {retailer} / {product}")
    log.debug(f"  URL: {url}")
    log.debug(f"  Mode: {'Playwright' if use_pw else 'Requests'}")
    
    # Try with specified method
    if use_pw and parser:
        # Playwright with custom parser
        html, code, p, handles = fetch_with_playwright(url, config)
        try:
            if handles:
                _, _, page = handles
                result = parser(page)
                
                if result.get("avg_rating") or result.get("total_ratings"):
                    log.info(f"  ✓ Found: {result['avg_rating']} stars, {result['total_ratings']} reviews")
                    return result
            
            # Fallback to JSON-LD on rendered HTML
            if html:
                soup = BeautifulSoup(html, "lxml")
                result = extract_jsonld(soup)
                if result.get("avg_rating") or result.get("total_ratings"):
                    log.info(f"  ✓ Found via JSON-LD: {result['avg_rating']} stars")
                    return result
        finally:
            # Save artifact if failed
            if config["DEBUG_ARTIFACTS"] and html:
                if not (result.get("avg_rating") or result.get("total_ratings")):
                    save_artifact(html, retailer, product)
            close_playwright(p, handles)
    
    else:
        # Try requests first
        html, code = fetch_with_requests(url, config)
        
        if html:
            soup = BeautifulSoup(html, "lxml")
            result = extract_jsonld(soup)
            
            if result.get("avg_rating") or result.get("total_ratings"):
                log.info(f"  ✓ Found: {result['avg_rating']} stars, {result['total_ratings']} reviews")
                return result
            
            # Try parser on HTML
            if parser:
                result = parser(html)
                if result.get("avg_rating") or result.get("total_ratings"):
                    log.info(f"  ✓ Found: {result['avg_rating']} stars")
                    return result
        
        # Fallback to Playwright
        if HAVE_PW:
            log.debug("  Falling back to Playwright...")
            html, code, p, handles = fetch_with_playwright(url, config)
            try:
                if html:
                    if parser and handles:
                        _, _, page = handles
                        result = parser(page)
                    else:
                        soup = BeautifulSoup(html, "lxml")
                        result = extract_jsonld(soup)
                    
                    if result.get("avg_rating") or result.get("total_ratings"):
                        log.info(f"  ✓ Found: {result['avg_rating']} stars")
                        return result
            finally:
                if config["DEBUG_ARTIFACTS"] and html:
                    save_artifact(html, retailer, product)
                close_playwright(p, handles)
    
    log.warning(f"  ✗ No ratings found")
    return {}

def save_artifact(html: str, retailer: str, product: str):
    """Save HTML for debugging"""
    try:
        artifact_dir = Path("/tmp/artifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{slugify(retailer)}__{slugify(product)}.html"
        filepath = artifact_dir / filename
        
        filepath.write_text(html, encoding="utf-8", errors="ignore")
        log.info(f"  💾 Saved artifact: {filename}")
    except Exception as e:
        log.warning(f"Failed to save artifact: {e}")

# --------------------------------------------------------------------------------------
# Google Sheets
# --------------------------------------------------------------------------------------

def connect_sheets() -> tuple[gspread.Client, gspread.Spreadsheet]:
    """Connect to Google Sheets"""
    sa_json = os.getenv("GOOGLE_SA_JSON", "")
    if not sa_json:
        raise ValueError("GOOGLE_SA_JSON not set")
    
    try:
        creds = json.loads(sa_json)
    except:
        raise ValueError("GOOGLE_SA_JSON is not valid JSON")
    
    gc = gspread.service_account_from_dict(creds)
    
    sheet_id = os.getenv("SHEET_ID", "")
    if not sheet_id:
        raise ValueError("SHEET_ID not set")
    
    ss = gc.open_by_key(sheet_id)
    return gc, ss

def read_input(ss: gspread.Spreadsheet) -> pd.DataFrame:
    """Read input data from sheet"""
    input_tab = os.getenv("INPUT_SHEET_NAME", "Input")
    ws = ss.worksheet(input_tab)
    
    df = get_as_dataframe(ws, evaluate_formulas=True, header=0)
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    
    # Validate columns
    required = ["retailer", "product_name", "url"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Input sheet missing column: {col}")
    
    return df

def write_results(ss: gspread.Spreadsheet, results: List[Dict]):
    """Append results to output sheet"""
    output_tab = os.getenv("OUTPUT_SHEET_NAME", "Retailer Results")
    
    try:
        ws = ss.worksheet(output_tab)
    except:
        ws = ss.add_worksheet(output_tab, rows=1000, cols=20)
    
    # Get existing data
    existing = get_as_dataframe(ws, evaluate_formulas=False, header=0)
    existing = existing.dropna(how="all", axis=0).dropna(how="all", axis=1)
    
    # Prepare new data
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
    
    # Combine
    if existing.empty:
        final_df = new_df
    else:
        for col in cols:
            if col not in existing.columns:
                existing[col] = None
        existing = existing[cols]
        final_df = pd.concat([existing, new_df], ignore_index=True)
    
    # Write back
    ws.clear()
    set_with_dataframe(ws, final_df, include_index=False, include_column_header=True)
    log.info(f"✓ Wrote {len(results)} results to '{output_tab}'")

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    """Main execution"""
    log.info("=" * 60)
    log.info("Apogee Retailer Monitor Starting")
    log.info("=" * 60)
    
    # Load config
    config = load_config()
    log.info(f"Config: {json.dumps(config, indent=2)}")
    
    # Connect to sheets
    log.info("Connecting to Google Sheets...")
    gc, ss = connect_sheets()
    
    # Read input
    log.info("Reading input data...")
    df = read_input(ss)
    log.info(f"Total products in sheet: {len(df)}")
    
    # Filter by retailer
    retailer_filter = config["RETAILER_FILTER"]
    if retailer_filter:
        df = filter_by_retailer(df, retailer_filter)
        log.info(f"Filtered to '{retailer_filter}': {len(df)} products")
    
    if df.empty:
        log.warning("No products to process. Exiting.")
        return
    
    # Process each product
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
        
        # Polite delay
        if idx > 0:
            delay = human_delay(config["REQUEST_DELAY"], config["JITTER"])
            log.info(f"Waiting {delay:.1f}s...")
            time.sleep(delay)
        
        # Scrape
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
                    "notes": "No rating data found"
                })
        
        except Exception as e:
            log.exception(f"Error processing {retailer}/{product}")
            results.append({
                "retailer": retailer,
                "product_name": product,
                "url": url,
                "status": "error",
                "notes": f"Exception: {str(e)[:200]}"
            })
    
    # Write results
    if results:
        log.info("Writing results to sheet...")
        write_results(ss, results)
    
    # Summary
    success = sum(1 for r in results if r["status"] == "ok")
    log.info("=" * 60)
    log.info(f"Complete! {success}/{len(results)} successful")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
