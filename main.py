# ---------- Imports ----------
import os
import sys
import csv
import re
import asyncio
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
from typing import Optional

import pandas as pd
import requests
import dns.resolver
import whois
from bs4 import BeautifulSoup

try:
    import aiohttp
except ImportError:
    print("❌ Missing dependency: aiohttp. Please install it: pip install aiohttp")
    sys.exit(1)

# ---------- Configuration ----------
def get_config():
    """Return configuration settings"""
    return {
        'input_file': r"sample.csv",
        'output_file': r"processed.csv",
        'MAX_PAGES_PER_DOMAIN': 250,
        'MAX_CONCURRENCY': 20,
        'REQUEST_TIMEOUT_SECONDS': 12,
        'USER_AGENT': "Mozilla/5.0 (compatible; LeadDiscoveryBot/1.0; +https://example.com/bot)",
        'RECENT_DAYS_THRESHOLD': 180,
        'MAX_EMAILS_IN_OUTPUT': 10,
        'MAX_SOCIALS_IN_OUTPUT': 15,
        'COMMON_PAGES': ["", "contact", "contact-us", "support", "help", "about", "about-us", "team"],
        'SOCIAL_DOMAINS': ["facebook.com", "linkedin.com", "twitter.com", "x.com", "instagram.com"]
    }

# ---------- Data Validation ----------
def validate_and_load_data(input_file):
    """Validate input file and load domain data"""
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found!")
        sys.exit(1)

    try:
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(df)} domains from {input_file}")
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        sys.exit(1)

    if df.empty:
        print("❌ Error: CSV file is empty!")
        sys.exit(1)

    # Normalize domain list
    domains = []
    try:
        if "domain" in [c.lower() for c in df.columns]:
            col = [c for c in df.columns if c.lower() == "domain"][0]
            domains = [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()]
        else:
            domains = [str(x).strip() for x in df.iloc[:,0].dropna().tolist() if str(x).strip()]
    except Exception as e:
        print(f"❌ Error parsing domains: {e}")
        sys.exit(1)

    if not domains:
        print("❌ No domains found in the CSV file!")
        sys.exit(1)

    return domains

# ---------- Helper Functions ----------
def normalize_url(base_url: str, href: str) -> str:
    url = urljoin(base_url, href)
    parsed = urlparse(url)
    clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if clean.endswith("/") and len(clean) > len(f"{parsed.scheme}://{parsed.netloc}/"):
        clean = clean[:-1]
    return clean

def is_internal(target_url: str, root_domain: str) -> bool:
    try:
        netloc = urlparse(target_url).netloc.lower()
        return netloc.endswith(root_domain.lower())
    except:
        return False

def looks_like_binary(path: str) -> bool:
    return any(path.lower().endswith(ext) for ext in (
        ".jpg",".jpeg",".png",".gif",".webp",".svg",".ico",".pdf",".zip",".rar",
        ".7z",".mp4",".mp3",".avi",".mov",".wmv",".mkv",".doc",".docx",".xls",
        ".xlsx",".ppt",".pptx",".css",".js",".woff",".woff2",".ttf",".eot",".otf"
    ))

def get_date_patterns_and_cutoff(recent_days_threshold):
    """Get date patterns and recent cutoff date"""
    RECENT_CUTOFF = datetime.now() - timedelta(days=recent_days_threshold)
    
    DATE_PATTERNS = [
        r"(20[0-9]{2})[-/\.](0[1-9]|1[0-2])(?:[-/\.](0[1-9]|[12][0-9]|3[01]))?",
        r"(?:(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December)\s+)([0-9]{1,2},?\s+)?(20[0-9]{2})",
    ]
    
    MONTH_MAP = {
        'jan':1,'january':1, 'feb':2,'february':2,'mar':3,'march':3,'apr':4,'april':4,
        'may':5,'jun':6,'june':6,'jul':7,'july':7,'aug':8,'august':8,'sep':9,'sept':9,'september':9,
        'oct':10,'october':10,'nov':11,'november':11,'dec':12,'december':12
    }
    
    return DATE_PATTERNS, MONTH_MAP, RECENT_CUTOFF

def any_recent_date_in_text(text: str, recent_days_threshold: int) -> bool:
    DATE_PATTERNS, MONTH_MAP, RECENT_CUTOFF = get_date_patterns_and_cutoff(recent_days_threshold)
    
    try:
        for m in re.finditer(DATE_PATTERNS[0], text):
            y = int(m.group(1)); mo = int(m.group(2)); d = m.group(3)
            day = int(d) if d else 1
            dt = datetime(y, mo, day)
            if dt >= RECENT_CUTOFF and dt <= datetime.now() + timedelta(days=3):
                return True
        for m in re.finditer(DATE_PATTERNS[1], text, flags=re.IGNORECASE):
            month_name = m.group(1).lower(); mo = MONTH_MAP.get(month_name, None)
            y = int(m.group(3)); day = 1
            if m.group(2):
                digits = re.findall(r"[0-9]{1,2}", m.group(2))
                if digits: day = int(digits[0])
            if mo:
                dt = datetime(y, mo, min(day,28))
                if dt >= RECENT_CUTOFF and dt <= datetime.now() + timedelta(days=3):
                    return True
    except:
        pass
    return False

def check_workspace(domain: str) -> str:
    try:
        answers = dns.resolver.resolve(domain, "MX")
        for rdata in answers:
            if "google.com" in str(rdata.exchange).lower():
                return "Yes"
        return "No"
    except:
        return "NA"

def get_whois(domain: str):
    try:
        w = whois.whois(domain)
        country = w.get("country") or "NA"
        created = w.get("creation_date", "NA")
        updated = w.get("updated_date", "NA")
        if isinstance(created, list) and created: created = created[0]
        if isinstance(updated, list) and updated: updated = updated[0]
        return (country or "NA", str(created) if created else "NA", str(updated) if updated else "NA")
    except:
        return ("NA","NA","NA")

# ---------- WordPress API Functions ----------
def get_last_wp_entry(domain: str, content_type: str = "posts"):
    """
    Fetch last created and last modified WordPress post/page via REST API.

    Args:
        domain (str): Domain name (example.com).
        content_type (str): "posts" or "pages".

    Returns:
        dict with created and modified info, or None values if unavailable.
    """
    base_urls = [
        f"https://{domain}/wp-json/wp/v2/{content_type}",
        f"https://www.{domain}/wp-json/wp/v2/{content_type}"
    ]

    result = {
        "domain": domain,
        "type": content_type,
        "last_created": None,
        "last_modified": None,
        "status": "API not available"
    }

    for base_url in base_urls:
        try:
            # Get latest created (orderby=date)
            created_url = f"{base_url}?per_page=1&orderby=date&order=desc"
            r_created = requests.get(created_url, timeout=10)
            
            # Get latest modified (orderby=modified)
            modified_url = f"{base_url}?per_page=1&orderby=modified&order=desc"
            r_modified = requests.get(modified_url, timeout=10)

            if r_created.status_code == 200 and r_modified.status_code == 200:
                data_created = r_created.json()
                data_modified = r_modified.json()

                if isinstance(data_created, list) and len(data_created) > 0:
                    item = data_created[0]
                    result["last_created"] = {
                        "title": item.get("title", {}).get("rendered"),
                        "slug": item.get("slug"),
                        "date": item.get("date"),         # created date
                        "modified": item.get("modified"), # last updated
                        "link": item.get("link"),
                        "api_url": created_url
                    }

                if isinstance(data_modified, list) and len(data_modified) > 0:
                    item = data_modified[0]
                    result["last_modified"] = {
                        "title": item.get("title", {}).get("rendered"),
                        "slug": item.get("slug"),
                        "date": item.get("date"),         # created date
                        "modified": item.get("modified"), # last updated
                        "link": item.get("link"),
                        "api_url": modified_url
                    }

                result["status"] = "success"
                return result  # success, stop trying further URLs

        except Exception:
            continue  # try next base URL

    return result

# ---------- Async Site Crawler ----------
class SiteScrapeResult:
    def __init__(self):
        self.emails = set()
        self.socials = set()
        self.has_contact_form = False
        self.pages_crawled = 0
        self.any_recent_blog_hint = False
        self.contact_page_link = None
        self.about_page_link = None

async def fetch_page(session, url: str, timeout: int) -> str:
    try:
        async with session.get(url, timeout=timeout, headers={"User-Agent": get_config()['USER_AGENT']}) as resp:
            if resp.status != 200: return ""
            text = await resp.text(errors="ignore")
            return text[:2_000_000]
    except:
        return ""

async def crawl_domain(session, domain: str, config: dict) -> SiteScrapeResult:
    result = SiteScrapeResult()

    # Candidate base URLs
    base_variants = [
        f"https://{domain}",
        f"http://{domain}",
        f"https://www.{domain}",
        f"http://www.{domain}",
    ]

    async def resolve_base(base_url: str) -> Optional[str]:
        """Fetch base URL once to see if it works & resolve redirects."""
        try:
            resp = await session.get(base_url, allow_redirects=True, timeout=10)
            if resp.status < 400:
                return str(resp.url)  # final resolved URL after redirects
        except Exception:
            return None
        return None

    # Step 1: Find the first working base URL
    resolved_base = None
    for base in base_variants:
        resolved = await resolve_base(base)
        if resolved:
            resolved_base = resolved.rstrip("/")
            break

    if not resolved_base:
        return result  # nothing worked

    # Step 2: Build candidate URLs using resolved base
    candidates = [f"{resolved_base}/{p}".rstrip("/") for p in config['COMMON_PAGES']]

    visited = set()
    sem = asyncio.Semaphore(5)

    async def process(url):
        if url in visited:
            return
        visited.add(url)

        async with sem:
            html = await fetch_page(session, url, config['REQUEST_TIMEOUT_SECONDS'])

        if not html:
            return
        soup = BeautifulSoup(html, "html.parser")

        # --- Extract emails ---
        for m in re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", html):
            if not m.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")):
                result.emails.add(m)

        # --- Extract social links ---
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if any(s in href for s in config['SOCIAL_DOMAINS']):
                result.socials.add(href)

        # --- Detect contact forms ---
        if not result.has_contact_form:
            for f in soup.find_all("form"):
                blob = " ".join([
                    (f.get("id") or ""),
                    (f.get("name") or ""),
                    f.get_text(" ", strip=True).lower()
                ])
                if any(k in blob for k in ("contact", "message", "email")):
                    result.has_contact_form = True
                    break

        # --- Detect contact and about pages ---
        if not result.contact_page_link:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "contact" in href.lower():
                    result.contact_page_link = href
                    break
        if not result.about_page_link:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "about" in href.lower():
                    result.about_page_link = href
                    break

    # Step 3: Process candidate pages concurrently
    await asyncio.gather(*(process(url) for url in candidates))

    return result

# ---------- Main Processing Function ----------
async def process_domains(domains, config):
    """Process all domains and return results"""
    results = []
    
    async with aiohttp.ClientSession() as session:
        for i, domain in enumerate(domains, 1):
            print(f"Processing {i}/{len(domains)}: {domain}")
            
            # These functions are synchronous and can be called directly
            workspace = check_workspace(domain)
            country, created, updated = get_whois(domain)
           
            # Call the async crawl_domain function using await
            crawl_result = await crawl_domain(session, domain, config)

            emails_out = "; ".join(sorted(crawl_result.emails)[:config['MAX_EMAILS_IN_OUTPUT']]) if crawl_result.emails else "NA"
            socials_out = "; ".join(sorted(crawl_result.socials)[:config['MAX_SOCIALS_IN_OUTPUT']]) if crawl_result.socials else "NA"

            # Get WordPress posts and pages data
            posts_data = get_last_wp_entry(domain, "posts")
            pages_data = get_last_wp_entry(domain, "pages")

            # Prepare posts columns
            posts_api_status = posts_data["status"]
            posts_last_created_link = posts_data["last_created"]["link"] if posts_data["last_created"] else "NA"
            posts_last_created_title = posts_data["last_created"]["title"] if posts_data["last_created"] else "NA"
            posts_last_created_date = posts_data["last_created"]["date"] if posts_data["last_created"] else "NA"
            posts_last_modified_title = posts_data["last_modified"]["title"] if posts_data["last_modified"] else "NA"
            posts_last_modified_date = posts_data["last_modified"]["modified"] if posts_data["last_modified"] else "NA"

            # Prepare pages columns
            pages_api_status = pages_data["status"]
            pages_last_created_link = pages_data["last_created"]["link"] if pages_data["last_created"] else "NA"
            pages_last_created_title = pages_data["last_created"]["title"] if pages_data["last_created"] else "NA"
            pages_last_created_date = pages_data["last_created"]["date"] if pages_data["last_created"] else "NA"
            pages_last_modified_title = pages_data["last_modified"]["title"] if pages_data["last_modified"] else "NA"
            pages_last_modified_date = pages_data["last_modified"]["modified"] if pages_data["last_modified"] else "NA"

            results.append({
                "Domain": domain,
                "Google Workspace": workspace,
                "Country of Origin": country,
                "Domain Created": created,
                "Domain Last Modified": updated,
                "Pages Crawled (Count)": crawl_result.pages_crawled,
                "Has Contact Form": "Yes" if crawl_result.has_contact_form else "No",
                "Emails (unique)": emails_out,
                "Social Links (unique)": socials_out,
                "Contact Page Link": crawl_result.contact_page_link or "NA",
                "About Page Link": crawl_result.about_page_link or "NA",
                "Posts API Status": posts_api_status,
                "Posts Last Created Title": posts_last_created_title,
                "Posts Last Created Link": posts_last_created_link,
                "Posts Last Created Date": posts_last_created_date,
                "Posts Last Modified Title": posts_last_modified_title,
                "Posts Last Modified Date": posts_last_modified_date,
                "Pages API Status": pages_api_status,
                "Pages Last Created Title": pages_last_created_title,
                "Pages Last Created Link": pages_last_created_link,
                "Pages Last Created Date": pages_last_created_date,
                "Pages Last Modified Title": pages_last_modified_title,
                "Pages Last Modified Date": pages_last_modified_date
            })
    
    return results

# ---------- Save Results Function ----------
def save_results(results, output_file):
    """Save results to CSV file"""
    try:
        out_df = pd.DataFrame(results)
        expected_cols = [
            "Domain",
            "Google Workspace","Country of Origin","Domain Created","Domain Last Modified",
            "Pages Crawled (Count)","Has Contact Form","Emails (unique)","Social Links (unique)",
            "Contact Page Link","About Page Link",
            "Posts API Status","Posts Last Created Title","Posts Last Created Link","Posts Last Created Date","Posts Last Modified Title","Posts Last Modified Date",
            "Pages API Status","Pages Last Created Title","Pages Last Created Link","Pages Last Created Date","Pages Last Modified Title","Pages Last Modified Date"
        ]
        for col in expected_cols:
            if col not in out_df.columns: out_df[col] = "NA"
        out_df = out_df[expected_cols]
        out_df.to_csv(output_file, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"✅ Results saved to {output_file}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

# ---------- Main Execution Function ----------
async def main():
    """Main execution function"""
    config = get_config()
    domains = validate_and_load_data(config['input_file'])
    results = await process_domains(domains, config)
    save_results(results, config['output_file'])

# ---------- Entry Point ----------
if __name__ == "__main__":
    asyncio.run(main())