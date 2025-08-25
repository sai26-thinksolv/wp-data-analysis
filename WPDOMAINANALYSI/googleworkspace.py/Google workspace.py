# ---------- Imports ----------
import os
import sys
import csv
import re
import asyncio
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse

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

# ---------- Paths (EDIT THESE) ----------
input_file  = r"Untitled spreadsheet - Sheet1.csv"
output_file = r"Final_All_Pages_Results.csv"

# ---------- Crawl Settings ----------
MAX_PAGES_PER_DOMAIN     = 250
MAX_CONCURRENCY          = 20
REQUEST_TIMEOUT_SECONDS  = 12
USER_AGENT = "Mozilla/5.0 (compatible; LeadDiscoveryBot/1.0; +https://example.com/bot)"
RECENT_DAYS_THRESHOLD    = 180
MAX_EMAILS_IN_OUTPUT     = 10
MAX_SOCIALS_IN_OUTPUT    = 15

# ---------- Validate Input ----------
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

results = []

# ---------- Helpers ----------
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

RECENT_CUTOFF = datetime.now() - timedelta(days=RECENT_DAYS_THRESHOLD)

DATE_PATTERNS = [
    r"(20[0-9]{2})[-/\.](0[1-9]|1[0-2])(?:[-/\.](0[1-9]|[12][0-9]|3[01]))?",
    r"(?:(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December)\s+)([0-9]{1,2},?\s+)?(20[0-9]{2})",
]

MONTH_MAP = {
    'jan':1,'january':1, 'feb':2,'february':2,'mar':3,'march':3,'apr':4,'april':4,
    'may':5,'jun':6,'june':6,'jul':7,'july':7,'aug':8,'august':8,'sep':9,'sept':9,'september':9,
    'oct':10,'october':10,'nov':11,'november':11,'dec':12,'december':12
}

def any_recent_date_in_text(text: str) -> bool:
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

def check_wordpress_recent(domain: str) -> str:
    urls = [
        f"https://{domain}/wp-json/wp/v2/posts?per_page=1&orderby=modified&order=desc",
        f"http://{domain}/wp-json/wp/v2/posts?per_page=1&orderby=modified&order=desc",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=8, headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    last_modified = data[0].get("modified")
                    if last_modified:
                        dt = None
                        try:
                            ts = last_modified.replace("Z", "+00:00")
                            dt = datetime.fromisoformat(ts)
                        except: pass
                        if dt is None:
                            try:
                                clean_ts = last_modified.split('+')[0].split('Z')[0]
                                dt = datetime.strptime(clean_ts, "%Y-%m-%dT%H:%M:%S")
                            except: pass
                        if dt is None:
                            try:
                                clean_ts = last_modified.replace('T', ' ').split('+')[0].split('Z')[0]
                                dt = datetime.strptime(clean_ts, "%Y-%m-%d %H:%M:%S")
                            except: pass
                        if dt is None:
                            try:
                                date_part = last_modified[:10]
                                dt = datetime.strptime(date_part, "%Y-%m-%d")
                            except: continue
                        if dt:
                            return "Yes (Active)" if dt >= RECENT_CUTOFF else "Yes (Inactive)"
        except:
            continue
    return "No"

def has_generic_blog(domain: str) -> str:
    for proto in ("https","http"):
        url = f"{proto}://{domain}/blog"
        try:
            r = requests.get(url, timeout=8, headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                return "Yes"
        except:
            pass
    return "No"

# ---------- Async Site Crawler ----------
class SiteScrapeResult:
    def __init__(self):
        self.emails = set()
        self.socials = set()
        self.has_contact_form = False
        self.pages_crawled = 0
        self.any_recent_blog_hint = False

async def fetch_page(session, url: str) -> str:
    try:
        async with session.get(url, timeout=REQUEST_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}) as resp:
            if resp.status != 200: return ""
            text = await resp.text(errors="ignore")
            return text[:2_000_000]
    except:
        return ""

async def crawl_domain(session, domain: str) -> SiteScrapeResult:
    result = SiteScrapeResult()
    start_urls = [f"https://{domain}", f"http://{domain}"]

    visited = set(); queue = asyncio.Queue()
    for u in start_urls: await queue.put(u); visited.add(u)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def worker():
        while True:
            try: url = await queue.get()
            except asyncio.CancelledError: break
            except: return

            if result.pages_crawled >= MAX_PAGES_PER_DOMAIN: queue.task_done(); continue
            if looks_like_binary(url): queue.task_done(); continue

            async with sem: html = await fetch_page(session, url)

            if not html: queue.task_done(); continue

            result.pages_crawled += 1

            for m in re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", html):
                if not m.lower().endswith((".png",".jpg",".jpeg",".gif",".webp",".svg")):
                    result.emails.add(m)

            soup = BeautifulSoup(html, "html.parser")

            for a in soup.find_all("a", href=True):
                href = a["href"]
                if any(s in href for s in ("facebook.com","linkedin.com","twitter.com","x.com","instagram.com")):
                    result.socials.add(href)

            if not result.has_contact_form:
                for f in soup.find_all("form"):
                    blob = " ".join([(f.get("id") or ""),(f.get("name") or ""),f.get_text(" ", strip=True).lower()])
                    if any(k in blob for k in ("contact","message","email")):
                        result.has_contact_form = True; break

            if not result.any_recent_blog_hint:
                if any_recent_date_in_text(soup.get_text(" ", strip=True)):
                    result.any_recent_blog_hint = True

            for a in soup.find_all("a", href=True):
                new = normalize_url(url, a["href"])
                if not new.startswith(("http://","https://")): continue
                if not is_internal(new, domain): continue
                if new in visited: continue
                if looks_like_binary(new): continue
                if result.pages_crawled + queue.qsize() >= MAX_PAGES_PER_DOMAIN: continue
                visited.add(new); await queue.put(new)

            queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(MAX_CONCURRENCY)]
    await queue.join()
    for w in workers: w.cancel()
    await asyncio.gather(*workers, return_exceptions=True) # Using await gather to ensure all workers are handled


    return result

def derive_blog_active_label(wp_status: str, generic_blog_exists: str, crawl_hint: bool) -> str:
    if wp_status == "Yes (Active)": return "Yes (Active)"
    if crawl_hint: return "Yes (Active)"
    if wp_status == "Yes (Inactive)": return "Yes (Inactive)"
    if generic_blog_exists == "Yes": return "Yes (Inactive)"
    return "No"

# ---------- Main Processing ----------
print("🔍 Processing domains...")

async def main():
    global results
    async with aiohttp.ClientSession() as session:
        for i, domain in enumerate(domains, 1):
            print(f"Processing {i}/{len(domains)}: {domain}")
            # These functions are synchronous and can be called directly
            workspace = check_workspace(domain)
            country, created, updated = get_whois(domain)
            wp_blog_status = check_wordpress_recent(domain)
            generic_blog_exists = has_generic_blog(domain)

            # Call the async crawl_domain function using await
            crawl_result = await crawl_domain(session, domain)

            blog_active_final = derive_blog_active_label(
                wp_blog_status,
                generic_blog_exists,
                crawl_result.any_recent_blog_hint
            )

            emails_out = "; ".join(sorted(crawl_result.emails)[:MAX_EMAILS_IN_OUTPUT]) if crawl_result.emails else "NA"
            socials_out = "; ".join(sorted(crawl_result.socials)[:MAX_SOCIALS_IN_OUTPUT]) if crawl_result.socials else "NA"

            results.append({
                "Domain": domain,
                "Blog Active? (Final)": blog_active_final,
                "WordPress Blog Status": wp_blog_status,
                "Generic /blog Exists": generic_blog_exists,
                "Google Workspace": workspace,
                "Country of Origin": country,
                "Domain Created": created,
                "Domain Last Modified": updated,
                "Pages Crawled (Count)": crawl_result.pages_crawled,
                "Has Contact Form": "Yes" if crawl_result.has_contact_form else "No",
                "Emails (unique)": emails_out,
                "Social Links (unique)": socials_out
            })

# Run the main async function
asyncio.run(main())

# ---------- Save ----------
try:
    out_df = pd.DataFrame(results)
    expected_cols = [
        "Domain","Blog Active? (Final)","WordPress Blog Status","Generic /blog Exists",
        "Google Workspace","Country of Origin","Domain Created","Domain Last Modified",
        "Pages Crawled (Count)","Has Contact Form","Emails (unique)","Social Links (unique)"
    ]
    for col in expected_cols:
        if col not in out_df.columns: out_df[col] = "NA"
    out_df = out_df[expected_cols]
    out_df.to_csv(output_file, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"✅ Results saved to {output_file}")
except Exception as e:
    print(f"❌ Error saving results: {e}")