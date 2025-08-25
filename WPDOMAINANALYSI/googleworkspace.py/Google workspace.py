
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
input_file  = r"C:\Users\sai\Desktop\Untitled spreadsheet - Sheet1.csv"
output_file = r"C:\Users\sai\Desktop\Final_All_Pages_Results.csv"

# ---------- Crawl Settings (tune for speed vs coverage) ----------
MAX_PAGES_PER_DOMAIN     = 250       # hard cap of pages to crawl per domain
MAX_CONCURRENCY          = 20        # simultaneous requests per domain
REQUEST_TIMEOUT_SECONDS  = 12
USER_AGENT = "Mozilla/5.0 (compatible; LeadDiscoveryBot/1.0; +https://example.com/bot)"
RECENT_DAYS_THRESHOLD    = 180       # for "Active?" decisions
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

# Normalize domain list (supports 'domain' header OR first column)
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
    """Resolve relative links and strip fragments/query for dedup stability."""
    url = urljoin(base_url, href)
    parsed = urlparse(url)
    # keep scheme, netloc, path; drop query/fragment to reduce duplicates
    clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    # remove trailing slash duplicates except root
    if clean.endswith("/") and len(clean) > len(f"{parsed.scheme}://{parsed.netloc}/"):
        clean = clean[:-1]
    return clean

def is_internal(target_url: str, root_domain: str) -> bool:
    """Check if URL belongs to the same registrable domain (incl. subdomains)."""
    try:
        netloc = urlparse(target_url).netloc.lower()
        return netloc.endswith(root_domain.lower())
    except:
        return False

def looks_like_binary(path: str) -> bool:
    """Skip common binary/static assets."""
    return any(path.lower().endswith(ext) for ext in (
        ".jpg",".jpeg",".png",".gif",".webp",".svg",".ico",".pdf",".zip",".rar",
        ".7z",".mp4",".mp3",".avi",".mov",".wmv",".mkv",".doc",".docx",".xls",
        ".xlsx",".ppt",".pptx",".css",".js",".woff",".woff2",".ttf",".eot",".otf"
    ))

RECENT_CUTOFF = datetime.now() - timedelta(days=RECENT_DAYS_THRESHOLD)

DATE_PATTERNS = [
    # 2025-08, 2025/08, 2025-08-23, 2025/08/23, 2025.08.23
    r"(20[0-9]{2})[-/\.](0[1-9]|1[0-2])(?:[-/\.](0[1-9]|[12][0-9]|3[01]))?",
    # Aug 2025, 23 Aug 2025, August 23, 2025
    r"(?:(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December)\s+)([0-9]{1,2},?\s+)?(20[0-9]{2})",
]

MONTH_MAP = {
    'jan':1,'january':1, 'feb':2,'february':2,'mar':3,'march':3,'apr':4,'april':4,
    'may':5,'jun':6,'june':6,'jul':7,'july':7,'aug':8,'august':8,'sep':9,'sept':9,'september':9,
    'oct':10,'october':10,'nov':11,'november':11,'dec':12,'december':12
}

def any_recent_date_in_text(text: str) -> bool:
    """Heuristic: find any date in text that lands within RECENT_DAYS_THRESHOLD."""
    try:
        # pattern 1: 2025-08(-23)?
        for m in re.finditer(DATE_PATTERNS[0], text):
            y = int(m.group(1))
            mo = int(m.group(2))
            d = m.group(3)
            day = int(d) if d else 1
            dt = datetime(y, mo, day)
            if dt >= RECENT_CUTOFF and dt <= datetime.now() + timedelta(days=3):
                return True

        # pattern 2: Month names
        for m in re.finditer(DATE_PATTERNS[1], text, flags=re.IGNORECASE):
            month_name = m.group(1).lower()
            mo = MONTH_MAP.get(month_name, None)
            y  = int(m.group(3))
            day = 1
            if m.group(2):
                # like "23 " or "23,"
                digits = re.findall(r"[0-9]{1,2}", m.group(2))
                if digits:
                    day = int(digits[0])
            if mo:
                dt = datetime(y, mo, min(day,28))  # safe day
                if dt >= RECENT_CUTOFF and dt <= datetime.now() + timedelta(days=3):
                    return True
    except:
        pass
    return False

def check_workspace(domain: str) -> str:
    """Check if MX records indicate Google Workspace."""
    try:
        answers = dns.resolver.resolve(domain, "MX")
        for rdata in answers:
            if "google.com" in str(rdata.exchange).lower():
                return "Yes"
        return "No"
    except:
        return "NA"

def get_whois(domain: str):
    """Get country, creation and updated date from WHOIS."""
    try:
        w = whois.whois(domain)
        country = w.get("country") or "NA"
        created = w.get("creation_date", "NA")
        updated = w.get("updated_date", "NA")
        # Normalize list -> first
        if isinstance(created, list) and created:
            created = created[0]
        if isinstance(updated, list) and updated:
            updated = updated[0]
        return (country or "NA", str(created) if created else "NA", str(updated) if updated else "NA")
    except:
        return ("NA","NA","NA")

def check_wordpress_recent(domain: str) -> str:
    """Check WP posts modified in last RECENT_DAYS_THRESHOLD days via REST API."""
    urls = [
        f"https://{domain}/wp-json/wp/v2/posts?per_page=1&orderby=modified&order=desc",
        f"https://www.{domain}/wp-json/wp/v2/posts?per_page=1&orderby=modified&order=desc",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=8, headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    last_modified = data[0].get("modified")
                    if last_modified:
                        # WP returns ISO8601; handle Z
                        ts = last_modified.replace("Z","+00:00")
                        try:
                            dt = datetime.fromisoformat(ts)
                        except:
                            # fallback rough parse
                            dt = datetime.strptime(last_modified[:19], "%Y-%m-%dT%H:%M:%S")
                        return "Yes (Active)" if dt >= RECENT_CUTOFF else "Yes (Inactive)"
            # non-200 -> try next
        except:
            continue
    return "No"  # WP API not present or no posts

def has_generic_blog(domain: str) -> str:
    """Quick yes/no if /blog exists (doesn't check recency)."""
    for proto in ("https","http"):
        url = f"{proto}://{domain}/blog"
        try:
            r = requests.get(url, timeout=8, headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                return "Yes"
        except:
            pass
    return "No"

# ---------- Async Site Crawler (FAST) ----------
class SiteScrapeResult:
    def __init__(self):
        self.emails = set()
        self.socials = set()
        self.has_contact_form = False
        self.pages_crawled = 0
        self.any_recent_blog_hint = False  # heuristic based on dates in content

async def fetch_page(session, url: str) -> str:
    try:
        async with session.get(url, timeout=REQUEST_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}) as resp:
            if resp.status != 200:
                return ""
            # limit size to avoid huge pages
            text = await resp.text(errors="ignore")
            if len(text) > 2_000_000:  # 2MB guard
                return text[:2_000_000]
            return text
    except:
        return ""

async def crawl_domain(domain: str) -> SiteScrapeResult:
    result = SiteScrapeResult()
    start_urls = [f"https://{domain}", f"http://{domain}"]

    visited = set()
    queue = asyncio.Queue()

    # seed
    for u in start_urls:
        await queue.put(u)
        visited.add(u)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async with aiohttp.ClientSession() as session:
        async def worker():
            while True:
                try:
                    url = await queue.get()
                except:
                    return
                if result.pages_crawled >= MAX_PAGES_PER_DOMAIN:
                    queue.task_done()
                    continue
                if looks_like_binary(url):
                    queue.task_done()
                    continue

                async with sem:
                    html = await fetch_page(session, url)

                if not html:
                    queue.task_done()
                    continue

                result.pages_crawled += 1

                # Extract emails
                for m in re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", html):
                    # filter obvious false positives
                    if not m.lower().endswith((".png",".jpg",".jpeg",".gif",".webp",".svg")):
                        result.emails.add(m)

                soup = BeautifulSoup(html, "html.parser")

                # Social links
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if any(s in href for s in ("facebook.com","linkedin.com","twitter.com","x.com","instagram.com")):
                        result.socials.add(href)

                # Contact-like forms
                if not result.has_contact_form:
                    for f in soup.find_all("form"):
                        blob = " ".join([
                            (f.get("id") or ""),
                            (f.get("name") or ""),
                            f.get_text(" ", strip=True).lower()
                        ])
                        if any(k in blob for k in ("contact","message","email")):
                            result.has_contact_form = True
                            break

                # Heuristic blog recency (scan visible text)
                if not result.any_recent_blog_hint:
                    text_sample = soup.get_text(" ", strip=True)
                    if any_recent_date_in_text(text_sample):
                        result.any_recent_blog_hint = True

                # Discover internal links
                for a in soup.find_all("a", href=True):
                    new = normalize_url(url, a["href"])
                    if not new.startswith(("http://","https://")):
                        continue
                    if not is_internal(new, domain):
                        continue
                    if new in visited:
                        continue
                    if looks_like_binary(new):
                        continue
                    if result.pages_crawled + queue.qsize() >= MAX_PAGES_PER_DOMAIN:
                        continue
                    visited.add(new)
                    await queue.put(new)

                queue.task_done()

        # launch workers
        workers = [asyncio.create_task(worker()) for _ in range(MAX_CONCURRENCY)]
        await queue.join()
        for w in workers:
            w.cancel()
        # swallow cancellations
        try:
            await asyncio.gather(*workers, return_exceptions=True)
        except:
            pass

    return result

def derive_blog_active_label(wp_status: str, generic_blog_exists: str, crawl_hint: bool) -> str:
    """
    Combine WP status + generic /blog existence + crawl date hints into one label.
    Priority:
      1) WP "Yes (Active)" -> Yes (Active)
      2) Otherwise, if crawl_hint True -> Yes (Active)
      3) If WP "Yes (Inactive)" -> Yes (Inactive)
      4) Else if generic exists -> Yes (Inactive) (unknown recency)
      5) Else -> No
    """
    if wp_status == "Yes (Active)":
        return "Yes (Active)"
    if crawl_hint:
        return "Yes (Active)"
    if wp_status == "Yes (Inactive)":
        return "Yes (Inactive)"
    if generic_blog_exists == "Yes":
        return "Yes (Inactive)"
    return "No"

# ---------- Main Processing ----------
print("🔍 Processing domains...")

# Windows event loop policy fix (if needed)
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
    except:
        pass

for i, domain in enumerate(domains, 1):
    print(f"Processing {i}/{len(domains)}: {domain}")

    # Fast metadata checks (sync)
    workspace = check_workspace(domain)
    country, created, updated = get_whois(domain)
    wp_blog_status = check_wordpress_recent(domain)
    generic_blog_exists = has_generic_blog(domain)

    # Full-site async crawl
    try:
        crawl_result: SiteScrapeResult = asyncio.run(crawl_domain(domain))
    except RuntimeError:
        # in case of nested loop environments
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        crawl_result = loop.run_until_complete(crawl_domain(domain))
        loop.close()

    blog_active_final = derive_blog_active_label(
        wp_blog_status,
        generic_blog_exists,
        crawl_result.any_recent_blog_hint
    )

    # Prepare output row (trim lists for CSV)
    emails_out  = "; ".join(sorted(crawl_result.emails))[:8000]  # guard against huge cells
    socials_out = "; ".join(sorted(crawl_result.socials))[:8000]

    # Slice to max counts while keeping headings non-blank
    if emails_out:
        emails_list = sorted(crawl_result.emails)[:MAX_EMAILS_IN_OUTPUT]
        emails_out = "; ".join(emails_list)
    else:
        emails_out = "NA"

    if socials_out:
        socials_list = sorted(crawl_result.socials)[:MAX_SOCIALS_IN_OUTPUT]
        socials_out = "; ".join(socials_list)
    else:
        socials_out = "NA"

    results.append({
        "Domain": domain,
        "Blog Active? (Final)": blog_active_final,           # combined label
        "WordPress Blog Status": wp_blog_status,             # WP-only view
        "Generic /blog Exists": generic_blog_exists,         # quick existence flag
        "Google Workspace": workspace,                       # MX check
        "Country of Origin": country,                        # WHOIS
        "Domain Created": created,                           # WHOIS
        "Domain Last Modified": updated,                     # WHOIS
        "Pages Crawled (Count)": crawl_result.pages_crawled, # crawl metric
        "Has Contact Form": "Yes" if crawl_result.has_contact_form else "No",
        "Emails (unique)": emails_out,
        "Social Links (unique)": socials_out
    })

# ---------- Save ----------
try:
    out_df = pd.DataFrame(results)
    # ensure all headings exist and are non-blank
    expected_cols = [
        "Domain",
        "Blog Active? (Final)",
        "WordPress Blog Status",
        "Generic /blog Exists",
        "Google Workspace",
        "Country of Origin",
        "Domain Created",
        "Domain Last Modified",
        "Pages Crawled (Count)",
        "Has Contact Form",
        "Emails (unique)",
        "Social Links (unique)",
    ]
    for col in expected_cols:
        if col not in out_df.columns:
            out_df[col] = "NA"
    out_df = out_df[expected_cols]
    out_df.to_csv(output_file, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"✅ Results saved to {output_file}")
except Exception as e:
    print(f"❌ Error saving results: {e}")
