# ---------- Imports ----------
import os
import sys
import csv
import re
import asyncio
import json
import signal
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
from typing import Optional
import time

import pandas as pd
import requests
import dns.resolver
import whois
from bs4 import BeautifulSoup


import gzip
import bz2
import argparse


def get_file_size_mb(file_path):
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)

def analyze_file(file_path):
    """Analyze processed file and show statistics"""
    print(f"\n📊 File Analysis: {file_path}")

    if not os.path.exists(file_path):
        print("❌ File not found!")
        return

    size_mb = get_file_size_mb(file_path)
    print(f"📁 File size: {size_mb:.2f} MB")

    # Determine file type and load data
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.jsonl'):
        records = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        df = pd.DataFrame(records)
    else:
        print("❌ Unsupported file format")
        return

    print(f"📈 Total domains: {len(df)}")

    # Analyze data completeness
    if 'Domain' in df.columns:
        # Count domains with meaningful data
        meaningful_domains = 0
        for _, row in df.iterrows():
            non_na_fields = sum(1 for v in row.values if str(v) not in ['NA', 'No', 'disabled', ''])
            if non_na_fields >= 3:  # Domain + at least 2 other fields
                meaningful_domains += 1

        print(f"✅ Domains with data: {meaningful_domains} ({meaningful_domains/len(df)*100:.1f}%)")

        # Show top data sources
        if 'Posts API Status' in df.columns:
            wp_success = len(df[df['Posts API Status'] == 'success'])
            print(f"🔌 WordPress API success: {wp_success} ({wp_success/len(df)*100:.1f}%)")

        if 'Has Contact Form' in df.columns:
            contact_forms = len(df[df['Has Contact Form'] == 'Yes'])
            print(f"📝 Contact forms found: {contact_forms} ({contact_forms/len(df)*100:.1f}%)")

def split_file_by_size(file_path, max_size_mb=50):
    """Split large file into smaller chunks"""
    print(f"\n✂️ Splitting {file_path} into {max_size_mb}MB chunks...")

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)

        # Calculate rows per chunk
        total_size = get_file_size_mb(file_path)
        rows_per_chunk = int(len(df) * (max_size_mb / total_size))

        chunks = [df[i:i+rows_per_chunk] for i in range(0, len(df), rows_per_chunk)]

        base_name = file_path.replace('.csv', '')
        for i, chunk in enumerate(chunks):
            chunk_file = f"{base_name}_part{i+1}.csv"
            chunk.to_csv(chunk_file, index=False)
            print(f"📁 Created: {chunk_file} ({len(chunk)} domains, {get_file_size_mb(chunk_file):.1f}MB)")

    elif file_path.endswith('.jsonl'):
        # For JSONL, split by line count
        with open(file_path, 'r') as f:
            lines = f.readlines()

        total_size = get_file_size_mb(file_path)
        lines_per_chunk = int(len(lines) * (max_size_mb / total_size))

        base_name = file_path.replace('.jsonl', '')
        chunk_num = 1

        for i in range(0, len(lines), lines_per_chunk):
            chunk_lines = lines[i:i+lines_per_chunk]
            chunk_file = f"{base_name}_part{chunk_num}.jsonl"

            with open(chunk_file, 'w') as f:
                f.writelines(chunk_lines)

            print(f"📁 Created: {chunk_file} ({len(chunk_lines)} domains, {get_file_size_mb(chunk_file):.1f}MB)")
            chunk_num += 1

# Updated compress_file to handle jsonl extension correctly
def compress_file(file_path, compression='gzip'):
    """Compress a file"""
    print(f"\n🗜️ Compressing {file_path} with {compression}...")

    original_size = get_file_size_mb(file_path)
    compressed_file = None

    if compression == 'gzip':
        with open(file_path, 'rb') as f_in:
            compressed_file = f"{file_path}.gz"
            with gzip.open(compressed_file, 'wb') as f_out:
                f_out.writelines(f_in)
    elif compression == 'bz2':
        with open(file_path, 'rb') as f_in:
            compressed_file = f"{file_path}.bz2"
            with bz2.open(compressed_file, 'wb') as f_out:
                f_out.writelines(f_in)
    else:
        print("❌ Unsupported compression format")
        return file_path # Return original path if no compression

    if compressed_file and os.path.exists(compressed_file):
        compressed_size = get_file_size_mb(compressed_file)
        compression_ratio = (1 - compressed_size/original_size) * 100 if original_size > 0 else 0

        print(f"✅ Compressed: {compressed_file}")
        print(f"📊 Size reduction: {original_size:.1f}MB → {compressed_size:.1f}MB ({compression_ratio:.1f}% smaller)")
        # Remove original file after successful compression
        os.remove(file_path)
        return compressed_file
    else:
        print(f"❌ Compression failed for {file_path}")
        return file_path # Return original path if compression failed


def clean_data(file_path, output_file=None):
    """Remove domains with minimal data to reduce file size"""
    print(f"\n🧹 Cleaning data in {file_path}...")

    if output_file is None:
        base_name, ext = os.path.splitext(file_path)
        output_file = f"{base_name}_cleaned{ext}"

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.jsonl'):
        records = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        df = pd.DataFrame(records)
    else:
        print("❌ Unsupported file format")
        return

    original_count = len(df)

    # Keep only domains with meaningful data
    cleaned_df = df.copy()
    rows_to_keep = []

    for idx, row in df.iterrows():
        meaningful_fields = 0
        for col, value in row.items():
            if col != 'Domain' and str(value) not in ['NA', 'No', 'disabled', '']:
                meaningful_fields += 1

        # Keep if has at least 3 meaningful fields (excluding domain)
        if meaningful_fields >= 3:
            rows_to_keep.append(idx)

    cleaned_df = df.loc[rows_to_keep]

    # Save cleaned data
    if output_file.endswith('.csv'):
        cleaned_df.to_csv(output_file, index=False)
    elif output_file.endswith('.jsonl'):
        with open(output_file, 'w') as f:
            for record in cleaned_df.to_dict('records'):
                f.write(json.dumps(record, default=str) + '\n')

    original_size = get_file_size_mb(file_path)
    cleaned_size = get_file_size_mb(output_file)

    print(f"✅ Cleaned file: {output_file}")
    print(f"📊 Domains: {original_count} → {len(cleaned_df)} ({len(cleaned_df)/original_count*100:.1f}% kept)")
    print(f"📊 Size: {original_size:.1f}MB → {cleaned_size:.1f}MB ({(1-cleaned_size/original_size)*100:.1f}% smaller)")


try:
    import aiohttp
except ImportError:
    print("❌ Missing dependency: aiohttp. Please install it: pip install aiohttp")
    sys.exit(1)

# ---------- Configuration ----------
def get_config(preset: str = "balanced"):
    """Return configuration settings with performance presets"""
    base_config = {
        'input_file': r"/content/wp-data-analysis/first_2000.csv",
        'output_file': r"processed.jsonl", # Changed default to jsonl
        'checkpoint_file': r"checkpoint.json",
        'shutdown_file': r"stop_processing.txt",
        'batch_save_interval': 100,  # Save progress every N domains
        'MAX_PAGES_PER_DOMAIN': 250,
        'MAX_CONCURRENCY': 20,
        'REQUEST_TIMEOUT_SECONDS': 12,
        'USER_AGENT': "Mozilla/5.0 (compatible; LeadDiscoveryBot/1.0; +https://example.com/bot)",
        'RECENT_DAYS_THRESHOLD': 180,
        'MAX_EMAILS_IN_OUTPUT': 10,
        'MAX_SOCIALS_IN_OUTPUT': 15,
        'COMMON_PAGES': ["", "contact", "contact-us", "support", "help", "about", "about-us", "team"],
        'SOCIAL_DOMAINS': ["facebook.com", "linkedin.com", "twitter.com", "x.com", "instagram.com"],

        # Feature flags - enable/disable specific operations
        'ENABLE_WHOIS': True,
        'ENABLE_DNS_MX': True,
        'ENABLE_CRAWLING': True,
        'ENABLE_WORDPRESS_API': True,
        'ENABLE_POSTS_API': True,
        'ENABLE_PAGES_API': True,

        # Output format options
        'OUTPUT_FORMAT': 'jsonl',  # 'csv', 'json', 'jsonl', 'sqlite'
        'OUTPUT_COMPRESSION': None,  # None, 'gzip', 'bz2'
        'ENABLE_DATA_DEDUPLICATION': True,  # Remove duplicate data
        'MAX_FILE_SIZE_MB': 100,  # Split files when they exceed this size

        # Record update behavior
        'UPDATE_EXISTING_RECORDS': True,  # Merge new data with existing records
        'FORCE_REPROCESS': False,  # Ignore checkpoint and process all domains

        # Shutdown check interval (seconds)
        'SHUTDOWN_CHECK_INTERVAL': 1.0
    }

    # Performance presets
    if preset == "speed":
        base_config.update({
            'MAX_CONCURRENCY': 50,
            'REQUEST_TIMEOUT_SECONDS': 8,
            'MAX_PAGES_PER_DOMAIN': 50,
            'ENABLE_WHOIS': False,  # Disable slow WHOIS lookups
            'ENABLE_PAGES_API': False,  # Only check posts, not pages
            'batch_save_interval': 10,
            'MAX_EMAILS_IN_OUTPUT': 5,
            'MAX_SOCIALS_IN_OUTPUT': 8,
            'COMMON_PAGES': ["", "contact", "about"],  # Fewer pages to crawl
            'OUTPUT_COMPRESSION': 'gzip',
            'MAX_FILE_SIZE_MB': 50,  # Smaller files for speed
            'ENABLE_DATA_DEDUPLICATION': True
        })
    elif preset == "complete":
        base_config.update({
            'MAX_CONCURRENCY': 10,
            'REQUEST_TIMEOUT_SECONDS': 20,
            'MAX_PAGES_PER_DOMAIN': 500,
            'batch_save_interval': 3,
            'MAX_EMAILS_IN_OUTPUT': 20,
            'MAX_SOCIALS_IN_OUTPUT': 25,
            'COMMON_PAGES': ["", "contact", "contact-us", "support", "help", "about", "about-us", "team", "services", "blog", "news"]
        })
    elif preset == "minimal":
        # Optimized for smallest file sizes
        base_config.update({
            'MAX_CONCURRENCY': 30,
            'REQUEST_TIMEOUT_SECONDS': 10,
            'MAX_PAGES_PER_DOMAIN': 20,
            'ENABLE_WHOIS': False,
            'ENABLE_CRAWLING': False,  # Skip crawling to reduce data
            'ENABLE_PAGES_API': False,  # Only posts
            'batch_save_interval': 20,
            'MAX_EMAILS_IN_OUTPUT': 3,
            'MAX_SOCIALS_IN_OUTPUT': 5,
            'COMMON_PAGES': ["", "contact"],
            'OUTPUT_COMPRESSION': 'gzip',
            'MAX_FILE_SIZE_MB': 25,
            'ENABLE_DATA_DEDUPLICATION': True,
            'OUTPUT_FORMAT': 'jsonl'  # More compact than CSV
        })
    # "balanced" preset uses base_config as-is

    return base_config

# ---------- Graceful Shutdown Handler ----------
class GracefulShutdown:
    def __init__(self, shutdown_file: str):
        self.shutdown_requested = False
        self.shutdown_file = shutdown_file
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception:
            pass  # Signal handling may not work in all environments (e.g., Jupyter)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received shutdown signal ({signum}). Finishing current batch...")
        self.shutdown_requested = True

    def should_shutdown(self) -> bool:
        """Check if shutdown has been requested via signal or file"""
        if self.shutdown_requested:
            return True

        # Check for shutdown file
        if os.path.exists(self.shutdown_file):
            print(f"🛑 Shutdown file '{self.shutdown_file}' detected. Finishing current batch...")
            self.shutdown_requested = True
            try:
                os.remove(self.shutdown_file)
            except Exception:
                pass
            return True

        return False

# ---------- Data Validation ----------
def validate_and_load_data(input_file):
    """Validate input file and load domain data"""
    if not os.path.exists(input_file):
        #print(f"❌ Error: Input file '{input_file}' not found!")
        sys.exit(1)

    try:
        df = pd.read_csv(input_file)
        #print(f"✅ Loaded {len(df)} domains from {input_file}")
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        sys.exit(1)

    if df.empty:
        print("❌ Error: CSV file is empty!")
        sys.exit(1)

    # Normalize domain list
    domains = []
    raw_domains = []
    try:
        if "domain" in [c.lower() for c in df.columns]:
            col = [c for c in df.columns if c.lower() == "domain"][0]
            raw_domains = [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()]
        else:
            raw_domains = [str(x).strip() for x in df.iloc[:,0].dropna().tolist() if str(x).strip()]

        # Normalize each domain and track invalid ones
        invalid_domains = []
        for raw_domain in raw_domains:
            normalized = normalize_domain(raw_domain)
            if normalized:
                domains.append(normalized)
                if normalized != raw_domain.lower():
                    print(f"📝 Normalized '{raw_domain}' → '{normalized}'")
            else:
                invalid_domains.append(raw_domain)

        # Report invalid domains
        if invalid_domains:
            print(f"⚠️ Skipped {len(invalid_domains)} invalid domain(s): {', '.join(invalid_domains[:5])}")
            if len(invalid_domains) > 5:
                print(f"   ... and {len(invalid_domains) - 5} more")

    except Exception as e:
        print(f"❌ Error parsing domains: {e}")
        sys.exit(1)

    if not domains:
        print("❌ No domains found in the CSV file!")
        sys.exit(1)

    return domains

# ---------- Helper Functions ----------
def normalize_domain(domain_input: str) -> str:
    """
    Normalize domain input to extract clean domain name.
    Handles URLs like https://domain.com, https://www.domain.com, etc.

    Args:
        domain_input (str): Raw domain input (can be URL or plain domain)

    Returns:
        str: Clean domain name (e.g., "example.com")
    """
    if not domain_input or not isinstance(domain_input, str):
        return ""

    domain_input = domain_input.strip()

    # If it looks like a URL, parse it
    if domain_input.startswith(('http://', 'https://', 'www.')):
        try:
            # Add protocol if missing but starts with www
            if domain_input.startswith('www.'):
                domain_input = 'https://' + domain_input

            parsed = urlparse(domain_input)
            domain = parsed.netloc.lower()

            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]

            # Remove port numbers from parsed netloc
            if ':' in domain and domain.count(':') == 1:
                domain = domain.split(':')[0]

            return domain
        except Exception:
            # If URL parsing fails, try to extract domain manually
            pass

    # Clean up common prefixes/suffixes
    domain = domain_input.lower()

    # Remove protocol prefixes
    for prefix in ['https://', 'http://', 'www.']:
        if domain.startswith(prefix):
            domain = domain[len(prefix):]

    # Remove trailing slashes and paths
    if '/' in domain:
        domain = domain.split('/')[0]

    # Remove port numbers (but avoid breaking IPv6 addresses)
    if ':' in domain and domain.count(':') == 1:  # Only single colon (not IPv6)
        domain = domain.split(':')[0]

    # Basic validation - should contain at least one dot
    if '.' not in domain or len(domain) < 3:
        return ""

    return domain

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
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://google.com",
    }
    for base_url in base_urls:
        try:
            # Get latest created (orderby=date)
            created_url = f"{base_url}?per_page=1&orderby=date&order=desc"
            #print(f"Fetching latest created entry for {domain} via {created_url}")
            r_created = requests.get(created_url, timeout=10, headers=headers)

            # Get latest modified (orderby=modified)
            modified_url = f"{base_url}?per_page=1&orderby=modified&order=desc"
            #print(f"Fetching latest modified entry for {domain} via {modified_url}")
            r_modified = requests.get(modified_url, timeout=10, headers=headers)

            #print(r_created.status_code, r_modified.status_code)
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
    start_time = time.time()
    results = []
    checkpoint_file = config['checkpoint_file']
    batch_save_interval = config['batch_save_interval']
    checkpoint_data = {}

    # Determine expected output file name based on format and compression
    output_file_base = config['output_file'].replace('.csv', '').replace('.json', '').replace('.jsonl', '').replace('.sqlite', '')
    output_format = config.get('OUTPUT_FORMAT', 'csv').lower()
    output_extension = {
        'csv': '.csv',
        'json': '.json',
        'jsonl': '.jsonl',
        'sqlite': '.db'
    }.get(output_format, '.csv')

    uncompressed_output_file = f"{output_file_base}{output_extension}"
    config['current_output_file'] = uncompressed_output_file # Store the expected uncompressed name

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            # Load from the correct output file name stored in checkpoint if available
            if 'current_output_file' in checkpoint_data:
                 config['current_output_file'] = checkpoint_data['current_output_file']
                 uncompressed_output_file = config['current_output_file']
                 # If output file was compressed, need to work with the compressed name for existence check
                 if config.get('OUTPUT_COMPRESSION') and config['OUTPUT_COMPRESSION'] != 'None':
                     compressed_output_file = uncompressed_output_file + ('.gz' if config['OUTPUT_COMPRESSION'] == 'gzip' else '.bz2')
                     if os.path.exists(compressed_output_file):
                         print(f"⚠️ Found compressed checkpoint file: {compressed_output_file}. Data will be merged into this.")
                         config['current_output_file'] = compressed_output_file # Use compressed name for append logic
        except Exception as e:
            print(f"⚠️ Error loading checkpoint file: {e}. Starting fresh.")
            checkpoint_data = {}


    processed_domains = checkpoint_data.get('processed_domains', [])
    failed_domains = checkpoint_data.get('failed_domains', [])
    next_domain_index = checkpoint_data.get('next_domain_index', 0)

    shutdown_handler = GracefulShutdown(config['shutdown_file'])

    async with aiohttp.ClientSession() as session:
        for i, domain in enumerate(domains[next_domain_index:], next_domain_index):
            # Skip domains only if not forcing reprocess
            if not config.get('FORCE_REPROCESS', False):
                if domain in processed_domains or domain in failed_domains:
                    continue

            # Check for shutdown at the beginning of each iteration
            if shutdown_handler.should_shutdown():
                print(f"🛑 Shutdown requested. Stopping at domain {i+1}/{len(domains)}")
                break

            domain_start_time = time.time()
            #print(f"Processing {i+1}/{len(domains)}: {domain}")

            try:
                # WHOIS and DNS timing
                whois_start = time.time()
                workspace = check_workspace(domain) if config['ENABLE_DNS_MX'] else "NA"
                country, created, updated = get_whois(domain) if config['ENABLE_WHOIS'] else ("NA","NA","NA")
                whois_dns_time = time.time() - whois_start

                # Crawling timing
                crawl_start = time.time()
                crawl_result = await crawl_domain(session, domain, config) if config['ENABLE_CRAWLING'] else SiteScrapeResult()
                crawl_time = time.time() - crawl_start

                emails_out = "; ".join(sorted(crawl_result.emails)[:config['MAX_EMAILS_IN_OUTPUT']]) if crawl_result.emails else "NA"
                socials_out = "; ".join(sorted(crawl_result.socials)[:config['MAX_SOCIALS_IN_OUTPUT']]) if crawl_result.socials else "NA"

                # WordPress API timing
                wp_start = time.time()

                posts_data = get_last_wp_entry(domain, "posts") if (config['ENABLE_WORDPRESS_API'] and config['ENABLE_POSTS_API']) else {"status": "disabled", "last_created": None, "last_modified": None}
                pages_data = get_last_wp_entry(domain, "pages") if (config['ENABLE_WORDPRESS_API'] and config['ENABLE_PAGES_API']) else {"status": "disabled", "last_created": None, "last_modified": None}
                wp_time = time.time() - wp_start

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
                    "Pages Crawled": crawl_result.pages_crawled,
                    "Has Contact Form": "Yes" if crawl_result.has_contact_form else "No",
                    "Emails": emails_out,
                    "Social Links": socials_out,
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

                # Mark as successfully processed
                processed_domains.append(domain)
                total_time = time.time() - domain_start_time

                # Show which operations were performed
                ops_performed = []
                if config['ENABLE_DNS_MX']: ops_performed.append("DNS")
                if config['ENABLE_WHOIS']: ops_performed.append("WHOIS")
                if config['ENABLE_CRAWLING']: ops_performed.append("Crawl")
                if config['ENABLE_WORDPRESS_API'] and (config['ENABLE_POSTS_API'] or config['ENABLE_PAGES_API']):
                    ops_performed.append("WP-API")

                ops_str = f" [{', '.join(ops_performed)}]" if ops_performed else ""
                #print(f"✅ Successfully processed {domain} in {total_time:.2f}s{ops_str} (WHOIS/DNS: {whois_dns_time:.2f}s, Crawl: {crawl_time:.2f}s, WP: {wp_time:.2f}s)")

            except Exception as e:
                total_time = time.time() - domain_start_time
                print(f"❌ Error processing {domain} in {total_time:.2f}s: {e}")
                # Mark as failed but still track progress
                failed_domains.append(domain)
                print(f"⚠️ Marked {domain} as failed, will not retry")

            # Update checkpoint data regardless of success/failure
            checkpoint_data['processed_domains'] = processed_domains
            checkpoint_data['failed_domains'] = failed_domains
            checkpoint_data['next_domain_index'] = i + 1
            checkpoint_data['last_updated'] = datetime.now().isoformat()
            checkpoint_data['current_output_file'] = config['current_output_file'] # Save the current output file name

            # Save checkpoint and results at regular intervals
            if (i + 1) % batch_save_interval == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                if results:  # Only save if there are successful results
                    save_results(results, config)
                    results = []
                print(f"💾 Checkpoint saved at domain {i+1} (Success: {len(processed_domains)}, Failed: {len(failed_domains)})")

    # Save any remaining results
    if results:
        save_results(results, config)

    # Final checkpoint save
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

    # Summary
    total_processing_time = time.time() - start_time
    total_processed = len(processed_domains) + len(failed_domains)

    print(f"\n📊 Processing Summary:")
    print(f"   ✅ Successful: {len(processed_domains)}")
    print(f"   ❌ Failed: {len(failed_domains)}")
    print(f"   📈 Total processed: {total_processed}/{len(domains)}")
    print(f"   ⏰ Total time: {total_processing_time:.2f} seconds ({total_processing_time/60:.1f} minutes)")

    if total_processed > 0:
        avg_time_per_domain = total_processing_time / total_processed
        #print(f"   📊 Average time per domain: {avg_time_per_domain:.2f} seconds")

        # Estimate remaining time if not complete
        remaining_domains = len(domains) - total_processed
        if remaining_domains > 0:
            estimated_remaining_time = remaining_domains * avg_time_per_domain
            #print(f"   🔮 Estimated time for remaining {remaining_domains} domains: {estimated_remaining_time/60:.1f} minutes")
            #print(f"   🎯 Estimated completion: {(datetime.now() + timedelta(seconds=estimated_remaining_time)).strftime('%H:%M:%S')}")

    # Performance rate
    if total_processing_time > 0:
        domains_per_second = total_processed / total_processing_time
        domains_per_minute = domains_per_second * 60
        #print(f"   🚀 Processing rate: {domains_per_minute:.1f} domains/minute ({domains_per_second:.2f} domains/second)")

    # Show configuration summary
    enabled_features = []
    if config['ENABLE_DNS_MX']: enabled_features.append("DNS MX")
    if config['ENABLE_WHOIS']: enabled_features.append("WHOIS")
    if config['ENABLE_CRAWLING']: enabled_features.append("Crawling")
    if config['ENABLE_WORDPRESS_API'] and config['ENABLE_POSTS_API']: enabled_features.append("Posts API")
    if config['ENABLE_WORDPRESS_API'] and config['ENABLE_PAGES_API']: enabled_features.append("Pages API")
    print(f"   🔧 Features enabled: {', '.join(enabled_features)}")

    print(f"   • Shutdown file: {config['shutdown_file']} (create this file to stop gracefully)")
    print()

    # Re-validate and load domains at the end for the summary count if needed
    # domains = validate_and_load_data(config['input_file']) # Removed duplicate call
    # await process_domains(domains, config) # Removed duplicate call

# ---------- File Management Functions ----------
def rotate_output_file(original_file):
    """Create a new output file when the current one gets too large"""
    import time
    timestamp = int(time.time())
    base_name, ext = os.path.splitext(original_file)
    # Handle compressed files
    if ext in ['.gz', '.bz2']:
        base_name = os.path.splitext(base_name)[0] # Get original base name
        new_file = f"{base_name}_{timestamp}{ext}"
    else:
         new_file = f"{base_name}_{timestamp}{ext}"
    return new_file

def compress_file(file_path, compression='gzip'):
    """Compress a file using the specified compression method"""
    print(f"\n🗜️ Compressing {file_path} with {compression}...")

    original_size = get_file_size_mb(file_path)
    compressed_file = None

    try:
        if compression == 'gzip':
            compressed_file = f"{file_path}.gz"
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    f_out.writelines(f_in)
        elif compression == 'bz2':
            compressed_file = f"{file_path}.bz2"
            with open(file_path, 'rb') as f_in:
                with bz2.open(compressed_file, 'wb') as f_out:
                    f_out.writelines(f_in)
        else:
            print("❌ Unsupported compression format")
            return file_path # Return original path if no compression

        if compressed_file and os.path.exists(compressed_file):
            compressed_size = get_file_size_mb(compressed_file)
            compression_ratio = (1 - compressed_size/original_size) * 100 if original_size > 0 else 0

            print(f"✅ Compressed: {compressed_file}")
            print(f"📊 Size reduction: {original_size:.1f}MB → {compressed_size:.1f}MB ({compression_ratio:.1f}% smaller)")
            # Remove original file after successful compression
            os.remove(file_path)
            return compressed_file
        else:
            print(f"❌ Compression failed for {file_path}")
            return file_path # Return original path if compression failed
    except Exception as e:
        print(f"❌ Error during compression of {file_path}: {e}")
        return file_path # Return original path on error


def optimize_data_for_storage(results, config):
    """Optimize data before storage to reduce file size"""
    optimized_results = []

    for result in results:
        optimized = {}
        for key, value in result.items():
            # Convert long strings to abbreviated versions
            if isinstance(value, str):
                # Truncate very long titles/content
                if 'Title' in key and len(value) > 100:
                    value = value[:97] + "..."
                # Remove redundant "NA" values for optional fields
                elif value == "NA" and key in ['Contact Page Link', 'About Page Link', 'Emails (unique)', 'Social Links (unique)']:
                    continue  # Skip storing NA values for optional fields
                # Compress URLs
                elif 'Link' in key and value.startswith('http'):
                    # Keep only essential part of URL
                    from urllib.parse import urlparse
                    parsed = urlparse(value)
                    value = f"{parsed.netloc}{parsed.path}"

            optimized[key] = value

        # Only include domains with meaningful data
        if config.get('ENABLE_DATA_DEDUPLICATION', True):
            # Skip if this is just a basic domain with no additional info
            meaningful_fields = [k for k, v in optimized.items()
                               if k != 'Domain' and v not in ['NA', 'No', 'disabled', '']]
            if len(meaningful_fields) >= 2:  # At least 2 meaningful fields
                optimized_results.append(optimized)
        else:
            optimized_results.append(optimized)

    return optimized_results

# ---------- Save Results Function ----------
def save_results(results, config):
    """Save results in the specified output format with update capability"""
    if not results:
        return

    output_format = config.get('OUTPUT_FORMAT', 'csv').lower()
    update_existing = config.get('UPDATE_EXISTING_RECORDS', True)
    compression = config.get('OUTPUT_COMPRESSION')
    max_size_mb = config.get('MAX_FILE_SIZE_MB', 100)

    # Determine the uncompressed file name based on format and original output_file config
    output_file_base = config['output_file'].replace('.csv', '').replace('.json', '').replace('.jsonl', '').replace('.sqlite', '')
    output_extension = {
        'csv': '.csv',
        'json': '.json',
        'jsonl': '.jsonl',
        'sqlite': '.db'
    }.get(output_format, '.csv')

    uncompressed_output_file = f"{output_file_base}{output_extension}"
    config['current_output_file'] = uncompressed_output_file # Store the expected uncompressed name

    # Check file size (check the compressed file if compression is enabled)
    file_to_check_size = uncompressed_output_file
    if compression and compression != 'None':
         compressed_ext = '.gz' if compression == 'gzip' else '.bz2'
         compressed_name_to_check = uncompressed_output_file + compressed_ext
         if os.path.exists(compressed_name_to_check):
             file_to_check_size = compressed_name_to_check

    if os.path.exists(file_to_check_size):
        file_size_mb = os.path.getsize(file_to_check_size) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            # Rotate the current output file (handling compressed files)
            rotated_file = rotate_output_file(file_to_check_size)
            # Update config to point to the new base file for the next batch
            config['output_file'] = os.path.splitext(rotated_file)[0].replace('.jsonl', '').replace('.json', '').replace('.csv', '').replace('.db', '') # Store base name for next rotation
            uncompressed_output_file = f"{config['output_file']}{output_extension}" # New uncompressed name
            config['current_output_file'] = uncompressed_output_file
            print(f"📁 File size exceeded {max_size_mb}MB, rotating to: {uncompressed_output_file}") # Log the new uncompressed name


    try:
        # Optimize data before processing
        optimized_results = optimize_data_for_storage(results, config)
        new_df = pd.DataFrame(optimized_results)
        expected_cols = [
            "Domain",
            "Google Workspace","Country of Origin","Domain Created","Domain Last Modified",
            "Pages Crawled (Count)","Has Contact Form","Emails (unique)","Social Links (unique)",
            "Contact Page Link","About Page Link",
            "Posts API Status","Posts Last Created Title","Posts Last Created Link","Posts Last Created Date","Posts Last Modified Title","Posts Last Modified Date",
            "Pages API Status","Pages Last Created Title","Pages Last Created Link","Pages Last Created Date","Pages Last Modified Title","Pages Last Modified Date"
        ]
        # Add missing columns if any
        for col in expected_cols:
            if col not in new_df.columns: new_df[col] = "NA"
        # Reorder columns to match expected order
        new_df = new_df[expected_cols]

        if output_format == 'csv':
            file_exists = os.path.exists(uncompressed_output_file)

            if update_existing and file_exists:
                try:
                    existing_df = pd.read_csv(uncompressed_output_file)
                    merged_df = merge_domain_records(existing_df, new_df)
                    merged_df.to_csv(uncompressed_output_file, index=False, quoting=csv.QUOTE_MINIMAL)
                except Exception as e:
                    print(f"⚠️ Error reading existing CSV file for update: {e}. Appending instead.")
                    new_df.to_csv(uncompressed_output_file, index=False, mode='a', header=not file_exists, quoting=csv.QUOTE_MINIMAL)
            else:
                new_df.to_csv(uncompressed_output_file, index=False, mode='a', header=not file_exists, quoting=csv.QUOTE_MINIMAL)
            print(f"✅ Results saved to {uncompressed_output_file} (CSV format)")

        elif output_format == 'json':
            existing_data = []
            if os.path.exists(uncompressed_output_file):
                try:
                    # Read existing data if file exists and is not empty
                    if os.path.getsize(uncompressed_output_file) > 0:
                         with open(uncompressed_output_file, 'r') as f:
                            existing_data = json.load(f)
                except json.JSONDecodeError:
                     print(f"⚠️ Existing JSON file '{uncompressed_output_file}' is empty or invalid. Starting fresh.")
                     existing_data = []
                except Exception as e:
                    print(f"⚠️ Error reading existing JSON file for update: {e}. Starting fresh.")
                    existing_data = []


            if update_existing and existing_data:
                existing_df = pd.DataFrame(existing_data)
                merged_df = merge_domain_records(existing_df, new_df)
                final_data = merged_df.to_dict('records')
            else:
                final_data = existing_data + new_df.to_dict('records')

            with open(uncompressed_output_file, 'w') as f:
                json.dump(final_data, f, indent=2, default=str)
            print(f"✅ Results saved to {uncompressed_output_file} (JSON format)")

        elif output_format == 'jsonl':
             # For JSONL, append directly to the uncompressed file first
             with open(uncompressed_output_file, 'a') as f:
                 for record in new_df.to_dict('records'):
                     f.write(json.dumps(record, default=str) + '\n')
             print(f"✅ Results saved to {uncompressed_output_file} (JSONL format)")

             # Note: For JSONL update_existing logic is handled by the fact that process_domains skips already processed domains
             # and we are appending new results. Full merging on JSONL would require reading the entire file, which can be slow for large files.
             # Appending is more efficient for incremental saving.

        elif output_format == 'sqlite':
            import sqlite3
            db_file = uncompressed_output_file # .db extension is already handled
            conn = sqlite3.connect(db_file)

            if update_existing:
                new_df.to_sql('domains', conn, if_exists='replace', index=False, method='multi')
            else:
                new_df.to_sql('domains', conn, if_exists='append', index=False)

            conn.close()
            print(f"✅ Results saved to {db_file} (SQLite format)")
            uncompressed_output_file = db_file # Update for compression step

        else:
            # Fallback to CSV
            file_exists = os.path.exists(uncompressed_output_file)
            new_df.to_csv(uncompressed_output_file, index=False, mode='a', header=not file_exists, quoting=csv.QUOTE_MINIMAL)
            print(f"✅ Results saved to {uncompressed_output_file} (CSV format)")

        # Apply compression if enabled and the file exists
        if compression and compression != 'None' and os.path.exists(uncompressed_output_file):
            compressed_file_name = compress_file(uncompressed_output_file, compression)
            config['current_output_file'] = compressed_file_name # Update config to the compressed file name for checkpoint

    except Exception as e:
        print(f"❌ Error saving results: {e}")
        # Fallback to CSV append mode
        try:
            fallback_file = config['output_file'].replace('.jsonl', '.csv') # Fallback to CSV
            file_exists = os.path.exists(fallback_file)
            # Try to append to CSV as a last resort
            new_df.to_csv(fallback_file, index=False, mode='a', header=not file_exists, quoting=csv.QUOTE_MINIMAL)
            print(f"✅ Fallback: Results saved to {fallback_file} (CSV format)")
        except Exception as e2:
            print(f"❌ Critical error: Could not save results even with fallback: {e2}")


def merge_domain_records(existing_df, new_df):
    """Merge new domain records with existing ones, updating fields that have new data"""
    if existing_df.empty:
        return new_df

    if new_df.empty:
        return existing_df

    # Ensure both DataFrames have the same columns
    all_cols = list(set(existing_df.columns) | set(new_df.columns))
    for col in all_cols:
        if col not in existing_df.columns:
            existing_df[col] = "NA"
        if col not in new_df.columns:
            new_df[col] = "NA"

    # Reorder columns to match
    existing_df = existing_df[all_cols]
    new_df = new_df[all_cols]

    # Create a merged DataFrame
    merged_df = existing_df.copy()

    for _, new_row in new_df.iterrows():
        domain = new_row['Domain']
        existing_mask = merged_df['Domain'] == domain

        if existing_mask.any():
            # Update existing record - only update fields that are not "NA" or "disabled" in new data
            existing_idx = merged_df[existing_mask].index[0]
            for col in new_df.columns:
                if col != 'Domain':  # Don't update the domain name itself
                    new_value = new_row[col]
                    # Update if new value is not NA/disabled and is different from existing
                    if (new_value not in ["NA", "disabled", ""] and
                        str(new_value).strip() != "" and
                        new_value != merged_df.loc[existing_idx, col]):
                        merged_df.loc[existing_idx, col] = new_value
        else:
            # Add new record
            merged_df = pd.concat([merged_df, new_row.to_frame().T], ignore_index=True)

    return merged_df

# ---------- Main Execution Function ----------
async def main(preset: str = "balanced", custom_config: dict = None):
    """Main execution function with preset support"""
    config = get_config(preset)

    # Apply custom configuration overrides if provided
    if custom_config:
        config.update(custom_config)

    print(f"🚀 Starting WordPress Domain Analysis with '{preset}' preset")
    print(f"📋 Configuration:")
    print(f"   • Max Concurrency: {config['MAX_CONCURRENCY']}")
    print(f"   • Request Timeout: {config['REQUEST_TIMEOUT_SECONDS']}s")
    print(f"   • Batch Save Interval: {config['batch_save_interval']}")
    print(f"   • Output Format: {config.get('OUTPUT_FORMAT', 'csv').upper()}")
    print(f"   • Update Existing Records: {'Yes' if config.get('UPDATE_EXISTING_RECORDS', True) else 'No'}")
    print(f"   • Force Reprocess: {'Yes' if config.get('FORCE_REPROCESS', False) else 'No'}")
    print(f"   • Output Compression: {config.get('OUTPUT_COMPRESSION')}")


    # Show enabled features
    enabled_features = []
    if config['ENABLE_DNS_MX']: enabled_features.append("DNS MX")
    if config['ENABLE_WHOIS']: enabled_features.append("WHOIS")
    if config['ENABLE_CRAWLING']: enabled_features.append("Crawling")
    if config['ENABLE_WORDPRESS_API'] and config['ENABLE_POSTS_API']: enabled_features.append("Posts API")
    if config['ENABLE_WORDPRESS_API'] and config['ENABLE_PAGES_API']: enabled_features.append("Pages API")
    print(f"   🔧 Features enabled: {', '.join(enabled_features)}")

    print(f"   • Shutdown file: {config['shutdown_file']} (create this file to stop gracefully)")
    print()

    domains = validate_and_load_data(config['input_file'])
    await process_domains(domains, config)

# ---------- Entry Point ----------
if __name__ == "__main__":
    import asyncio
    # Run the async main function
    asyncio.run(main("balanced", {"ENABLE_PAGES_API": True}))

    # Run 1: Basic info only
#await main("balanced", {"ENABLE_WORDPRESS_API": False})

# Run 2: Add WordPress Posts
#await main("balanced", {"ENABLE_WHOIS": False, "ENABLE_CRAWLING": False, "ENABLE_PAGES_API": False})

# Run 3: Add WordPress Pages
#await main("balanced", {"ENABLE_WHOIS": False, "ENABLE_CRAWLING": False, "ENABLE_POSTS_API": False})