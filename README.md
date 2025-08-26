# WordPress Domain Analysis Tool

A comprehensive Python script for analyzing WordPress domains to extract valuable business intelligence data including contact information, social media links, WordPress API data, and technical details.

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- Internet connection for domain analysis

### Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd wp-domain-data-analysis
   ```

2. **Install required dependencies**
   ```bash
   pip install pandas requests dnspython python-whois beautifulsoup4 aiohttp
   ```

3. **Prepare your domain list**
   - Create a CSV file with domains (see [Input Format](#input-format))
   - Or use the provided `sample.csv` as a template

4. **Run the analysis**
   ```bash
   python main.py
   ```

## 📋 Input Format

Create a CSV file with your domains. The script accepts two formats:

**Option 1: Column named "Domain" (recommended)**
```csv
Domain
jamesclear.com
vikram.im
masterblogging.com
```

**Option 2: First column (any name)**
```csv
Domain Name
jamesclear.com
vikram.im
masterblogging.com
```

Save your file as `sample.csv` in the same directory as `main.py`.

## 🎯 What This Tool Analyzes

For each domain, the script extracts:

### Technical Information
- **Google Workspace Detection**: Checks if domain uses Google Workspace (Gmail for Business)
- **WHOIS Data**: Country of origin, domain creation date, last modified date
- **DNS MX Records**: Mail server configuration

### Content Analysis
- **WordPress API Integration**: 
  - Latest blog posts (title, date, link)
  - Latest pages (title, date, link)
  - Last modified content dates
- **Contact Information**:
  - Email addresses found on the website
  - Contact forms detection
  - Contact and About page links
- **Social Media Links**: Facebook, LinkedIn, Twitter/X, Instagram profiles

### Website Crawling
- **Page Analysis**: Crawls common pages (home, contact, about, etc.)
- **Email Extraction**: Finds email addresses across multiple pages
- **Social Link Discovery**: Identifies social media profiles

## 🔧 Configuration & Performance Presets

The script offers three performance presets to balance speed vs. completeness:

### Speed Preset (Fastest)
```python
asyncio.run(main("speed"))
```
- **Best for**: Large domain lists (1000+ domains)
- **Features**: Basic WordPress posts, email extraction, social links
- **Disabled**: WHOIS lookups, WordPress pages API
- **Concurrency**: 50 simultaneous requests
- **Timeout**: 8 seconds per request

### Balanced Preset (Default)
```python
asyncio.run(main("balanced"))  # or just python main.py
```
- **Best for**: Medium domain lists (100-1000 domains)
- **Features**: All features enabled
- **Concurrency**: 20 simultaneous requests
- **Timeout**: 12 seconds per request

### Complete Preset (Most Thorough)
```python
asyncio.run(main("complete"))
```
- **Best for**: Small, high-value domain lists (< 100 domains)
- **Features**: All features + extended page crawling
- **Concurrency**: 10 simultaneous requests
- **Timeout**: 20 seconds per request

## 🛠️ Advanced Usage

### Custom Configuration

You can override any settings by passing a custom configuration:

```python
import asyncio
from main import main

# Custom configuration example
custom_config = {
    'MAX_CONCURRENCY': 15,
    'REQUEST_TIMEOUT_SECONDS': 10,
    'ENABLE_WHOIS': False,  # Skip slow WHOIS lookups
    'OUTPUT_FORMAT': 'json',  # Output as JSON instead of CSV
    'MAX_EMAILS_IN_OUTPUT': 5,
    'input_file': 'my_domains.csv',
    'output_file': 'my_results.csv'
}

asyncio.run(main("balanced", custom_config))
```

### Feature Toggles

Enable/disable specific features:

```python
# Only WordPress API analysis (fastest)
config = {
    'ENABLE_WHOIS': False,
    'ENABLE_DNS_MX': False,
    'ENABLE_CRAWLING': False,
    'ENABLE_WORDPRESS_API': True
}

# Only contact information extraction
config = {
    'ENABLE_WHOIS': False,
    'ENABLE_WORDPRESS_API': False,
    'ENABLE_CRAWLING': True
}
```

### Output Formats

Choose your preferred output format:

```python
# CSV (default)
{'OUTPUT_FORMAT': 'csv'}

# JSON
{'OUTPUT_FORMAT': 'json'}

# JSON Lines (one JSON object per line)
{'OUTPUT_FORMAT': 'jsonl'}

# SQLite database
{'OUTPUT_FORMAT': 'sqlite'}
```

## 🔄 Resume & Checkpoint System

The script automatically saves progress and can resume from interruptions:

- **Checkpoint File**: `checkpoint.json` tracks processed domains
- **Graceful Shutdown**: Press `Ctrl+C` or create `stop_processing.txt` file
- **Auto-Resume**: Restart the script to continue from where it left off
- **Force Reprocess**: Set `FORCE_REPROCESS: True` to ignore checkpoints

### Manual Stop
Create a file named `stop_processing.txt` in the script directory to stop gracefully:
```bash
touch stop_processing.txt
```

## 📊 Output Data

The script generates a CSV file with these columns:

| Column | Description |
|--------|-------------|
| Domain | The analyzed domain name |
| Google Workspace | Whether domain uses Google Workspace (Yes/No/NA) |
| Country of Origin | Domain registration country |
| Domain Created | Domain creation date |
| Domain Last Modified | Domain last update date |
| Pages Crawled (Count) | Number of pages analyzed |
| Has Contact Form | Contact form detected (Yes/No) |
| Emails (unique) | Email addresses found (semicolon-separated) |
| Social Links (unique) | Social media profiles (semicolon-separated) |
| Contact Page Link | URL to contact page |
| About Page Link | URL to about page |
| Posts API Status | WordPress posts API availability |
| Posts Last Created Title | Latest blog post title |
| Posts Last Created Link | Latest blog post URL |
| Posts Last Created Date | Latest blog post date |
| Posts Last Modified Title | Most recently updated post title |
| Posts Last Modified Date | Most recent post update date |
| Pages API Status | WordPress pages API availability |
| Pages Last Created Title | Latest page title |
| Pages Last Created Link | Latest page URL |
| Pages Last Created Date | Latest page creation date |
| Pages Last Modified Title | Most recently updated page title |
| Pages Last Modified Date | Most recent page update date |

## 🚦 Performance Guidelines

### For Different Domain List Sizes

| Domain Count | Recommended Preset | Estimated Time | Memory Usage |
|--------------|-------------------|----------------|--------------|
| 1-50 | complete | 5-15 minutes | Low |
| 51-500 | balanced | 30-90 minutes | Medium |
| 501-2000 | speed | 1-4 hours | Medium |
| 2000+ | speed + custom config | 4+ hours | High |

### Optimization Tips

1. **Large Lists**: Use "speed" preset and disable WHOIS
2. **Network Issues**: Reduce `MAX_CONCURRENCY` to 5-10
3. **Timeouts**: Increase `REQUEST_TIMEOUT_SECONDS` for slow sites
4. **Memory**: Process in batches using custom domain lists

## 🔍 Troubleshooting

### Common Issues

**"Missing dependency" error**
```bash
pip install aiohttp pandas requests dnspython python-whois beautifulsoup4
```

**"CSV file not found"**
- Ensure `sample.csv` exists in the script directory
- Check file path in configuration: `input_file: "your_file.csv"`

**Slow performance**
- Use "speed" preset: `asyncio.run(main("speed"))`
- Disable WHOIS: `{'ENABLE_WHOIS': False}`
- Reduce concurrency: `{'MAX_CONCURRENCY': 10}`

**Network timeouts**
- Increase timeout: `{'REQUEST_TIMEOUT_SECONDS': 20}`
- Reduce concurrency: `{'MAX_CONCURRENCY': 5}`

**Memory issues with large lists**
- Process in smaller batches
- Use "speed" preset
- Reduce `batch_save_interval` to save more frequently

### Error Handling

The script handles various errors gracefully:
- **Network timeouts**: Continues with next domain
- **Invalid domains**: Logs error and continues
- **API failures**: Marks as "API not available"
- **Interruptions**: Saves progress and allows resume

## 🎛️ Google Colab Usage

The script is optimized for Google Colab environments:

```python
# In Google Colab
import asyncio
from main import main, get_config, process_domains, validate_and_load_data

# Run entire analysis
await main("balanced")

# Or run components separately
config = get_config("speed")
config['MAX_CONCURRENCY'] = 10  # Adjust for Colab
domains = validate_and_load_data("sample.csv")
results = await process_domains(domains, config)
```

## 📈 Use Cases

### Lead Generation
- Extract contact emails from competitor domains
- Find social media profiles for outreach
- Identify active WordPress sites for plugin/theme sales

### Market Research
- Analyze competitor content freshness
- Identify market trends through blog activity
- Map competitive landscape

### SEO Analysis
- Find contact information for link building
- Identify active blogs in your niche
- Analyze content publication patterns

### Business Intelligence
- Validate domain authenticity
- Check technical infrastructure (Google Workspace usage)
- Assess website activity levels

## 🔒 Ethical Usage

This tool is designed for legitimate business purposes:
- ✅ Lead generation and market research
- ✅ Competitive analysis
- ✅ SEO and content marketing
- ❌ Spam or unsolicited marketing
- ❌ Data harvesting for malicious purposes

Always respect website terms of service and applicable privacy laws.

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review error messages in the console output
3. Verify your input CSV format
4. Test with a small domain list first

## 📄 License

This tool is provided as-is for educational and business purposes. Use responsibly and in compliance with applicable laws and website terms of service.
