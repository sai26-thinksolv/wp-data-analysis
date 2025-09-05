# File Size Management for WordPress Domain Analysis

## Overview
This guide provides strategies to prevent your processed files from getting too large during WordPress domain analysis.

## Quick Solutions

### 1. Use the "minimal" preset for smallest files
```python
# Run with minimal data collection
await main("minimal")
```

### 2. Enable automatic file rotation and compression
```python
# Custom config for size management
config = {
    'MAX_FILE_SIZE_MB': 50,        # Split files at 50MB
    'OUTPUT_COMPRESSION': 'gzip',   # Compress files automatically
    'ENABLE_DATA_DEDUPLICATION': True,  # Remove low-value records
    'OUTPUT_FORMAT': 'jsonl'       # More compact than CSV
}
await main("balanced", config)
```

## Configuration Options

### File Size Limits
- `MAX_FILE_SIZE_MB`: Automatically create new files when size exceeds limit
- Files are named with timestamps: `processed_1704067200.csv`

### Compression Options
- `OUTPUT_COMPRESSION`: `'gzip'`, `'bz2'`, or `None`
- Typically reduces file size by 60-80%
- Files are automatically compressed after saving

### Data Optimization
- `ENABLE_DATA_DEDUPLICATION`: Removes domains with minimal data
- Only keeps domains with at least 2 meaningful fields
- Truncates long titles to 100 characters
- Removes redundant "NA" values for optional fields

### Output Formats (by efficiency)
1. **JSONL** (most compact) - One JSON object per line
2. **JSON** - Standard JSON format
3. **CSV** - Human readable but larger
4. **SQLite** - Database format, good for queries

## Preset Configurations

### "minimal" - Smallest Files
```python
await main("minimal")
```
- Skips crawling and WHOIS data
- Only WordPress Posts API
- Limits emails/socials to 3/5 items
- 25MB file size limit
- GZIP compression enabled

### "speed" - Balanced Size/Speed
```python
await main("speed")
```
- Faster processing, moderate file sizes
- 50MB file size limit
- GZIP compression
- Reduced crawling

### "balanced" - Default
```python
await main("balanced")
```
- 100MB file size limit
- GZIP compression
- All features enabled

## File Management Utility

Use the included `file_manager.py` script to manage existing files:

### Analyze file statistics
```bash
python file_manager.py processed.csv --analyze
```

### Split large files
```bash
python file_manager.py processed.csv --split 25
```

### Compress files
```bash
python file_manager.py processed.csv --compress gzip
```

### Clean data (remove low-value records)
```bash
python file_manager.py processed.csv --clean
```

## Data Reduction Strategies

### 1. Selective Feature Disabling
```python
config = {
    'ENABLE_WHOIS': False,      # Skip domain registration data
    'ENABLE_CRAWLING': False,   # Skip website crawling
    'ENABLE_PAGES_API': False,  # Only check posts, not pages
}
```

### 2. Limit Data Collection
```python
config = {
    'MAX_EMAILS_IN_OUTPUT': 3,    # Limit emails per domain
    'MAX_SOCIALS_IN_OUTPUT': 5,   # Limit social links
    'COMMON_PAGES': ["", "contact"]  # Crawl fewer pages
}
```

### 3. Batch Processing
```python
config = {
    'batch_save_interval': 20,  # Save every 20 domains
}
```

## File Size Estimates

For 10,000 domains:
- **Minimal preset**: ~15-25MB (compressed)
- **Speed preset**: ~30-50MB (compressed)  
- **Balanced preset**: ~60-100MB (compressed)
- **Complete preset**: ~150-300MB (compressed)

## Best Practices

1. **Start with "minimal" preset** to test your domain list
2. **Use JSONL format** for better compression
3. **Enable automatic compression** for long-term storage
4. **Monitor file sizes** during processing
5. **Clean data periodically** to remove low-value records
6. **Split large datasets** by domain count or alphabetically

## Example Usage

```python
# For large domain lists (10k+ domains)
config = {
    'MAX_FILE_SIZE_MB': 25,
    'OUTPUT_COMPRESSION': 'gzip',
    'OUTPUT_FORMAT': 'jsonl',
    'ENABLE_DATA_DEDUPLICATION': True,
    'ENABLE_WHOIS': False,
    'ENABLE_CRAWLING': False,
    'MAX_EMAILS_IN_OUTPUT': 3,
    'batch_save_interval': 50
}
await main("minimal", config)

# For detailed analysis (smaller lists)
await main("complete")

# For production monitoring
await main("balanced", {
    'MAX_FILE_SIZE_MB': 100,
    'OUTPUT_COMPRESSION': 'gzip'
})
```

## Troubleshooting

### File too large despite settings?
- Check if `ENABLE_DATA_DEDUPLICATION` is enabled
- Reduce `MAX_EMAILS_IN_OUTPUT` and `MAX_SOCIALS_IN_OUTPUT`
- Use "minimal" preset
- Enable compression

### Need to process existing large file?
```bash
# Split into smaller files
python file_manager.py large_file.csv --split 50

# Clean and compress
python file_manager.py large_file.csv --clean
python file_manager.py large_file_cleaned.csv --compress gzip
```

### Out of disk space?
- Enable compression immediately: `'OUTPUT_COMPRESSION': 'gzip'`
- Reduce file size limit: `'MAX_FILE_SIZE_MB': 25`
- Clean existing files: `python file_manager.py *.csv --clean`
