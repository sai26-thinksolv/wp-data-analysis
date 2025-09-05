#!/usr/bin/env python3
"""
File Management Utility for WordPress Domain Analysis
Helps manage large processed files by splitting, compressing, and analyzing them.
"""

import os
import json
import gzip
import bz2
import pandas as pd
from datetime import datetime
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

def compress_file(file_path, compression='gzip'):
    """Compress a file"""
    print(f"\n🗜️ Compressing {file_path} with {compression}...")
    
    original_size = get_file_size_mb(file_path)
    
    if compression == 'gzip':
        with open(file_path, 'rb') as f_in:
            with gzip.open(f"{file_path}.gz", 'wb') as f_out:
                f_out.writelines(f_in)
        compressed_file = f"{file_path}.gz"
    elif compression == 'bz2':
        with open(file_path, 'rb') as f_in:
            with bz2.open(f"{file_path}.bz2", 'wb') as f_out:
                f_out.writelines(f_in)
        compressed_file = f"{file_path}.bz2"
    else:
        print("❌ Unsupported compression format")
        return
    
    compressed_size = get_file_size_mb(compressed_file)
    compression_ratio = (1 - compressed_size/original_size) * 100
    
    print(f"✅ Compressed: {compressed_file}")
    print(f"📊 Size reduction: {original_size:.1f}MB → {compressed_size:.1f}MB ({compression_ratio:.1f}% smaller)")

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

def main():
    parser = argparse.ArgumentParser(description='Manage WordPress domain analysis files')
    parser.add_argument('file', help='Path to the processed file')
    parser.add_argument('--analyze', action='store_true', help='Analyze file statistics')
    parser.add_argument('--split', type=int, metavar='SIZE_MB', help='Split file into chunks of specified size (MB)')
    parser.add_argument('--compress', choices=['gzip', 'bz2'], help='Compress file')
    parser.add_argument('--clean', action='store_true', help='Remove domains with minimal data')
    parser.add_argument('--output', help='Output file path (for clean operation)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        return
    
    if args.analyze:
        analyze_file(args.file)
    
    if args.split:
        split_file_by_size(args.file, args.split)
    
    if args.compress:
        compress_file(args.file, args.compress)
    
    if args.clean:
        clean_data(args.file, args.output)
    
    if not any([args.analyze, args.split, args.compress, args.clean]):
        # Default to analysis
        analyze_file(args.file)

if __name__ == "__main__":
    main()
