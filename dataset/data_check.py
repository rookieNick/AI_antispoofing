import os
import re
import cv2
from PIL import Image
import hashlib
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import time

def extract_subject(filename):
    idx = filename.find('v')
    if idx > 0:
        return filename[:idx]
    return None

def process_single_image(args):
    """Process a single image and return all check results"""
    filepath, filename, label, target_size, target_mode = args
    
    results = {
        'filepath': filepath,
        'filename': filename,
        'label': label,
        'subject': extract_subject(filename),
        'corrupted': False,
        'shape': None,
        'mode': None,
        'hash': None,
        'irregular_name': False,
        'low_quality': False,
        'label_error': False
    }
    
    # Check if image is readable (combined check)
    try:
        with Image.open(filepath) as img:
            results['shape'] = img.size
            results['mode'] = img.mode
            # Quick verification
            img.verify()
    except Exception:
        results['corrupted'] = True
        return results
    
    # Hash calculation
    try:
        with open(filepath, 'rb') as f:
            results['hash'] = hashlib.md5(f.read()).hexdigest()
    except Exception:
        pass
    
    # Filename pattern check removed as requested
    
    # Label validation
    if label == 'live' and 'spoof' in filename.lower():
        results['label_error'] = True
    if label == 'spoof' and 'live' in filename.lower():
        results['label_error'] = True
    
    # Low quality check (only if not corrupted)
    if not results['corrupted']:
        try:
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                variance = cv2.Laplacian(img, cv2.CV_64F).var()
                if variance < 20 or np.mean(img) < 10:
                    results['low_quality'] = True
        except Exception:
            results['low_quality'] = True
    
    return results

def analyze_dataset(base_path, target_size=(112, 112), target_mode='RGB', max_workers=None):
    if max_workers is None:
        max_workers = min(cpu_count(), 8)  # Limit to 8 cores max
    
    print(f"Starting analysis with {max_workers} workers...")
    start_time = time.time()
    
    # Collect all image files first
    all_files = []
    for split in ['train', 'test']:
        for label in ['live', 'spoof']:
            folder = os.path.join(base_path, split, label)
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        filepath = os.path.join(folder, f)
                        all_files.append((filepath, f, label, target_size, target_mode))
    
    print(f"Found {len(all_files)} images to process...")
    
    # Process files in parallel
    with Pool(processes=max_workers) as pool:
        results = pool.map(process_single_image, all_files)
    
    # Aggregate results
    result = {'train': {'live': {'images': 0, 'subjects': set()}, 
                       'spoof': {'images': 0, 'subjects': set()}},
              'test': {'live': {'images': 0, 'subjects': set()}, 
                      'spoof': {'images': 0, 'subjects': set()}}}
    
    hash_to_file = {}
    duplicate_files = []
    corrupted_files = []
    inconsistent_shape_files = []
    inconsistent_mode_files = []
    irregular_name_files = []
    low_quality_files = []
    label_errors = []
    class_counts = {'live': 0, 'spoof': 0}
    
    for res in results:
        # Determine split from filepath
        split = 'train' if 'train' in res['filepath'] else 'test'
        label = res['label']
        
        # Count images and subjects
        result[split][label]['images'] += 1
        if res['subject']:
            result[split][label]['subjects'].add(res['subject'])
        class_counts[label] += 1
        
        # Check for issues
        if res['corrupted']:
            corrupted_files.append(res['filepath'])
        
        if res['shape'] and res['shape'] != target_size:
            inconsistent_shape_files.append((res['filepath'], res['shape']))
        
        if res['mode'] and res['mode'] != target_mode:
            inconsistent_mode_files.append((res['filepath'], res['mode']))
        
        if res['hash']:
            if res['hash'] in hash_to_file:
                duplicate_files.append((res['filepath'], hash_to_file[res['hash']]))
            else:
                hash_to_file[res['hash']] = res['filepath']
        
        if res['irregular_name']:
            irregular_name_files.append(res['filepath'])
        
        if res['low_quality']:
            low_quality_files.append(res['filepath'])
        
        if res['label_error']:
            label_errors.append(res['filepath'])
    
    # Convert sets to counts
    for split in result:
        for label in result[split]:
            result[split][label]['subjects'] = len(result[split][label]['subjects'])
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\n=== Dataset Analysis Complete ({elapsed_time:.2f}s) ===")
    for split in result:
        for label in result[split]:
            print(f"{split}/{label}: {result[split][label]['images']} images, {result[split][label]['subjects']} subjects")
    
    print(f"\nClass balance: Live={class_counts['live']}, Spoof={class_counts['spoof']}")
    
    print(f"\n✅ Corrupted images: {len(corrupted_files)}")
    if corrupted_files and len(corrupted_files) <= 10:
        for f in corrupted_files[:10]:
            print(f"  {f}")
    elif len(corrupted_files) > 10:
        print(f"  (Showing first 10 of {len(corrupted_files)})")
        for f in corrupted_files[:10]:
            print(f"  {f}")
    
    print(f"\n✅ Images with inconsistent shape: {len(inconsistent_shape_files)}")
    if inconsistent_shape_files and len(inconsistent_shape_files) <= 10:
        for f, shape in inconsistent_shape_files[:10]:
            print(f"  {f} shape={shape}")
    elif len(inconsistent_shape_files) > 10:
        print(f"  (Showing first 10 of {len(inconsistent_shape_files)})")
        for f, shape in inconsistent_shape_files[:10]:
            print(f"  {f} shape={shape}")
    
    print(f"\n✅ Images with inconsistent color mode: {len(inconsistent_mode_files)}")
    if inconsistent_mode_files and len(inconsistent_mode_files) <= 10:
        for f, mode in inconsistent_mode_files[:10]:
            print(f"  {f} mode={mode}")
    elif len(inconsistent_mode_files) > 10:
        print(f"  (Showing first 10 of {len(inconsistent_mode_files)})")
        for f, mode in inconsistent_mode_files[:10]:
            print(f"  {f} mode={mode}")
    
    print(f"\n✅ Duplicate images: {len(duplicate_files)}")
    if duplicate_files and len(duplicate_files) <= 10:
        for dup, orig in duplicate_files[:10]:
            print(f"  {dup} (duplicate of {orig})")
    elif len(duplicate_files) > 10:
        print(f"  (Showing first 10 of {len(duplicate_files)})")
        for dup, orig in duplicate_files[:10]:
            print(f"  {dup} (duplicate of {orig})")
    
    print(f"\n✅ Files with irregular names: {len(irregular_name_files)}")
    if irregular_name_files and len(irregular_name_files) <= 10:
        for f in irregular_name_files[:10]:
            print(f"  {f}")
    elif len(irregular_name_files) > 10:
        print(f"  (Showing first 10 of {len(irregular_name_files)})")
        for f in irregular_name_files[:10]:
            print(f"  {f}")
    
    print(f"\n✅ Low quality or blank images: {len(low_quality_files)}")
    if low_quality_files and len(low_quality_files) <= 10:
        for f in low_quality_files[:10]:
            print(f"  {f}")
    elif len(low_quality_files) > 10:
        print(f"  (Showing first 10 of {len(low_quality_files)})")
        for f in low_quality_files[:10]:
            print(f"  {f}")
    
    print(f"\n✅ Label errors: {len(label_errors)}")
    if label_errors and len(label_errors) <= 10:
        for f in label_errors[:10]:
            print(f"  {f}")
    elif len(label_errors) > 10:
        print(f"  (Showing first 10 of {len(label_errors)})")
        for f in label_errors[:10]:
            print(f"  {f}")
    
    print("=== End of Analysis ===")
    
    return {
        'corrupted': corrupted_files,
        'inconsistent_shape': inconsistent_shape_files,
        'inconsistent_mode': inconsistent_mode_files,
        'duplicates': duplicate_files,
        'irregular_names': irregular_name_files,
        'low_quality': low_quality_files,
        'label_errors': label_errors
    }

if __name__ == "__main__":
    base_path = r"casia-fasd"
    issues = analyze_dataset(base_path)
