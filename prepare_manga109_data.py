"""
Prepare Manga109s dataset by extracting text region crops from XML annotations.

This script processes the Manga109s dataset to create a CSV file with image crops
and their corresponding text labels for training.
"""

import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

load_dotenv()

MANGA109_ROOT = Path(os.getenv("MANGA109_ROOT", "")).expanduser()
ANNOTATIONS_DIR = MANGA109_ROOT / "annotations"
IMAGES_DIR = MANGA109_ROOT / "images"
CROPS_DIR = MANGA109_ROOT / "crops"
OUTPUT_CSV = MANGA109_ROOT / "data.csv"

# Train/test split ratio
TRAIN_RATIO = 0.9
RANDOM_SEED = 42


def parse_manga109_xml(xml_path):
    """Parse Manga109 XML annotation file and extract text regions."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    book_title = root.attrib['title']
    
    text_regions = []
    for page in root.findall('.//page'):
        page_index = page.attrib['index']
        page_width = int(page.attrib['width'])
        page_height = int(page.attrib['height'])
        
        for text_elem in page.findall('.//text'):
            try:
                xmin = int(text_elem.attrib['xmin'])
                ymin = int(text_elem.attrib['ymin'])
                xmax = int(text_elem.attrib['xmax'])
                ymax = int(text_elem.attrib['ymax'])
                text_content = text_elem.text or ""
                
                # Skip empty text
                if not text_content.strip():
                    continue
                
                # Skip very small boxes (likely noise)
                if (xmax - xmin) < 10 or (ymax - ymin) < 10:
                    continue
                
                text_regions.append({
                    'book': book_title,
                    'page': page_index,
                    'page_width': page_width,
                    'page_height': page_height,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'text': text_content.strip()
                })
            except (KeyError, ValueError) as e:
                print(f"Warning: Skipping invalid text element in {book_title} page {page_index}: {e}")
                continue
    
    return text_regions


def crop_and_save_regions(text_regions, output_dir):
    """Crop text regions from images and save them."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    crop_records = []
    
    for region in tqdm(text_regions, desc="Cropping images"):
        book = region['book']
        page = region['page']
        
        # Convert page to int if it's a string
        if isinstance(page, str):
            page = int(page)
        
        # Construct image path
        image_path = IMAGES_DIR / book / f"{page:03d}.jpg"
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        try:
            # Load image
            img = Image.open(image_path)
            
            # Crop region
            box = (region['xmin'], region['ymin'], region['xmax'], region['ymax'])
            cropped = img.crop(box)
            
            # Generate unique crop filename
            crop_filename = f"{book}_{page:03d}_{region['xmin']}_{region['ymin']}_{region['xmax']}_{region['ymax']}.jpg"
            crop_path = output_dir / crop_filename
            
            # Save crop
            cropped.save(crop_path, quality=95)
            
            crop_records.append({
                'crop_path': f"crops/{crop_filename}",
                'text': region['text'],
                'book': book,
                'page': page,
                'xmin': region['xmin'],
                'ymin': region['ymin'],
                'xmax': region['xmax'],
                'ymax': region['ymax']
            })
            
        except Exception as e:
            print(f"Error processing {book} page {page}: {e}")
            continue
    
    return crop_records


def main():
    """Main function to prepare Manga109s dataset."""
    print("=" * 60)
    print("Preparing Manga109s Dataset")
    print("=" * 60)
    
    print(f"\nManga109 Root: {MANGA109_ROOT}")
    print(f"Annotations: {ANNOTATIONS_DIR}")
    print(f"Images: {IMAGES_DIR}")
    print(f"Output Crops: {CROPS_DIR}")
    print(f"Output CSV: {OUTPUT_CSV}")
    
    # Check if directories exist
    if not ANNOTATIONS_DIR.exists():
        print(f"\nError: Annotations directory not found: {ANNOTATIONS_DIR}")
        return
    
    if not IMAGES_DIR.exists():
        print(f"\nError: Images directory not found: {IMAGES_DIR}")
        return
    
    # Parse all XML files
    print("\n" + "=" * 60)
    print("Step 1: Parsing XML annotations")
    print("=" * 60)
    
    all_text_regions = []
    xml_files = list(ANNOTATIONS_DIR.glob("*.xml"))
    
    print(f"Found {len(xml_files)} XML annotation files")
    
    for xml_path in tqdm(xml_files, desc="Parsing XMLs"):
        regions = parse_manga109_xml(xml_path)
        all_text_regions.extend(regions)
    
    print(f"\nTotal text regions found: {len(all_text_regions)}")
    
    if len(all_text_regions) == 0:
        print("Error: No text regions found in annotations!")
        return
    
    # Crop and save regions
    print("\n" + "=" * 60)
    print("Step 2: Cropping and saving text regions")
    print("=" * 60)
    
    crop_records = crop_and_save_regions(all_text_regions, CROPS_DIR)
    
    print(f"\nTotal crops saved: {len(crop_records)}")
    
    # Create train/test split
    print("\n" + "=" * 60)
    print("Step 3: Creating train/test split")
    print("=" * 60)
    
    random.seed(RANDOM_SEED)
    random.shuffle(crop_records)
    
    split_idx = int(len(crop_records) * TRAIN_RATIO)
    train_records = crop_records[:split_idx]
    test_records = crop_records[split_idx:]
    
    # Add split column
    for record in train_records:
        record['split'] = 'train'
    for record in test_records:
        record['split'] = 'test'
    
    # Combine and create DataFrame
    all_records = train_records + test_records
    df = pd.DataFrame(all_records)
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\nTrain samples: {len(train_records)}")
    print(f"Test samples: {len(test_records)}")
    print(f"Total samples: {len(all_records)}")
    print(f"\nDataset saved to: {OUTPUT_CSV}")
    
    # Display sample
    print("\n" + "=" * 60)
    print("Sample data:")
    print("=" * 60)
    print(df.head())
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
