#!/usr/bin/env python3
"""
Dataset Verification Script
Verifies the existing dataset structure and provides statistics
"""

import os
from pathlib import Path

def verify_dataset_structure():
    """Verify the dataset structure and provide statistics"""
    print("🔍 Verifying Dataset Structure")
    print("=" * 50)
    
    # Dataset paths
    base_dir = "archive/real_vs_fake/real-vs-fake"
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'valid')
    test_dir = os.path.join(base_dir, 'test')
    
    # Check if directories exist
    directories = {
        "Base Directory": base_dir,
        "Training Directory": train_dir,
        "Validation Directory": val_dir,
        "Test Directory": test_dir
    }
    
    all_exist = True
    for name, path in directories.items():
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} - NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n❌ Some directories are missing!")
        return False
    
    # Count images in each category
    print("\n📊 Dataset Statistics:")
    print("-" * 30)
    
    categories = ['fake', 'real']
    splits = ['train', 'valid', 'test']
    
    total_images = 0
    
    for split in splits:
        print(f"\n{split.upper()} SET:")
        split_path = os.path.join(base_dir, split)
        split_total = 0
        
        for category in categories:
            category_path = os.path.join(split_path, category)
            if os.path.exists(category_path):
                # Count image files
                image_files = [f for f in os.listdir(category_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                count = len(image_files)
                print(f"  {category.capitalize()}: {count:,} images")
                split_total += count
            else:
                print(f"  {category.capitalize()}: Directory not found")
        
        print(f"  Total: {split_total:,} images")
        total_images += split_total
    
    print(f"\n🎯 TOTAL DATASET: {total_images:,} images")
    
    # Verify expected structure
    print("\n🔍 Verifying Expected Structure:")
    expected_structure = [
        "archive/real_vs_fake/real-vs-fake/train/fake",
        "archive/real_vs_fake/real-vs-fake/train/real",
        "archive/real_vs_fake/real-vs-fake/valid/fake",
        "archive/real_vs_fake/real-vs-fake/valid/real",
        "archive/real_vs_fake/real-vs-fake/test/fake",
        "archive/real_vs_fake/real-vs-fake/test/real"
    ]
    
    structure_ok = True
    for path in expected_structure:
        if os.path.exists(path):
            file_count = len([f for f in os.listdir(path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"✅ {path}: {file_count:,} images")
        else:
            print(f"❌ {path}: Not found")
            structure_ok = False
    
    if structure_ok:
        print("\n🎉 Dataset structure is perfect!")
        print("✅ Ready for training!")
        return True
    else:
        print("\n⚠️  Dataset structure has issues.")
        return False

def check_config():
    """Check if config.py is properly set up"""
    print("\n🔧 Checking Configuration:")
    print("-" * 30)
    
    try:
        import sys
        sys.path.append('Pretrained_Models')
        from config import BASE_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR
        
        print(f"✅ BASE_DIR: {BASE_DIR}")
        print(f"✅ TRAIN_DIR: {TRAIN_DIR}")
        print(f"✅ VAL_DIR: {VAL_DIR}")
        print(f"✅ TEST_DIR: {TEST_DIR}")
        
        # Check if paths exist
        if os.path.exists(BASE_DIR):
            print("✅ Base directory exists")
        else:
            print("❌ Base directory not found")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False

def main():
    print("🚀 DeepFake Detection Dataset Verification")
    print("=" * 50)
    
    # Verify dataset structure
    dataset_ok = verify_dataset_structure()
    
    # Check configuration
    config_ok = check_config()
    
    print("\n" + "=" * 50)
    if dataset_ok and config_ok:
        print("🎉 Everything is ready for training!")
        print("\nNext steps:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Start training: python train_models.py")
    else:
        print("❌ Setup issues detected. Please fix them before training.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
