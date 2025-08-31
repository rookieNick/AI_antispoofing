#!/usr/bin/env python3
"""
Test script to verify folder increment logic works correctly
"""
import os
import tempfile
import shutil
from plot_utils import get_next_index

def test_folder_increment():
    """Test the get_next_index function with different prefixes"""
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing in temporary directory: {temp_dir}")
        
        # Test 1: Empty directory - should return 1
        print("\n=== Test 1: Empty directory ===")
        index = get_next_index(temp_dir, "train")
        print(f"get_next_index(temp_dir, 'train') = {index}")
        assert index == 1, f"Expected 1, got {index}"
        
        index = get_next_index(temp_dir, "test")
        print(f"get_next_index(temp_dir, 'test') = {index}")
        assert index == 1, f"Expected 1, got {index}"
        
        # Test 2: Create some train folders
        print("\n=== Test 2: Create train folders ===")
        os.makedirs(os.path.join(temp_dir, "train_1_20250101"))
        os.makedirs(os.path.join(temp_dir, "train_2_20250102"))
        os.makedirs(os.path.join(temp_dir, "train_5_20250105"))  # Gap to test max logic
        
        index = get_next_index(temp_dir, "train")
        print(f"After creating train_1, train_2, train_5: get_next_index = {index}")
        assert index == 6, f"Expected 6, got {index}"
        
        # Test 3: Test folders should still start from 1
        print("\n=== Test 3: Test folders independent ===")
        index = get_next_index(temp_dir, "test")
        print(f"get_next_index(temp_dir, 'test') = {index}")
        assert index == 1, f"Expected 1, got {index}"
        
        # Test 4: Create some test folders
        print("\n=== Test 4: Create test folders ===")
        os.makedirs(os.path.join(temp_dir, "test_1_20250101"))
        os.makedirs(os.path.join(temp_dir, "test_3_20250103"))
        
        index = get_next_index(temp_dir, "test")
        print(f"After creating test_1, test_3: get_next_index = {index}")
        assert index == 4, f"Expected 4, got {index}"
        
        # Test 5: Train folders should be unaffected
        print("\n=== Test 5: Train folders unaffected ===")
        index = get_next_index(temp_dir, "train")
        print(f"get_next_index(temp_dir, 'train') = {index}")
        assert index == 6, f"Expected 6, got {index}"
        
        # Test 6: Test with some invalid folder names mixed in
        print("\n=== Test 6: Invalid folder names ignored ===")
        os.makedirs(os.path.join(temp_dir, "train_invalid"))
        os.makedirs(os.path.join(temp_dir, "train_abc_20250106"))
        os.makedirs(os.path.join(temp_dir, "result_1_20250101"))  # Old format
        
        index = get_next_index(temp_dir, "train")
        print(f"After creating invalid folders: get_next_index = {index}")
        assert index == 6, f"Expected 6, got {index}"
        
        print("\n=== All tests passed! ===")
        print("✅ Folder increment logic works correctly")
        print("✅ Train and test prefixes are handled independently")
        print("✅ Invalid folder names are ignored")
        print("✅ Gaps in numbering are handled correctly")

if __name__ == "__main__":
    test_folder_increment()
