import os


import re

def extract_subject(filename):
    # Subject is everything before the first 'v' in the filename
    idx = filename.find('v')
    if idx > 0:
        return filename[:idx]
    return None

def count_images_and_subjects(base_path):
    result = {}
    for split in ['train', 'test']:
        result[split] = {}
        for label in ['live', 'spoof']:
            folder = os.path.join(base_path, split, label)
            image_files = []
            subjects = set()
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(f)
                        subj = extract_subject(f)
                        if subj:
                            subjects.add(subj)
            result[split][label] = {
                'images': len(image_files),
                'subjects': len(subjects)
            }
    return result

if __name__ == "__main__":
    base_path = r"casia-fasd"
    counts = count_images_and_subjects(base_path)
    for split in counts:
        for label in counts[split]:
            print(f"{split}/{label}: {counts[split][label]['images']} images, {counts[split][label]['subjects']} subjects")