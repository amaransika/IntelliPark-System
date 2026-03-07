import os
import glob
import random
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm  # You might need to install this: pip install tqdm

# --- CONFIGURATION ---
# The folder where your current mixed images/xmls are
INPUT_FOLDER = '../datasets/Numberplate' 

# The folder where we will create the clean, YOLO-ready dataset
OUTPUT_FOLDER = '../datasets/SriLankan_YOLO'

# Split ratio (80% for training, 20% for testing)
TRAIN_RATIO = 0.8 

def convert_bbox(size, box):
    """Converts XML box (xmin, xmax, ymin, ymax) to YOLO (x, y, w, h)."""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_file, output_txt_file):
    """Reads an XML file and writes a YOLO TXT file."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        with open(output_txt_file, 'w') as out_file:
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls != "license-plate" and cls != "number plate": 
                    # Sometimes datasets have different names, we assume ID 0 for plates
                    continue
                
                # YOLO class ID for license plate is 0
                cls_id = 0 
                
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert_bbox((w, h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        return True
    except Exception as e:
        print(f"Error converting {xml_file}: {e}")
        return False

def main():
    # 1. Create Directory Structure
    for split in ['train', 'valid']:
        for dtype in ['images', 'labels']:
            os.makedirs(os.path.join(OUTPUT_FOLDER, split, dtype), exist_ok=True)

    # 2. Get all XML files
    xml_files = glob.glob(os.path.join(INPUT_FOLDER, '*.xml'))
    if not xml_files:
        print(f"Error: No XML files found in {INPUT_FOLDER}")
        return

    print(f"Found {len(xml_files)} XML files. processing...")

    # 3. Shuffle and Split
    random.shuffle(xml_files)
    split_index = int(len(xml_files) * TRAIN_RATIO)
    train_files = xml_files[:split_index]
    valid_files = xml_files[split_index:]

    # 4. Process Files
    def process_batch(files, split_name):
        for xml_path in files:
            # Define paths
            base_name = os.path.basename(xml_path).replace('.xml', '')
            
            # Find corresponding image (could be .jpg, .jpeg, .png)
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                potential_path = os.path.join(INPUT_FOLDER, base_name + ext)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if image_path is None:
                print(f"Warning: No image found for {xml_path}")
                continue

            # Destination paths
            dest_img_path = os.path.join(OUTPUT_FOLDER, split_name, 'images', os.path.basename(image_path))
            dest_txt_path = os.path.join(OUTPUT_FOLDER, split_name, 'labels', base_name + '.txt')

            # Convert XML -> TXT
            if convert_annotation(xml_path, dest_txt_path):
                # Copy Image
                shutil.copy(image_path, dest_img_path)

    print("Creating Training Set...")
    process_batch(train_files, 'train')
    print("Creating Validation Set...")
    process_batch(valid_files, 'valid')

    # 5. Create data.yaml
    yaml_content = f"""
path: {os.path.abspath(OUTPUT_FOLDER)} # Absolute path to dataset
train: train/images
val: valid/images

# Classes
names:
  0: license-plate
"""
    with open(os.path.join(OUTPUT_FOLDER, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

    print(f"\nSuccess! Dataset prepared at: {OUTPUT_FOLDER}")
    print(f"You can now train your model.")

if __name__ == "__main__":
    main()