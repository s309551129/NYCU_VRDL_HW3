import json
import os
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

root = "./dataset/dataset/train/"
img_save_train_root = "./dataset/dataset/all_train_images/"
img_save_valid_root = "./dataset/dataset/valid_images/"
ann_save_root = "./dataset/dataset/"

CATEGORIES = [
    {
        'id': 1,
        'name': 'nucleus',
    },
]

def main():
    coco_output_train = {
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }
    coco_output_valid = {
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }
    train = True
    image_id = 1
    segmentation_id = 1
    
    root_dirs = os.listdir(root)
    root_dirs.sort()
    for dir in root_dirs:
        """
        if train == True and image_id == 24:
            train = False
            image_id = 1
            segmentation_id = 1
        """ 
        image_filename = os.path.join(root, dir, "images", dir)+".png"
        image = Image.open(image_filename)
        image_info = pycococreatortools.create_image_info(image_id, os.path.basename(image_filename), image.size)
        if train == True: # training
            coco_output_train["images"].append(image_info)
            image.save(os.path.join(img_save_train_root, dir)+".png")
        else: # validation
            coco_output_valid["images"].append(image_info)
            image.save(os.path.join(img_save_valid_root, dir)+".png")
        
        mask_path = os.path.join(root, dir, "masks")
        mask_files = os.listdir(mask_path)
        mask_files.sort()
        for filename in mask_files:
            if filename[-4:] != ".png":
                continue
            print(dir, filename)
            mask_filename = os.path.join(mask_path, filename)
            category_info = {'id': 1, 'is_crowd': 0}
            binary_mask = np.asarray(Image.open(mask_filename).convert('1')).astype(np.uint8)
            annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)
            if annotation_info is not None:
                if train == True: # training
                    coco_output_train["annotations"].append(annotation_info)
                else: # validation
                    coco_output_valid["annotations"].append(annotation_info)

            segmentation_id = segmentation_id + 1
        
        image_id = image_id + 1
    
    with open('{}/all_train_coco_format.json'.format(ann_save_root), 'w') as output_json_file:
        json.dump(coco_output_train, output_json_file, indent=4)
    #with open('{}/valid_coco_format.json'.format(ann_save_root), 'w') as output_json_file:
    #    json.dump(coco_output_valid, output_json_file, indent=4)

if __name__ == "__main__":
    main()