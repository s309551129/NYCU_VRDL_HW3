import json


ann_save_root = "./dataset/dataset/"


CATEGORIES = [
    {
        'id': 1,
        'name': 'nucleus',
    },
]


def get_coco_test():
    json_file = open("./dataset/dataset/test_img_ids.json", 'r')
    test_img_ids = json.load(json_file)
    coco_output_test = {
        "images": test_img_ids,
        "annotations": [],
        "categories": CATEGORIES,
    }
    
    return coco_output_test
    

def main():
    coco_output_test = get_coco_test()
    with open('{}/test_coco_format.json'.format(ann_save_root), 'w') as output_json_file:
        json.dump(coco_output_test, output_json_file, indent=4)


if __name__ == "__main__":
    main()