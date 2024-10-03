import json

import tqdm
from coco_dataset import CocoDataset

# Example Usage:
if __name__ == "__main__":

    file_root = "/home/kanchana/data/mscoco/coco_2014"
    anno_file = f"{file_root}/annotations/instances_val2014.json"
    
    dataset = CocoDataset(file_root, anno_file)
    
    # Load the dataset
    dataset.load_dataset()
    
    per_image_unique_categories = {}

    for image_id in tqdm.tqdm(dataset.annotations.keys()):
        image, annotation = dataset.get_image_annotations(image_id)
        
        # Filter 
        categories = [anno['category_id'] for anno in annotation]
        counter = {}
        for cat in categories:
            counter[cat] = counter.get(cat, 0) + 1
        per_image_unique_categories[image_id] = [x for x,y in counter.items() if y == 1]

    filtered_image_category = {x:y for x,y in per_image_unique_categories.items() if len(y) >= 2}

    new_annotation_dict = {}
    new_image_info_dict = {}
    for image_id, category_list in tqdm.tqdm(filtered_image_category.items()):
        image, annotation = dataset.get_image_annotations(image_id)
        image_info = [img for img in dataset.coco_data['images'] if img['id'] == image_id][0]
        cur_annotation = []
        for anno in annotation:
            if anno['category_id'] in category_list:
                if 'segmentation' in anno:
                    anno.pop('segmentation')
                cur_annotation.append(anno)
        new_annotation_dict[image_id] = cur_annotation
        new_image_info_dict[image_id] = image_info

    combined_dict = {}
    for key in new_annotation_dict.keys():
        combined_dict[key] = {
            'annotations': new_annotation_dict[key],
            'file_name': new_image_info_dict[key]['file_name'],
            'height': new_image_info_dict[key]['height'],
            'width': new_image_info_dict[key]['width']
        }
    save_dict = {
        'categories': dataset.coco_data['categories'],
        'data': combined_dict
    }
    save_path = "/home/kanchana/repo/locvlm/data/coco_spatial.json"
    json.dump(save_dict, open(save_path, "w"), indent=2)