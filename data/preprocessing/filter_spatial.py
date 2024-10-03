import json
from collections import defaultdict

import tqdm
from coco_spatial_dataset import CocoSpatialDataset

file_root = "/home/kanchana/data/mscoco/coco_2014"
anno_file = "/home/kanchana/repo/locvlm/data/coco_spatial_unique_obj.json"

dataset = CocoSpatialDataset(file_root, anno_file)

good_pairs = defaultdict(list)
idx = 0
for image_idx in tqdm.tqdm(dataset.image_id_list):
    image, annotation = dataset.get_image_annotations(image_idx)
    sorted_annotation = sorted(annotation['annotation'], key=lambda x: x['bbox'][0])
    for right_idx in range(len(sorted_annotation) - 1):
        right_object = sorted_annotation[right_idx]
        right_end_left_obj = right_object['bbox'][0] + right_object['bbox'][2]

        for left_idx in range(right_idx + 1, len(sorted_annotation)):
            left_object = sorted_annotation[left_idx]
            left_end_right_obj = left_object['bbox'][0]
            if right_end_left_obj < left_end_right_obj:
                good_pairs[image_idx].append((right_idx, left_idx))
                idx += 1

image_idx_list = list(good_pairs.keys())
pair_count = len([y for x in good_pairs.values() for y in x])

data_dict = dataset.coco_data
save_dict = {
    'categories': data_dict['categories'],
    'data': {}
}

for image_idx in tqdm.tqdm(good_pairs.keys()):
    annotation = data_dict['data'][image_idx]
    annotation['annotations'] = sorted(annotation['annotations'], key=lambda x: x['bbox'][0])
    save_dict['data'][image_idx] = annotation
    save_dict['data'][image_idx]['good_pairs'] = good_pairs[image_idx]

save_path = "/home/kanchana/repo/locvlm/data/coco_spatial.json"
json.dump(data_dict, open(save_path, "w"), indent=2)