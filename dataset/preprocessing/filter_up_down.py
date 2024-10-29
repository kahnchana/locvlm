import json
from collections import defaultdict

import tqdm
from dataset.coco_spatial_dataset import CocoSpatialDataset

file_root = "/home/kanchana/data/mscoco/coco_2014"
anno_file = "/home/kanchana/repo/locvlm/data/coco_spatial_unique_obj.json"

dataset = CocoSpatialDataset(file_root, anno_file)

good_pairs = defaultdict(list)
idx = 0
for image_idx in tqdm.tqdm(dataset.image_id_list):
    image, annotation = dataset.get_image_annotations(image_idx)
    sorted_annotation = sorted(annotation['annotation'], key=lambda x: x['bbox'][1])
    for up_idx in range(len(sorted_annotation) - 1):
        up_object = sorted_annotation[up_idx]
        down_end_up_obj = up_object['bbox'][1] + up_object['bbox'][3]

        for down_idx in range(up_idx + 1, len(sorted_annotation)):
            down_object = sorted_annotation[down_idx]
            up_end_down_obj = down_object['bbox'][1]
            if down_end_up_obj < up_end_down_obj:
                good_pairs[image_idx].append((up_idx, down_idx))
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
    annotation['annotations'] = sorted(annotation['annotations'], key=lambda x: x['bbox'][1])
    save_dict['data'][image_idx] = annotation
    save_dict['data'][image_idx]['good_pairs'] = good_pairs[image_idx]

save_path = "/home/kanchana/repo/locvlm/data/coco_up_down.json"
json.dump(save_dict, open(save_path, "w"), indent=2)