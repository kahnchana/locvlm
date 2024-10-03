import json

import requests
from PIL import Image, ImageDraw, ImageFont


class CocoSpatialDataset:
    def __init__(self, dataset_root, annotation_file, autoload=True):
        """
        Initializes the CocoDataset with the root directory of the dataset and the annotation file.
        
        dataset_root: The root directory where the dataset is stored.
        annotation_file: The path to the COCO annotations file (JSON format).
        autoload: Whether to load the dataset automatically. Defaults to False.
        """
        self.dataset_root = dataset_root
        self.annotation_file = annotation_file
        self.image_id_list = None
        self.coco_data = None
        self.categories = None
        self.annotations = None

        if autoload:
            self.load_dataset()

    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, index):
        image_id = self.image_id_list[index]
        image, annotations = self.get_image_annotations(image_id)
        return image, annotations

    def load_dataset(self):
        """
        Loads the COCO dataset annotations from the specified JSON file.
        """
        if self.annotation_file.startswith('https://'):
            response = requests.get(self.annotation_file)
            if response.status_code == 200:
                self.coco_data = response.json()  # This automatically parses the JSON content
            else:
                raise Exception(f"Failed to download JSON file. Status code: {response.status_code}")
        else:
            with open(self.annotation_file, 'r') as f:
                self.coco_data = json.load(f)
        # Create a mapping from category ID to category name
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.annotations = self.coco_data['data']
        self.image_id_list = list(self.annotations.keys())

    def get_image_annotations(self, image_id):
        """
        Retrieves the image and its annotations given an image ID.
        
        image_id: The ID of the image to retrieve.

        Return: 
            A tuple (image, annotations) where `image` is a PIL image object,
            `annotations` is a list of bounding boxes and category IDs for the given image.
        """
        # Find the image information by image ID
        datum = self.annotations[image_id]
        
        image_path = f'{self.dataset_root}/val2014/{datum["file_name"]}'
        image = Image.open(image_path)
        
        # Get the annotations for the given image ID
        annotations = datum['annotations']
        good_pairs = datum['good_pairs'] if 'good_pairs' in datum else None
        data = {'annotation': annotations, 'good_pairs': good_pairs}
        
        return image, data

    def visualize_image(self, image, annotations, font_path=None, font_size=25):
        """
        Visualizes the image by drawing bounding boxes and category labels on it.
        
        :param image: The PIL image object to visualize.
        :param annotations: A list of annotations with bounding boxes and category IDs.
        :param font_path: The path to the font file for rendering text. Defaults to "arial.ttf" if available.
        :param font_size: The font size to use for the text.
        """
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)

        if isinstance(annotations, dict):
            annotations = annotations['annotation']

        # Load the font for text labels
        try:
            font = ImageFont.truetype(font_path or "arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        # Loop through each annotation and draw the bounding box and label
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, width, height = bbox
            category_id = ann['category_id']
            category_name = self.categories[category_id]

            # Draw the bounding box
            draw.rectangle([x, y, x + width, y + height], outline='green', width=3)

            # Draw the category label
            text_position = (x, y)
            draw.text(text_position, category_name, fill="red", font=font)

        return vis_image

    def generate_spatial_questions(self, image, annotation):
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        object_list = [self.categories[x['category_id']] for x in annotation['annotation']]
        object_pairs = annotation['good_pairs']

        question_list = []
        answer_list = []
        for (obj_left, obj_right) in object_pairs:
            name_left = object_list[obj_left]
            name_right = object_list[obj_right]
            question = f"Which side of the {name_left} is the {name_right}?"
            # correct and wrong answers respectively
            answers = [
                f"The {name_right} is on the right side of the {name_left}.",
                f"The {name_right} is on the left side of the {name_left}.",
            ]
            question_list.append(question)
            answer_list.append(answers)
        
        return {
            'image': image,
            'image_flipped': flipped_image,
            'questions': question_list,
            'answers': answer_list
        }

    def generate_object_questions(self, annotation):
        object_list = [self.categories[x['category_id']] for x in annotation['annotation']]
        question_list = []
        answer_list = []
        for obj_name in object_list:
            if obj_name.startswith(tuple("aeiou")):
                question = f"Is there an {obj_name} in the image?"
                # correct and wrong answers respectively
                answer = [
                    f"Yes, there is a {obj_name} in the image.",
                    f"No, there is no {obj_name} in the image.",
                ]
            else:
                question = f"Is there a {obj_name} in the image?"
                # correct and wrong answers respectively
                answer = [
                    f"Yes, there is a {obj_name} in the image.",
                    f"No, there is no {obj_name} in the image.",
                ]
            
            question_list.append(question)
            answer_list.append(answer)
        
        return {
            'questions': question_list,
            'answers': answer_list
        }


if __name__ == "__main__":
    # Sample usage code.
    file_root = "/home/kanchana/data/mscoco/coco_2014"
    anno_file = "https://github.com/kahnchana/locvlm/releases/download/v1.0/coco_spatial.json"

    dataset = CocoSpatialDataset(file_root, anno_file)
    
    image, annotation = dataset[5]
    vis_image = dataset.visualize_image(image, annotation)
    object_eval_data = dataset.generate_object_questions(annotation)
    spatial_eval_data = dataset.generate_spatial_questions(image, annotation)
