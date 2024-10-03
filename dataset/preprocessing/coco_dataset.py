import json

from PIL import Image, ImageDraw, ImageFont


class CocoDataset:
    def __init__(self, dataset_root, annotation_file, autoload=False):
        """
        Initializes the CocoDataset with the root directory of the dataset and the annotation file.
        
        dataset_root: The root directory where the dataset is stored.
        annotation_file: The path to the COCO annotations file (JSON format).
        autoload: Whether to load the dataset automatically. Defaults to False.
        """
        self.dataset_root = dataset_root
        self.annotation_file = annotation_file
        self.coco_data = None
        self.categories = None
        self.annotations = None

        if autoload:
            self.load_dataset()

    def load_dataset(self):
        """
        Loads the COCO dataset annotations from the specified JSON file.
        """
        with open(self.annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        # Create a mapping from category ID to category name
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.annotations = self.create_annotations_dict()

    def create_annotations_dict(self):
        """
        Creates a dictionary that maps image_id to the list of annotations for that image.
        This will allow for faster lookups of annotations by image_id.
        
        Return: 
            A dictionary where keys are image IDs and values are lists of annotations for that image.
        """
        annotations_dict = {}
        
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_dict:
                annotations_dict[image_id] = []
            annotations_dict[image_id].append(ann)
        
        return annotations_dict

    def get_image_annotations(self, image_id):
        """
        Retrieves the image and its annotations given an image ID.
        
        image_id: The ID of the image to retrieve.

        Return: 
            A tuple (image, annotations) where `image` is a PIL image object,
            `annotations` is a list of bounding boxes and category IDs for the given image.
        """
        # Find the image information by image ID
        image_info = next((img for img in self.coco_data['images'] if img['id'] == image_id), None)
        if image_info is None:
            raise ValueError(f"Image ID {image_id} not found in the dataset.")
        
        image_path = f'{self.dataset_root}/val2014/{image_info["file_name"]}'
        image = Image.open(image_path)
        
        # Get the annotations for the given image ID
        annotations = self.annotations[image_id]
        
        return image, annotations

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

# Example Usage:
if __name__ == "__main__":
    # Initialize the visualizer
    file_root = "/home/kanchana/data/mscoco/coco_2014"
    anno_file = f"{file_root}/annotations/instances_val2014.json"
    
    dataset = CocoDataset(file_root, anno_file)
    
    # Load the dataset
    dataset.load_dataset()
    
    # Get image and annotations for a specific image ID
    image_id = dataset.coco_data['images'][0]['id']  # Using the first image in the dataset

    image, annotation = dataset.get_image_annotations(image_id)
    vis_image = dataset.visualize_image(image, annotation)
