import numpy as np
from PIL import Image
from transformers import CLIPProcessor, TFCLIPModel
import tensorflow as tf


class ImgClassifier:
    _instance = None

    def __init__(self):
        self.model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ImgClassifier, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def process_img_classifications(self, image: Image, labels: list[str]):
        inputs = self.processor(
            text=labels, images=image, return_tensors="tf", padding=True
        )
        return inputs

    def predict_label(self, inputs):
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = tf.nn.softmax(logits_per_image, axis=-1)
        np_probs = probs.numpy()
        max_idx = np.argmax(np_probs[0])
        return max_idx


model = ImgClassifier()
