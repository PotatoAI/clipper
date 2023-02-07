from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Load ViT-B-32 CLIP model
device = torch.device("mps")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("input.jpg")

image_feature = processor(images=image, return_tensors="pt",
                          padding=True)['pixel_values']
print(image_feature)

text_feature = processor(text=["a photo of a cat", "a photo of a dog"],
                         return_tensors="pt",
                         padding=True)['input_ids']
print(text_feature)

# cosine similarity
sim = cosine_similarity(image_feature, text_feature)
print(sim)
