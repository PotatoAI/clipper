from PIL import Image
from sentence_transformers import SentenceTransformer, util
import torch

device = torch.device("mps")
model = SentenceTransformer('clip-ViT-B-32').to(device)

image = Image.open("input.jpg")

image_feature = model.encode(image)
print(image_feature)

text_feature = model.encode([
    "drawing of a shoe", "a photo of a cat", "a picture of shark like monster",
    "a photo of shark like monster"
])
print(text_feature)

# cosine similarity
cos_scores = util.cos_sim(image_feature, text_feature)
print(cos_scores)
