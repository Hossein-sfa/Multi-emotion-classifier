from transformers import AutoModel, DistilBertTokenizer
from fastapi import FastAPI, File, UploadFile
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import uvicorn
import torch
import io

app = FastAPI()
registered_embeddings = []
emotions = ['praise', 'amusement', 'anger', 'disapproval', 'confusion', 'interest', 'sadness', 'fear', 'joy', 'love']

class MyModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(MyModel, self).__init__()
        self.base_model = models.efficientnet_b0(weights=None)
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.base_model.features[-5:].parameters():
            param.requires_grad = True 
        self.base_model.classifier = nn.Identity()
        self.batch_norm = nn.BatchNorm1d(1280)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_embedding = nn.Linear(1280, embedding_dim)
        self.dropout = nn.Dropout(p=0.4)

    def forward_one(self, x):
        x = self.base_model.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.batch_norm(x)
        x = self.fc_embedding(x)
        x = self.dropout(x)
        return F.normalize(x, p=2, dim=1)  # Normalize embeddings

    def forward(self, input1, input2):
        embed1 = self.forward_one(input1)
        embed2 = self.forward_one(input2)
        return embed1, embed2

def tokenize_text(texts, tokenizer, max_length=512):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_token_type_ids=False 
    )

class EmotionClassifier(nn.Module):
    def __init__(self, num_labels):
        super(EmotionClassifier, self).__init__()
        self.albert = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_6L_768D")
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(self.albert.config.hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(32, num_labels)
    def forward(self, input_ids, attention_mask):
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout1(outputs.last_hidden_state[:, 0, :])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

model = MyModel()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu'), weights_only=False))
model.eval()

nlp_model = EmotionClassifier(10)
nlp_model.load_state_dict(torch.load("nlp_model.pth", map_location=torch.device('cpu'), weights_only=True))
nlp_model.eval()

transform = transforms.Compose([
    transforms.Resize(456),                                                      # Resize shortest edge to 456px
    transforms.CenterCrop(456),                                                  # Center crop to 456x456
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

def get_embedding(image: Image.Image) -> torch.Tensor:
    image = transform(image).unsqueeze(0)     # Add batch dimension
    with torch.no_grad():
        embedding = model.forward_one(image)  # Use forward_one to extract embedding
    return embedding.squeeze(0)               # Remove batch dimension

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def predict_emotions(text, threshold=0.1):
    inputs = tokenizer(text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True,
        max_length=64,
        return_token_type_ids=False
    )
    with torch.no_grad():
        outputs = nlp_model(**inputs)
    probs = F.softmax(outputs, dim=1).squeeze().tolist()
    return {emotions[i]: float(probs[i]) for i in range(len(probs)) if probs[i] > 0.05}

@app.post("/register/")
async def register(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    embedding = get_embedding(image)
    registered_embeddings.append(embedding)
    return {"message": f"Registered successfully!"}

@app.post("/verify/")
async def verify(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_embedding = get_embedding(image)
    min_distance = 10000000
    threshold = 1.0
    for registered_embedding in registered_embeddings:
        distance = torch.norm(input_embedding - registered_embedding)
        if distance < min_distance:
            min_distance = distance
    is_match = min_distance < threshold
    return {
    "euclidean_distance": float(min_distance),
    "is_match": bool(is_match),
    }

@app.post("/emotion/")
async def get_emotion(request: dict):
    return predict_emotions(request["text"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
