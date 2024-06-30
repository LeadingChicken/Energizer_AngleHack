import os
from ultralytics import YOLO
import torch

torch.cuda.empty_cache()

HOME = '/home/lucy/Documents/ai-ml/yolo/yolov10'
MODEL_PATH = os.path.join(HOME, 'weights', 'yolov10m.pt')
DATA_PATH = '/home/lucy/Documents/ai-ml/code/angelhack24heineiken/src/train/Hackathon-2/data.yaml'
EPOCHS = 200
BATCH_SIZE = 16

model = YOLO(MODEL_PATH)

train_params = {
    'task': 'detect',
    'mode': 'train',
    'epochs': EPOCHS,
    'batch': BATCH_SIZE,
    'plots': True,
    'single_cls': False,
    'imgsz': 640,
    'amp': True,  # Enable Automatic Mixed Precision
}

# Train the model
model.train(data=DATA_PATH, **train_params)

print("Training complete.")
