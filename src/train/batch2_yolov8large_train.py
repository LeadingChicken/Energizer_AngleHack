import torch
from ultralytics import YOLO

torch.cuda.empty_cache()

DATA_PATH = '/root/angelhack24heineiken/src/train/Hackathon-2/data.yaml'
EPOCHS = 200
BATCH_SIZE = 64

model = YOLO("yolov8l.yaml").load("yolov8l.pt")

train_params = {

    'task': 'detect',
    'mode': 'train',
    'epochs': EPOCHS,
    'batch': BATCH_SIZE,
    'plots': True,
    'single_cls': False,
    'imgsz': 640,
    'amp': True,
}

model.train(data=DATA_PATH, **train_params)

print("Training complete.")
