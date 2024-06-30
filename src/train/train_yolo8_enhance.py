import optuna
from ultralytics import YOLO
import torch

DATA_PATH = '/home/lucy/Documents/ai-ml/code/angelhack24heineiken/src/train/Hackathon-6/data.yaml'
EPOCHS = 300


def objective(trial):
    # Define hyperparameters to tune
    lr0 = trial.suggest_float('lr0', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    momentum = trial.suggest_float('momentum', 0.7, 0.99)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

    # Clear the GPU cache
    torch.cuda.empty_cache()

    # Load the YOLOv8 model
    model = YOLO("yolov8m.yaml").load("yolov8m.pt")

    # Set training parameters
    train_params = {
        'task': 'detect',
        'mode': 'train',
        'epochs': EPOCHS,
        'batch': batch_size,
        'plots': True,
        'single_cls': False,
        'imgsz': 640,
        'amp': True,
        'optimizer': 'Adam',  # Specify the optimizer type
        'lr0': lr0,
        'momentum': momentum,
        'weight_decay': weight_decay,
    }

    # Train the model
    model.train(data=DATA_PATH, **train_params)

    # Evaluate the model
    results = model.val(data=DATA_PATH)
    map50 = results['       metrics/mAP50(B)']

    return map50


# Optimize hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print(f"Best trial: {study.best_trial.params}")
