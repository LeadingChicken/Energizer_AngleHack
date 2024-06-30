import optuna
from ultralytics import YOLO
import torch

DATA_PATH = '/home/lucy/Documents/ai-ml/code/angelhack24heineiken/src/train/Hackathon-6/data.yaml'
EPOCHS = 500


def objective(trial):
    # Define hyperparameters to tune
    lr0 = trial.suggest_loguniform('lr0', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 20])

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
        'lr0': lr0,
        'amp': True,
    }

    # Train the model
    model.train(data=DATA_PATH, **train_params)

    # Evaluate the model
    results = model.val(data=DATA_PATH)
    print(results)


# Optimize hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print(f"Best trial: {study.best_trial.params}")
