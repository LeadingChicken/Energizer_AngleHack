from ultralytics import YOLOv10

model = YOLOv10(f'{HOME}/weights/yolov10n.pt')
results = model(source=f'{HOME}/data/dog.jpeg', conf=0.25)
results[0].boxes.xyxy
results[0].boxes.conf
results[0].boxes.cls