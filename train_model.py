from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=25,
    imgsz=640,
    batch=16,
    project="runs",
    name="indian_number_plate"
)