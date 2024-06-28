from ultralytics import YOLOv10

model = YOLOv10("runs/detect/train/weights/best.pt")

model.export(format="onnx", opset=13)
