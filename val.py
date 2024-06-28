from ultralytics import YOLOv10

model = YOLOv10('runs/detect/train/weights/best.pt')

if __name__ == '__main__':
    model.val(data='data/data.yaml', batch=96)
    input('Press enter to continue.')
