from ultralytics import YOLO

IMAGE_SIZE = 1000
BATCH_SIZE = 8
EPOCHS = 800


 
output_dir = '/data/output'

model_x = YOLO('yolov10x.pt')
model_x.train(data='/data/HERIDAL_COMBINED/data.yaml', epochs=EPOCHS, project=output_dir, imgsz = IMAGE_SIZE, batch = BATCH_SIZE, device = 0,plots = True)
