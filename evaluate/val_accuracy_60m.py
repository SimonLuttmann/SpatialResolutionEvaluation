from ultralytics import YOLO
import shutil
import os
import json

image_height = 60
model = YOLO('yolov10x_heridal_combined.pt')

image_size = 128

docker_directory = "/ultralytics/runs/detect"


destination_directory = f"/ultralytics/benchmarks/validate_{image_height}m"


json_file_name = f"stats_{image_height}m.json"
stats = []
while image_size < 2033:
    image_size_directory = os.path.join(destination_directory,f"{image_size}")
    os.makedirs(image_size_directory, exist_ok=True)
    
    result = model.val(data = '/ultralytics/benchmarks/60m_dataset_60_10_30/data.yaml', imgsz = image_size, plots = True, batch=1, device = "cuda:0")
    
    
    items = os.listdir(docker_directory)

    for item in items:
        source_item = os.path.join(docker_directory,item)
        destination_item = os.path.join(image_size_directory,item)
        shutil.move(source_item, destination_item)
    
    

    try:
        with open(json_file_name, 'r') as file:
            data = json.load(file)  # Load existing data
    except FileNotFoundError:
        data = []  # If the file doesn't exist, start with an empty list
    data.append({'image_size': image_size, 'AP50': result.box.map50, 'AP75': result.box.map75, 'AP50-95': result.box.map, 'inference_time': result.speed["inference"], 'fps': round(1000/result.speed["inference"], 2)})
    with open(json_file_name, mode = "w") as file:
        json.dump(data, file)

    image_size += 32
print("finished")