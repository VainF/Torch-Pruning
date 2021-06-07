# Pruning example for Yolo-v5 (In progress)

### 1. Clone the yolo-v5 repository and copy the training script
```bash
cd examples/yolo_example
git clone https://github.com/ultralytics/yolov5.git

cp train_with_pruning ./yolov5/
cd yolov5
```

### 2. Prepare datasets  
Please follow the instructions from the [official repo](https://github.com/ultralytics/yolov5)


### 3. Run the training script
```bash
python train_with_pruning.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
```
