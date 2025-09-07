# YOLOv7 Pruning

## 0. Requirements

```bash
pip install -r requirements.txt
```
Tested environment:
```
Pytorch==2.5.1
torch-pruning==1.6.1
```

## 1. Pruning
The following scripts (adapted from [yolov7/detect.py](https://github.com/WongKinYiu/yolov7/blob/main/detect.py) and [yolov7/train.py](https://github.com/WongKinYiu/yolov7/blob/main/train.py)) provide the basic examples of pruning YOLOv7. It is important to note that the training part has not been validated yet due to the time-consuming training process.

Note: [yolov7_detect_pruned.py](https://github.com/VainF/Torch-Pruning/blob/master/benchmarks/prunability/yolov7_detect_pruned.py) does not include any code for fine-tuning. 

```bash
git clone https://github.com/WongKinYiu/yolov7.git
cp yolov7_detect_pruned.py yolov7/
cp yolov7_train_pruned.py yolov7/
cd yolov7 

# Test only: We only prune and test the YOLOv7 model in this script. COCO dataset is not required.
python yolov7_detect_pruned.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg

# Training with pruned yolov7 (The training part is not validated)
# Please download the pretrained yolov7_training.pt from https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt.
python yolov7_train_pruned.py --workers 8 --device 0 --batch-size 1 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'yolov7.pt' --name yolov7 --hyp data/hyp.scratch.p5.yaml
```

#### Screenshot for yolov7_train_pruned.py:
![image](https://user-images.githubusercontent.com/18592211/232129303-18a61be1-b505-4950-b6a1-c60b4974291b.png)


#### Outputs of yolov7_detect_pruned.py:
```
Model(
  (model): Sequential(
    (0): Conv(
      (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) => (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU(inplace=True)
    )
    (1): Conv(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) => (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (act): SiLU(inplace=True)
    )
    (2): Conv(
      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) => (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU(inplace=True)
    )
    (3): Conv(
      (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) => (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (act): SiLU(inplace=True)
    )
    (4): Conv(
      (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1)) => (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU(inplace=True)
    )
    (5): Conv(
      (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1)) => (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU(inplace=True)
    )
    (6): Conv(
      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) => (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU(inplace=True)
    )
    (7): Conv(
      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) => (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU(inplace=True)
    )
    (8): Conv(
      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) => (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU(inplace=True)
    )
    (9): Conv(
      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) => (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    ...
  
    (86): Conv(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) => (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU(inplace=True)
    )
    (87): Concat()
    (88): Conv(
      (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1)) => (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU(inplace=True)
    )
    (89): MP(
      (m): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (90): Conv(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)) => (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU(inplace=True)
    )
    (91): Conv(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)) => (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU(inplace=True)
    )
    (92): Conv(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) => (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (act): SiLU(inplace=True)
    )
    (93): Concat()
    (94): Conv(
      (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1)) => (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU(inplace=True)
    )
    (95): Conv(
      (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1)) => (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU(inplace=True)
    )
    (96): Conv(
      (conv): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) => (conv): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU(inplace=True)
    )
    (97): Conv(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) => (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU(inplace=True)
    )
    (98): Conv(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) => (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU(inplace=True)
    )
    (99): Conv(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) => (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (act): SiLU(inplace=True)
    )
    (100): Concat()
    (101): Conv(
      (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1)) => (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU(inplace=True)
    )
    (102): RepConv(
      (act): SiLU(inplace=True)
      (rbr_reparam): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) => (rbr_reparam): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (103): RepConv(
      (act): SiLU(inplace=True)
      (rbr_reparam): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) => (rbr_reparam): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (104): RepConv(
      (act): SiLU(inplace=True)
      (rbr_reparam): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) => (rbr_reparam): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (105): Detect(
      (m): ModuleList(
        (0): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1)) => (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
        (1): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1)) => (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
        (2): Conv2d(1024, 255, kernel_size=(1, 1), stride=(1, 1)) => (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
)

Before Pruning: MACs=6.413721 G, #Params=0.036905 G
After Pruning: MACs=1.639895 G, #Params=0.009347 G
```

