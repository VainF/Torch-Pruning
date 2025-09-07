# YOLOv5 Pruning

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


```bash
git clone https://github.com/ultralytics/yolov5
cp detect_after_pruning.py yolov5/
cd yolov5

# Test only: We only prune and test the YOLOv5 model in this script. COCO dataset is not required.
python detect_after_pruning.py --weights yolov5s.pt --source  data/images/bus.jpg
```

#### Outputs of detect_after_pruning.py:
```
DetectMultiBackend(
  (model): DetectionModel(
    (model): Sequential(
      (0): Conv(
        (conv): Conv2d(3, 32, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2)) => (conv): Conv2d(3, 16, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
        (act): SiLU(inplace=True)
      )
      (1): Conv(
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) => (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (act): SiLU(inplace=True)
      )
      (2): C3(
        (cv1): Conv(
          (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1)) => (conv): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU(inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1)) => (conv): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU(inplace=True)
        )
        (cv3): Conv(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1)) => (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU(inplace=True)
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1)) => (conv): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) => (conv): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU(inplace=True)
            )
          )
        )
      )
...

      (21): Conv(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) => (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (act): SiLU(inplace=True)
      )
      (22): Concat()
      (23): C3(
        (cv1): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)) => (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU(inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)) => (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU(inplace=True)
        )
        (cv3): Conv(
          (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1)) => (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): SiLU(inplace=True)
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)) => (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) => (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (act): SiLU(inplace=True)
            )
          )
        )
      )
      (24): Detect(
        (m): ModuleList(
          (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1)) => (0): Conv2d(64, 255, kernel_size=(1, 1), stride=(1, 1))
          (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1)) => (1): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
          (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1)) => (2): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
  )
)
Before Pruning: MACs=1.009904 G, #Params=0.007226 G
After Pruning: MACs=0.275478 G, #Params=0.001867 G
```

