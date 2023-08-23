# YOLOv8 Pruning

## 0. Requirements

```bash
pip install -r requirements.txt
```
Tested environment:
```
Pytorch==1.12.1
Torchvision==0.13.1
```

## 1. Pruning

This example was implemented by [@Hyunseok-Kim0 (Hyunseok Kim)](https://github.com/Hyunseok-Kim0). Please refer to Issue [#147](https://github.com/VainF/Torch-Pruning/issues/147#issuecomment-1507475657) for more details.

#### Ultralytics
```bash
git clone https://github.com/ultralytics/ultralytics.git 
cp yolov8_pruning.py ultralytics/
cd ultralytics 
git checkout 44c7c3514d87a5e05cfb14dba5a3eeb6eb860e70 # for compatibility
```

#### Modification
Some functions will be automatically modified by the yolov8_pruning.py to prevent performance loss during model saving.

##### 1. ```train``` in class ```YOLO```
This function creates new trainer when called. Trainer loads model based on config file and reassign it to current model, which should be avoided for pruning.

##### 2. ```save_model``` in class ```BaseTrainer```
YOLO v8 saves trained model with half precision. Due to this precision loss, saved model shows different performance with validation result during fine-tuning.
This is modified to save the model with full precision because changing model to half precision can be done easily whenever after the pruning.

##### 3. ```final_eval``` in class ```BaseTrainer```
YOLO v8 replaces saved checkpoint file to half precision after training is done using ```strip_optimizer```. Half precision saving is changed with same reason above.

#### Training
```bash
# This example will craft yolov8-half and fine-tune it on the coco128 toy set.
python yolov8_pruning.py
```

#### Screenshot for coco128 post-training:
<img width="876" alt="image" src="https://user-images.githubusercontent.com/18592211/234308234-771e1895-06a3-43a7-8967-b3736ee06d87.png">


#### Outputs of yolov8_pruning.py:
```
DetectionModel(
  (model): Sequential(
    (0): Conv(
      (conv): Conv2d(3, 80, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (1): Conv(
      (conv): Conv2d(80, 160, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(160, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
...
        (2): Sequential(
          (0): Conv(
            (conv): Conv2d(640, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (2): Conv2d(320, 80, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (dfl): DFL(
        (conv): Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
    )
  )
)


DetectionModel(
  (model): Sequential(
    (0): Conv(
      (conv): Conv2d(3, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (1): Conv(
      (conv): Conv2d(40, 80, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
...
        (2): Sequential(
          (0): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (2): Conv2d(320, 80, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (dfl): DFL(
        (conv): Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
    )
  )
)
Before Pruning: MACs=129.092051 G, #Params=68.229648 M
After Pruning: MACs=41.741203 G, #Params=20.787528 M
```
