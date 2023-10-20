import cv2
import torch

from mmpose.utils import register_all_modules
register_all_modules()

from mmengine import Config

config_path = 'configs/body_2d_keypoint/edpose/coco/edpose_res50_8xb2-50e_coco-800x1333.py'
config = Config.fromfile(config_path)
config.load_from = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/edpose/coco/edpose_res50_coco_3rdparty.pth'

from mmpose.apis import init_model

model = init_model(config, config.load_from, device='cpu')

img = cv2.imread('tests/data/coco/000000000785.jpg')
print(img.shape)
img = cv2.resize(img, (1333, 800))
print(img.shape)
img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

torch.onnx.export(
    model,
    img, 
    'edpose_r50.onnx',
    input_names=["input"], 
    output_names=["output"],  
    opset_version=16,
    verbose=False,
    # dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}}, 
)
