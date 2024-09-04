import torch
import cv2
from common import DetectMultiBackend, read_img
# from augmentations import letterbox
import numpy as np
from general import non_max_suppression, scale_boxes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class predict(object):
    def __init__(self):
        self.weights = 'best.pt'
        self.model = DetectMultiBackend(self.weights, device=device, dnn=True, data='plane_trained.yaml', fp16=True)
        self.img_size = [512, 512]
        self.stride = 32
        self.auto = True
        self.namesdic = read_img()
        print(self.model)

    def detect_image(self, image_path):
        im = self.imread(image_path)
        im = torch.from_numpy(im).to(device)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]
        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.45,
                                   classes=None, max_det=1000)
        results = []
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    xmin, ymin, xmax, ymax = xyxy
                    results.append([int(ymin.item()), int(xmin.item()), int(ymax.item()), int(xmax.item()), float(conf), c])
        return results
    
    # def detect_image(self, image_path:str):
    #     results = self.model([image_path], imgsz=640)  # return a list of Results objects
    #     for result in results:
    #         boxes = result.boxes  # Boxes object for bbox outputs
    #     xyxy = boxes.xyxy
    #     if len(boxes.conf) == 0:
    #         return []
    #     cls_pred = [float(i) for i in boxes.cls]
    #     confs = [float(i) for i in boxes.conf]
    #     # cls_pred = boxes.cls
    #     # confs = boxes.conf
    #     #
    #     maxconf_id = confs.index(max(confs))
    #     main_cls = cls_pred[maxconf_id]
    #     print(f'This img has the main class: {main_cls}')
        
    #     output = []
    #     for loc, pred, conf in zip(xyxy, cls_pred, confs):
    #         #
    #         if pred == main_cls and conf >= self.score_thres[int(pred)]:
    #             conf = round(float(conf), 1)
    #             cur = [loc[1], loc[0], loc[3], loc[2], conf, pred]
    #         else:
    #             print(f"the {pred} class object is removed from {main_cls},",
    #                   f"whose confidence {conf} is smaller than {self.score_thres[int(pred)]}")
    #             continue
                
    #         # cur = [loc[0], loc[1], loc[2], loc[3], conf, pred]
    #         # cur = [float(loc[0]), float(loc[1]), float(loc[2]), float(loc[3]), conf, int(pred)]
            
    #         output.append(cur)

    #     return output
    def pre_transform(self, im):
        same_shapes = all(x.shape == im[0].shape for x in im)
        im = letterbox(im, self.img_size, stride=self.stride, auto=self.auto)[0] 
        return [x for x in im]
    def imread(self, path):
        im = self.namesdic[path]
        im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        return im



