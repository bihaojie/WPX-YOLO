import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(f'train_results/weights/WPX-YOLO.pt')
    model.val(data='dataset/data.yaml',
              split='test',
              imgsz=640,
              batch=16,
              rect=True,
              save_json=True, # if you need to cal coco metric
              project='runs/test',
              name=f'WPX-YOLO'
              )