import argparse
from ultralytics import YOLO
from PIL import Image
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='weights/yolov8n-face-lindevs.pt', help='Weights path')
    parser.add_argument('--source', default='data/images/bus.jpg')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--conf', type=float, default=0.75, help='Object confidence threshold for detection')
    parser.add_argument('--iou', type=float, default=0.7, help='Intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='0', help='CUDA device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--output', default='prediction.png', help='Output filename for saved image')
    opt = parser.parse_args()

    model = YOLO(opt.weights)
    results = model.predict(
        opt.source,
        imgsz=opt.imgsz,
        conf=opt.conf,
        iou=opt.iou,
        device=opt.device,
        verbose=False
    )

    img = results[0].plot()  # BGR numpy array
    img = Image.fromarray(img[..., ::-1])  # Convert to RGB for PIL

    # Derive output filename from input
    base, _ = os.path.splitext(os.path.basename(opt.source))
    output_filename = f"{base}-faces-detected.png"
    output_path = os.path.abspath(output_filename)

    img.save(output_path, format="PNG")
    print(f"Saved prediction to {output_path}")
