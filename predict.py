# predict.py
import cv2, matplotlib.pyplot as plt, logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

GREEN = "Green"

def predict_one(model_path, img_path, out_dir):
    model = YOLO(model_path)
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    res = model.predict(img_path, conf=0.5, save=False)[0]

    cnt = 0
    for box in res.boxes:
        cls = int(box.cls.item())
        if res.names[cls] == GREEN: cnt += 1
        x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, f"{res.names[cls]} {box.conf.item():.2f}",
                    (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    plt.figure(figsize=(12,8))
    plt.imshow(img); plt.axis('off')
    plt.title(f"{cnt} {GREEN} berries")
    out = Path(out_dir) / f"pred_{img_path.name}"
    plt.savefig(out, bbox_inches='tight'); plt.close()
    log.info(f"Saved → {out}")

if __name__ == "__main__":
    BEST   = "/home/root/dataset/runs/yolov8m_dataset2k/weights/best.pt"
    TEST   = Path("/home/root/dataset/Fotos_Test")
    OUT    = Path("/home/root/dataset/runs/predictions")
    OUT.mkdir(parents=True, exist_ok=True)

    for p in TEST.glob("*.[jp][pn]g"):
        predict_one(BEST, p, OUT)