from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.predict(
    source = r".venv\Lib\site-packages\ultralytics\assets",
    save = True,
    show = False,
)