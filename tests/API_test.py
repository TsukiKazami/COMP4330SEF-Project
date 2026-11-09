import requests
from pathlib import Path
from ultralytics import YOLO


ACCESS_TOKEN = "MLY|28615767494790846|fc350d6709f2d055f9a34bf0a10c5a48"

# Hong Kong example (Central area) — change to your area
# bbox format: west,south,east,north
BBOX = "114.157,22.281,114.162,22.286"

LIMIT = 50
SAVE_FOLDER = "mapillary_images"
RESULTS_FOLDER = "yolo_results"

# YOLO model
MODEL_NAME = "yolov8n.pt"


def search_images(bbox: str, access_token: str, limit: int = 50):
    """Search for image IDs inside a bounding box."""
    url = (
        "https://graph.mapillary.com/images"
        f"?access_token={access_token}"
        f"&fields=id,captured_at,thumb_1024_url"
        f"&bbox={bbox}"
        f"&limit={limit}"
    )
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Search failed: {response.status_code}")
        print(response.text)
        return []
    
    data = response.json()
    images = data.get("data", [])
    print(f"Found {len(images)} images")
    return [img["id"] for img in images]


def download_image(image_id: str, access_token: str, save_path: str):
    """Download one image by ID."""
    url = (
        f"https://graph.mapillary.com/{image_id}"
        f"?access_token={access_token}"
        f"&fields=thumb_original_url,thumb_2048_url,thumb_1024_url"
    )
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to get info for {image_id}")
        return None
    
    img_info = response.json()
    img_url = (
        img_info.get("thumb_original_url")
        or img_info.get("thumb_2048_url")
        or img_info.get("thumb_1024_url")
    )
    
    if not img_url:
        print(f"No URL for {image_id}")
        return None
    
    Path(save_path).mkdir(exist_ok=True)
    img_data = requests.get(img_url).content
    filename = Path(save_path) / f"{image_id}.jpg"
    filename.write_bytes(img_data)
    print(f"Downloaded: {filename.name}")
    return filename


def run_yolo_on_folder(image_folder: str, model_name: str, results_folder: str):
    """Run YOLO detection on all images in a folder."""
    print(f"\nLoading YOLO model: {model_name}")
    model = YOLO(model_name)
    
    image_folder = Path(image_folder)
    image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))
    
    if not image_files:
        print("No images found to run YOLO on.")
        return
    
    print(f"Running YOLO on {len(image_files)} images...\n")
    
    results = model.predict(
        source=str(image_folder),
        save=True,
        project=results_folder,
        name="predict",
        exist_ok=True,
        conf=0.25,
        device=0,
    )
    
    print(f"\nYOLO finished! Results saved in: {results_folder}/predict/")
    print("Annotated images are in that folder.")


def main():
    image_ids = search_images(BBOX, ACCESS_TOKEN, LIMIT)
    
    if not image_ids:
        print("No images found. Try a different bbox or larger area.")
        return
    
    print(f"\nDownloading {len(image_ids)} images...\n")
    for i, image_id in enumerate(image_ids, 1):
        print(f"[{i}/{len(image_ids)}] ", end="")
        download_image(image_id, ACCESS_TOKEN, SAVE_FOLDER)
    
    print(f"\nDownload done! Images saved in: {SAVE_FOLDER}")
    
    run_yolo_on_folder(SAVE_FOLDER, MODEL_NAME, RESULTS_FOLDER)


if __name__ == "__main__":
    main()