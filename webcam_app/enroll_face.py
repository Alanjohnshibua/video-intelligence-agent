import argparse
import sys
from pathlib import Path

import cv2

# Ensure we can import the project core
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from video_intelligence_agent.config import FaceIdentifierConfig
from video_intelligence_agent.core import FaceIdentifier

def main():
    parser = argparse.ArgumentParser(description="Enroll a new face into the webcam database")
    parser.add_argument("--name", type=str, required=True, help="Name of the person")
    parser.add_argument("--image", type=str, required=True, help="Path to the face image")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERROR] Image path does not exist: {img_path}")
        return 1

    print(f"[INFO] Initializing database to add: {args.name}...")
    config = FaceIdentifierConfig(
        database_path=str(ROOT / "data" / "webcam_embeddings.pkl"),
        unknown_dir=str(ROOT / "outputs" / "unknown_faces")
    )
    identifier = FaceIdentifier(config=config)
    
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"[ERROR] OpenCV could not read the image file: {img_path}")
        return 1
        
    try:
        record = identifier.add_person(name=args.name, image=frame, source_image=img_path)
        bbox = record.get("bbox", {})
        print(f"[SUCCESS] Successfully enrolled {args.name}!")
        print(f"         Detected face bounding box: x={bbox.get('x')}, y={bbox.get('y')}")
        print(f"         The face is now actively tracked in webcam_app/main.py!")
    except Exception as e:
        print(f"[ERROR] Failed to enroll face: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
