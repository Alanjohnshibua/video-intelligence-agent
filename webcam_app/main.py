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
    parser = argparse.ArgumentParser(description="Live Webcam Face Recognition")
    parser.add_argument("--device", type=int, default=0, help="Webcam device ID (default: 0)")
    args = parser.parse_args()

    print("[INFO] Initializing face recognition... this may take a few seconds to load models...")
    config = FaceIdentifierConfig(
        database_path=str(ROOT / "data" / "webcam_embeddings.pkl"),
        unknown_dir=str(ROOT / "outputs" / "unknown_faces")
    )
    identifier = FaceIdentifier(config=config)
    
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam with device ID: {args.device}")
        return 1

    print("[INFO] Webcam active! Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from webcam.")
            break

        # Process frame for faces
        try:
            results = identifier.process_frame(frame)
        except Exception as e:
            print(f"[ERROR] Inference error: {e}")
            results = []

        # Draw bounds
        for res in results:
            if not res.bbox:
                continue
                
            x, y, w, h = res.bbox.x, res.bbox.y, res.bbox.w, res.bbox.h
            
            # Change color if unknown
            color = (0, 0, 255) if res.name == "Unknown" else (0, 255, 0)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{res.name} (Conf: {res.confidence:.2f})" if res.name != "Unknown" else "Unknown"
            
            cv2.putText(
                frame, label, (x, max(10, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        cv2.imshow("Live Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    sys.exit(main())
