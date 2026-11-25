import cv2
import os

from detector import PersonDetector
from tracker import IdentityTracker


INPUT_VIDEO = "input/video.mp4"
OUTPUT_VIDEO = "output/tracked_video.mp4"


def draw_results(frame, objects):

    for obj in objects:

        x1, y1, x2, y2 = obj["bbox"]

        label = (
            f'{obj["identity_label"]} '
            f'{obj["confidence"]:.2f} '
            f'{obj["movement"]}'
        )

        cv2.rectangle(
            frame, (x1, y1), (x2, y2),
            (0, 255, 0), 2
        )

        cv2.putText(
            frame, label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 0), 2
        )

    return frame


def main():

    os.makedirs("output", exist_ok=True)

    if not os.path.exists(INPUT_VIDEO):
        print("Put a video at input/video.mp4")
        return

    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        print("Cannot open input video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    detector = PersonDetector()
    tracker = IdentityTracker()

    frame_number = 0

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            detections = detector.detect(frame, track=True)

            objects = tracker.update(
                detections, frame_number
            )

            frame = draw_results(frame, objects)

            writer.write(frame)

            cv2.imshow("ARGUS-X", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if frame_number % 100 == 0:
                print("Processed:", frame_number)

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print("Processing completed.")
    print("Saved:", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
