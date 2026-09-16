"""
tracker.py

Smart Cart human-following vision test.

Stage:
    Camera
        ↓
    YOLO person detection
        ↓
    Target acquisition
        ↓
    OpenCV lightweight tracking
        ↓
    Target state

Motor control is intentionally NOT connected yet.
"""

import cv2
import time

from camera import Camera
from detector import YOLODetector
from target_tracker import TargetTracker
from utils import create_logger


def draw_tracking_overlay(frame, result):

    state = result["state"]
    bbox = result["bbox"]

    if bbox is not None:

        x, y, w, h = bbox

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        center_x = x + w // 2
        center_y = y + h // 2

        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            "TARGET",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"State: {state}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


def main():

    logger = create_logger()

    camera = Camera(logger)

    detector = YOLODetector(logger)

    tracker = TargetTracker(
        detector,
        logger
    )

    camera.start()

    logger.info(
        "Smart Cart visual tracker started."
    )

    logger.info(
        "No Arduino/motor control is active."
    )

    last_time = time.time()

    frame_counter = 0

    try:

        while True:

            frame = camera.read()

            if frame is None:
                logger.warning(
                    "Camera returned no frame."
                )
                continue

            frame_counter += 1

            result = tracker.update(frame)

            draw_tracking_overlay(
                frame,
                result
            )

            # -------------------------------------------------
            # FPS
            # -------------------------------------------------

            now = time.time()

            elapsed = now - last_time

            if elapsed >= 1.0:

                fps = frame_counter / elapsed

                logger.info(
                    f"State={result['state']} "
                    f"FPS={fps:.1f}"
                )

                frame_counter = 0
                last_time = now

            # -------------------------------------------------
            # Save latest frame
            # -------------------------------------------------

            cv2.imwrite(
                "latest_tracking.jpg",
                frame
            )

            # -------------------------------------------------
            # Terminal control
            # -------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                logger.info(
                    "Exit requested."
                )

                break

    except KeyboardInterrupt:

        logger.info(
            "Keyboard interrupt."
        )

    finally:

        camera.stop()

        logger.info(
            "Visual tracker stopped."
        )


if __name__ == "__main__":
    main()
