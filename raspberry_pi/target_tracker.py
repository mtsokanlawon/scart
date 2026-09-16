"""
target_tracker.py

Target acquisition and short-term visual tracking.

YOLO is used periodically to acquire/re-acquire a person.
An OpenCV tracker is used between YOLO detections.

No horizontal steering logic is implemented here.
"""

import cv2
import time


class TargetTracker:

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    YOLO_INTERVAL = 5

    MIN_CONFIDENCE = 0.45

    MAX_MISSED_FRAMES = 12

    MIN_BOX_WIDTH = 30
    MIN_BOX_HEIGHT = 50

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------

    def __init__(self, detector, logger):

        self.detector = detector
        self.logger = logger

        self.tracker = None

        self.target_bbox = None

        self.locked = False
        self.tracking = False

        self.frame_count = 0
        self.missed_frames = 0

        self.last_detection_time = 0.0

        self.logger.info("Target tracker initialized.")

    # ---------------------------------------------------------
    # Create OpenCV tracker
    # ---------------------------------------------------------

    def create_tracker(self):

        # Prefer KCF because it is generally lighter than CSRT.
        if hasattr(cv2, "legacy"):

            if hasattr(cv2.legacy, "TrackerKCF_create"):
                return cv2.legacy.TrackerKCF_create()

            if hasattr(cv2.legacy, "TrackerCSRT_create"):
                return cv2.legacy.TrackerCSRT_create()

        if hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()

        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()

        raise RuntimeError(
            "No suitable OpenCV tracker is available."
        )

    # ---------------------------------------------------------
    # Validate detection
    # ---------------------------------------------------------

    def valid_detection(self, detection):

        if detection is None:
            return False

        confidence = detection.get("confidence", 0.0)

        if confidence < self.MIN_CONFIDENCE:
            return False

        x, y, w, h = detection["bbox"]

        if w < self.MIN_BOX_WIDTH:
            return False

        if h < self.MIN_BOX_HEIGHT:
            return False

        return True

    # ---------------------------------------------------------
    # Select target
    # ---------------------------------------------------------

    def select_target(self, detections):

        valid = [
            d for d in detections
            if self.valid_detection(d)
        ]

        if not valid:
            return None

        # Select the strongest detection.
        valid.sort(
            key=lambda d: d["confidence"],
            reverse=True
        )

        return valid[0]

    # ---------------------------------------------------------
    # Start tracker
    # ---------------------------------------------------------

    def start_tracking(self, frame, detection):

        x, y, w, h = detection["bbox"]

        try:

            tracker = self.create_tracker()

            success = tracker.init(
                frame,
                (float(x), float(y), float(w), float(h))
            )

            if success is False:

                self.logger.warning(
                    "Tracker initialization failed."
                )

                return False

            self.tracker = tracker
            self.target_bbox = (x, y, w, h)

            self.locked = True
            self.tracking = True

            self.missed_frames = 0

            self.last_detection_time = time.time()

            self.logger.info(
                "TARGET LOCKED "
                f"confidence={detection['confidence']:.2f} "
                f"bbox={self.target_bbox}"
            )

            return True

        except Exception as exc:

            self.logger.error(
                f"Could not create tracker: {exc}"
            )

            self.tracker = None
            self.tracking = False

            return False

    # ---------------------------------------------------------
    # Run tracker
    # ---------------------------------------------------------

    def update_tracker(self, frame):

        if self.tracker is None:

            self.tracking = False

            return None

        try:

            success, bbox = self.tracker.update(frame)

        except Exception as exc:

            self.logger.warning(
                f"Tracker update error: {exc}"
            )

            success = False
            bbox = None

        if not success:

            self.missed_frames += 1

            self.logger.info(
                f"Tracker miss "
                f"({self.missed_frames}/{self.MAX_MISSED_FRAMES})"
            )

            if self.missed_frames >= self.MAX_MISSED_FRAMES:

                self.logger.warning(
                    "TARGET LOST"
                )

                self.reset()

            return None

        x, y, w, h = bbox

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        # Reject obviously invalid tracker boxes.
        if w < self.MIN_BOX_WIDTH or h < self.MIN_BOX_HEIGHT:

            self.missed_frames += 1

            return None

        # Clamp coordinates to frame.
        frame_h, frame_w = frame.shape[:2]

        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))

        w = min(w, frame_w - x)
        h = min(h, frame_h - y)

        self.target_bbox = (
            x,
            y,
            w,
            h
        )

        self.missed_frames = 0
        self.tracking = True

        return self.target_bbox

    # ---------------------------------------------------------
    # Main update
    # ---------------------------------------------------------

    def update(self, frame):

        self.frame_count += 1

        # -----------------------------------------------------
        # If currently tracking, update lightweight tracker.
        # -----------------------------------------------------

        tracked_bbox = None

        if self.tracking:

            tracked_bbox = self.update_tracker(frame)

        # -----------------------------------------------------
        # Periodically run YOLO.
        # -----------------------------------------------------

        should_detect = (
            not self.locked
            or
            self.frame_count % self.YOLO_INTERVAL == 0
            or
            tracked_bbox is None
        )

        if should_detect:

            detections = self.detector.detect(frame)

            target = self.select_target(detections)

            if target is not None:

                # If no target currently exists,
                # acquire one.

                if not self.locked:

                    self.start_tracking(
                        frame,
                        target
                    )

                    tracked_bbox = self.target_bbox

                else:

                    # YOLO has reacquired the person.
                    #
                    # Reinitialize the tracker so that
                    # tracker drift does not accumulate.

                    self.start_tracking(
                        frame,
                        target
                    )

                    tracked_bbox = self.target_bbox

            else:

                if not self.locked:

                    self.logger.info(
                        "SEARCHING: no person detected."
                    )

        # -----------------------------------------------------
        # Return tracking state.
        # -----------------------------------------------------

        if tracked_bbox is not None:

            return {
                "state": "TRACKING",
                "bbox": tracked_bbox
            }

        if self.locked:

            return {
                "state": "SEARCHING",
                "bbox": None
            }

        return {
            "state": "SEARCHING",
            "bbox": None
        }

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------

    def reset(self):

        self.tracker = None

        self.target_bbox = None

        self.locked = False
        self.tracking = False

        self.missed_frames = 0

        self.logger.info(
            "Tracker reset. Returning to SEARCHING."
        )
