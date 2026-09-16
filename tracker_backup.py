import cv2
import time
import logging
import numpy as np
from picamera2 import Picamera2
import onnxruntime as ort


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/yolov8n.onnx"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

YOLO_SIZE = 640

PERSON_CLASS_ID = 0

# YOLO confidence threshold
YOLO_CONFIDENCE = 0.40

# Run YOLO periodically rather than every frame.
YOLO_INTERVAL = 1.5

# Maximum number of consecutive tracker failures
MAX_TRACKER_MISSES = 12

# Minimum bounding-box dimensions
MIN_BOX_WIDTH = 30
MIN_BOX_HEIGHT = 50

# KCF tracker
TRACKER_TYPE = "KCF"

# Logging
LOG_INTERVAL = 1.0


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)

log = logging.getLogger("SmartCart")


# ============================================================
# KCF CREATION
# ============================================================

def create_kcf_tracker():

    # Modern OpenCV
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()

    # OpenCV legacy namespace
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()

    raise RuntimeError("KCF tracker is not available in this OpenCV installation.")


# ============================================================
# YOLO DETECTOR
# ============================================================

class YOLODetector:

    def __init__(self, model_path):

        log.info("Loading YOLO model...")

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        input_shape = self.session.get_inputs()[0].shape

        log.info(
            f"YOLO loaded: input={input_shape}, "
            f"output={self.session.get_outputs()[0].shape}"
        )

    def preprocess(self, frame):

        image = cv2.resize(
            frame,
            (YOLO_SIZE, YOLO_SIZE),
            interpolation=cv2.INTER_LINEAR
        )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32) / 255.0

        image = np.transpose(image, (2, 0, 1))

        image = np.expand_dims(image, axis=0)

        return image

    def detect(self, frame):

        original_h, original_w = frame.shape[:2]

        input_tensor = self.preprocess(frame)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )

        output = outputs[0]

        # ----------------------------------------------------
        # YOLO export with NMS
        #
        # Expected:
        # [1, 300, 6]
        #
        # x1 y1 x2 y2 confidence class
        # ----------------------------------------------------

        if output.ndim == 3:
            detections = output[0]
        else:
            detections = output

        results = []

        scale_x = original_w / YOLO_SIZE
        scale_y = original_h / YOLO_SIZE

        for detection in detections:

            x1, y1, x2, y2, confidence, class_id = detection

            confidence = float(confidence)
            class_id = int(class_id)

            if confidence < YOLO_CONFIDENCE:
                continue

            if class_id != PERSON_CLASS_ID:
                continue

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            x1 = max(0, min(original_w - 1, x1))
            y1 = max(0, min(original_h - 1, y1))
            x2 = max(0, min(original_w - 1, x2))
            y2 = max(0, min(original_h - 1, y2))

            width = x2 - x1
            height = y2 - y1

            if width < MIN_BOX_WIDTH:
                continue

            if height < MIN_BOX_HEIGHT:
                continue

            results.append(
                {
                    "bbox": (x1, y1, width, height),
                    "confidence": confidence
                }
            )

        return results


# ============================================================
# TARGET TRACKER
# ============================================================

class TargetTracker:

    def __init__(self):

        self.tracker = None

        self.locked = False

        self.bbox = None

        self.confidence = 0.0

        self.missed_frames = 0

        self.last_yolo_time = 0.0

        self.last_log_time = 0.0

        self.state = "SEARCHING"

        self.last_detection_time = 0.0

        self.frame_count = 0

        self.start_time = time.time()

        log.info("Target tracker initialized.")

    # --------------------------------------------------------
    # Select target
    # --------------------------------------------------------

    def select_target(self, detections, frame):

        if not detections:
            return False

        # ----------------------------------------------------
        # If no target exists, select the most confident person.
        # ----------------------------------------------------

        if not self.locked:

            selected = max(
                detections,
                key=lambda d: d["confidence"]
            )

            self.lock_target(
                frame,
                selected["bbox"],
                selected["confidence"]
            )

            return True

        # ----------------------------------------------------
        # Target already exists.
        #
        # Prefer a detection that overlaps the current target.
        # ----------------------------------------------------

        current = self.bbox

        best_detection = None
        best_score = -1.0

        for detection in detections:

            bbox = detection["bbox"]

            iou = self.calculate_iou(
                current,
                bbox
            )

            score = iou * 0.7 + detection["confidence"] * 0.3

            if score > best_score:

                best_score = score
                best_detection = detection

        # ----------------------------------------------------
        # Only replace/reinitialize target when there is
        # reasonable spatial agreement.
        # ----------------------------------------------------

        if best_detection is not None:

            bbox = best_detection["bbox"]

            iou = self.calculate_iou(
                current,
                bbox
            )

            if iou >= 0.15:

                self.lock_target(
                    frame,
                    bbox,
                    best_detection["confidence"]
                )

                return True

        return False

    # --------------------------------------------------------
    # Lock target
    # --------------------------------------------------------

    def lock_target(self, frame, bbox, confidence):

        x, y, w, h = bbox

        # Clamp bbox
        x = max(0, x)
        y = max(0, y)

        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        if w <= 0 or h <= 0:
            return False

        self.tracker = create_kcf_tracker()

        self.tracker.init(
            frame,
            (x, y, w, h)
        )

        self.bbox = (x, y, w, h)

        self.confidence = confidence

        self.locked = True

        self.state = "TRACKING"

        self.missed_frames = 0

        self.last_detection_time = time.time()

        log.info(
            f"TARGET LOCKED "
            f"confidence={confidence:.2f} "
            f"bbox={self.bbox}"
        )

        return True

    # --------------------------------------------------------
    # KCF update
    # --------------------------------------------------------

    def update_tracker(self, frame):

        if not self.locked or self.tracker is None:
            return False

        success, bbox = self.tracker.update(frame)

        if not success:

            self.missed_frames += 1

            log.info(
                f"Tracker miss "
                f"({self.missed_frames}/{MAX_TRACKER_MISSES})"
            )

            if self.missed_frames >= MAX_TRACKER_MISSES:

                self.locked = False

                self.tracker = None

                self.bbox = None

                self.confidence = 0.0

                self.state = "SEARCHING"

                log.info("Target lost. Returning to SEARCHING.")

            return False

        x, y, w, h = [int(v) for v in bbox]

        # Validate tracker result
        if w < MIN_BOX_WIDTH or h < MIN_BOX_HEIGHT:

            self.missed_frames += 1

            return False

        # Clamp
        x = max(0, min(frame.shape[1] - 1, x))
        y = max(0, min(frame.shape[0] - 1, y))

        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        self.bbox = (x, y, w, h)

        self.missed_frames = 0

        self.state = "TRACKING"

        return True

    # --------------------------------------------------------
    # YOLO verification
    # --------------------------------------------------------

    def verify_with_yolo(self, frame, detector):

        detections = detector.detect(frame)

        if not detections:

            log.info("YOLO verification: no person detected.")

            return False

        current = self.bbox

        # If currently tracking, find the detection closest
        # to the current tracker position.
        if current is not None:

            best = None
            best_iou = 0.0

            for detection in detections:

                iou = self.calculate_iou(
                    current,
                    detection["bbox"]
                )

                if iou > best_iou:

                    best_iou = iou
                    best = detection

            if best is not None and best_iou >= 0.10:

                self.lock_target(
                    frame,
                    best["bbox"],
                    best["confidence"]
                )

                log.info(
                    f"YOLO verification successful "
                    f"IoU={best_iou:.2f}"
                )

                return True

        # ----------------------------------------------------
        # If tracker has been lost, acquire strongest person.
        # ----------------------------------------------------

        selected = max(
            detections,
            key=lambda d: d["confidence"]
        )

        self.lock_target(
            frame,
            selected["bbox"],
            selected["confidence"]
        )

        log.info(
            f"YOLO target reacquired "
            f"confidence={selected['confidence']:.2f}"
        )

        return True

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    @staticmethod
    def calculate_iou(box_a, box_b):

        if box_a is None or box_b is None:
            return 0.0

        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        ax2 = ax + aw
        ay2 = ay + ah

        bx2 = bx + bw
        by2 = by + bh

        intersection_x1 = max(ax, bx)
        intersection_y1 = max(ay, by)

        intersection_x2 = min(ax2, bx2)
        intersection_y2 = min(ay2, by2)

        intersection_w = max(
            0,
            intersection_x2 - intersection_x1
        )

        intersection_h = max(
            0,
            intersection_y2 - intersection_y1
        )

        intersection_area = (
            intersection_w * intersection_h
        )

        area_a = aw * ah
        area_b = bw * bh

        union = area_a + area_b - intersection_area

        if union <= 0:
            return 0.0

        return intersection_area / union


# ============================================================
# SMART CART APPLICATION
# ============================================================

class SmartCart:

    def __init__(self):

        self.detector = YOLODetector(
            MODEL_PATH
        )

        self.target_tracker = TargetTracker()

        self.camera = Picamera2()

        config = self.camera.create_preview_configuration(
            main={
                "size": (
                    CAMERA_WIDTH,
                    CAMERA_HEIGHT
                ),
                "format": "BGR888"
            }
        )

        self.camera.configure(config)

        self.running = False

        self.frame_count = 0

        self.last_frame_time = time.time()

        self.fps = 0.0

        log.info(
            f"Camera configured "
            f"({CAMERA_WIDTH}x{CAMERA_HEIGHT})"
        )

    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    def start(self):

        self.camera.start()

        time.sleep(1)

        self.running = True

        log.info("Camera started.")

        log.info(
            "Smart Cart visual tracker started."
        )

        log.info(
            "Arduino/motor control is DISABLED."
        )

    # --------------------------------------------------------
    # Process frame
    # --------------------------------------------------------

    def process_frame(self):

        frame = self.camera.capture_array()

        self.frame_count += 1

        now = time.time()

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        elapsed = now - self.last_frame_time

        if elapsed > 0:

            instant_fps = 1.0 / elapsed

            if self.fps == 0:

                self.fps = instant_fps

            else:

                self.fps = (
                    self.fps * 0.8
                    + instant_fps * 0.2
                )

        self.last_frame_time = now

        # ----------------------------------------------------
        # If target is already locked:
        #
        # KCF gets priority.
        # ----------------------------------------------------

        if self.target_tracker.locked:

            tracking_success = (
                self.target_tracker.update_tracker(
                    frame
                )
            )

            # ------------------------------------------------
            # Periodic YOLO verification
            # ------------------------------------------------

            if (
                now -
                self.target_tracker.last_yolo_time
                >= YOLO_INTERVAL
            ):

                self.target_tracker.last_yolo_time = now

                self.target_tracker.verify_with_yolo(
                    frame,
                    self.detector
                )

        # ----------------------------------------------------
        # No target:
        #
        # Run YOLO acquisition.
        # ----------------------------------------------------

        else:

            if (
                now -
                self.target_tracker.last_yolo_time
                >= YOLO_INTERVAL
            ):

                self.target_tracker.last_yolo_time = now

                detections = self.detector.detect(
                    frame
                )

                log.info(
                    f"YOLO acquisition: "
                    f"{len(detections)} person(s)"
                )

                self.target_tracker.select_target(
                    detections,
                    frame
                )

        # ----------------------------------------------------
        # Diagnostic information
        # ----------------------------------------------------

        if now - self.target_tracker.last_log_time >= LOG_INTERVAL:

            self.target_tracker.last_log_time = now

            if self.target_tracker.bbox is not None:

                x, y, w, h = (
                    self.target_tracker.bbox
                )

                center_x = x + w // 2

                center_y = y + h // 2

                log.info(
                    f"State={self.target_tracker.state} "
                    f"Tracker={TRACKER_TYPE} "
                    f"FPS={self.fps:.1f} "
                    f"bbox=({x},{y},{w},{h}) "
                    f"center=({center_x},{center_y}) "
                    f"miss={self.target_tracker.missed_frames}"
                )

            else:

                log.info(
                    f"State={self.target_tracker.state} "
                    f"FPS={self.fps:.1f}"
                )

        return frame

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    def run(self):

        self.start()

        try:

            while self.running:

                self.process_frame()

        except KeyboardInterrupt:

            log.info("Keyboard interrupt.")

        finally:

            self.stop()

    # --------------------------------------------------------
    # Stop
    # --------------------------------------------------------

    def stop(self):

        if not self.running:
            return

        self.running = False

        log.info("Stopping...")

        try:

            self.camera.stop()

            log.info("Camera stopped.")

        except Exception as e:

            log.error(
                f"Camera stop error: {e}"
            )

        log.info(
            "Visual tracker stopped."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    log.info("Starting Smart Cart...")

    try:

        app = SmartCart()

        app.run()

    except Exception as e:

        log.exception(
            f"Fatal error: {e}"
        )


if __name__ == "__main__":

    main()
