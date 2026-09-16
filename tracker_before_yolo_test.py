import cv2
import time
import logging
import numpy as np
import threading
import serial
import os

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
YOLO_CONFIDENCE = 0.20

# How often YOLO is requested
YOLO_INTERVAL = 2.0

# Maximum consecutive KCF failures
MAX_TRACKER_MISSES = 12

# Minimum bounding-box dimensions
MIN_BOX_WIDTH = 30
MIN_BOX_HEIGHT = 50

# Minimum IoU before YOLO is allowed to correct KCF
YOLO_CORRECTION_IOU = 0.30

# Minimum IoU for a stronger correction
YOLO_STRONG_IOU = 0.50

# YOLO result older than this is ignored for correction
MAX_YOLO_RESULT_AGE = 1.0

# Maximum fraction of image occupied by KCF box
MAX_BOX_AREA_RATIO = 0.75

# Maximum allowed width/height relative to camera
MAX_BOX_WIDTH_RATIO = 0.90
MAX_BOX_HEIGHT_RATIO = 0.98

# Edge boxes are suspicious when they become very large
EDGE_MIN_WIDTH = 100
EDGE_MIN_HEIGHT = 150

# KCF
TRACKER_TYPE = "KCF"

# Logging
LOG_INTERVAL = 1.0

# ============================================================
# DEBUG IMAGE OUTPUT
# ============================================================

DEBUG_SAVE_ENABLED = True

DEBUG_DIRECTORY = "/home/mts/smart_cart/debug_images"

# Save approximately one debug image every second at 15 FPS.
DEBUG_SAVE_INTERVAL = 15

# ============================================================
# ARDUINO SERIAL CONTROL
# ============================================================

ARDUINO_PORT = "/dev/ttyACM0"
ARDUINO_BAUDRATE = 115200
ARDUINO_TIMEOUT = 1.0

# Arduino expects a heartbeat at least once within its
# 1000 ms timeout period.
#
# We send the state every 200 ms.
ARDUINO_HEARTBEAT_INTERVAL = 0.20


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

    # Legacy OpenCV namespace
    if hasattr(cv2, "legacy") and hasattr(
        cv2.legacy,
        "TrackerKCF_create"
    ):
        return cv2.legacy.TrackerKCF_create()

    raise RuntimeError(
        "KCF tracker is not available in this OpenCV installation."
    )


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
        output_shape = self.session.get_outputs()[0].shape

        self.last_raw_person_candidates = []

        log.info(
            f"YOLO loaded: input={input_shape}, "
            f"output={output_shape}"
        )

    def preprocess(self, frame):

        original_h, original_w = frame.shape[:2]

        # --------------------------------------------------------
        # Preserve aspect ratio
        # --------------------------------------------------------

        scale = min(
            YOLO_SIZE / original_w,
            YOLO_SIZE / original_h
        )

        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        resized = cv2.resize(
            frame,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )

        # --------------------------------------------------------
        # Letterbox
        # --------------------------------------------------------

        canvas = np.full(
            (YOLO_SIZE, YOLO_SIZE, 3),
            114,
            dtype=np.uint8
        )

        pad_x = (YOLO_SIZE - new_w) // 2
        pad_y = (YOLO_SIZE - new_h) // 2

        canvas[
            pad_y:pad_y + new_h,
            pad_x:pad_x + new_w
        ] = resized

        # --------------------------------------------------------
        # BGR -> RGB
        # --------------------------------------------------------

        image = cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2RGB
        )

        image = image.astype(
           np.float32
        ) / 255.0

        image = np.transpose(
           image,
           (2, 0, 1)
        )

        image = np.expand_dims(
           image,
           axis=0
        )

        return image, scale, pad_x, pad_y

    def detect(self, frame):

        original_h, original_w = frame.shape[:2]

        input_tensor, scale, pad_x, pad_y = (
            self.preprocess(frame)
        )

        outputs = self.session.run(
            [self.output_name],
            {
                self.input_name: input_tensor
            }
        )

        output = outputs[0]

        if output.ndim == 3:
            detections = output[0]
        else:
            detections = output

        results = []

        # --------------------------------------------------------
        # Diagnostic information
        # --------------------------------------------------------

        if len(detections) > 0:

            try:

                max_confidence = max(
                    float(d[4])
                    for d in detections
                )

                log.info(
                    f"YOLO raw detections={len(detections)} "
                    f"max_confidence={max_confidence:.3f}"
                )

            except Exception as e:

                log.warning(
                    f"Could not inspect YOLO output: {e}"
                )

        # --------------------------------------------------------
        # Process detections
        # --------------------------------------------------------

        for detection in detections:

            if len(detection) < 6:
                continue

            x1, y1, x2, y2, confidence, class_id = detection[:6]

            confidence = float(confidence)
            class_id = int(class_id)

            # ----------------------------------------------------
            # Confidence filter
            # ----------------------------------------------------

            if confidence < YOLO_CONFIDENCE:
                continue

            # ----------------------------------------------------
            # Person only
            # ----------------------------------------------------

            if class_id != PERSON_CLASS_ID:
                continue

            # ----------------------------------------------------
            # Convert letterboxed coordinates back to
            # original camera coordinates.
            # ----------------------------------------------------

            x1 = (float(x1) - pad_x) / scale
            y1 = (float(y1) - pad_y) / scale

            x2 = (float(x2) - pad_x) / scale
            y2 = (float(y2) - pad_y) / scale

            x1 = int(x1)
            y1 = int(y1)

            x2 = int(x2)
            y2 = int(y2)

            # ----------------------------------------------------
            # Clamp to camera image
            # ----------------------------------------------------

            x1 = max(
                0,
                min(original_w - 1, x1)
            )

            y1 = max(
                0,
                min(original_h - 1, y1)
            )

            x2 = max(
                0,
                min(original_w - 1, x2)
            )

            y2 = max(
                0,
                min(original_h - 1, y2)
            )

            width = x2 - x1
            height = y2 - y1

            if width < MIN_BOX_WIDTH:
                continue

            if height < MIN_BOX_HEIGHT:
                continue

            results.append(
                {
                    "bbox": (
                        x1,
                        y1,
                        width,
                        height
                    ),
                    "confidence": confidence
                }
            )

        # --------------------------------------------------------
        # Final detection diagnostic
        # --------------------------------------------------------

        if results:

            log.info(
                f"YOLO PERSON DETECTION: "
                f"{len(results)} person(s), "
                f"best_confidence="
                f"{max(r['confidence'] for r in results):.3f}"
            )

        else:

            log.info(
                "YOLO PERSON DETECTION: none"
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

        self.state = "SEARCHING"

        self.last_log_time = 0.0

        self.last_detection_time = 0.0

        log.info("Target tracker initialized.")

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

        intersection_x1 = max(
            ax,
            bx
        )

        intersection_y1 = max(
            ay,
            by
        )

        intersection_x2 = min(
            ax2,
            bx2
        )

        intersection_y2 = min(
            ay2,
            by2
        )

        intersection_w = max(
            0,
            intersection_x2 - intersection_x1
        )

        intersection_h = max(
            0,
            intersection_y2 - intersection_y1
        )

        intersection_area = (
            intersection_w *
            intersection_h
        )

        area_a = aw * ah
        area_b = bw * bh

        union = (
            area_a +
            area_b -
            intersection_area
        )

        if union <= 0:
            return 0.0

        return intersection_area / union

    # --------------------------------------------------------
    # Validate KCF bounding box
    # --------------------------------------------------------

    def validate_tracker_bbox(self, bbox, frame):

        if bbox is None:
            return False

        x, y, w, h = [
            int(v)
            for v in bbox
        ]

        frame_h, frame_w = frame.shape[:2]

        # Basic dimensions
        if w < MIN_BOX_WIDTH:
            return False

        if h < MIN_BOX_HEIGHT:
            return False

        # Allow box to touch image edges
        if x < 0:
            return False

        if y < 0:
            return False

        if x >= frame_w:
            return False

        if y >= frame_h:
            return False

        if x + w <= 0:
            return False

        if y + h <= 0:
            return False

        # Prevent KCF from expanding excessively
        if w > frame_w * MAX_BOX_WIDTH_RATIO:
            return False

        if h > frame_h * MAX_BOX_HEIGHT_RATIO:
            return False

        # Area check
        area_ratio = (
            (w * h) /
            float(frame_w * frame_h)
        )

        if area_ratio > MAX_BOX_AREA_RATIO:
            return False

        return True

    # --------------------------------------------------------
    # Target selection
    # --------------------------------------------------------

    def select_initial_target(
        self,
        detections,
        frame
    ):

        if not detections:
            return False

        selected = max(
            detections,
            key=lambda d: d["confidence"]
        )

        return self.lock_target(
            frame,
            selected["bbox"],
            selected["confidence"]
        )

    # --------------------------------------------------------
    # Lock target
    # --------------------------------------------------------

    def lock_target(
        self,
        frame,
        bbox,
        confidence
    ):

        x, y, w, h = bbox

        x = max(
            0,
            min(frame.shape[1] - 1, x)
        )

        y = max(
            0,
            min(frame.shape[0] - 1, y)
        )

        w = min(
            w,
            frame.shape[1] - x
        )

        h = min(
            h,
            frame.shape[0] - y
        )

        if (
            w < MIN_BOX_WIDTH or
            h < MIN_BOX_HEIGHT
        ):
            return False

        self.tracker = create_kcf_tracker()

        self.tracker.init(
            frame,
            (
                x,
                y,
                w,
                h
            )
        )

        self.bbox = (
            x,
            y,
            w,
            h
        )

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
    # Update KCF
    # --------------------------------------------------------

    def update_tracker(self, frame):

        if (
            not self.locked or
            self.tracker is None
        ):
            return False

        success, bbox = self.tracker.update(
            frame
        )

        if not success:

            self.missed_frames += 1

            log.info(
                f"Tracker miss "
                f"({self.missed_frames}/"
                f"{MAX_TRACKER_MISSES})"
            )

            if (
                self.missed_frames >=
                MAX_TRACKER_MISSES
            ):

                self.locked = False
                self.tracker = None
                self.bbox = None
                self.confidence = 0.0
                self.state = "SEARCHING"

                log.info(
                    "Target lost. "
                    "Returning to SEARCHING."
                )

            return False

        x, y, w, h = [
            int(v)
            for v in bbox
        ]

        candidate_bbox = (
            x,
            y,
            w,
            h
        )

        if not self.validate_tracker_bbox(
            candidate_bbox,
            frame
        ):

            self.missed_frames += 1

            log.info(
                f"KCF bbox rejected "
                f"bbox={candidate_bbox} "
                f"miss={self.missed_frames}/"
                f"{MAX_TRACKER_MISSES}"
            )

            if (
                self.missed_frames >=
                MAX_TRACKER_MISSES
            ):

                self.locked = False
                self.tracker = None
                self.bbox = None
                self.confidence = 0.0
                self.state = "SEARCHING"

                log.info(
                    "KCF target lost due to "
                    "invalid/drifting bounding box."
                )

            return False

        x = max(
            0,
            min(frame.shape[1] - 1, x)
        )

        y = max(
            0,
            min(frame.shape[0] - 1, y)
        )

        w = min(
            w,
            frame.shape[1] - x
        )

        h = min(
            h,
            frame.shape[0] - y
        )

        self.bbox = (
            x,
            y,
            w,
            h
        )

        self.missed_frames = 0

        self.state = "TRACKING"

        return True

    # --------------------------------------------------------
    # YOLO correction
    # --------------------------------------------------------

    def correct_with_yolo(
        self,
        detections,
        frame
    ):

        if not detections:

            log.info(
                "YOLO verification: no person detected."
            )

            return False

        if (
            not self.locked or
            self.bbox is None
        ):

            selected = max(
                detections,
                key=lambda d: d["confidence"]
            )

            return self.lock_target(
                frame,
                selected["bbox"],
                selected["confidence"]
            )

        current = self.bbox

        best = None
        best_iou = 0.0

        for detection in detections:

            bbox = detection["bbox"]

            iou = self.calculate_iou(
                current,
                bbox
            )

            if iou > best_iou:

                best_iou = iou
                best = detection

        if (
            best is None or
            best_iou < YOLO_CORRECTION_IOU
        ):

            log.info(
                f"YOLO verification rejected "
                f"best_IoU={best_iou:.2f}"
            )

            return False

        bbox = best["bbox"]

        self.missed_frames = 0

        self.confidence = best["confidence"]

        self.last_detection_time = time.time()

        self.state = "TRACKING"

        if best_iou >= YOLO_STRONG_IOU:

            if self.validate_tracker_bbox(
                bbox,
                frame
            ):

                try:

                    x, y, w, h = [
                        int(v)
                        for v in bbox
                    ]

                    new_tracker = create_kcf_tracker()

                    new_tracker.init(
                        frame,
                        (
                            x,
                            y,
                            w,
                            h
                        )
                    )

                    self.tracker = new_tracker

                    self.bbox = (
                        x,
                        y,
                        w,
                        h
                    )

                    log.info(
                        f"YOLO verification accepted "
                        f"IoU={best_iou:.2f} "
                        f"(KCF reinitialized)"
                    )

                    return True

                except Exception as e:

                    log.error(
                        f"KCF reinitialization failed: {e}"
                    )

                    return True

            else:

                log.info(
                    f"YOLO verification accepted "
                    f"IoU={best_iou:.2f} "
                    f"but YOLO bbox rejected; "
                    f"KCF retained"
                )

                return True

        log.info(
            f"YOLO verification accepted "
            f"IoU={best_iou:.2f} "
            f"(KCF retained)"
        )

        return True


# ============================================================
# ASYNCHRONOUS YOLO WORKER
# ============================================================

class YOLOWorker:

    def __init__(self, detector):

        self.detector = detector

        self.frame = None

        self.result = []

        self.result_time = 0.0

        self.request_time = 0.0

        self.requested = False

        self.running = False

        self.lock = threading.Lock()

        self.thread = None

    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    def start(self):

        self.running = True

        self.thread = threading.Thread(
            target=self.worker_loop,
            daemon=True
        )

        self.thread.start()

    # --------------------------------------------------------
    # Submit frame
    # --------------------------------------------------------

    def submit(self, frame):

        with self.lock:

            self.frame = frame.copy()

            self.request_time = time.time()

            self.requested = True

    # --------------------------------------------------------
    # Worker loop
    # --------------------------------------------------------

    def worker_loop(self):

        while self.running:

            frame = None

            with self.lock:

                if (
                    self.requested and
                    self.frame is not None
                ):

                    frame = self.frame

                    self.frame = None

                    self.requested = False

            if frame is None:

                time.sleep(0.005)

                continue

            try:

                detections = self.detector.detect(
                    frame
                )

                with self.lock:

                    self.result = detections

                    self.result_time = time.time()

            except Exception as e:

                log.error(
                    f"YOLO worker error: {e}"
                )

    # --------------------------------------------------------
    # Get latest result
    # --------------------------------------------------------

    def get_result(self):

        with self.lock:

            return (
                list(self.result),
                self.result_time,
                self.request_time
            )

    # --------------------------------------------------------
    # Stop
    # --------------------------------------------------------

    def stop(self):

        self.running = False

        if self.thread is not None:

            self.thread.join(
                timeout=2.0
            )

# ============================================================
# ARDUINO MOTOR CONTROLLER
# ============================================================

class ArduinoController:

    def __init__(self):

        self.ser = None

        # Arduino state:
        # '1' = TRACKING
        # '0' = SEARCHING / STOP
        self.current_command = '0'

        self.running = False

        self.lock = threading.Lock()

        self.heartbeat_thread = None

        try:

            self.ser = serial.Serial(
                ARDUINO_PORT,
                ARDUINO_BAUDRATE,
                timeout=ARDUINO_TIMEOUT
            )

            time.sleep(2)

            log.info(
                f"Arduino connected on {ARDUINO_PORT}"
            )

            # ------------------------------------------------
            # Establish safe STOP state
            # ------------------------------------------------

            self._send('0')

            log.info(
                "Arduino STOP command sent."
            )

            # ------------------------------------------------
            # Start continuous heartbeat
            # ------------------------------------------------

            self.running = True

            self.heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True
            )

            self.heartbeat_thread.start()

        except Exception as e:

            log.error(
                f"Could not connect to Arduino: {e}"
            )

            self.ser = None

    # --------------------------------------------------------
    # Raw serial transmission
    # --------------------------------------------------------

    def _send(self, command):

        if self.ser is None:
            return

        try:

            message = command + "\n"

            self.ser.write(
                message.encode("utf-8")
            )

            self.ser.flush()

        except Exception as e:

            log.error(
                f"Arduino communication error: {e}"
            )

    # --------------------------------------------------------
    # Set desired Arduino state
    # --------------------------------------------------------

    def set_state(self, tracking):

        with self.lock:

            if tracking:

                self.current_command = '1'

            else:

                self.current_command = '0'

    # --------------------------------------------------------
    # Dedicated heartbeat loop
    # --------------------------------------------------------

    def _heartbeat_loop(self):

        log.info(
            "Arduino heartbeat thread started."
        )

        while self.running:

            with self.lock:

                command = self.current_command

            self._send(command)

            time.sleep(
                ARDUINO_HEARTBEAT_INTERVAL
            )

        log.info(
            "Arduino heartbeat thread stopped."
        )

    # --------------------------------------------------------
    # Immediate STOP
    # --------------------------------------------------------

    def send_immediate_stop(self):

        with self.lock:

            self.current_command = '0'

        # Send immediately without waiting for
        # the heartbeat thread.

        self._send('0')

        log.info(
            "Arduino STOP command sent."
        )

    # --------------------------------------------------------
    # Close
    # --------------------------------------------------------

    def close(self):

        try:

            # ------------------------------------------------
            # Stop state immediately
            # ------------------------------------------------

            self.send_immediate_stop()

            # ------------------------------------------------
            # Stop heartbeat thread
            # ------------------------------------------------

            self.running = False

            if self.heartbeat_thread is not None:

                self.heartbeat_thread.join(
                    timeout=1.0
                )

            # ------------------------------------------------
            # Send one final STOP
            # ------------------------------------------------

            self._send('0')

            time.sleep(0.1)

            # ------------------------------------------------
            # Close serial
            # ------------------------------------------------

            if self.ser is not None:

                self.ser.close()

                log.info(
                    "Arduino serial connection closed."
                )

        except Exception as e:

            log.error(
                f"Arduino close error: {e}"
            )


# ============================================================
# SMART CART
# ============================================================

class SmartCart:

    def __init__(self):

        self.detector = YOLODetector(
            MODEL_PATH
        )

        self.target_tracker = TargetTracker()

        self.yolo_worker = YOLOWorker(
            self.detector
        )

        self.arduino = ArduinoController()

        self.camera = Picamera2()

        config = (
            self.camera
            .create_preview_configuration(
                main={
                    "size": (
                        CAMERA_WIDTH,
                        CAMERA_HEIGHT
                    ),
                    "format": "BGR888"
                }
            )
        )

        self.camera.configure(
            config
        )

        self.running = False

        self.frame_count = 0

        self.last_frame_time = time.time()

        self.fps = 0.0

        self.last_yolo_request = 0.0

        self.last_yolo_result_time = 0.0

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

        self.yolo_worker.start()

        self.running = True

        self.last_yolo_request = (
            time.time() -
            YOLO_INTERVAL
        )

        log.info(
            "Camera started."
        )

        log.info(
            "Smart Cart visual tracker started."
        )

        log.info(
            "Arduino/motor control is ENABLED."
        )

    # --------------------------------------------------------
    # Request YOLO
    # --------------------------------------------------------

    def request_yolo(self, frame):

        now = time.time()

        if (
            now -
            self.last_yolo_request
            >= YOLO_INTERVAL
        ):

            self.last_yolo_request = now

            self.yolo_worker.submit(
                frame
            )

            log.info(
                "YOLO detection requested."
            )

    # --------------------------------------------------------
    # Process frame
    # --------------------------------------------------------

    def process_frame(self):

        frame = self.camera.capture_array()

        # Camera is physically mounted upside down
        frame = cv2.rotate(
            frame,
            cv2.ROTATE_180
        )

        # ----------------------------------------------------
        # SAVE RAW TEST IMAGE
        # ----------------------------------------------------

        if self.frame_count % 30 == 0:

            filename = (
                f"/home/mts/smart_cart/"
                f"test_{self.frame_count}.jpg"
            )

            cv2.imwrite(
                filename,
                frame
            )

        self.frame_count += 1

        now = time.time()

        # ====================================================
        # 1. FPS
        # ====================================================

        elapsed = (
            now -
            self.last_frame_time
        )

        if elapsed > 0:

            instant_fps = 1.0 / elapsed

            if self.fps == 0:

                self.fps = instant_fps

            else:

                self.fps = (
                    self.fps * 0.8 +
                    instant_fps * 0.2
                )

        self.last_frame_time = now

        # ====================================================
        # 2. UPDATE KCF TRACKER
        # ====================================================

        if self.target_tracker.locked:

            tracking_success = (
                self.target_tracker
                .update_tracker(frame)
            )

            if not tracking_success:

                self.request_yolo(
                    frame
                )

            else:

                self.request_yolo(
                    frame
                )

        # ====================================================
        # 3. SEARCH FOR TARGET
        # ====================================================

        else:

            self.request_yolo(
                frame
            )

        # ====================================================
        # 4. GET LATEST YOLO RESULT
        # ====================================================

        detections, result_time, request_time = (
            self.yolo_worker.get_result()
        )

        # ====================================================
        # 5. PROCESS NEW YOLO RESULT
        # ====================================================

        if (
            result_time >
            self.last_yolo_result_time
        ):

            result_age = (
                now -
                result_time
            )

            if result_age > MAX_YOLO_RESULT_AGE:

                log.info(
                    f"YOLO result ignored "
                    f"age={result_age:.2f}s"
                )

                self.last_yolo_result_time = (
                    result_time
                )

            else:

                self.last_yolo_result_time = (
                    result_time
                )

                # ------------------------------------------------
                # NO CURRENT TARGET
                # ------------------------------------------------

                if not self.target_tracker.locked:

                    if detections:

                        log.info(
                            f"YOLO acquisition: "
                            f"{len(detections)} "
                            f"person(s)"
                        )

                        self.target_tracker.select_initial_target(
                            detections,
                            frame
                        )

                    else:

                        log.info(
                            "YOLO acquisition: "
                            "no person detected."
                        )

                # ------------------------------------------------
                # EXISTING TARGET
                # ------------------------------------------------

                else:

                    if detections:

                        self.target_tracker.correct_with_yolo(
                            detections,
                            frame
                        )

                    else:

                        log.info(
                            "YOLO verification: "
                            "no person detected."
                        )

        # ====================================================
        # 6. DIAGNOSTIC LOGGING
        # ====================================================

        if (
            now -
            self.target_tracker.last_log_time
            >= LOG_INTERVAL
        ):

            self.target_tracker.last_log_time = now

            if (
                self.target_tracker.bbox
                is not None
            ):

                x, y, w, h = (
                    self.target_tracker.bbox
                )

                # ------------------------------------------------
                # Diagnostic only.
                #
                # These coordinates are NOT used for steering
                # or motor control.
                # ------------------------------------------------

                center_x = (
                    x + w // 2
                )

                center_y = (
                    y + h // 2
                )

                log.info(
                    f"State="
                    f"{self.target_tracker.state} "
                    f"Tracker={TRACKER_TYPE} "
                    f"FPS={self.fps:.1f} "
                    f"bbox="
                    f"({x},{y},{w},{h}) "
                    f"center="
                    f"({center_x},{center_y}) "
                    f"miss="
                    f"{self.target_tracker.missed_frames}"
                )

            else:

                log.info(
                    f"State="
                    f"{self.target_tracker.state} "
                    f"FPS={self.fps:.1f}"
                )

        # ====================================================
        # 7. ARDUINO STATE CONTROL
        # ====================================================

        self.update_motor_control()

        # ====================================================
        # 8. SAVE ANNOTATED DEBUG IMAGE
        #
        # BLUE  = YOLO detection
        # GREEN = KCF tracking
        # ====================================================

        self.save_debug_frame(
            frame,
            detections,
            result_time
        )

    # --------------------------------------------------------
    # Save annotated debug image
    # --------------------------------------------------------

    def save_debug_frame(
        self,
        frame,
        detections,
        result_time
    ):

        if not DEBUG_SAVE_ENABLED:
            return

        if (
            self.frame_count %
            DEBUG_SAVE_INTERVAL != 0
        ):
            return

        try:

            # ----------------------------------------------------
            # Make sure debug directory exists
            # ----------------------------------------------------

            os.makedirs(
                DEBUG_DIRECTORY,
                exist_ok=True
            )

            # ----------------------------------------------------
            # Copy frame so original camera frame is untouched
            # ----------------------------------------------------

            debug = frame.copy()

            frame_h, frame_w = (
                debug.shape[:2]
            )

            # ====================================================
            # 1. DRAW YOLO DETECTIONS
            # ====================================================

            yolo_count = len(
                detections
            )

            best_confidence = 0.0

            for detection in detections:

                x, y, w, h = (
                    detection["bbox"]
                )

                confidence = (
                    detection["confidence"]
                )

                if (
                    confidence >
                    best_confidence
                ):

                    best_confidence = (
                        confidence
                    )

                # ------------------------------------------------
                # YOLO BOX
                #
                # BLUE
                # ------------------------------------------------

                cv2.rectangle(
                    debug,
                    (x, y),
                    (
                        x + w,
                        y + h
                    ),
                    (255, 0, 0),
                    2
                )

                # ------------------------------------------------
                # YOLO LABEL
                # ------------------------------------------------

                cv2.putText(
                    debug,
                    f"YOLO PERSON "
                    f"{confidence:.2f}",
                    (
                        x,
                        max(
                            20,
                            y - 8
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )

            # ====================================================
            # 2. DRAW KCF TRACKING BOX
            # ====================================================

            if (
                self.target_tracker.bbox
                is not None
            ):

                x, y, w, h = (
                    self.target_tracker.bbox
                )

                # ------------------------------------------------
                # KCF BOX
                #
                # GREEN
                # ------------------------------------------------

                cv2.rectangle(
                    debug,
                    (x, y),
                    (
                        x + w,
                        y + h
                    ),
                    (0, 255, 0),
                    3
                )

                # ------------------------------------------------
                # KCF LABEL
                # ------------------------------------------------

                cv2.putText(
                    debug,
                    "KCF TRACKING",
                    (
                        x,
                        min(
                            frame_h - 10,
                            y + h + 20
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # ====================================================
            # 3. PI COMMAND
            # ====================================================

            if self.target_tracker.locked:

                pi_command = "FOLLOW"

            else:

                pi_command = "STOP"

            # ====================================================
            # 4. YOLO RESULT AGE
            # ====================================================

            if result_time > 0:

                result_age = (
                    time.time() -
                    result_time
                )

            else:

                result_age = -1.0

            # ====================================================
            # 5. INFORMATION PANEL
            # ====================================================

            panel_lines = [

                f"STATE: "
                f"{self.target_tracker.state}",

                f"PI COMMAND: "
                f"{pi_command}",

                f"YOLO PERSONS: "
                f"{yolo_count}",

                f"YOLO BEST CONF: "
                f"{best_confidence:.2f}",

                f"YOLO AGE: "
                f"{result_age:.2f}s",

                f"KCF MISSES: "
                f"{self.target_tracker.missed_frames}",

                f"FPS: "
                f"{self.fps:.1f}",

                "BLUE = YOLO",

                "GREEN = KCF",

                "MOTOR DIRECTION: ARDUINO"
            ]

            panel_x = 10
            panel_y = 25
            line_spacing = 22

            for index, text in enumerate(
                panel_lines
            ):

                y_position = (
                    panel_y +
                    index * line_spacing
                )

                cv2.putText(
                    debug,
                    text,
                    (
                        panel_x,
                        y_position
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

            # ====================================================
            # 6. FRAME COUNTER
            # ====================================================

            cv2.putText(
                debug,
                f"FRAME: "
                f"{self.frame_count}",
                (
                    10,
                    frame_h - 15
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            # ====================================================
            # 7. SAVE IMAGE
            # ====================================================

            filename = os.path.join(
                DEBUG_DIRECTORY,
                f"debug_"
                f"{self.frame_count:06d}.jpg"
            )

            success = cv2.imwrite(
                filename,
                debug
            )

            if not success:

                log.error(
                    f"Failed to write "
                    f"debug image: "
                    f"{filename}"
                )

        except Exception as e:

            log.error(
                f"Debug image save error: "
                f"{e}"
            )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    def run(self):

        self.start()

        try:

            while self.running:

                self.process_frame()

        except KeyboardInterrupt:

            log.info(
                "Keyboard interrupt."
            )

        finally:

            self.stop()

    # --------------------------------------------------------
    # Stop
    # --------------------------------------------------------

    def stop(self):

        if not self.running:
            return

        self.running = False

        log.info(
            "Stopping..."
        )

        # ----------------------------------------------------
        # STOP ARDUINO FIRST
        #
        # Sends '0' immediately and stops the heartbeat.
        # ----------------------------------------------------

        try:

            self.arduino.send_immediate_stop()

            time.sleep(0.1)

            self.arduino.close()

        except Exception as e:

            log.error(
                f"Arduino stop error: {e}"
            )

        # ----------------------------------------------------
        # STOP YOLO
        # ----------------------------------------------------

        try:

            self.yolo_worker.stop()

        except Exception as e:

            log.error(
                f"YOLO worker stop error: {e}"
            )

        # ----------------------------------------------------
        # STOP CAMERA
        # ----------------------------------------------------

        try:

            self.camera.stop()

            log.info(
                "Camera stopped."
            )

        except Exception as e:

            log.error(
                f"Camera stop error: {e}"
            )

        log.info(
            "Visual tracker stopped."
        )

    # --------------------------------------------------------
    # Arduino state decision
    # --------------------------------------------------------

    def update_motor_control(self):

        if self.target_tracker.locked:

            # TRACKING
            # Arduino continuously receives '1'
            self.arduino.set_state(True)

        else:

            # SEARCHING
            # Arduino continuously receives '0'
            self.arduino.set_state(False)


# ============================================================
# MAIN
# ============================================================

def main():

    log.info(
        "Starting Smart Cart..."
    )

    try:

        app = SmartCart()

        app.run()

    except Exception as e:

        log.exception(
            f"Fatal error: {e}"
        )


if __name__ == "__main__":

    main()
