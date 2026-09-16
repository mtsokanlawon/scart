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
# SMART CART
# KCF PRIMARY + ASYNCHRONOUS YOLO + RAPID REACQUISITION
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/yolov8n.onnx"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

YOLO_SIZE = 640
PERSON_CLASS_ID = 0
YOLO_CONFIDENCE = 0.20


# ============================================================
# MAIN PROCESSING
# ============================================================

MAIN_LOOP_FPS = 30.0
MAIN_LOOP_INTERVAL = 1.0 / MAIN_LOOP_FPS

KCF_FPS = 20.0
KCF_INTERVAL = 1.0 / KCF_FPS


# ============================================================
# YOLO TIMING
# ============================================================

YOLO_INTERVAL = 1.20

YOLO_REACQUISITION_INTERVAL = 0.25

MAX_YOLO_RESULT_AGE = 1.50

MAX_REACQUISITION_RESULT_AGE = 0.80


# ============================================================
# KCF FAILURE HANDLING
# ============================================================

MAX_TRACKER_MISSES = 12

KCF_LOST_GRACE_TIME = 0.60


# ============================================================
# BOUNDING BOX LIMITS
# ============================================================

MIN_BOX_WIDTH = 30
MIN_BOX_HEIGHT = 50

YOLO_CORRECTION_IOU = 0.30
YOLO_STRONG_IOU = 0.60

MAX_BOX_AREA_RATIO = 0.75

MAX_BOX_WIDTH_RATIO = 0.90
MAX_BOX_HEIGHT_RATIO = 0.98


# ============================================================
# COLOUR TARGET
# ============================================================

COLOUR_SAMPLE_COUNT = 10

COLOUR_SAMPLE_INTERVAL = 0.15

COLOUR_HUE_TOLERANCE = 12
COLOUR_SAT_TOLERANCE = 55
COLOUR_VALUE_TOLERANCE = 65

COLOUR_MIN_SATURATION = 40
COLOUR_MIN_VALUE = 35

MIN_VALID_COLOUR_RATIO = 0.20


# ============================================================
# COLOUR VERIFICATION
# ============================================================

COLOUR_VERIFY_INTERVAL = 0.30

COLOUR_CONFIDENCE_INITIAL = 1.0

COLOUR_CONFIDENCE_MATCH_GAIN = 0.12

COLOUR_CONFIDENCE_WEAK_LOSS = 0.06

COLOUR_CONFIDENCE_MISMATCH_LOSS = 0.18

COLOUR_CONFIDENCE_LOST_THRESHOLD = 0.05

COLOUR_WEAK_MATCH_THRESHOLD = 0.45

COLOUR_STRONG_MATCH_THRESHOLD = 0.70


# ============================================================
# COLOUR REGION
# ============================================================

COLOUR_REGION_X1 = 0.25
COLOUR_REGION_X2 = 0.75

COLOUR_REGION_Y1 = 0.20
COLOUR_REGION_Y2 = 0.65


# ============================================================
# DEBUG
# ============================================================

DEBUG_SAVE_ENABLED = True

DEBUG_DIRECTORY = "/home/mts/smart_cart/debug_images"

DEBUG_SAVE_INTERVAL = 5.0


# ============================================================
# ARDUINO
# ============================================================

ARDUINO_PORT = "/dev/ttyACM0"

ARDUINO_BAUDRATE = 115200

ARDUINO_TIMEOUT = 1.0

ARDUINO_HEARTBEAT_INTERVAL = 0.20


# ============================================================
# LOGGING
# ============================================================

LOG_INTERVAL = 1.0

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

    if hasattr(cv2, "TrackerKCF_create"):

        return cv2.TrackerKCF_create()

    if (
        hasattr(cv2, "legacy")
        and
        hasattr(
            cv2.legacy,
            "TrackerKCF_create"
        )
    ):

        return cv2.legacy.TrackerKCF_create()

    raise RuntimeError(
        "KCF tracker is not available in this OpenCV installation."
    )


# ============================================================
# COLOUR TARGET IDENTIFIER
# ============================================================

class ColourTargetIdentifier:

    def __init__(self):

        self.locked = False

        self.target_h = None
        self.target_s = None
        self.target_v = None

        self.samples = []

        self.sample_count = 0

        self.colour_confidence = 0.0

        self.last_sample_time = 0.0

        self.last_verify_time = 0.0

        self.state = "NOT LOCKED"

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.locked = False

        self.target_h = None
        self.target_s = None
        self.target_v = None

        self.samples = []

        self.sample_count = 0

        self.colour_confidence = 0.0

        self.last_sample_time = 0.0

        self.last_verify_time = 0.0

        self.state = "NOT LOCKED"

    # ========================================================
    # BEGIN ACQUISITION
    # ========================================================

    def begin_acquisition(self):

        self.locked = False

        self.target_h = None
        self.target_s = None
        self.target_v = None

        self.samples = []

        self.sample_count = 0

        self.colour_confidence = 0.0

        self.last_sample_time = 0.0

        self.last_verify_time = 0.0

        self.state = "SAMPLING"

        log.info(
            "Starting colour target acquisition."
        )

    # ========================================================
    # GET COLOUR REGION
    # ========================================================

    def get_colour_region(
        self,
        frame,
        bbox
    ):

        if bbox is None:

            return None

        x, y, w, h = bbox

        frame_h, frame_w = frame.shape[:2]

        rx1 = int(
            x +
            w *
            COLOUR_REGION_X1
        )

        rx2 = int(
            x +
            w *
            COLOUR_REGION_X2
        )

        ry1 = int(
            y +
            h *
            COLOUR_REGION_Y1
        )

        ry2 = int(
            y +
            h *
            COLOUR_REGION_Y2
        )

        rx1 = max(
            0,
            min(
                frame_w - 1,
                rx1
            )
        )

        rx2 = max(
            0,
            min(
                frame_w,
                rx2
            )
        )

        ry1 = max(
            0,
            min(
                frame_h - 1,
                ry1
            )
        )

        ry2 = max(
            0,
            min(
                frame_h,
                ry2
            )
        )

        if (
            rx2 <= rx1
            or
            ry2 <= ry1
        ):

            return None

        return frame[
            ry1:ry2,
            rx1:rx2
        ]

    # ========================================================
    # EXTRACT COLOUR
    # ========================================================

    def extract_colour(
        self,
        frame,
        bbox
    ):

        region = self.get_colour_region(
            frame,
            bbox
        )

        if region is None:

            return None

        hsv = cv2.cvtColor(
            region,
            cv2.COLOR_BGR2HSV
        )

        h = hsv[:, :, 0]

        s = hsv[:, :, 1]

        v = hsv[:, :, 2]

        valid_mask = (
            (s >= COLOUR_MIN_SATURATION)
            &
            (v >= COLOUR_MIN_VALUE)
        )

        valid_count = np.count_nonzero(
            valid_mask
        )

        total_count = valid_mask.size

        if total_count == 0:

            return None

        valid_ratio = (
            valid_count /
            float(total_count)
        )

        if (
            valid_ratio <
            MIN_VALID_COLOUR_RATIO
        ):

            return None

        valid_h = h[valid_mask]

        valid_s = s[valid_mask]

        valid_v = v[valid_mask]

        return (
            float(np.median(valid_h)),
            float(np.median(valid_s)),
            float(np.median(valid_v))
        )

    # ========================================================
    # HUE DISTANCE
    # ========================================================

    @staticmethod
    def hue_distance(
        h1,
        h2
    ):

        difference = abs(
            float(h1) -
            float(h2)
        )

        return min(
            difference,
            180.0 - difference
        )

    # ========================================================
    # COLOUR SIMILARITY
    # ========================================================

    def colour_similarity(
        self,
        colour
    ):

        if (
            colour is None
            or
            not self.locked
        ):

            return 0.0

        h, s, v = colour

        hue_difference = (
            self.hue_distance(
                h,
                self.target_h
            )
        )

        saturation_difference = abs(
            s -
            self.target_s
        )

        value_difference = abs(
            v -
            self.target_v
        )

        hue_score = max(
            0.0,
            1.0 -
            (
                hue_difference /
                COLOUR_HUE_TOLERANCE
            )
        )

        saturation_score = max(
            0.0,
            1.0 -
            (
                saturation_difference /
                COLOUR_SAT_TOLERANCE
            )
        )

        value_score = max(
            0.0,
            1.0 -
            (
                value_difference /
                COLOUR_VALUE_TOLERANCE
            )
        )

        score = (
            hue_score * 0.50
            +
            saturation_score * 0.25
            +
            value_score * 0.25
        )

        return float(
            max(
                0.0,
                min(
                    1.0,
                    score
                )
            )
        )

    # ========================================================
    # COLOUR MATCH
    # ========================================================

    def colour_matches(
        self,
        colour
    ):

        return (
            self.colour_similarity(
                colour
            )
            >=
            COLOUR_WEAK_MATCH_THRESHOLD
        )

    # ========================================================
    # SAMPLE
    # ========================================================

    def sample(
        self,
        frame,
        bbox
    ):

        if self.locked:

            return True

        now = time.monotonic()

        if (
            now -
            self.last_sample_time
            <
            COLOUR_SAMPLE_INTERVAL
        ):

            return False

        self.last_sample_time = now

        colour = self.extract_colour(
            frame,
            bbox
        )

        if colour is None:

            return False

        self.samples.append(
            colour
        )

        self.sample_count = len(
            self.samples
        )

        h, s, v = colour

        log.info(
            f"Colour sample "
            f"{self.sample_count}/"
            f"{COLOUR_SAMPLE_COUNT}: "
            f"H={h:.1f} "
            f"S={s:.1f} "
            f"V={v:.1f}"
        )

        if (
            self.sample_count
            >=
            COLOUR_SAMPLE_COUNT
        ):

            self.lock_colour()

        return self.locked

    # ========================================================
    # LOCK COLOUR
    # ========================================================

    def lock_colour(self):

        if not self.samples:

            return False

        values = np.array(
            self.samples,
            dtype=np.float32
        )

        self.target_h = float(
            np.median(
                values[:, 0]
            )
        )

        self.target_s = float(
            np.median(
                values[:, 1]
            )
        )

        self.target_v = float(
            np.median(
                values[:, 2]
            )
        )

        self.locked = True

        self.colour_confidence = (
            COLOUR_CONFIDENCE_INITIAL
        )

        self.state = "LOCKED"

        log.info(
            f"COLOUR TARGET LOCKED: "
            f"H={self.target_h:.1f} "
            f"S={self.target_s:.1f} "
            f"V={self.target_v:.1f}"
        )

        return True

    # ========================================================
    # VERIFY
    # ========================================================

    def verify(
        self,
        frame,
        bbox
    ):

        if not self.locked:

            return False

        now = time.monotonic()

        if (
            now -
            self.last_verify_time
            <
            COLOUR_VERIFY_INTERVAL
        ):

            return True

        self.last_verify_time = now

        colour = self.extract_colour(
            frame,
            bbox
        )

        if colour is None:

            self.colour_confidence -= (
                COLOUR_CONFIDENCE_WEAK_LOSS
            )

            self.colour_confidence = max(
                0.0,
                self.colour_confidence
            )

            self.state = "UNCERTAIN"

        else:

            score = (
                self.colour_similarity(
                    colour
                )
            )

            if (
                score
                >=
                COLOUR_STRONG_MATCH_THRESHOLD
            ):

                self.colour_confidence += (
                    COLOUR_CONFIDENCE_MATCH_GAIN
                )

                self.colour_confidence = min(
                    1.0,
                    self.colour_confidence
                )

                self.state = "LOCKED"

            elif (
                score
                >=
                COLOUR_WEAK_MATCH_THRESHOLD
            ):

                self.colour_confidence += (
                    COLOUR_CONFIDENCE_MATCH_GAIN *
                    0.50
                )

                self.colour_confidence = min(
                    1.0,
                    self.colour_confidence
                )

                self.state = "LOCKED"

            else:

                self.colour_confidence -= (
                    COLOUR_CONFIDENCE_MISMATCH_LOSS
                )

                self.colour_confidence = max(
                    0.0,
                    self.colour_confidence
                )

                self.state = "UNCERTAIN"

                log.warning(
                    f"Colour verification weak: "
                    f"score={score:.2f} "
                    f"confidence="
                    f"{self.colour_confidence:.2f}"
                )

        if (
            self.colour_confidence
            <=
            COLOUR_CONFIDENCE_LOST_THRESHOLD
        ):

            self.state = "IDENTITY LOST"

            log.warning(
                "Learned colour identity confidence "
                "has fallen below the loss threshold."
            )

            return False

        return True


# ============================================================
# YOLO DETECTOR
# ============================================================

class YOLODetector:

    def __init__(
        self,
        model_path
    ):

        log.info(
            "Loading YOLO model..."
        )

        self.session = (
            ort.InferenceSession(
                model_path,
                providers=[
                    "CPUExecutionProvider"
                ]
            )
        )

        self.input_name = (
            self.session
            .get_inputs()[0]
            .name
        )

        self.output_name = (
            self.session
            .get_outputs()[0]
            .name
        )

        log.info(
            f"YOLO loaded: "
            f"input="
            f"{self.session.get_inputs()[0].shape}, "
            f"output="
            f"{self.session.get_outputs()[0].shape}"
        )

    # ========================================================
    # PREPROCESS
    # ========================================================

    def preprocess(
        self,
        frame
    ):

        original_h, original_w = (
            frame.shape[:2]
        )

        scale = min(
            YOLO_SIZE / original_w,
            YOLO_SIZE / original_h
        )

        new_w = int(
            original_w *
            scale
        )

        new_h = int(
            original_h *
            scale
        )

        resized = cv2.resize(
            frame,
            (
                new_w,
                new_h
            ),
            interpolation=cv2.INTER_LINEAR
        )

        canvas = np.full(
            (
                YOLO_SIZE,
                YOLO_SIZE,
                3
            ),
            114,
            dtype=np.uint8
        )

        pad_x = (
            YOLO_SIZE -
            new_w
        ) // 2

        pad_y = (
            YOLO_SIZE -
            new_h
        ) // 2

        canvas[
            pad_y:
            pad_y + new_h,
            pad_x:
            pad_x + new_w
        ] = resized

        image = cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2RGB
        )

        image = (
            image.astype(
                np.float32
            )
            /
            255.0
        )

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image = np.expand_dims(
            image,
            axis=0
        )

        return (
            image,
            scale,
            pad_x,
            pad_y
        )

    # ========================================================
    # DETECT
    # ========================================================

    def detect(
        self,
        frame
    ):

        original_h, original_w = (
            frame.shape[:2]
        )

        (
            input_tensor,
            scale,
            pad_x,
            pad_y
        ) = self.preprocess(
            frame
        )

        outputs = self.session.run(
            [self.output_name],
            {
                self.input_name:
                input_tensor
            }
        )

        output = outputs[0]

        if output.ndim == 3:

            output = output[0]

        if (
            output.ndim == 2
            and
            output.shape[0]
            <
            output.shape[1]
        ):

            output = output.T

        detections = output

        results = []

        for detection in detections:

            if len(detection) < 6:

                continue

            # ------------------------------------------------
            # YOLOv8 output
            # ------------------------------------------------

            if len(detection) > 6:

                cx, cy, bw, bh = (
                    detection[:4]
                )

                class_scores = (
                    detection[4:]
                )

                class_id = int(
                    np.argmax(
                        class_scores
                    )
                )

                confidence = float(
                    class_scores[
                        class_id
                    ]
                )

                if (
                    confidence <
                    YOLO_CONFIDENCE
                ):

                    continue

                if (
                    class_id !=
                    PERSON_CLASS_ID
                ):

                    continue

                x1_model = (
                    float(cx) -
                    float(bw) / 2.0
                )

                y1_model = (
                    float(cy) -
                    float(bh) / 2.0
                )

                x2_model = (
                    float(cx) +
                    float(bw) / 2.0
                )

                y2_model = (
                    float(cy) +
                    float(bh) / 2.0
                )

            # ------------------------------------------------
            # Legacy six-column output
            # ------------------------------------------------

            else:

                x1_model = float(
                    detection[0]
                )

                y1_model = float(
                    detection[1]
                )

                x2_model = float(
                    detection[2]
                )

                y2_model = float(
                    detection[3]
                )

                confidence = float(
                    detection[4]
                )

                class_id = int(
                    detection[5]
                )

                if (
                    confidence <
                    YOLO_CONFIDENCE
                ):

                    continue

                if (
                    class_id !=
                    PERSON_CLASS_ID
                ):

                    continue

            # ------------------------------------------------
            # Undo letterbox
            # ------------------------------------------------

            x1 = (
                x1_model -
                pad_x
            ) / scale

            y1 = (
                y1_model -
                pad_y
            ) / scale

            x2 = (
                x2_model -
                pad_x
            ) / scale

            y2 = (
                y2_model -
                pad_y
            ) / scale

            x1 = int(x1)
            y1 = int(y1)

            x2 = int(x2)
            y2 = int(y2)

            x1 = max(
                0,
                min(
                    original_w - 1,
                    x1
                )
            )

            y1 = max(
                0,
                min(
                    original_h - 1,
                    y1
                )
            )

            x2 = max(
                0,
                min(
                    original_w - 1,
                    x2
                )
            )

            y2 = max(
                0,
                min(
                    original_h - 1,
                    y2
                )
            )

            width = (
                x2 -
                x1
            )

            height = (
                y2 -
                y1
            )

            if (
                width <
                MIN_BOX_WIDTH
            ):

                continue

            if (
                height <
                MIN_BOX_HEIGHT
            ):

                continue

            results.append(
                {
                    "bbox": (
                        x1,
                        y1,
                        width,
                        height
                    ),
                    "confidence":
                    confidence
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

        self.state = "SEARCHING"

        self.last_log_time = 0.0

        self.last_detection_time = 0.0

        self.failure_start_time = None

    # ========================================================
    # RELEASE
    # ========================================================

    def release_target(
        self,
        state="SEARCHING"
    ):

        self.tracker = None

        self.locked = False

        self.bbox = None

        self.confidence = 0.0

        self.missed_frames = 0

        self.failure_start_time = None

        self.state = state

    # ========================================================
    # IoU
    # ========================================================

    @staticmethod
    def calculate_iou(
        box_a,
        box_b
    ):

        if (
            box_a is None
            or
            box_b is None
        ):

            return 0.0

        ax, ay, aw, ah = box_a

        bx, by, bw, bh = box_b

        ax2 = ax + aw
        ay2 = ay + ah

        bx2 = bx + bw
        by2 = by + bh

        ix1 = max(
            ax,
            bx
        )

        iy1 = max(
            ay,
            by
        )

        ix2 = min(
            ax2,
            bx2
        )

        iy2 = min(
            ay2,
            by2
        )

        iw = max(
            0,
            ix2 - ix1
        )

        ih = max(
            0,
            iy2 - iy1
        )

        intersection = (
            iw *
            ih
        )

        area_a = (
            aw *
            ah
        )

        area_b = (
            bw *
            bh
        )

        union = (
            area_a +
            area_b -
            intersection
        )

        if union <= 0:

            return 0.0

        return (
            intersection /
            union
        )

    # ========================================================
    # VALIDATE BBOX
    # ========================================================

    def validate_tracker_bbox(
        self,
        bbox,
        frame
    ):

        if bbox is None:

            return False

        x, y, w, h = [
            int(v)
            for v in bbox
        ]

        frame_h, frame_w = (
            frame.shape[:2]
        )

        if (
            w <
            MIN_BOX_WIDTH
        ):

            return False

        if (
            h <
            MIN_BOX_HEIGHT
        ):

            return False

        if x < 0 or y < 0:

            return False

        if (
            x >= frame_w
            or
            y >= frame_h
        ):

            return False

        if (
            w >
            frame_w *
            MAX_BOX_WIDTH_RATIO
        ):

            return False

        if (
            h >
            frame_h *
            MAX_BOX_HEIGHT_RATIO
        ):

            return False

        area_ratio = (
            (w * h)
            /
            float(
                frame_w *
                frame_h
            )
        )

        if (
            area_ratio >
            MAX_BOX_AREA_RATIO
        ):

            return False

        return True

    # ========================================================
    # LOCK TARGET
    # ========================================================

    def lock_target(
        self,
        frame,
        bbox,
        confidence
    ):

        if not self.validate_tracker_bbox(
            bbox,
            frame
        ):

            return False

        x, y, w, h = [
            int(v)
            for v in bbox
        ]

        x = max(
            0,
            min(
                frame.shape[1] - 1,
                x
            )
        )

        y = max(
            0,
            min(
                frame.shape[0] - 1,
                y
            )
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
            w <
            MIN_BOX_WIDTH
            or
            h <
            MIN_BOX_HEIGHT
        ):

            return False

        try:

            new_tracker = (
                create_kcf_tracker()
            )

            new_tracker.init(
                frame,
                (
                    x,
                    y,
                    w,
                    h
                )
            )

            self.tracker = (
                new_tracker
            )

        except Exception as e:

            log.error(
                f"KCF initialization failed: {e}"
            )

            return False

        self.bbox = (
            x,
            y,
            w,
            h
        )

        self.confidence = (
            confidence
        )

        self.locked = True

        self.state = "TRACKING"

        self.missed_frames = 0

        self.failure_start_time = None

        self.last_detection_time = (
            time.monotonic()
        )

        log.info(
            f"TARGET LOCKED "
            f"confidence={confidence:.2f} "
            f"bbox={self.bbox}"
        )

        return True

    # ========================================================
    # UPDATE KCF
    # ========================================================

    def update_tracker(
        self,
        frame
    ):

        if (
            not self.locked
            or
            self.tracker is None
        ):

            return False

        try:

            success, bbox = (
                self.tracker.update(
                    frame
                )
            )

        except Exception as e:

            log.error(
                f"KCF update error: {e}"
            )

            success = False

            bbox = None

        # ----------------------------------------------------
        # KCF FAILURE
        # ----------------------------------------------------

        if not success:

            self.missed_frames += 1

            if (
                self.failure_start_time
                is None
            ):

                self.failure_start_time = (
                    time.monotonic()
                )

            return self.handle_tracker_failure()

        candidate_bbox = tuple(
            int(v)
            for v in bbox
        )

        if not self.validate_tracker_bbox(
            candidate_bbox,
            frame
        ):

            self.missed_frames += 1

            if (
                self.failure_start_time
                is None
            ):

                self.failure_start_time = (
                    time.monotonic()
                )

            return self.handle_tracker_failure()

        x, y, w, h = (
            candidate_bbox
        )

        x = max(
            0,
            min(
                frame.shape[1] - 1,
                x
            )
        )

        y = max(
            0,
            min(
                frame.shape[0] - 1,
                y
            )
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

        self.failure_start_time = None

        self.state = "TRACKING"

        return True

    # ========================================================
    # KCF FAILURE HANDLING
    # ========================================================

    def handle_tracker_failure(self):

        now = time.monotonic()

        # Keep the physical KCF lock during the grace period.
        if (
            self.failure_start_time
            is not None
            and
            (
                now -
                self.failure_start_time
            )
            <
            KCF_LOST_GRACE_TIME
            and
            self.missed_frames
            <
            MAX_TRACKER_MISSES
        ):

            self.state = (
                "TRACKING-UNCERTAIN"
            )

            return False

        self.release_target(
            state="TARGET-LOST"
        )

        log.warning(
            "KCF target genuinely lost."
        )

        return False

    # ========================================================
    # YOLO CORRECTION
    # ========================================================
    #
    # Once the target colour has been learned, YOLO detections
    # must also match that colour before they can influence KCF.
    #
    # Moderate IoU validates KCF.
    #
    # Strong IoU can reinitialize KCF.
    # ========================================================

    def correct_with_yolo(
        self,
        detections,
        frame,
        colour_identifier=None
    ):

        if not detections:

            return False

        if (
            not self.locked
            or
            self.bbox is None
        ):

            return False

        current = self.bbox

        best = None

        best_iou = 0.0

        for detection in detections:

            # Identity gate.
            if (
                colour_identifier is not None
                and
                colour_identifier.locked
            ):

                colour = (
                    colour_identifier
                    .extract_colour(
                        frame,
                        detection["bbox"]
                    )
                )

                if colour is None:

                    continue

                colour_score = (
                    colour_identifier
                    .colour_similarity(
                        colour
                    )
                )

                if (
                    colour_score
                    <
                    COLOUR_WEAK_MATCH_THRESHOLD
                ):

                    continue

            iou = self.calculate_iou(
                current,
                detection["bbox"]
            )

            if iou > best_iou:

                best_iou = iou

                best = detection

        if (
            best is None
            or
            best_iou <
            YOLO_CORRECTION_IOU
        ):

            return False

        self.last_detection_time = (
            time.monotonic()
        )

        self.missed_frames = 0

        self.failure_start_time = None

        # Moderate overlap:
        # validate without moving the KCF box.
        if (
            best_iou <
            YOLO_STRONG_IOU
        ):

            self.state = "TRACKING"

            return True

        # Strong overlap:
        # safely reinitialize KCF.
        bbox = best["bbox"]

        if not self.validate_tracker_bbox(
            bbox,
            frame
        ):

            return True

        try:

            x, y, w, h = [
                int(v)
                for v in bbox
            ]

            new_tracker = (
                create_kcf_tracker()
            )

            new_tracker.init(
                frame,
                (
                    x,
                    y,
                    w,
                    h
                )
            )

            self.tracker = (
                new_tracker
            )

            self.bbox = (
                x,
                y,
                w,
                h
            )

            self.confidence = (
                best["confidence"]
            )

            self.state = "TRACKING"

            log.info(
                f"KCF corrected by strong "
                f"YOLO match IoU={best_iou:.2f}"
            )

        except Exception as e:

            log.error(
                f"KCF correction failed: {e}"
            )

        return True


# ============================================================
# ASYNCHRONOUS YOLO WORKER
# ============================================================

class YOLOWorker:

    def __init__(
        self,
        detector
    ):

        self.detector = detector

        self.frame = None

        self.frame_request_time = 0.0

        self.result = []

        self.result_time = 0.0

        self.result_request_time = 0.0

        self.requested = False

        self.running = False

        self.lock = threading.Lock()

        self.event = threading.Event()

        self.thread = None

        self.inference_count = 0

        self.last_inference_duration = 0.0

    # ========================================================
    # START
    # ========================================================

    def start(self):

        self.running = True

        self.thread = threading.Thread(
            target=self.worker_loop,
            daemon=True,
            name="YOLOWorker"
        )

        self.thread.start()

    # ========================================================
    # SUBMIT LATEST FRAME
    # ========================================================

    def submit(
        self,
        frame
    ):

        with self.lock:

            self.frame = frame.copy()

            self.frame_request_time = (
                time.monotonic()
            )

            self.requested = True

        self.event.set()

    # ========================================================
    # WORKER LOOP
    # ========================================================

    def worker_loop(self):

        log.info(
            "YOLO worker started."
        )

        while self.running:

            self.event.wait(
                timeout=0.10
            )

            self.event.clear()

            if not self.running:

                break

            with self.lock:

                if (
                    not self.requested
                    or
                    self.frame is None
                ):

                    continue

                frame = self.frame

                request_time = (
                    self.frame_request_time
                )

                self.frame = None

                self.requested = False

            try:

                start = (
                    time.monotonic()
                )

                detections = (
                    self.detector.detect(
                        frame
                    )
                )

                duration = (
                    time.monotonic()
                    -
                    start
                )

                result_time = (
                    time.monotonic()
                )

                with self.lock:

                    self.result = (
                        detections
                    )

                    self.result_time = (
                        result_time
                    )

                    self.result_request_time = (
                        request_time
                    )

                    self.last_inference_duration = (
                        duration
                    )

                    self.inference_count += 1

            except Exception as e:

                log.error(
                    f"YOLO worker error: {e}"
                )

        log.info(
            "YOLO worker stopped."
        )

    # ========================================================
    # GET RESULT
    # ========================================================

    def get_result(self):

        with self.lock:

            return (
                list(self.result),
                self.result_time,
                self.result_request_time
            )

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        self.running = False

        self.event.set()

        if (
            self.thread
            is not None
        ):

            self.thread.join(
                timeout=2.0
            )


# ============================================================
# ARDUINO CONTROLLER
# ============================================================

class ArduinoController:

    def __init__(self):

        self.ser = None

        self.current_command = "0"

        self.running = False

        self.lock = threading.Lock()

        self.heartbeat_thread = None

        try:

            self.ser = serial.Serial(
                ARDUINO_PORT,
                ARDUINO_BAUDRATE,
                timeout=ARDUINO_TIMEOUT,
                write_timeout=0.5
            )

            time.sleep(2)

            log.info(
                f"Arduino connected on "
                f"{ARDUINO_PORT}"
            )

            self._send("0")

            self.running = True

            self.heartbeat_thread = (
                threading.Thread(
                    target=self._heartbeat_loop,
                    daemon=True,
                    name="ArduinoHeartbeat"
                )
            )

            self.heartbeat_thread.start()

        except Exception as e:

            log.error(
                f"Could not connect to Arduino: {e}"
            )

            self.ser = None

    # ========================================================
    # SEND
    # ========================================================

    def _send(
        self,
        command
    ):

        if self.ser is None:

            return

        try:

            self.ser.write(
                (
                    command +
                    "\n"
                ).encode(
                    "utf-8"
                )
            )

            self.ser.flush()

        except Exception as e:

            log.error(
                f"Arduino communication error: {e}"
            )

    # ========================================================
    # SET MOVEMENT STATE
    # ========================================================

    def set_state(
        self,
        tracking
    ):

        new_command = (
            "1"
            if tracking
            else
            "0"
        )

        with self.lock:

            if (
                new_command ==
                self.current_command
            ):

                return

            self.current_command = (
                new_command
            )

        log.info(
            f"Arduino command changed: "
            f"{new_command} "
            f"("
            f"{'TRACKING/MOVE' if tracking else 'STOP'}"
            f")"
        )

    # ========================================================
    # HEARTBEAT
    # ========================================================

    def _heartbeat_loop(self):

        heartbeat_count = 0

        while self.running:

            with self.lock:

                command = (
                    self.current_command
                )

            self._send(
                command
            )

            heartbeat_count += 1

            if (
                heartbeat_count %
                25
                ==
                0
            ):

                log.info(
                    f"Arduino heartbeat active: "
                    f"command={command}"
                )

            time.sleep(
                ARDUINO_HEARTBEAT_INTERVAL
            )

    # ========================================================
    # IMMEDIATE STOP
    # ========================================================

    def send_immediate_stop(self):

        with self.lock:

            self.current_command = "0"

        self._send("0")

        log.info(
            "Arduino STOP command sent."
        )

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        try:

            self.send_immediate_stop()

            self.running = False

            if (
                self.heartbeat_thread
                is not None
            ):

                self.heartbeat_thread.join(
                    timeout=1.0
                )

            self._send("0")

            time.sleep(
                0.1
            )

            if (
                self.ser
                is not None
            ):

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

        # ----------------------------------------------------
        # AI
        # ----------------------------------------------------

        self.detector = (
            YOLODetector(
                MODEL_PATH
            )
        )

        self.target_tracker = (
            TargetTracker()
        )

        self.colour_identifier = (
            ColourTargetIdentifier()
        )

        self.yolo_worker = (
            YOLOWorker(
                self.detector
            )
        )

        # ----------------------------------------------------
        # ARDUINO
        # ----------------------------------------------------

        self.arduino = (
            ArduinoController()
        )

        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        self.camera = (
            Picamera2()
        )

        config = (
            self.camera
            .create_preview_configuration(
                main={
                    "size": (
                        CAMERA_WIDTH,
                        CAMERA_HEIGHT
                    ),
                    "format":
                    "BGR888"
                },
                buffer_count=4
            )
        )

        self.camera.configure(
            config
        )

        # ----------------------------------------------------
        # RUNTIME
        # ----------------------------------------------------

        self.running = False

        self.frame_count = 0

        self.last_frame_time = (
            time.monotonic()
        )

        self.fps = 0.0

        # ----------------------------------------------------
        # TIMERS
        # ----------------------------------------------------

        now = time.monotonic()

        self.last_yolo_request = (
            now -
            YOLO_INTERVAL
        )

        self.last_reacquisition_request = (
            now -
            YOLO_REACQUISITION_INTERVAL
        )

        self.last_yolo_result_time = 0.0

        self.last_kcf_update = now

        self.last_debug_save = now

        # ----------------------------------------------------
        # REACQUISITION
        # ----------------------------------------------------

        self.reacquisition_active = False

        self.reacquisition_start_time = 0.0

        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------

        self.loop_counter = 0

        self.loop_cpu_time = 0.0

        log.info(
            f"Camera configured "
            f"({CAMERA_WIDTH}x{CAMERA_HEIGHT})"
        )

    # ========================================================
    # START
    # ========================================================

    def start(self):

        self.camera.start()

        time.sleep(1)

        self.yolo_worker.start()

        self.running = True

        now = time.monotonic()

        self.last_yolo_request = (
            now -
            YOLO_INTERVAL
        )

        self.last_reacquisition_request = (
            now -
            YOLO_REACQUISITION_INTERVAL
        )

        self.last_kcf_update = now

        self.last_debug_save = now

        log.info(
            "Camera started."
        )

        log.info(
            "Smart Cart visual tracker started."
        )

        log.info(
            "KCF PRIMARY tracking mode ENABLED."
        )

        log.info(
            "YOLO ASYNCHRONOUS validation/reacquisition mode ENABLED."
        )

    # ========================================================
    # REQUEST NORMAL YOLO
    # ========================================================

    def request_yolo(
        self,
        frame
    ):

        now = time.monotonic()

        if (
            now -
            self.last_yolo_request
            >=
            YOLO_INTERVAL
        ):

            self.last_yolo_request = now

            self.yolo_worker.submit(
                frame
            )

    # ========================================================
    # REQUEST RAPID YOLO
    # ========================================================

    def request_reacquisition_yolo(
        self,
        frame
    ):

        now = time.monotonic()

        if (
            now -
            self.last_reacquisition_request
            >=
            YOLO_REACQUISITION_INTERVAL
        ):

            self.last_reacquisition_request = (
                now
            )

            self.yolo_worker.submit(
                frame
            )

    # ========================================================
    # ENTER REACQUISITION
    # ========================================================

    def enter_reacquisition(self):

        if self.reacquisition_active:

            return

        self.reacquisition_active = True

        self.reacquisition_start_time = (
            time.monotonic()
        )

        self.target_tracker.state = (
            "REACQUIRING"
        )

        self.arduino.send_immediate_stop()

        log.warning(
            "Entering rapid YOLO target reacquisition."
        )

    # ========================================================
    # RELEASE PHYSICAL TARGET
    # ========================================================

    def release_physical_target(
        self,
        reason
    ):

        log.warning(
            f"Physical target released: "
            f"{reason}"
        )

        self.target_tracker.release_target(
            state="TARGET-LOST"
        )

        # Learned colour is intentionally preserved.
        self.arduino.send_immediate_stop()

        if (
            self.colour_identifier.locked
        ):

            self.enter_reacquisition()

    # ========================================================
    # COMPLETE REACQUISITION
    # ========================================================

    def complete_reacquisition(self):

        self.reacquisition_active = False

        self.target_tracker.state = (
            "TRACKING"
        )

        self.last_kcf_update = (
            time.monotonic()
        )

        log.info(
            "Target reacquisition completed."
        )

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(self):

        loop_start = (
            time.monotonic()
        )

        # ----------------------------------------------------
        # CAPTURE
        # ----------------------------------------------------

        frame = (
            self.camera.capture_array()
        )

        frame = cv2.rotate(
            frame,
            cv2.ROTATE_180
        )

        self.frame_count += 1

        now = time.monotonic()

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        elapsed = (
            now -
            self.last_frame_time
        )

        if elapsed > 0:

            instant_fps = (
                1.0 /
                elapsed
            )

            if self.fps == 0:

                self.fps = instant_fps

            else:

                self.fps = (
                    self.fps * 0.9
                    +
                    instant_fps * 0.1
                )

        self.last_frame_time = now

        # ====================================================
        # 1. KCF PRIMARY TRACKING
        # ====================================================

        if (
            self.target_tracker.locked
            and
            now -
            self.last_kcf_update
            >=
            KCF_INTERVAL
        ):

            self.last_kcf_update = now

            kcf_success = (
                self.target_tracker
                .update_tracker(
                    frame
                )
            )

            if (
                not kcf_success
                and
                self.target_tracker.state
                ==
                "TARGET-LOST"
            ):

                self.enter_reacquisition()

        # ====================================================
        # 2. YOLO REQUEST
        # ====================================================

        if self.reacquisition_active:

            self.request_reacquisition_yolo(
                frame
            )

        else:

            self.request_yolo(
                frame
            )

        # ====================================================
        # 3. COLOUR PROCESSING
        # ====================================================

        if (
            self.target_tracker.locked
            and
            self.target_tracker.bbox
            is not None
        ):

            bbox = (
                self.target_tracker.bbox
            )

            # ------------------------------------------------
            # Initial colour learning.
            #
            # This does NOT stop movement.
            # ------------------------------------------------

            if (
                not
                self.colour_identifier.locked
            ):

                if (
                    self.colour_identifier.state
                    ==
                    "NOT LOCKED"
                ):

                    self.colour_identifier.begin_acquisition()

                self.colour_identifier.sample(
                    frame,
                    bbox
                )

            # ------------------------------------------------
            # Colour verification.
            #
            # Do not verify during a short KCF recovery.
            # ------------------------------------------------

            elif (
                self.target_tracker.state
                !=
                "TRACKING-UNCERTAIN"
            ):

                colour_ok = (
                    self.colour_identifier
                    .verify(
                        frame,
                        bbox
                    )
                )

                if not colour_ok:

                    self.release_physical_target(
                        "target colour identity confidence lost"
                    )

        # ====================================================
        # 4. GET YOLO RESULT
        # ====================================================

        (
            detections,
            result_time,
            request_time
        ) = (
            self.yolo_worker.get_result()
        )

        # ====================================================
        # 5. PROCESS NEW YOLO RESULT
        # ====================================================

        if (
            result_time >
            self.last_yolo_result_time
        ):

            self.last_yolo_result_time = (
                result_time
            )

            result_age = (
                now -
                result_time
            )

            # =================================================
            # RAPID REACQUISITION
            # =================================================

            if self.reacquisition_active:

                if (
                    result_age
                    <=
                    MAX_REACQUISITION_RESULT_AGE
                ):

                    selected = (
                        self.select_colour_target(
                            detections,
                            frame
                        )
                    )

                    if selected is not None:

                        locked = (
                            self.target_tracker
                            .lock_target(
                                frame,
                                selected["bbox"],
                                selected["confidence"]
                            )
                        )

                        if locked:

                            log.info(
                                "TARGET REACQUIRED "
                                "using learned colour."
                            )

                            self.complete_reacquisition()

                    elif detections:

                        log.info(
                            "Reacquisition: people detected, "
                            "but none match learned colour."
                        )

            # =================================================
            # NORMAL OPERATION
            # =================================================

            elif (
                result_age
                <=
                MAX_YOLO_RESULT_AGE
            ):

                # ---------------------------------------------
                # NO KCF TARGET
                # ---------------------------------------------

                if not self.target_tracker.locked:

                    selected = (
                        self.select_colour_target(
                            detections,
                            frame
                        )
                    )

                    if selected is not None:

                        locked = (
                            self.target_tracker
                            .lock_target(
                                frame,
                                selected["bbox"],
                                selected["confidence"]
                            )
                        )

                        if (
                            locked
                            and
                            not
                            self.colour_identifier.locked
                        ):

                            self.colour_identifier.begin_acquisition()

                            log.info(
                                "Initial human target acquired. "
                                "Beginning colour sampling."
                            )

                # ---------------------------------------------
                # KCF TARGET EXISTS
                # ---------------------------------------------

                elif detections:

                    self.target_tracker.correct_with_yolo(
                        detections,
                        frame,
                        self.colour_identifier
                    )

        # ====================================================
        # 6. GENUINE TARGET LOSS
        # ====================================================

        if (
            not self.target_tracker.locked
            and
            self.colour_identifier.locked
            and
            not self.reacquisition_active
        ):

            self.enter_reacquisition()

        # ====================================================
        # 7. MOTOR PERMISSION
        # ====================================================
        #
        # This is the major smoothing change.
        #
        # TRACKING-UNCERTAIN still retains the physical lock
        # during the KCF recovery grace period.
        #
        # Therefore:
        #
        # TRACKING            -> FOLLOW
        # TRACKING-UNCERTAIN  -> FOLLOW
        # TARGET-LOST         -> STOP
        # REACQUIRING         -> STOP
        #
        # Initial colour sampling also does not stop movement.
        # ====================================================

        movement_allowed = (
            self.target_tracker.locked
            and
            self.target_tracker.bbox
            is not None
            and
            not self.reacquisition_active
            and
            self.target_tracker.state
            in
            (
                "TRACKING",
                "TRACKING-UNCERTAIN"
            )
        )

        # ====================================================
        # 8. DIAGNOSTICS
        # ====================================================

        if (
            now -
            self.target_tracker.last_log_time
            >=
            LOG_INTERVAL
        ):

            self.target_tracker.last_log_time = now

            if (
                self.target_tracker.bbox
                is not None
            ):

                x, y, w, h = (
                    self.target_tracker.bbox
                )

                center_x = (
                    x +
                    w // 2
                )

                center_y = (
                    y +
                    h // 2
                )

                log.info(
                    f"State="
                    f"{self.target_tracker.state} "
                    f"FPS={self.fps:.1f} "
                    f"bbox="
                    f"({x},{y},{w},{h}) "
                    f"center="
                    f"({center_x},{center_y}) "
                    f"miss="
                    f"{self.target_tracker.missed_frames} "
                    f"colour="
                    f"{self.colour_identifier.state} "
                    f"colour_conf="
                    f"{self.colour_identifier.colour_confidence:.2f} "
                    f"motor="
                    f"{'FOLLOW' if movement_allowed else 'STOP'}"
                )

            else:

                log.info(
                    f"State="
                    f"{self.target_tracker.state} "
                    f"FPS={self.fps:.1f} "
                    f"colour="
                    f"{self.colour_identifier.state} "
                    f"colour_conf="
                    f"{self.colour_identifier.colour_confidence:.2f} "
                    f"motor="
                    f"{'FOLLOW' if movement_allowed else 'STOP'}"
                )

        # ====================================================
        # 9. MOTOR CONTROL
        # ====================================================

        self.arduino.set_state(
            movement_allowed
        )

        # ====================================================
        # 10. DEBUG IMAGE
        # ====================================================

        if (
            DEBUG_SAVE_ENABLED
            and
            now -
            self.last_debug_save
            >=
            DEBUG_SAVE_INTERVAL
        ):

            self.last_debug_save = now

            self.save_debug_frame(
                frame,
                detections
            )

        # ====================================================
        # PERFORMANCE
        # ====================================================

        processing_time = (
            time.monotonic()
            -
            loop_start
        )

        self.loop_cpu_time += (
            processing_time
        )

        self.loop_counter += 1

        return frame

    # ========================================================
    # SELECT TARGET
    # ========================================================

    def select_colour_target(
        self,
        detections,
        frame
    ):

        if not detections:

            return None

        # ----------------------------------------------------
        # INITIAL TARGET
        # ----------------------------------------------------

        if not self.colour_identifier.locked:

            selected = max(
                detections,
                key=lambda d:
                d["confidence"]
            )

            log.info(
                "Initial human target acquisition: "
                f"confidence="
                f"{selected['confidence']:.2f}"
            )

            return selected

        # ----------------------------------------------------
        # REACQUISITION
        # ----------------------------------------------------

        best = None

        best_score = -1.0

        for detection in detections:

            colour = (
                self.colour_identifier
                .extract_colour(
                    frame,
                    detection["bbox"]
                )
            )

            if colour is None:

                continue

            colour_score = (
                self.colour_identifier
                .colour_similarity(
                    colour
                )
            )

            if (
                colour_score
                <
                COLOUR_WEAK_MATCH_THRESHOLD
            ):

                continue

            score = (
                colour_score * 0.75
                +
                detection["confidence"] * 0.25
            )

            if score > best_score:

                best_score = score

                best = detection

        if best is None:

            return None

        log.info(
            "Matching target colour found: "
            f"score={best_score:.2f} "
            f"confidence="
            f"{best['confidence']:.2f}"
        )

        return best

    # ========================================================
    # MOTOR CONTROL
    # ========================================================

    def update_motor_control(self):

        movement_allowed = (
            self.target_tracker.locked
            and
            self.target_tracker.bbox
            is not None
            and
            not self.reacquisition_active
            and
            self.target_tracker.state
            in
            (
                "TRACKING",
                "TRACKING-UNCERTAIN"
            )
        )

        self.arduino.set_state(
            movement_allowed
        )

    # ========================================================
    # DEBUG FRAME
    # ========================================================

    def save_debug_frame(
        self,
        frame,
        detections
    ):

        try:

            os.makedirs(
                DEBUG_DIRECTORY,
                exist_ok=True
            )

            debug = frame.copy()

            frame_h, frame_w = (
                debug.shape[:2]
            )

            # ------------------------------------------------
            # YOLO DETECTIONS
            # ------------------------------------------------

            for detection in detections:

                x, y, w, h = (
                    detection["bbox"]
                )

                confidence = (
                    detection["confidence"]
                )

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

                cv2.putText(
                    debug,
                    f"YOLO {confidence:.2f}",
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

            # ------------------------------------------------
            # KCF TARGET
            # ------------------------------------------------

            if (
                self.target_tracker.bbox
                is not None
            ):

                x, y, w, h = (
                    self.target_tracker.bbox
                )

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

                cv2.putText(
                    debug,
                    "KCF TARGET",
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

                # ------------------------------------------------
                # COLOUR REGION
                # ------------------------------------------------

                rx1 = int(
                    x +
                    w *
                    COLOUR_REGION_X1
                )

                rx2 = int(
                    x +
                    w *
                    COLOUR_REGION_X2
                )

                ry1 = int(
                    y +
                    h *
                    COLOUR_REGION_Y1
                )

                ry2 = int(
                    y +
                    h *
                    COLOUR_REGION_Y2
                )

                cv2.rectangle(
                    debug,
                    (rx1, ry1),
                    (rx2, ry2),
                    (0, 255, 255),
                    2
                )

            # ------------------------------------------------
            # MOTOR STATUS
            # ------------------------------------------------

            movement_allowed = (
                self.target_tracker.locked
                and
                self.target_tracker.bbox
                is not None
                and
                not self.reacquisition_active
                and
                self.target_tracker.state
                in
                (
                    "TRACKING",
                    "TRACKING-UNCERTAIN"
                )
            )

            motor_status = (
                "FOLLOW"
                if movement_allowed
                else
                "STOP"
            )

            # ------------------------------------------------
            # COLOUR STATUS
            # ------------------------------------------------

            if (
                self.colour_identifier.locked
            ):

                colour_text = (
                    f"{self.colour_identifier.target_h:.0f},"
                    f"{self.colour_identifier.target_s:.0f},"
                    f"{self.colour_identifier.target_v:.0f}"
                )

            else:

                colour_text = (
                    "NOT LOCKED"
                )

            # ------------------------------------------------
            # REACQUISITION STATUS
            # ------------------------------------------------

            reacquisition_text = (
                "ACTIVE"
                if self.reacquisition_active
                else
                "NO"
            )

            # ------------------------------------------------
            # STATUS PANEL
            # ------------------------------------------------

            panel_lines = [

                f"STATE: "
                f"{self.target_tracker.state}",

                f"MOTOR: "
                f"{motor_status}",

                f"COLOUR: "
                f"{self.colour_identifier.state}",

                f"COLOUR CONF: "
                f"{self.colour_identifier.colour_confidence:.2f}",

                f"TARGET HSV: "
                f"{colour_text}",

                f"SAMPLES: "
                f"{self.colour_identifier.sample_count}/"
                f"{COLOUR_SAMPLE_COUNT}",

                f"YOLO PERSONS: "
                f"{len(detections)}",

                f"REACQUIRE: "
                f"{reacquisition_text}",

                f"FPS: "
                f"{self.fps:.1f}",

                f"YOLO: "
                f"{self.yolo_worker.last_inference_duration:.2f}s",

                f"KCF MISSES: "
                f"{self.target_tracker.missed_frames}",

                "BLUE = YOLO",

                "GREEN = KCF",

                "YELLOW = COLOUR"
            ]

            panel_x = 10

            panel_y = 25

            line_spacing = 20

            for index, text in enumerate(
                panel_lines
            ):

                y_position = (
                    panel_y
                    +
                    index *
                    line_spacing
                )

                cv2.putText(
                    debug,
                    text,
                    (
                        panel_x,
                        y_position
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    2
                )

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

            filename = os.path.join(
                DEBUG_DIRECTORY,
                f"debug_"
                f"{self.frame_count:06d}.jpg"
            )

            cv2.imwrite(
                filename,
                debug,
                [
                    int(
                        cv2.IMWRITE_JPEG_QUALITY
                    ),
                    80
                ]
            )

        except Exception as e:

            log.error(
                f"Debug image save error: {e}"
            )

    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        self.start()

        try:

            while self.running:

                loop_start = (
                    time.monotonic()
                )

                self.process_frame()

                elapsed = (
                    time.monotonic()
                    -
                    loop_start
                )

                remaining = (
                    MAIN_LOOP_INTERVAL
                    -
                    elapsed
                )

                if remaining > 0:

                    time.sleep(
                        remaining
                    )

        except KeyboardInterrupt:

            log.info(
                "Keyboard interrupt."
            )

        finally:

            self.stop()

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        if not self.running:

            return

        self.running = False

        log.info(
            "Stopping..."
        )

        # ----------------------------------------------------
        # STOP MOTOR FIRST
        # ----------------------------------------------------

        try:

            self.arduino.send_immediate_stop()

            time.sleep(
                0.1
            )

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
