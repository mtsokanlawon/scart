"""
detector.py

YOLOv8 ONNX detector using ONNX Runtime.
Model exported with nms=True.
"""

import cv2
import numpy as np
import onnxruntime as ort

from config import (
    MODEL_PATH,
    PERSON_CLASS_ID,
    CONFIDENCE_THRESHOLD
)


class YOLODetector:

    def __init__(self, logger):

        self.logger = logger

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

        self.input_width = 640
        self.input_height = 640

        self.logger.info("YOLO detector initialized.")

    # -----------------------------------------------------

    def preprocess(self, frame):

        h, w = frame.shape[:2]

        scale = min(
            self.input_width / w,
            self.input_height / h
        )

        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h))

        canvas = np.full(
            (640, 640, 3),
            114,
            dtype=np.uint8
        )

        pad_x = (640 - new_w) // 2
        pad_y = (640 - new_h) // 2

        canvas[
            pad_y:pad_y + new_h,
            pad_x:pad_x + new_w
        ] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        rgb = rgb.astype(np.float32) / 255.0

        rgb = np.transpose(rgb, (2, 0, 1))

        rgb = np.expand_dims(rgb, axis=0)

        return rgb, scale, pad_x, pad_y

    # -----------------------------------------------------

    def detect(self, frame):

        tensor, scale, pad_x, pad_y = self.preprocess(frame)

        outputs = self.session.run(
            None,
            {self.input_name: tensor}
        )[0]

        if not hasattr(self, "_printed_sample"):
            self._printed_sample = True

            self.logger.info("First 5 raw detections:")

            for det in outputs[0][:5]:
                self.logger.info(det)

        detections = []

        for det in outputs[0]:

            x1, y1, x2, y2, conf, cls = det

            if conf < CONFIDENCE_THRESHOLD:
                continue

            if int(cls) != PERSON_CLASS_ID:
                continue

            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            x1 = max(0, min(frame.shape[1] - 1, x1))
            y1 = max(0, min(frame.shape[0] - 1, y1))
            x2 = max(0, min(frame.shape[1] - 1, x2))
            y2 = max(0, min(frame.shape[0] - 1, y2))

            detections.append({

                "bbox": (
                    int(x1),
                    int(y1),
                    int(x2 - x1),
                    int(y2 - y1)
                ),

                "confidence": float(conf)

            })

        return detections
