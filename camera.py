"""
camera.py

Picamera2 interface for the Smart Follower Shopping Cart.
"""

import cv2
from picamera2 import Picamera2

from config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_ROTATE_180
)


class Camera:

    def __init__(self, logger):

        self.logger = logger
        self.picam2 = None
        self.running = False

    def start(self):

        if self.running:
            return

        self.picam2 = Picamera2()

        config = self.picam2.create_preview_configuration(
            main={
                "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
                "format": "BGR888"
            }
        )

        self.picam2.configure(config)

        self.picam2.start()

        self.running = True

        self.logger.info(
            f"Camera started ({CAMERA_WIDTH}x{CAMERA_HEIGHT})"
        )

    def read(self):

        if not self.running:
            return None

        frame = self.picam2.capture_array()

        if CAMERA_ROTATE_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        return frame

    def stop(self):

        if self.picam2 is not None:

            self.picam2.stop()

            self.running = False

            self.logger.info("Camera stopped")
