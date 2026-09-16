"""
arduino.py

Reliable serial communication with Arduino.
"""

import time
import serial

from config import (
    SERIAL_PORT,
    BAUDRATE,
    RECONNECT_INTERVAL
)


class ArduinoInterface:

    def __init__(self, logger):

        self.logger = logger

        self.serial = None

        self.last_command = None

        self.last_attempt = 0

        self.connect()

    def connect(self):

        self.last_attempt = time.time()

        try:

            self.serial = serial.Serial(
                SERIAL_PORT,
                BAUDRATE,
                timeout=1
            )

            time.sleep(2)

            self.logger.info(
                f"Arduino connected ({SERIAL_PORT})"
            )

        except Exception as e:

            self.serial = None

            self.logger.warning(
                f"Arduino unavailable: {e}"
            )

    def reconnect(self):

        if self.serial is not None:
            return

        if time.time() - self.last_attempt < RECONNECT_INTERVAL:
            return

        self.logger.info("Attempting Arduino reconnect...")

        self.connect()

    def send(self, command):

        if self.serial is None:

            self.reconnect()

            return

        if command == self.last_command:
            return

        try:

            self.serial.write((command + "\n").encode())

            self.last_command = command

            self.logger.info(f"Arduino <- {command}")

        except Exception as e:

            self.logger.warning(
                f"Serial error: {e}"
            )

            try:
                self.serial.close()
            except Exception:
                pass

            self.serial = None

    def close(self):

        if self.serial is not None:

            try:
                self.serial.close()
            except Exception:
                pass

            self.serial = None

            self.logger.info("Arduino disconnected")
