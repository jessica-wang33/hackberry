from gpiozero import Button
import time
from ADCDevice import *


class Joystick:
    X_CHANNEL = 1   # ADC channel 1 = X axis  (matches original main.py analogRead(1))
    Y_CHANNEL = 0   # ADC channel 0 = Y axis  (matches original main.py analogRead(0))
    Z_PIN     = 18
    CENTER    = 128
    DEADZONE  = 10  # ADC units either side of centre that count as "no movement"

    def __init__(self):
        self.button = Button(self.Z_PIN)
        self.adc = ADCDevice()
        if self.adc.detectI2C(0x48):
            self.adc = PCF8591()
        elif self.adc.detectI2C(0x4b):
            self.adc = ADS7830()
        else:
            raise RuntimeError(
                "No correct I2C address found. "
                "Run 'i2cdetect -y 1' to check the I2C address."
            )

    def read_raw(self) -> tuple[int, int, bool]:
        """Return (x_raw 0-255, y_raw 0-255, z_pressed bool)."""
        x = self.adc.analogRead(self.X_CHANNEL)
        y = self.adc.analogRead(self.Y_CHANNEL)
        z = not self.button.value
        return x, y, z

    def get_deflection(self) -> tuple[float, float]:
        """Return (dx, dy) each in -1.0…+1.0, with deadzone applied."""
        x, y, _ = self.read_raw()
        dx = x - self.CENTER
        dy = y - self.CENTER
        if abs(dx) < self.DEADZONE:
            dx = 0
        if abs(dy) < self.DEADZONE:
            dy = 0
        scale = self.CENTER - self.DEADZONE
        return (
            max(-1.0, min(1.0, dx / scale)),
            max(-1.0, min(1.0, dy / scale)),
        )

    def get_pan_delta(self, scale: float = 0.0005) -> tuple[float, float]:
        """Return (dlon, dlat) to add to current map centre each poll tick.

        scale=0.0005 at 10 Hz → full deflection ≈ 55 m/s at the equator,
        which feels responsive at zoom 11 (~100 m/pixel).
        Y axis is inverted: pushing up (ADC value decreases → dy < 0) → +lat (north).
        """
        dx, dy = self.get_deflection()
        return dx * scale, -dy * scale

    def close(self):
        self.adc.close()
        self.button.close()

    @staticmethod
    def test():
        """Print raw + normalised readings at 10 Hz until Ctrl+C."""
        j = Joystick()
        print("Joystick test — press Ctrl+C to stop")
        try:
            while True:
                x, y, z = j.read_raw()
                dx, dy = j.get_deflection()
                print(f"raw=({x:3d},{y:3d},{int(z)})  deflection=({dx:+.2f},{dy:+.2f})")
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            j.close()
