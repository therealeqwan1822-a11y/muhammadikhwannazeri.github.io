# anti_spoof.py
import cv2
import numpy as np

class PresentationAttackDetector:
    """
    Presentation Attack Detector (PAD).
    Uses Gaussian-filtered FFT spectrum analysis on RAW face crops to detect 
    digital screen replay attacks (Moiré patterns) without false-triggering 
    on live human facial sharpness or camera ISO noise.
    """
    def __init__(self, fft_grid_threshold=52.0):
        self.fft_grid_threshold = fft_grid_threshold

    def inspect_face_roi(self, raw_face_bgr):
        """
        Inspects un-enhanced face crop for digital screen lattice artifacts.
        Returns: (is_spoof: bool, reason: str)
        """
        if raw_face_bgr is None or raw_face_bgr.size == 0:
            return False, "INVALID_ROI"

        # Standardize ROI dimensions
        resized = cv2.resize(raw_face_bgr, (128, 128))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized.copy()

        # 1. Smooth ISO webcam gain noise (preserves periodic screen subpixel grids)
        smoothed = cv2.GaussianBlur(gray, (3, 3), 0)

        # 2. 2D Fourier Transform
        f_transform = np.fft.fft2(smoothed.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        # Zero out low frequencies (DC component and center facial structures)
        cy, cx = 64, 64
        mag_high_freq = magnitude_spectrum.copy()
        mag_high_freq[cy - 16:cy + 16, cx - 16:cx + 16] = 0.0

        # Calculate high-frequency energy concentration ratio
        mean_power = np.mean(mag_high_freq) + 1e-6
        max_peak_power = np.max(mag_high_freq)
        peak_ratio = max_peak_power / mean_power

        # Screen displays (phones/monitors) exhibit distinct periodic lattice spikes (> 52.0)
        if peak_ratio > self.fft_grid_threshold:
            return True, f"SCREEN_DISPLAY_GRID ({peak_ratio:.1f} > {self.fft_grid_threshold})"

        return False, "LIVE_REAL_SUBJECT"