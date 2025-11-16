"""
Example unit test for the preprocessing logic.
"""
import numpy as np
from bsort.preprocess import _classify_color

# Dummy thresholds for testing
TEST_THRESHOLDS = {
  'light_blue_h': [90, 110],
  'dark_blue_h': [110, 130],
  'min_saturation': 70,
  'min_value': 70
}

def test_classify_color():
    """Tests the color classification logic."""
    
    # 0: light_blue
    light_blue_hsv = (100, 150, 150)
    assert _classify_color(light_blue_hsv, TEST_THRESHOLDS) == 0
    
    # 1: dark_blue
    dark_blue_hsv = (120, 150, 150)
    assert _classify_color(dark_blue_hsv, TEST_THRESHOLDS) == 1
    
    # 2: other (green)
    green_hsv = (60, 150, 150)
    assert _classify_color(green_hsv, TEST_THRESHOLDS) == 2
    
    # 2: other (red)
    red_hsv = (0, 150, 150)
    assert _classify_color(red_hsv, TEST_THRESHOLDS) == 2
    
    # 2: other (low saturation = gray)
    gray_hsv = (100, 20, 150)
    assert _classify_color(gray_hsv, TEST_THRESHOLDS) == 2
    
    # 2: other (low value = black)
    black_hsv = (100, 150, 20)
    assert _classify_color(black_hsv, TEST_THRESHOLDS) == 2