from enum import Enum


class DetectionType(Enum):
    NOTHING = 'Nobody'
    MASK_ON = 'Mask On'
    MASK_OFF = 'Mask Off'


class DetectionResult:
    def __init__(self, result: DetectionType, point_start=None, point_end=None):
        self.result = result
        self.point_start = point_start
        self.point_end = point_end

    def __repr__(self):
        return f'{self.result.value}: {self.point_start}, {self.point_end}'


def limit_detection_output(detection):
    return min(detection, 1)