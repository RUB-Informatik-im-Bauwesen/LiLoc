import json
from json import JSONEncoder
from typing import Any, Type

import cv2
import numpy as np
# =======
# Helpers
# =======


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class KeypointEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, cv2.KeyPoint):
            kp_dict = {"pt": obj.pt, "size": obj.size, "angle": obj.angle, "response": obj.response,
                       "octave": obj.octave, "class_id": obj.class_id}
            return kp_dict
        return JSONEncoder.default(self, obj)


class DMatchEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, cv2.DMatch):
            dm_dict = {"distance": obj.distance, "imgIdx": obj.imgIdx, "queryIdx": obj.queryIdx, "trainIdx": obj.trainIdx}
            return dm_dict
        return JSONEncoder.default(self, obj)


def multiencoder_factory(*encoders):
    class MultipleJsonEncoders(json.JSONEncoder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.encoders = [encoder(*args, **kwargs) for encoder in encoders]

        def default(self, o):
            for encoder in self.encoders:
                try:
                    return encoder.default(o)
                except TypeError:
                    pass
            return super().default(o)

    return MultipleJsonEncoders
