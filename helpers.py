import json
from json import JSONEncoder
from os import PathLike
from typing import Any, Type

import cv2
import numpy as np

import image_tools


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

class DataSource(dict):
    def __init__(self, matches_file: (str | PathLike), image_type="Localized", name=""):
        super().__init__()
        self.matches_file = matches_file
        self.update(self._load_data())
        self.image_type = image_type
        if name:
            self.name = name
        else:
            self.name = str(matches_file)
        self.data = None
        self._img_cache = {}

    def _load_data(self):
        data = {}
        with open(self.matches_file, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error parsing {self.matches_file}: {e}")
        return data

    def get_image(self, img_key):
        if img_key in self._img_cache:
            return self._img_cache[img_key]
        img_set: dict = self.get("image_set", {})
        if img_key in img_set:
            img = image_tools.read_image(img_set[img_key]["filepath"], max_size=2048)
            self._img_cache[img_key] = img
            return img
        img_set_a: dict = self.get("image_set_a", {})
        if img_key in img_set_a:
            img = image_tools.read_image(img_set_a[img_key]["filepath"], max_size=2048)
            self._img_cache[img_key] = img
            return img
        img_set_b: dict = self.get("image_set_b", {})
        if img_key in img_set_b:
            img = image_tools.read_image(img_set_b[img_key]["filepath"], max_size=2048)
            self._img_cache[img_key] = img
            return img
        raise KeyError(img_key)
