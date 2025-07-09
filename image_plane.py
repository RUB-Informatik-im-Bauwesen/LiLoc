import numpy as np
from trimesh.primitives import Primitive, PrimitiveAttributes
from trimesh.visual import TextureVisuals
from PIL import Image


class ImagePlane(Primitive):
    def to_dict(self):
        return {
            "kind": "cylinder",
            "transform": self.primitive.transform.tolist(),
            "image_path": self.primitive.image_path
        }

    def __init__(self, image_path: str, center=None, transform=None):
        super().__init__()

        defaults = {"image_path": None, "transform": np.eye(4)}
        constructor = {"image_path": image_path}
        # center is a helper method for "transform"
        # since a sphere is rotationally symmetric
        if center is not None:
            if transform is not None:
                raise ValueError("only one of `center` and `transform` may be passed!")
            translate = np.eye(4)
            translate[:3, 3] = center
            constructor["transform"] = translate
        elif transform is not None:
            constructor["transform"] = transform

        # create the attributes object
        self.primitive = PrimitiveAttributes(
            self, defaults=defaults, kwargs=constructor
        )

    def _create_mesh(self):
        image: Image = Image.open(self.primitive.image_path)
        texture_viz = TextureVisuals(np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]), image=image)
        self.visual = texture_viz
        vertices = np.array([[-0.5, -0.5, 0, 0], [0.5, -0.5, 0, 0], [0.5, 0.5, 0, 0], [-0.5, 0.5, 0, 0]])
        ratio = image.width / image.height
        vertices *= np.array([ratio, 1, 1, 0])
        vertices = (vertices @ self.primitive.transform)[..., :3]
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        face_normals = np.array([[0, 0, 1], [0, 0, 1]])
        # apply the center offset here
        self._cache["vertices"] = vertices + self.primitive.center
        self._cache["faces"] = faces
        self._cache["face_normals"] = face_normals


if __name__ == '__main__':
    import trimesh
    image_viz = ImagePlane("IMG_6355(1).JPEG")
    # image_viz = image_viz.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]))

    sc = trimesh.Scene()
    sc.add_geometry(image_viz)
    sc.show()
