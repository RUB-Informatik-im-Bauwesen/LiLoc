import numpy as np
from trimesh.primitives import Primitive, PrimitiveAttributes
from trimesh.visual import TextureVisuals
from PIL import Image


class ImagePlane(Primitive):
    def to_dict(self):
        return {
            "kind": "cylinder",
            "transform": self.primitive.transform.tolist(),
            "image": self.primitive.image
        }

    def __init__(self, image: str, center=None, transform=None):
        super().__init__()

        defaults = {"image": None, "transform": np.eye(4), "npimage": None}
        constructor = {"image": image}
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
        if type(self.primitive.image) is str:
            image: Image = Image.open(self.primitive.image)
        else:
            imagearr = np.array(self.primitive.image).astype(np.uint8)
            imagearr = imagearr[...,::-1]  # flip color channel to convert BGR to RGB
            image = Image.fromarray(imagearr)
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

def test():
    import trimesh
    image_viz = ImagePlane("example/defect_img/IMG_9280.JPG")
    import cv2
    img2 = cv2.imdecode(np.fromfile(str("example/defect_img/IMG_9285.JPG"), np.uint8), cv2.IMREAD_COLOR)
    image_viz2 = ImagePlane(img2)
    image_viz2 = image_viz2.apply_transform(trimesh.transformations.translation_matrix([1, 0, 0]))

    sc = trimesh.Scene()
    sc.add_geometry(image_viz)
    sc.add_geometry(image_viz2)
    sc.show()


if __name__ == '__main__':
    test()