import itertools

import numpy as np
import pye57
import trimesh
import trimesh.scene
import trimesh.viewer
import trimesh.visual
from trimesh.transformations import translation_matrix, rotation_matrix, translation_from_matrix

from image_plane import ImagePlane

scanner_viz_scale = 1.5

def _read_e57(pc_e57: pye57.E57):
    header = pc_e57.get_header(0)
    print("Point count:", header.point_count)
    print("PC Rotation matrix:", header.rotation_matrix)
    print("PC Translation matrix:", header.translation)
    scanner_tr = header.translation

    print("Reading data...")
    pc_data = pc_e57.read_scan(0, colors=True, ignore_missing_fields=True)
    print("Fields available:", pc_data.keys())
    print(len(pc_data["cartesianX"]), "points read, type", pc_data["cartesianX"].dtype)
    points = np.vstack((
        pc_data["cartesianX"],
        pc_data["cartesianY"],
        pc_data["cartesianZ"])).transpose()
    colors = np.vstack((
        pc_data["colorRed"],
        pc_data["colorGreen"],
        pc_data["colorBlue"],
        np.ones(pc_data["colorRed"].shape, dtype=np.uint8) * 255)).transpose()

    print("points:", points.shape)
    print("colors:", colors.shape)
    bb_size = 15  # Bounding box with size 2*bb_size
    bounding_box = np.array(((scanner_tr[0] - bb_size, scanner_tr[1] - bb_size, scanner_tr[2] - bb_size),
                             (scanner_tr[0] + bb_size, scanner_tr[1] + bb_size, scanner_tr[2] + bb_size)))
    print(bounding_box)
    points_in_bb = np.where((points[..., 0] > bounding_box[0][0]) & (points[..., 0] < bounding_box[1][0]) &
                            (points[..., 1] > bounding_box[0][1]) & (points[..., 1] < bounding_box[1][1]) &
                            (points[..., 2] > bounding_box[0][2]) & (points[..., 2] < bounding_box[1][2])
                            )
    print(points_in_bb, len(points_in_bb))
    points = points[points_in_bb]
    colors = colors[points_in_bb]

    return points, colors


def find_rays(rays: list, point_cloud_filename: str, show=False, images=None, reference_model_path=None, reference_model_transform=None):
    pc_e57 = pye57.E57(point_cloud_filename)

    header = pc_e57.get_header(0)
    scanner_tr = header.translation
    points, colors = _read_e57(pc_e57)


    base_rot = np.identity(3)
    base_rot[:3, :3] = header.rotation_matrix


    #yz_flip = np.array([[1,0,0],[0,0,-1],[0,1,0]])
    #rays = [yz_flip @ ray for ray in rays]
    print(f"Rays: {rays}")

    out_3d_points = [find_ray(ray, points, scanner_tr, base_rot) for ray in rays]

    if show:
        pc = trimesh.points.PointCloud(points, colors)
        visualize(pc, rays, out_3d_points, scanner_tr, base_rot, images, reference_model_path=reference_model_path,
                  reference_model_transform=reference_model_transform)

    return [o[0] if o is not None else None for o in out_3d_points]


def find_all_points(in_2d_points: list, point_cloud_filename: str, show=False, images=None, reference_model_path=None, reference_model_transform=None):
    pc_e57 = pye57.E57(point_cloud_filename)

    header = pc_e57.get_header(0)
    scanner_tr = header.translation
    points, colors = _read_e57(pc_e57)

    base_rot = np.identity(4)
    base_rot[:3, :3] = header.rotation_matrix

    out_3d_points = [find_point(in_2d_point, points, scanner_tr, base_rot) for in_2d_point in in_2d_points]

    if show:
        pc = trimesh.points.PointCloud(points, colors)
        visualize(pc, rays, out_3d_points, scanner_tr, base_rot, images, reference_model_path=reference_model_path, reference_model_transform=reference_model_transform)

    return [o[0] if o is not None else None for o in out_3d_points]


def visualize(pc, rays, out_3d_points, scanner_tr, scanner_rot, images=None, reference_model_path=None, reference_model_transform=None):
    laser_scanner_viz = trimesh.load_mesh("example/rtc_360_model.obj").apply_scale(scanner_viz_scale).apply_translation(scanner_tr)
    trimesh.Trimesh()
    scene: trimesh.scene.Scene = trimesh.scene.Scene()
    scene.add_geometry(pc, "pointcloud")
    scene.add_geometry(laser_scanner_viz)
    for index, (out, ray) in enumerate(itertools.zip_longest(out_3d_points, rays)):
        if out is None:
            ray /= np.linalg.norm(ray)
            ray_path = trimesh.load_path([scanner_tr, scanner_tr + ray * 10])
            scene.add_geometry(ray_path)
            continue
        (out_pt, out_normal) = out
        out_normal = -ray
        ray_path = trimesh.load_path([scanner_tr, out_pt])
        print("Placing image at", out_pt, out_normal)

        closest_point_viz = trimesh.primitives.Sphere(radius=0.1, center=out_pt)
        closest_point_viz.visual = trimesh.visual.ColorVisuals()
        closest_point_viz.visual.face_colors = np.array([255, 0, 0, 255] * len(closest_point_viz.faces)).reshape(
            len(closest_point_viz.faces), 4)
        if images is not None and len(images) > index:
            image_viz = (ImagePlane(images[index])
                        # .apply_transform(rotation_matrix(-np.pi / 2, [1, 0, 0]))
                        # .apply_transform(rotation_matrix(-np.pi / 2, [0, 1, 0]))
                        # .apply_transform(rotation_matrix(np.pi, [0, 0, 1]))
                        .apply_transform(look_at_matrix(out_normal, [0, 0, 1]))
                        .apply_transform(translation_matrix(out_pt))
                        .apply_transform(translation_matrix(out_normal))
                        )
            scene.add_geometry(image_viz)
            

        # scene.add_geometry(cast_cylinder)
        scene.add_geometry(closest_point_viz)
        scene.add_geometry(ray_path)

    arr_x = trimesh.load_path([np.array((0, 0, 0)), np.array((1, 0, 0))])
    arr_x.colors = [(255, 0, 0, 255)]
    arr_y = trimesh.load_path([np.array((0, 0, 0)), np.array((0, 1, 0))])
    arr_y.colors = [(0, 255, 0, 255)]
    arr_z = trimesh.load_path([np.array((0, 0, 0)), np.array((0, 0, 1))])
    arr_z.colors = [(0, 0, 255, 255)]
    scene.add_geometry([arr_x, arr_y, arr_z])

    if reference_model_path is not None:
        reference_model = trimesh.load_mesh(reference_model_path)
        if reference_model_transform is not None:
            if type(reference_model) is list:
                for m in reference_model:
                    m.apply_transform(reference_model_transform)
            else:
                reference_model.apply_transform(reference_model_transform)
        scene.add_geometry(reference_model, "reference_model")
        

    viewer: trimesh.viewer.SceneViewer = trimesh.viewer.SceneViewer(scene, start_loop=False)
    import pyglet
    global show_pc
    show_pc = True

    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.X:
            global show_pc
            show_pc = not show_pc
            set_pc_visibility(show_pc)
            return True
        
    def set_pc_visibility(show_pc):
            if not show_pc:
                viewer.hide_geometry("pointcloud")
                viewer.unhide_geometry("reference_model")
            else:
                viewer.unhide_geometry("pointcloud")
                viewer.hide_geometry("reference_model")

    set_pc_visibility(show_pc)
    viewer.push_handlers(on_key_press)
    viewer.reset_view()
    pyglet.app.run()


def find_ray(ray, points, scanner_tr, scanner_rot):
    # image_dimensions = np.array([8192, 3393])
    # image_point = in_2d_point
    # image_point = np.array([0, image_dimensions[1]/4+100])*2
    # Spherical to cartesian
    ray_direction = ray
    ray_direction /= np.linalg.norm(ray_direction)
    #ray_direction @= np.linalg.inv(scanner_rot)
    ray_origin = scanner_tr

    print("Ray:", ray_origin, ray_direction)
    # ray_path = trimesh.load_path([ray_origin, ray_origin + ray_direction * 15])

    cast_length = 20
    cast_radius = 0.3
    cast_origin = translation_matrix(ray_origin + ray_direction * (cast_length / 2))
    print("base_rot", trimesh.transformations.euler_from_matrix(scanner_rot))
    cast_cylinder = (trimesh.primitives.Cylinder(radius=cast_radius, height=cast_length)
                     .apply_transform(rotation_matrix(-np.pi / 2, [0, 1, 0]))
                     .apply_transform(cast_origin))

    print("Finding closest point...")
    points_in_cylinder = points[cast_cylinder.contains(points)]
    n_points_in_cylinder = points_in_cylinder.shape[0]
    if n_points_in_cylinder > 0:
        print("Points in cylinder =", n_points_in_cylinder)
        point_distance = trimesh.points.point_plane_distance(points_in_cylinder, ray_origin, ray_direction)
        closest_points_indices = np.argsort(point_distance)
        closest_point = points_in_cylinder[closest_points_indices[0], ...]

        # quick and dirty normal estimation
        try:
            second_closest_point = points_in_cylinder[closest_points_indices[1], ...]
            third_closest_point = points_in_cylinder[closest_points_indices[2], ...]
            normal = np.cross(closest_point - second_closest_point, closest_point - third_closest_point)
            normal /= np.linalg.norm(normal)
            if np.dot(normal, ray_origin - closest_point) < 0:
                normal *= -1  # flip if facing away from the ray origin
        except:
            print("Not enough points for normals")
            normal = np.array([1,0,0])

        print("Closest point =", closest_point)
        return closest_point, normal
    print("No points in cylinder for", ray_direction)
    return None


def find_point(in_2d_point, points, scanner_tr, scanner_rot):
    # image_dimensions = np.array([8192, 3393])
    # image_point = in_2d_point 
    # image_point = np.array([0, image_dimensions[1]/4+100])*2
    yaw_offset = 0
    ray_yaw = np.pi / 2 + in_2d_point[0] * np.pi * 2 + np.deg2rad(yaw_offset)
    ray_pitch = in_2d_point[1] * np.pi - np.pi / 2
    # Spherical to cartesian
    ray_direction = np.array([1, 0, 0, 0])
    print("Ray pitch/yaw (deg):", np.rad2deg(ray_pitch), np.rad2deg(ray_yaw))
    ray_rotation = rotation_matrix(ray_pitch, [0, 1, 0]) @ np.linalg.inv(scanner_rot) @ rotation_matrix(ray_yaw,
                                                                                                        [0, 0, 1])
    ray_direction = (ray_direction @ ray_rotation)[:3]
    ray_origin = scanner_tr

    print("Ray:", ray_origin, ray_direction)
    # ray_path = trimesh.load_path([ray_origin, ray_origin + ray_direction * 15])

    cast_length = 6
    cast_radius = 0.1
    cast_origin = translation_matrix(ray_origin + ray_direction * (cast_length / 2))
    print("base_rot", trimesh.transformations.euler_from_matrix(scanner_rot))
    cast_cylinder = (trimesh.primitives.Cylinder(radius=cast_radius, height=cast_length)
                     .apply_transform(rotation_matrix(-np.pi / 2, [0, 1, 0]))
                     .apply_transform(cast_origin @ np.linalg.inv(ray_rotation)))

    print("Finding closest point...")
    points_in_cylinder = points[cast_cylinder.contains(points)]
    n_points_in_cylinder = points_in_cylinder.shape[0]
    if n_points_in_cylinder > 0:
        print("Points in cylinder =", n_points_in_cylinder)
        point_distance = trimesh.points.point_plane_distance(points_in_cylinder, ray_origin, ray_direction)
        closest_points_indices = np.argsort(point_distance)
        closest_point = points_in_cylinder[closest_points_indices[0], ...]
        
        # quick and dirty normal estimation
        second_closest_point = points_in_cylinder[closest_points_indices[1], ...]
        third_closest_point = points_in_cylinder[closest_points_indices[2], ...]
        normal = np.cross(closest_point - second_closest_point, closest_point - third_closest_point)
        normal /= np.linalg.norm(normal)
        if np.dot(normal, ray_origin - closest_point) < 0:
            normal *= -1  # flip if facing away from the ray origin

        print("Closest point =", closest_point)
        return closest_point, normal
    return None

def look_at_matrix(vec: np.ndarray, up: np.ndarray) -> np.ndarray:
        z = vec / np.linalg.norm(vec)
        up = up / np.linalg.norm(up)
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        m = np.array(
            [[x[0], y[0], z[0], 0],
            [x[1], y[1], z[1], 0],
            [x[2], y[2], z[2], 0],
            [0, 0, 0, 1]]).transpose()
        return m

if __name__ == '__main__':
    import argparse
    import pathlib
    import json

    parser = argparse.ArgumentParser(description="Runs Services")
    parser.add_argument("-i", "--inputPath", type=pathlib.Path, nargs='+', default=".",
                        help="Path to input data")
    parser.add_argument("-o", "--outputPath", type=pathlib.Path, default="/tmp/outputData",
                        help="Path to output directory")

    parser.add_argument("point_cloud_filename", metavar="pc_file", type=pathlib.Path,
                        help="Point cloud file in e57 format")
    parser.add_argument("image_points_json", metavar="img_loc_file", type=pathlib.Path,
                        help="")

    args = parser.parse_args()
    input_path = pathlib.Path(args.inputPath)
    output_path = pathlib.Path(args.outputPath)

    pc_path = input_path / pathlib.Path(args.point_cloud_filename)
    json_input_path = input_path / pathlib.Path(args.image_points_json)
    with open(json_input_path) as f:
        image_points_json = json.load(f)

    in_points = [np.array(j["center_point"]) for j in image_points_json]
    out_points = find_all_points(in_points, str(pc_path), show=False)
    print(out_points)

    for j, pt in zip(image_points_json, out_points):
        j["center_3d"] = pt.tolist()

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "out.json", 'w') as f:
        json.dump(image_points_json, f)
