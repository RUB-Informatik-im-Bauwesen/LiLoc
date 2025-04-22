LiLoc: LiDAR Image Localization
=====

Scripts for localizing image sets in point clouds based on feature matching.

## Usage

*Quickstart:* Matches all images in a given folder and writes to `/path/to/image/folder/matches`

```bash
python liloc.py match /path/to/image/folder
```

To match all images from one folder against all images from another folder (e.g. camera photos against LiDAR photos).
```bash
python liloc.py cross_match image/folder/one image/folder/two
```

For more flags and information use the help function:
```bash
python liloc.py match --help
python liloc.py cross_match --help
```

---

LiLoc can also extract rectified images from point clouds. 
```bash
python extract_images_from_e57.py pointcloud.e57
```
If there are no images, use the `--rgb` flag to render rectified images from the point clouds rgb values.

## Examples

TBD

## Paper

*To be published at ISARC 2025 in Montréal, Canada*

