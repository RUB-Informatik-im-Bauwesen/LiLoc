import argparse
import logging, coloredlogs
import pathlib

# Create a logger object.
log = logging.getLogger("LiLoc")
coloredlogs.install(logger=log, level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="LiLoc Image Feature Matcher Tool")

    parser.add_argument("-d", "--debug",  action='store_true')

    subparsers = parser.add_subparsers(help='Select an operation', dest="action")

    # Generic options
    generic_parser = argparse.ArgumentParser(add_help=False)
    generic_parser.add_argument("-o", "--output-dir", type=pathlib.Path, help="Path to output directory")
    generic_parser.add_argument("-m", "--matcher", choices=["XFeatLighterglue", "SIFTkNN"], default="XFeatLighterglue", help="Keypoint extraction and matching algorithms to use")
    generic_parser.add_argument("-r", "--recurse-dirs",  action='store_true', help="Recurse subdirectories for images")
    generic_parser.add_argument("-c", "--cache-features",  action='store_true', help="Store features in a cache directory (SIFT matching only)")

    # Cross-matching
    cross_match_parser = subparsers.add_parser('cross_match', help="Match all images from one folder against all images from a second folder.", parents=[generic_parser])

    cross_match_parser.add_argument("panoramic_image_folder", metavar="panIMG", type=pathlib.Path)
    cross_match_parser.add_argument("input_image_folder", metavar="IMG", type=pathlib.Path)


    # Exhaustive matching
    match_parser = subparsers.add_parser('match', help="Match all images from one folder against each other.", parents=[generic_parser])

    match_parser.add_argument("input_image_folder", metavar="IMG", type=pathlib.Path)


    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)
        coloredlogs.install(logger=log, level=logging.DEBUG)

    from feature_matching import start_cross_match, start_exhaustive_match

    if args.action == "cross_match":
        start_cross_match(args)
    elif args.action == "match":
        start_exhaustive_match(args)


if __name__ == "__main__":
    main()
