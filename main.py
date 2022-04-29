"""Entrypoint of the package."""
import sys
import argparse

import matplotlib.pyplot as plt

from core.dataset import Coloradar
from core.utils.common import info, success


def main () -> None:
    """Main entry point for the cli interface."""
    parser = argparse.ArgumentParser(
        prog="Coloradar Dataset loader",
        description="Facility for interacting with the Coloradar dataset."
    )
    parser.add_argument(
        "-v", "--version",
        help="Print software version and information about the dataset.",
        action="store_true"
    )
    parser.add_argument(
        "-o", "--overview",
        help="Print the dataset JSON description file",
        action="store_true"
    )
    parser.add_argument(
        "--codename",
        help="Render the available sub-datasets and their codename",
        action="store_true"
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        help="Codename of the sub-dataset record to load"
    )
    parser.add_argument(
        "-i", "--index",
        type=int,
        help="Index of the dataset entry to read"
    )
    parser.add_argument(
        "--lidar",
        help="Render the lidar pointcloud of a given dataset entry",
        action="store_true"
    )
    parser.add_argument(
        "--bird-eye-view",
        "-b",
        help="Request a bird eye view rendering",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--resolution",
        "-r",
        help="Bird eye view resolution",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--width",
        "-w",
        help="Bird eye view image width",
        type=float,
        default=80.0,
    )
    parser.add_argument(
        "--height",
        "-h",
        help="Bird eye view image height",
        type=float,
        default=80.0,
    )
    parser.add_argument(
        "--groundtruth",
        help="Render the bounding box wrapping the groundtruth objects "
            "a given dataset entry",
        action="store_true"
    )
    args = parser.parse_args()

    coloradar = Coloradar()

    if args.overview:
        if args.codename:
            coloradar.printCodenames()
            sys.exit(0)
        print(coloradar)
        sys.exit(0)

    if args.dataset and args.index:
        record = coloradar.getRecord(args.dataset, args.index)

        if args.lidar:
            if args.bird_eye_view:
                info("Rendering lidar pointcloud bird eye view ...")
                bev = record.lidar.getBirdEyeView(
                    args.resolution,
                    (-args.width/2, args.width/2),
                    (-args.height/2, args.height/2),
                )
                success("Bird Eye View successfully rendred!")
                plt.imshow(bev)
                plt.show()
                info("Bird Eye View closed!")
                sys.exit(0)
            info("Rendering lidar pointcloud ...")
            record.lidar.show()
            success("Successfully closed!")
            sys.exit(0)

    parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    main()
