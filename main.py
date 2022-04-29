"""Entrypoint of the package."""
import sys
import argparse

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
            info("Rendering lidar pointcloud ...")
            record.lidar.show()
            success("Successfully closed!")
            sys.exit(0)

    parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    main()
