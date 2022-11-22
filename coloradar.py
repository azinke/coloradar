"""Entrypoint of the package."""
import sys
import argparse

import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt

from core.dataset import Coloradar
from core.utils.common import info, success


def main () -> None:
    """Main entry point for the cli interface."""
    parser = argparse.ArgumentParser(
        prog="coloradar.py",
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
        help="Render the lidar pointcloud",
        action="store_true"
    )
    parser.add_argument(
        "--scradar",
        help="Render the single chip radar pointcloud",
        action="store_true"
    )
    parser.add_argument(
        "--ccradar",
        help="Render the cascade chip radar pointcloud",
        action="store_true"
    )
    parser.add_argument(
        "--raw",
        help="Consider radar ADC measurement (only for scradar and ccradar)",
        action="store_true"
    )
    parser.add_argument(
        "--heatmap",
        help="Render heatmap (only for scradar and ccradar)",
        action="store_true"
    )
    parser.add_argument(
        "-pcl", "--pointcloud",
        help="Render 3D pointcloud (only for scradar and ccradar)",
        action="store_true"
    )
    parser.add_argument(
        "--heatmap-2d",
        help="Render 2D heatmap (only for scradar and ccradar)",
        action="store_true"
    )
    parser.add_argument(
        "--threshold",
        help="Threshold for filtering heatmap pointcloud",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--no-sidelobe",
        help="Skip the data within the first couple of meters to avoid sidelobes",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--velocity-view",
        help=(
            "Render the radar heatmap using velocity as the fourth dimension."
            " By default, this parameter is false and the gain in dB is used"
            "instead. However, it's only available for the processed raw ADC "
            "samples."
        ),
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--bird-eye-view",
        "-bev",
        help="Request a bird eye view rendering",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--resolution",
        "-r",
        help="Bird eye view resolution",
        type=float,
        default=0.025,
    )
    parser.add_argument(
        "--width",
        help="Bird eye view image width",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--height",
        help="Bird eye view image height",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--groundtruth",
        help="Render the bounding box wrapping the groundtruth objects "
            "a given dataset entry",
        action="store_true"
    )

    # Parameters to control the rendering of the heatmap
    parser.add_argument(
        "--range",
        help="Range to focus the heatmap on",
        type=float,
        default=None
    )
    parser.add_argument(
        "--min-range",
        help="Min Range to render in the heatmap",
        type=float,
        default=None
    )
    parser.add_argument(
        "--max-range",
        help="Max Range to render in the heatmap",
        type=float,
        default=None
    )
    parser.add_argument(
        "--azimuth",
        help="Azimuth to focus the heatmap on",
        type=float,
        default=None
    )
    parser.add_argument(
        "--min-azimuth",
        help="Azimuth to focus the heatmap on",
        type=float,
        default=None
    )
    parser.add_argument(
        "--max-azimuth",
        help="Azimuth to focus the heatmap on",
        type=float,
        default=None
    )
    parser.add_argument(
        "--polar",
        help="Render heatmap in polar coordinate. "
        "Carterian coordinate is used by default otherwise",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "-s", "--save-as",
        help="Save post-processed ADC samples in files. Values: 'bin', 'csv'",
        type=str,
        default="bin",
    )
    parser.add_argument(
        "--save-to",
        help="Destination folder to save the processed data to.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--animate",
        help="Folder to read input images from",
        type=str,
        default=None,
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
        # Get an instance of the Record class
        record = coloradar.getRecord(args.dataset, args.index)

        if args.lidar:
            record.load("lidar")
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
        elif args.scradar:
            record.load("scradar")
            if args.bird_eye_view and (not args.raw):
                info("Rendering single chip radar pointcloud bird eye view ...")
                bev = record.scradar.getBirdEyeView(
                    args.resolution,
                    (-args.width/2, args.width/2),
                    (-args.height/2, args.height/2),
                )
                success("Bird Eye View successfully rendred!")
                plt.imshow(bev)
                plt.show()
                info("Bird Eye View closed!")
                sys.exit(0)
            elif args.heatmap:
                info("Rendering single chip radar heatmap ...")
                record.scradar.showHeatmap(args.threshold, args.no_sidelobe)
                success("Heatmap closed!")
                sys.exit(0)
            elif args.raw:
                info("Processing raw ADC samples ...")
                if args.pointcloud:
                    info("Rendering Radar pointcloud ...")
                    record.scradar.showPointcloudFromRaw(
                        args.velocity_view,
                        args.bird_eye_view,
                        args.polar
                    )
                    success("Successfully closed!")
                    sys.exit(0)
                elif args.heatmap_2d:
                    info("Rendering 2D heatmap ...")
                    record.scradar.show2dHeatmap()
                    sys.exit(0)
                info("Rendering processed raw radar ADC samples ...")
                record.scradar.showHeatmapFromRaw(
                    args.threshold,
                    args.no_sidelobe,
                    args.velocity_view,
                    args.polar,
                    (args.min_range, args.max_range),
                    (args.min_azimuth, args.max_azimuth),
                )
                success("Successfully closed!")
                sys.exit(0)
            info("Rendering single chip radar pointcloud ...")
            record.scradar.show()
            success("Successfully closed!")
            sys.exit(0)
        elif args.ccradar:
            record.load("ccradar")
            if args.bird_eye_view and (not args.raw):
                info("Rendering cascaded chip radar pointcloud bird eye view ...")
                bev = record.ccradar.getBirdEyeView(
                    args.resolution,
                    (-args.width/2, args.width/2),
                    (-args.height/2, args.height/2),
                )
                success("Bird Eye View successfully rendred!")
                plt.imshow(bev)
                plt.show()
                info("Bird Eye View closed!")
                sys.exit(0)
            elif args.heatmap:
                info("Rendering cascade chip radar heatmap ...")
                record.ccradar.showHeatmap(args.threshold, args.no_sidelobe)
                success("Heatmap closed!")
                sys.exit(0)
            elif args.raw:
                info("Processing raw ADC samples.")
                if args.pointcloud:
                    info("Rendering Radar pointcloud ...")
                    record.ccradar.showPointcloudFromRaw(
                        args.velocity_view,
                        args.bird_eye_view,
                        args.polar
                    )
                    success("Successfully closed!")
                    sys.exit(0)
                elif args.heatmap_2d:
                    info("Rendering 2D heatmap ...")
                    record.ccradar.show2dHeatmap(args.polar)
                    sys.exit(0)
                info("Rendering 4D heatmap ...")
                record.ccradar.showHeatmapFromRaw(
                    args.threshold,
                    args.no_sidelobe,
                    args.velocity_view,
                    args.polar,
                    (args.min_range, args.max_range),
                    (args.min_azimuth, args.max_azimuth),
                )
                success("Successfully closed!")
                sys.exit(0)
            info("Rendering cascaded chip radar pointcloud ...")
            record.ccradar.show()
            success("Successfully closed!")
            sys.exit(0)
    elif args.dataset and args.save_to:
        record = coloradar.getRecord(args.dataset, 0)
        if args.lidar:
            info(f"Generated batches of lidar heatmap from '{args.dataset}'")
            record.process_and_save(
                "lidar",
                resolution=args.resolution,
                srange=(-args.width/2, args.width/2),
                frange=(-args.height/2, args.height/2),
                output=args.save_to,
            )
            success("BEV generated with success!")
            sys.exit(0)
        elif args.ccradar and args.raw:
            info(f"Generating batches of radar heatmap from '{args.dataset}'")
            if args.heatmap_2d:
                record.process_and_save(
                    "ccradar",
                    heatmap_3d=False,
                    output=args.save_to,
                )
                success("Radar 2D heatmap generated with success!")
                sys.exit(0)
            if args.pointcloud:
                record.process_and_save(
                    "ccradar",
                    output=args.save_to,
                    velocity_view=args.velocity_view,
                    bird_eye_view=args.bird_eye_view,
                    polar=args.polar,
                    pointcloud=True,
                    save_as=args.save_as,
                )
                success("Radar pointcloud generated with success!")
                sys.exit(0)
            record.process_and_save(
                "ccradar",
                output=args.save_to,
                threshold=args.threshold,
                no_sidelobe=args.no_sidelobe,
                velocity_view=args.velocity_view,
                heatmap_3d=True,
            )
            success("Radar 3D heatmap generated with success!")
            sys.exit(0)

    elif args.dataset and args.animate:
        record = coloradar.getRecord(args.dataset, 0)
        record.make_video(args.animate)
        success("Animation generated with success!")
        sys.exit(0)
    parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    main()
