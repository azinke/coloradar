"""Dataset.

Root module loading the dataset
"""
import json

from core.config import ROOTDIR, DATASET
from core.calibration import Calibration
from core.record import Record


class Coloradar:
    """Coloradar dataset loader.

    Attributes:
        index: JSON file describing the dataset
        rootdir: Root directory containing the dataset
        calibration: Object holding sensor calibration details
    """

    def __init__(self, rootdir: str = ROOTDIR) -> None:
        """Load dataset paths.

        Argument:
            rootdir: Root directory marking the entry  point of the dataset
        """
        self.index: str = DATASET
        self.rootdir: str = rootdir
        self.config: dict = {}
        with open(DATASET, "r") as fh:
            self.config = json.load(fh)
        self.calibration = Calibration(self.config["calibration"])

    def getRecord(self, codename: str, index: int) -> Record:
        """Read a sensor recording frame from the dataset."""
        return Record(
            self.config["datastore"], self.calibration, codename, index)

    def printCodenames(self) -> None:
        """Print the available sub-dataset and their codename."""
        print(" Coloradar dataset")
        print("   [CODENAME]        [DIRECTORY NAME]")
        for dataset in self.config["datastore"]["folders"]:
            print(f"   {dataset['codename']}             {dataset['path']}")
        print()

    def __str__(self) -> str:
        return json.dumps(self.config, indent=2)
