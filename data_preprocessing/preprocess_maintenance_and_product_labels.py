"""
To ease the use of the log files, we convert the timezone unaware timestamps to aware ones that are consistent with the
labeled appliance timestamps. When using the convenience functions provided in the CREAM class (located in the data_utility
folder), the timestamps are returned aligned and in the same format.
This pre-processing was performed, before the events were refined by the labelers with the maintenance_and_product_events
_adjustment.tool jupyter notebook.
"""

# TODO noch auf die neuen Gegebenheiten anpassen mit den nachverfeinerten events

import os
import sys
import pandas as pd
from datetime import datetime

# Add private moduls and import them
project_path = os.path.abspath("..")
if project_path not in sys.path:
    sys.path.append(project_path)

# Add module path to path for import
module_path = os.path.abspath("../data_utility/data_utility.py")
if module_path not in sys.path:
    sys.path.append(module_path)

project_path = os.path.abspath("..")
if project_path not in sys.path:
    sys.path.append(project_path)
# Add module path to path for import
module_path = os.path.abspath("../data_utility/data_utility.py")
if module_path not in sys.path:
    sys.path.append(module_path)
from data_utility import CREAM_Day  # class to work with a day of the CREAM Dataset

# Global Variables, only edit here
DATA_PATH = os.path.abspath(os.path.join("..", "..", "Datasets", "CREAM", "CREAM_6400"))
SAVE_PATH = os.path.abspath(os.path.join("..", "..", "Datasets", "CREAM", "CREAM_6400"))

# Create Day Object, day does not matter, needed for initialization
CREAM_Day = CREAM_Day(os.path.join(DATA_PATH, "2018-08-24"))

timezone = CREAM_Day.get_datetime_from_filepath(CREAM_Day.files[0]).tzinfo
product_events = pd.read_csv(os.path.join(DATA_PATH, "product_events.csv"))
product_events.Timestamp = product_events.Timestamp.apply(pd.Timestamp)
product_events.Timestamp = product_events.Timestamp.dt.tz_localize(timezone)
product_events.sort_values("Timestamp", inplace=True)


maintenance_events = pd.read_csv(os.path.join(DATA_PATH, "maintenance_events.csv"))
maintenance_events.Timestamp = maintenance_events.Timestamp.apply(pd.Timestamp)
maintenance_events.Timestamp = maintenance_events.Timestamp.dt.tz_localize(timezone)
maintenance_events.sort_values("Timestamp", inplace=True)

# Now we save the new timestamps
product_events.to_csv(os.path.join(SAVE_PATH, "product_events.csv"), index=False)
maintenance_events.to_csv(os.path.join(SAVE_PATH, "maintenance_events.csv"), index=False)
