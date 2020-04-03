#!/usr/bin/env python3
"""
Run as a python script!
This script is used to validate the actual sampling rate in the CREAM files and to check
if the actual sampling rate differs from the nominal one.
Please specify the data location and run the script to reproduce the results.
The dataset location is handed over to the validation function in the main method.
"""
import os
import glob
import h5py
import numpy as np
import sys
# add data utility module to path for import
module_path = os.path.abspath("../data_utility/data_utility.py")
if module_path not in sys.path:
    sys.path.append(module_path)
from data_utility import CREAM_Day  # class to work with a day of the CREAM Dataset


def validate_sampling_rate(dataset_location: str):
    # This should be the root folder of CREAM

    # Iterate over the individual days

    cream_days = [d for d in os.listdir(dataset_location) if "2018" in d] #only the day folders
    average_sampling_rates = []

    for day in cream_days:

        # Initialize a CREAM_Day for the respective day (from the data_utilities)
        cream_day = CREAM_Day(cream_day_location=os.path.join(dataset_location, day),
                              use_buffer=True,
                              buffer_size_files=2)

        #load the ‚metadata -> the nominal sampling rate is stored in the frequency attribute
        metadata_test = cream_day.load_file_metadata(file_path=cream_day.files[0])

        metadata_dict = cream_day.load_file_metadata(file_path=cream_day.files[0],
                                                     attribute_list=["frequency"])
        average_sampling_rate = cream_day.compute_average_sampling_rate()
        average_sampling_rates.append(average_sampling_rate)

        print("Day %s : Nominal sampling rate: %i | Average sampling rate: %f " % (
              cream_day.day_date,
              metadata_dict["frequency"],
              average_sampling_rate
                ))

if __name__ == "__main__":

    # TODO: Specify the dataset location here and run the script
    DATASET_LOCATION = os.path.abspath(os.path.join("..", "..", "Datasets", "CREAM"))

    validate_sampling_rate(dataset_location=DATASET_LOCATION)