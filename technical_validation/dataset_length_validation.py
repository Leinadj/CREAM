#!/usr/bin/env python3
"""
This script is used to check if the dataset is complete.

Check if all files have the expected length
1. With a nominal sampling frequency of 6400 samples per second,
    and a duration of one hour, the files should have 6400 * 60 * 60 samples.
    This is checked for the voltage signal, and the two current signals (coffee maker and noise signal)
2. Per day, we recorded 16 hours of data, except for the last day 08-10-2018 with only 8 hours.
3. Furthermore, there should be no interruption in the days.
    Hence, we check if all days are available in the data.

You need to specify the dataset location before running the script. This can be done in the main funciton
of the script.
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


def validate_dataset_length(dataset_location: str):

    # Load the days available in the dataset
    cream_days = [d for d in os.listdir(dataset_location) if "2018" in d]  # only the day folders

    check_passed = True # set to true if all files, days, etc. have passed the checks

    for day in cream_days:
        print("Checking day %s " %(day))

        # Initialize a CREAM_Day for the respective day (from the data_utilities)
        cream_day = CREAM_Day(cream_day_location=os.path.join(dataset_location, day),
                              use_buffer=True,
                              buffer_size_files=2)

        # Check 1
        nominal_file_length = 60 * 60 * 6400

        for file in cream_day.files:
            voltage, current = cream_day.load_file(file, return_noise=True)


            if len(voltage) != nominal_file_length or len(current[0]) != nominal_file_length or len(current[1]) != nominal_file_length:
                print("File %s has not passed check 1" %(file))
                check_passed = False

        # Check 2
        nominal_file_number = 16
        if day == "2018-10-08":  # except for this day, has 8 files
            nominal_file_number = 8

        if len(cream_day.files) != nominal_file_number:
            print("Day %s has not passed check 2" % (cream_day.day_date))
            check_passed = False

    # Check 3:
    all_nominal_days = set(["2018-08-23" , "2018-08-24" , "2018-08-25",  "2018-08-26" , "2018-08-27" , "2018-08-28" ,
    "2018-08-29", "2018-08-30", "2018-08-31", "2018-09-01", "2018-09-02" , "2018-09-03" ,  "2018-09-04" ,
    "2018-09-05", "2018-09-06", "2018-09-07", "2018-09-08" , "2018-09-09" , "2018-09-10", "2018-09-11", "2018-09-12",
    "2018-09-13" ,"2018-09-14" ,"2018-09-15" ,  "2018-09-16", "2018-09-17", "2018-09-18", "2018-09-19"  , "2018-09-20" ,
    "2018-09-21" , "2018-09-22" ,  "2018-09-23" ,"2018-09-24" ,"2018-09-25" ,"2018-09-26" , "2018-09-27", "2018-09-28" ,
    "2018-09-29" , "2018-09-30" , "2018-10-01" ,"2018-10-02" , "2018-10-03" ,"2018-10-04", "2018-10-05" , "2018-10-06" ,
    "2018-10-07", "2018-10-08"])
    if set(cream_days) != all_nominal_days:
        print("Check 3 not passed")
        check_passed = False

    if check_passed is True:
        print("All checks passed successfully!")
    else:
        print("The checks failed!")

if __name__ == "__main__":

    # TODO: Specify the dataset location here and run the script
    DATASET_LOCATION = os.path.abspath(os.path.join("..", "..", "Datasets", "CREAM"))

    validate_dataset_length(dataset_location=DATASET_LOCATION)