#!/usr/bin/env python3
"""
This script is used to check the mains frequency in the signal.
It should be 50 Hz. Inspired by the analysis by Thomas Kriechbaumer on the BLOND dataset, we
perform a fourier transform on the voltage signal and copmare the base frequency to the nominal one of 50 Hz.
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


def validate_mains_frequency(file, voltage: np.ndarray):
    """
    Function using a fourier transform to compare the base frequency in the signal to the nominal mains frequency
    of 50 Hz (inspired by the evaluations of Thomas Kriechbaumer in BLOND).

    Parameters
    ----------
    file (str): filename the voltage signal belongs to
    voltage (np.ndarray): voltage signal to be tested

    Returns
    -------

    """

    filename = os.path.basename(file) # get the filename formt he filepath

    sampling_frequency = 6400 # samples per second
    hamming_window = np.hamming(len(voltage)) # use a hamming window for the fourier transform

    # We perform the fft per 10 seconds, hence we reshape the voltage signal into bins of 6400 * 10 samples.
    # We do this after applying the hamming window function to the signal
    voltage = (voltage * hamming_window).reshape(-1, sampling_frequency * 10)
    fourier_transform = np.fft.rfft(voltage)
    frequency_bins = np.fft.rfftfreq(sampling_frequency * 10, d=1 / sampling_frequency)

    # now we get the mains frequency. We get the maximum frequency bin, by using the argmax function (returns the index of the
    # maximum element)
    # One mains frequency every 10 seconds
    actual_mains_frequencies = frequency_bins[np.argmax(np.abs(fourier_transform), axis=1)]

    if not all(mains_frequency >= 49.0 and mains_frequency <= 51.0 for mains_frequency in actual_mains_frequencies):
        raise ValueError("The mains frequency in the signal differs from the nominal 50 Hz mains frequency in file %s" % (filename))

def run_checks(dataset_location: str):
    """
    Run this function to execute the mains frequency check per file

    Parameters
    ----------
    dataset_location (str): location of the root folder of CREAM

    Returns
    -------

    """
    # Load the days available in the dataset
    cream_days = [d for d in os.listdir(dataset_location) if "2018" in d]  # only the day folders


    for day in cream_days:
        print("Checking day %s " % (day))

        # Initialize a CREAM_Day for the respective day (from the data_utilities)
        cream_day = CREAM_Day(cream_day_location=os.path.join(dataset_location, day),
                              use_buffer=True,
                              buffer_size_files=2)

        # Perform the checks per file
        for file in cream_day.files:
            voltage, current = cream_day.load_file(file, return_noise=False)

            validate_mains_frequency(file=file, voltage=voltage)





if __name__ == "__main__":

    # TODO: Specify the dataset location here and run the script
    DATASET_LOCATION = os.path.abspath(os.path.join("..", "..", "Datasets", "CREAM"))

    # Run the voltage and current signal checks
    run_checks(dataset_location=DATASET_LOCATION)
