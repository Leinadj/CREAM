#!/usr/bin/env python3
"""
Run as a python script!
This script is used to check if the signals (voltage and current (coffee maker channel, noise channel)
are recorded as expected.
The checks are inspired by the ones carried out by Thomas Kriechbaumer for the BLOND dataset. See
the BLOND repository for details.
The pareameters checked are listed in an overview Table in the technical validation section of
the data descriptor.
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


def validate_voltage(file: str, voltage: np.ndarray):
    """
    This method checks the voltage signal.
    Check 1: Check if the voltage rms (root-mean-square) value is in the expected range of 210 - 240,
    the mean should be <5 and the crest_factor > 1.2. These values originate from the tests
    carried out on the BLOND dataset (same environment basically as for CREAM, values can be used for CREAM too)

    # Check 2: MEDAl uses 12bit unsigned integers with a DC-offset.
    When using half of the value range, there should be at least more then 2000 different voltage values.
    Parameters

    # Check 3: check the voltage bandwith

    # Check 4: we check the signal for flat_regions

    ----------
    file (str): filename the voltage signal belongs to
    voltage (np.ndarray): voltage signal to be tested

    Returns
    -------

    """
    filename = os.path.basename(file) #get the filename from the full path

    # Check 1:
    # Check RMS
    voltage_rms = np.sqrt(np.mean(np.square(voltage)))
    voltage_mean = np.abs(np.mean(voltage))
    voltage_crest_factor = np.percentile(np.abs(voltage), 99) / voltage_rms

    if not (voltage_rms >= 210 and voltage_rms <= 240):
        raise ValueError("Voltage RMS range violated in file %s" %(filename))

    if not (voltage_mean <= 5):
        raise ValueError("Voltage mean violated in file %s" % (filename))

    if not (voltage_crest_factor >= 1.2 and voltage_crest_factor <= 1.6):
        raise ValueError("Voltage crest_factor violated in file %s" % (filename))

    # Check 2 : check the voltage values. The voltage values have to be below 2000 (way below it)
    voltage_unique_values = len(np.unique(voltage))
    if not (voltage_unique_values >= 2000):
        raise ValueError("Number of unique voltage values violated in file %s" % (filename))

    # Check 3: check the bandwith of the voltage signal
    # The MEDAL measurement unit uses a 12-bit unsigned integers with DC-offset
    # the computations is similiar to the one done for BLOND.
    # Hence, we take the absolute max and min of the voltage signal
    # and divide it by 2 to the power of 11 (12 bits - 1) to get the number of steps
    # and multiply this by 100 to increase the step size
    # the bandwith should be bigger then 50 = x / 204800, so x (the min + max of the voltage,
    # should be bigger then 10240000 when added up

    threshold = 50

    voltage_bandwidth = np.abs(int(np.max(voltage))) + np.abs(np.min(voltage)) / (2 ** 11) * 100

    if not (voltage_bandwidth >= threshold):
        raise ValueError("Voltage bandwith violated in file %s" % (filename))

    # get the respective negative or positive value range, by clipping the corresponding other value range
    # e.g. for the neg. range, clip (set) all values above 0 to be 0.
    # the negative ones remain ontouched (i.e. - 999 is rediciously low, so there will be no clipping for the neg. values)
    # then we analyize the smallest one percent of values

    min_value = np.percentile(np.clip(voltage, -999, 0), 1)
    if not (min_value < -300 and min_value > -355):
        raise ValueError("Voltage minimum values violated in file %s" % (filename))

    max_value = np.percentile(np.clip(voltage, 0, 999), 99)
    if not (max_value > 300 and max_value < 355):
        raise ValueError("Voltage minimum values violated in file %s" % (filename))

    # We furthermore check the signals for flat regions, also inspired by the BLOND evaluation
    # For every period, we compute the absolute difference between consecutive values within this period
    # If there is a flat period, i.e. a period where a differences betweeen consecutive samples are 0.
    # the check is not passed

    # Every period has 6400 / 50 samples
    voltage_periods = np.array_split(voltage, int(6400 / 50))
    for period in voltage_periods:
        if np.sum(np.abs(np.diff(period))) == 0:
            raise ValueError("Voltage contains flat regions in file %s" % (filename))

def validate_current(file: str, current_signals: np.ndarray):
    """
    Validate the current signals, both the coffee maker and the noise channel signals.
    The RMS value, the mean and the crest_factor are validated, inspired by the analysis carried out
    by Thomas Kriechbaumer for the BLOND dataset.
    Furthermore, we check the current signals for flat regions (i.e. regions with contant values per period).

    Parameters
    ----------
    file (str): filename the current_signals belong to
    current_signals (np.ndarray): shape=(2,n) with n being the numbers of samples. The first element should
    be the coffee maker signal, the second element should be the noise channel, as it is provided by the
    load_file function in the data_utiltiy CREAM_Day class.


    Returns
    -------

    """

    filename = os.path.basename(file) #get the filename from the full path

    # MEDAL uses ACS712-5B and ACS712-30A with a 16A mains fuse
    threshold = 16

    # For the current signal, we compute the rms values per second (we have 6400 samples per second)
    # Then we use the maximum rms per second and compare it to the threshold
    for current in current_signals: # first the coffee maker signal, then the noise channel
        current_rms = np.max(np.sqrt(np.mean(np.square(current).reshape(-1, 6400), axis=1)))
        current_mean = np.abs(np.mean(current))
        current_crest_factor = np.max(np.abs(current)) / current_rms

    if not (current_rms <= threshold):
        raise ValueError("Current RMS range violated in file %s" % (filename))

    if not (current_mean <= 1):
        raise ValueError("Current mean violated in file %s" % (filename))

    if not (current_crest_factor >= 1.2):
        raise ValueError("Current crest_factor violated in file %s" % (filename))


    # We furthermore check the signals for flat regions, also inspired by the BLOND evaluation
    # For every period, we compute the absolute difference between consecutive values within this period
    # If there is a flat period, i.e. a period where a differences betweeen consecutive samples are 0.
    # the check is not passed

    # Every period has 6400 / 50 samples
    current_periods = np.array_split(current, int(6400 / 50))
    for period in current_periods:
        if np.sum(np.abs(np.diff(period))) == 0:
            raise ValueError("Current contains flat regions in file %s" % (filename))

def run_checks(dataset_location: str):
    """
    Run this function to execute the voltage and current signal checks.

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
            voltage, current = cream_day.load_file(file, return_noise=True)

            validate_voltage(file=file, voltage=voltage)

            validate_current(file=file, current_signals=current)

if __name__ == "__main__":

    # TODO: Specify the dataset location here and run the script
    DATASET_LOCATION = os.path.abspath(os.path.join("..", "..", "Datasets", "CREAM"))

    # Run the voltage and current signal checks
    run_checks(dataset_location=DATASET_LOCATION)