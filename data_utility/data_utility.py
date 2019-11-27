import os
from pathlib import Path
import glob

import h5py

from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd

import scipy
from scipy import interpolate
import pdb
#-------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------#
# TODO Die Funktionen auch einfacher machen

class CREAM_Day():
    """
    A class representing one day of the CREAM dataset
    1. create the Object - sets the following properties:
        - dataset_location
        - day_date
        - clear_medal_mapping

    2. Call the populate() method to read all the meta-data into the object --> no real data yet, just filenames etc.
        If the immediately_populate parameter of the constructor is set to True --> populate is called automatically during object initalization
        - saved in the "metadata" dictionary property:
            {
            "files":files,
            "length":length,
            "clear_phase":clear_phase,
            "len_files":len_files,
            "start_timestamp":start_timestamp,
            "average_frequency":average_frequency,
            "name":name,
            "frequency":frequency,  i.e. the sampling frequency = sampling_rate
            "length":length,
            "seconds_per_file":seconds_per_file,
            "delay_after_midnight":delay_after_midnight
            }
        - computes limit of the day and sets the following properties: the timestamps vary between devices --> the smallest time frame device is the reference one
            - gap_to_start_of_day
            - gap_to_end_of_day
            - minimum_request_timestamp
            - minimum_request_timestamp_device i.e. the reference device
            - maximum_request_timestamp
            - minimum_request_timestamp_device
    """

    def __init__(self, cream_day_location: str, use_buffer : bool =False, buffer_size_files : int =0):
        """


            cream_dataset_location: location of the dataset

        """
        self.dataset_location = cream_day_location
        self.use_buffer = use_buffer
        self.buffer_size_files = buffer_size_files

        if self.buffer_size_files == 0 and use_buffer is True:
            self.buffer_size_files = 5
            raise Warning("Buffer size was specified with size 0: a minimum buffer size of 5 files was set therefore")

        # Initiate the file buffer dictionary
        self.file_cache = {}

        # Get all the files of the respective day
        self.files = glob.glob(os.path.join(self.dataset_location, "*.hdf5"))
        self.files.sort()


        # We use the first file and the timestamps in the filenames in the dataset (of this day) to get the metadata information
        # Get the timezone information from the filename timestamp
        timezone = self.get_datetime_from_filepath(self.files[0])
        timezone = timezone.tzinfo

        # Load Metadata from the first file of the respective device --> same for all of the device --> STATIC METADATA
        with h5py.File(self.files[0], 'r', driver='core') as f:

            self.sampling_rate = int(f.attrs['frequency'])  # get the sampling rate
            self.samples_per_file = len(f["voltage"])  # get the length of the signal

            # get the start timestamp
            start_timestamp = datetime(
                year=int(f.attrs['year']),
                month=int(f.attrs['month']),
                day=int(f.attrs['day']),
                hour=int(f.attrs['hours']),
                minute=int(f.attrs['minutes']),
                second=int(f.attrs['seconds']),
                microsecond=int(f.attrs['microseconds']),
                tzinfo=timezone)

            self.file_duration_sec = self.samples_per_file / self.sampling_rate
            self.number_of_files = len(self.files)


        # Some file metadata for every file
        file_start_times = [self.get_datetime_from_filepath(f) for f in self.files]

        file_end_times = [timedelta(seconds=self.file_duration_sec) + ts for ts in file_start_times]
        self.files_metadata_df = pd.DataFrame({"Start_timestamp": file_start_times,
                                               "Filename": self.files,
                                               "End_timestamp": file_end_times})


        if self.sampling_rate == 6400:
            self.dataset_name = "CREAM_12000Hz"
        elif self.sampling_rate == 12000:
            self.dataset_name = "CREAM6400Hz"
        else:
            raise ValueError(
                "Check the metadata, it does not match the original CREAM dataset (e.g. the frequency attribute)")

        # Compute the average_sampling_rate on this particular day
        self._compute_average_sampling_rate(start_timestamp)

        # Compute the minimum and maximum time for this day, and the respective differences to the day before
        self.minimum_request_timestamp = self.files_metadata_df.iloc[0].Start_timestamp
        self.maximum_request_timestamp = self.files_metadata_df.iloc[-1].Start_timestamp + timedelta(
            seconds=(self.file_duration_sec / self.sampling_rate))

        # Find the day of the dataset
        folder_path = os.path.basename(os.path.normpath(self.dataset_location))  # name of the folder
        date = folder_path.split("-")
        self.day_date = datetime(year=int(date[0]), month=int(date[1]), day=int(date[2]))


        # initiate cache if set to true

    def load_maintenance_events(self, file_path: str = None, filter_day : bool = False) -> pd.DataFrame:
        """
        Load the maintenance event file. The events are sorted by the time they occur.

        Parameters
        ----------
        file_path

        Returns
        -------

        """

        if file_path is None:
            raise ValueError("Specify a file_path, containing the events file")

        data = pd.read_csv(file_path)

        # The timezone of the timestamps need to be from the same type
        # We use the first file of the day_object to get
        timezone = self.get_datetime_from_filepath(self.files[0]).tzinfo

        data.Timestamp = pd.to_datetime(data.Timestamp, box=True)



        ts_array = []
        for i, row in data.iterrows():
            ts = row.Timestamp.tz_convert(timezone)
            ts_array.append(ts)
        data["Timestamp"] = ts_array

        data.sort_values("Timestamp", inplace=True)

        data["Date"] = data.Timestamp.apply(lambda x: x.date())

        if filter_day is True:  # only return the event of the corresponding CREAM day
            data = data[data["Date"] == self.day_date.date()]

        return data

    def load_product_events(self, file_path: str = None, filter_day : bool = False) -> pd.DataFrame:
        """
        Load the product event file. The events are sorted by the time they occur.

        Parameters
        ----------
        file_path

        Returns
        -------

        """

        if file_path is None:
            raise ValueError("Specify a file_path, containing the events file.")

        data = pd.read_csv(file_path)

        # The timezone of the timestamps need to be from the same type
        # We use the first file of the day_object to get
        timezone = self.get_datetime_from_filepath(self.files[0]).tzinfo

        data.Timestamp = pd.to_datetime(data.Timestamp, box=True)

        ts_array = []

        for i, row in data.iterrows():
            ts = row.Timestamp.tz_convert(timezone)
            ts_array.append(ts)
        data["Timestamp"] = ts_array

        data.sort_values("Timestamp", inplace=True)

        data["Date"] = data.Timestamp.apply(lambda x: x.date())

        if filter_day is True:  # only return the event of the corresponding CREAM day
            data = data[data["Date"] == self.day_date.date()]

        return data

    def load_event_labels(self, file_path: str = None, filter_day : bool = False) -> pd.DataFrame:
        """
        Load the labeled electrical events file. The events are sorted by the time they occur.

        Parameters
        ----------
        file_path

        Returns
        -------

        """
        if file_path is None:
            raise ValueError("Specify a file_path, containing the events file.")

        data = pd.read_csv(file_path)
        # The timezone of the timestamps need to be from the same type
        # We use the first file of the day_object to get
        timezone = self.get_datetime_from_filepath(self.files[0]).tzinfo

        data.Timestamp = pd.to_datetime(data.Timestamp, box=True)

        ts_array = []
        for i, row in data.iterrows():
            ts = row.Timestamp.tz_convert(timezone)
            ts_array.append(ts)
        data["Timestamp"] = ts_array

        data.sort_values("Timestamp", inplace=True)

        data["Date"] = data.Timestamp.apply(lambda x: x.date())

        if filter_day is True:  # only return the event of the corresponding CREAM day
            data = data[data["Date"] == self.day_date.date()]



        return data

    def load_file(self, file_path: str, return_noise: bool = False) -> (np.ndarray, np.ndarray):
        """
        Load a file of the CREAM dataset
        If return_noise is specified, the noise channel is also returned. The current is 2-dimensional then.
        The signals get pre-processed before they are returned by this function:
        1. y-direction calibration: we center the signal around zero
        2. calibration_factor: we calibrate the signal by the measurement device specific calibration_factor.
        This calibration_factor is included in the metadata of the files.

        Parameters
        ----------
        file_path (string): path to the file to be loaded
        return_noise (boolean): default=False. If set to True, the current of the noise socket is also returned.

        Returns
        -------
        voltage (ndarray): voltage signal with shape=(1, file_length,). In case of an empty file None is returned.
        current (ndarray): current signal either with shape (1, file_length) or (2, file_length)
                            In case of an empty file None is returned


        """

        voltage = None
        current = None

        # Check if the file is already in the file cache
        if self.use_buffer is True and file_path in self.file_cache:

            voltage = self.file_cache[file_path]["voltage"]
            current = self.file_cache[file_path]["current"]
            return voltage, current

        else:


            # Check if the file is empty (zero bytes): if so return and empty current and voltage array
            if os.stat(file_path).st_size > 0:  # if not empty

                with h5py.File(file_path, 'r', driver='core') as f:

                    voltage_offset, current_offset = self._adjust_amplitude_offset(f)  # y value offset adjustment
                    for name in list(f):
                        signal = f[name][:] * 1.0

                        if name == 'voltage' and voltage_offset is not None:  # the voltage signal

                            voltage = signal - voltage_offset
                            calibration_factor = f[name].attrs['calibration_factor']
                            voltage = np.multiply(voltage, calibration_factor)

                        elif "current1" in name and current_offset is not None:  # the current signal of the coffee maker

                            current = signal - current_offset
                            calibration_factor = f[name].attrs['calibration_factor']
                            current = np.multiply(current, calibration_factor)

                        elif return_noise == True and "current6" in name and current_offset is not None:  # the current signal of the noise channel

                            current_noise = signal - current_offset
                            calibration_factor = f[name].attrs['calibration_factor']
                            current_noise = np.multiply(current_noise, calibration_factor)

                if return_noise is True:
                    current = np.array([current, current_noise])

                # Before returning, check if we store the file in the cache and if we need to delete one instead from the cache
                if self.use_buffer is True:
                    if len(self.file_cache) < self.buffer_size_files:
                        self.file_cache[file_path] = {"voltage" : np.array([voltage]), "current": np.array([current])}
                    else:
                        sorted_filenames = list(self.file_cache.keys())
                        sorted_filenames.sort()
                        del self.file_cache[sorted_filenames[0]] #delete the oldest file
                        # TODO check if worked

                return np.array([voltage]), np.array([current])

            else:  # if empty
                return None, None


    def load_time_frame(self, start_datetime: datetime, duration : float, return_noise: bool = False):

        # Perform initial checks
        if start_datetime < self.minimum_request_timestamp:
            raise ValueError(
                "The requested Time window is smaller then the minimum_request_timestamp of the day object")


        # TODO Duration in seconds only
        end_datetime = start_datetime + timedelta(seconds=duration)


        if end_datetime > self.maximum_request_timestamp:
            raise ValueError("The requested Time window is bigger then the maximum_request_timestamp of the day object")



        # determine all the files that are relevant for the requested time window

        # The index of the first relevant_file: i.e. the last file that is smaller then the start_datetime
        first_file_idx = self.files_metadata_df[self.files_metadata_df.Start_timestamp <= start_datetime].index[-1]

        # The last relevant_file: i.e. the first file that has and End_timestamp that is bigger then the one we need
        last_file_idx = self.files_metadata_df[self.files_metadata_df.End_timestamp >= end_datetime].index[0]


        # Get all the files in between the first and the last file needed
        relevant_files_df = self.files_metadata_df.loc[first_file_idx:last_file_idx]

        if len(relevant_files_df) == 0:
            raise ValueError("The timeframe requested does not lie within the current day!")

        relevant_voltage = []
        relevant_current = []

        for i, row in relevant_files_df.iterrows():

            voltage, current = self.load_file(row.Filename, return_noise=return_noise)

            relevant_voltage.append(voltage)
            relevant_current.append(current)

        # now stack together the relevant signals
        relevant_voltage = np.concatenate(relevant_voltage, axis=1)
        relevant_current = np.concatenate(relevant_current, axis=1)

        # Compute the start_index

        # 1.1 Compute the offset in the first file
        start_index = int(self.get_index_from_timestamp(relevant_files_df.iloc[0].Start_timestamp, start_datetime))
        end_index = int(self.get_index_from_timestamp(relevant_files_df.iloc[-1].Start_timestamp, end_datetime))


        # Get the voltage and current window
        voltage = relevant_voltage[0][start_index:end_index] #there is only one voltage channel

        current = []
        # there are multiple current channels
        for curr in relevant_current: #for each dimension
            current.append(curr[start_index:end_index])

        voltage = np.array(voltage)
        current = np.array(current)


        return voltage, current

    def _compute_average_sampling_rate(self, start_timestamp: datetime):
        """
        Estimate the average sampling rate per day.

        Calculate the difference between the first and last sample of a day based on
        the timestamps of the files.
        """

        duration = self.number_of_files * self.file_duration_sec

        # get the timestamp of last file
        end_file = self.files[-1]
        timezone = self.get_datetime_from_filepath(end_file)
        timezone = timezone.tzinfo

        # get the end file
        with h5py.File(end_file, 'r', driver='core') as f:
            end_timestamp = datetime(
                year=int(f.attrs['year']),
                month=int(f.attrs['month']),
                day=int(f.attrs['day']),
                hour=int(f.attrs['hours']),
                minute=int(f.attrs['minutes']),
                second=int(f.attrs['seconds']),
                microsecond=int(f.attrs['microseconds']),
                tzinfo=timezone)

        self.average_sampling_rate = duration / (end_timestamp - start_timestamp).total_seconds() * self.sampling_rate

    def get_datetime_from_filepath(self, filepath: str):
        """
        Args:
            filepath:

        Returns:
            datetime_object

        """

        filename = os.path.basename(filepath)  # get the filename
        string_timestamp = "-".join(filename.split("-")[2:-1])

        datetime_object = datetime.strptime(string_timestamp, '%Y-%m-%dT%H-%M-%S.%fT%z')  # string parse time

        return datetime_object

    def get_index_from_timestamp(self, start_timestamp: datetime, event_timestamp: datetime):
        """
        Returns the index of the event, represented by the event_timestamp, relativ to the start_timestamp (i.e. start timestamp of the file of interest e.g.)
        The event_timestamp is expected to be a pandas Timestam
        """

        sec_since_start = event_timestamp - start_timestamp
        event_index = sec_since_start.total_seconds() * (self.sampling_rate)  # and # multiply by samples per second

        return int(event_index)


    def get_timestamp_from_index(self, start_timestamp: datetime, event_index: int):
        seconds_per_sample = 1 / self.sampling_rate # 1 second / samples = seconds per sample
        time_since_start = event_index * seconds_per_sample
        event_ts = start_timestamp + timedelta(seconds=time_since_start)

        return event_ts


    def _adjust_amplitude_offset(self, file: h5py.File):
        """
        Computes the mean per period to get an estimate for the offset in each period.
        This is done for the voltage signal.
        The period length is computed using the average_sampling_rate, hence this can deviate from the
        theoretical period length. Therefore, we zero pad the voltage signal to get full periods again before computing
        the mean.
        Then we use the estimate per period, to linearly interpolate the mean values per period, to get an offset value
        per sample point in the signal. We then use the offset of the voltage to compute the offset of the current by multiplying
        it by the crest-coefficient of 1/sqrt(2), i.e., approx. 0.7 .

        """

        length = len(file['voltage'])

        # Compute the average period_length, using the average_sampling_rate
        period_length = round(self.average_sampling_rate / 50)

        # Get the missing samples, opposed to the optimal number of periods in the signal
        remainder = divmod(length, period_length)[1]

        if remainder != 0:  # no zero padding necessary
            voltage = np.pad(file['voltage'][:], (0, period_length - remainder), 'constant',
                             constant_values=0)  # zero padding

        voltage = voltage.reshape(-1, period_length)  # the single periods, period wise reshape
        mean_values_per_period = voltage.mean(axis=1)  # compute the mean per period

        # Create x values for the interpolation
        x_per_period = np.linspace(1, length, len(mean_values_per_period), dtype=np.int)  # number of periods
        x_original = np.linspace(1, length, length, dtype=np.int)

        # build a linear interpolation, that interpolates for each period witch offset it should have
        # for each of the datapoints, interpolate the offset
        voltage_offset = interpolate.interp1d(x_per_period, mean_values_per_period)(x_original)
        current_offset = voltage_offset * 1 / np.sqrt(2)  # roughly * 0.7

        return voltage_offset, current_offset

