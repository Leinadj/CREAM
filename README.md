# CREAM Code Repository
#### Fully labeled dataset for evaluating condition monitoring and other manufacturing related tasks, based on electrical signal analysis.
#### The dataset contains continuous voltage and current measurements at high sampling rates of two industry-grade coffeemakers, mimicking industrial processes.
#### The coffeemakers closely resemble industrial machinery and can be used to develop and evaluate manufacturing related algorithms, such as for condition monitoring, event detection, etc. .
#### We provide the data sampled at 6.4 kSps and additional event information: 370600 expert-labeled component-level electrical events, 1735 machine-generated product events and 3646 machine-generated maintenance-related events. 

This repository contains Utility classes and functions for the CREAM Dataset and the corresponding data descriptor.

Folder structure:

1. *data_utility*: Utility class for loading and processing the CREAM dataset
2. *data_colelction*: Data collection scripts of the MEDAL data acquisition dataset. Original scripts can be found in the BLOND dataset by Thomas Kriechbaumer.
3. *labeling_tools*: Jupyter notebook based tools that were used to label CREAM
4. *manuscript*: Scripts for creating the plots in the data descriptor
5. *technical_validation*: Scripts used to perform the technical validation of the data and the labels.
6. *requirements.txt*: Requirements necessary to execute the code in this repository.
    
## For questions please contact daniel.jorde@tum.de
