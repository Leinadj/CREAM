# CREAM Code Repository
## Dataset Download: https://mediatum.ub.tum.de/1534850
### Fully labeled dataset for evaluating condition monitoring and other manufacturing related tasks, based on electrical signal analysis.
### The dataset contains continuous voltage and current measurements at high sampling rates of an industry-grade coffeemaker, mimicking industrial processes.
### The coffeemaker closely resembles industrial machinery and can be used to develop and evaluate manufacturing related algorithms, such as for condition monitoring, event detection, etc. .
### We provide the data sampled at 6.4 kSps and additional event information: 92449 expert-labeled component-level electrical events, 1476 machine-generated product events and 2868 machine-generated maintenance-related events.

Utility classes and functions for the CREAM Dataset and the corresponding data descriptor.

Folder structure:

1. *data_utility*: Utility class for loading and processing the CREAM dataset
2. *labeling_tools*: Jupyter notebook based tools that were used to label CREAM
3. *manuscript*: Scripts for creating the plots in the data descriptor
4. *technical_validation*: Scripts used to perform the technical validation of the data and the labels.
5. *requirements.txt*: Requirements necessary to execute the code in this repository.
    
## For questions please contact daniel.jorde@tum.de
