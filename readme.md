


*This documentation will be expanded.*


## Quantification of thrip damage patterns to leafs

This project analyzes multi-channel leaf images to quantify thrip feeding damage patterns. The pipeline detects leaf and damage masks, computes spatial metrics (including island counts/distances, radial distributions, autocorrelation, roundness, and total damage area in pixels and optional cm²), and exports both summary tables and diagnostic plots for synthetic and real datasets.

## To install

To find out how to get started with Python and related required software to 
conveniently run scripts, please check out our [blog post](https://www.biodsc.nl/posts/installing_conda_python.html) about this.

Assuming you already have Conda installed and your preferred environment set up, install the following libraries to be able to run the scripts in this repository:

```bash

conda install -c conda-forge numpy pandas scipy scikit-image matplotlib seaborn pillow opencv openpyxl -y
```


## To run

To run this script, check out the files:
- `leafstats_project_example_1channel.py`, which shows how to analyze a dataset where 1 channel was recorded to identify the leaf and the damage done by thrips.
- `leafstats_projects_example_3channels.py`, which shows how to analyze a dataset where 3 channels were taken, 1 for identifying the leaf, and 1 for quantifying the damage.
