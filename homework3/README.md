# README: How to run our code & Experiement observations

This guide will help you run the code for the project on GitHub Codespaces, using the `24Aug14` branch. It covers both Part 1 (running a single file) and Part 2 (running experiments over multiple datasets).

Sanjit Verma (skverma), Arul Sharma (asharm52), Sarvesh Soma (ssomasu)

## Prerequisites

Ensure you have the correct environment set up for Python 3.13 & Codespaces as per the instructions provided in <https://txt.github.io/se24fall/03code.html#try-it-for-yourself>.

### Setting Up GitHub Codespace

1.  Boot up the GitHub Codespace in the `24Aug14` branch.

2.  Ensure that the correct version of Python (Python 3.13) is available and active in the Codespace.

## Running Part 1: Single File Execution

To run the experiment on a single file from the `/workspaces/ezr` directory, use the following command:

```bash
python3.13 -B homework3/extend_hw3.py -t data/optimize/{folder_name}/{file_name}
```

Replace `{folder_name}` and `{file_name}` with the actual folder and file you want to test.


## Running Part 2: Running Experiments Over Many Datasets

### Step 1: Sorting Files by Dimension

To run the experiments over multiple datasets, you first need to sort the files into high and low dimensions. Use the following script to sort the files:

1.  Navigate to the `homework3` directory.

2.  Run script named `sort.sh`. It will go throughs all the files in `data/optimize` and filter them for high or low. Run the script by navigating to the `/workspaces/ezr/homework3` directory and executing the following command:

    ```bash
    bash sort.sh
    ```

3.  This will generate a folder called `sortedFiles` inside `homework3`, with two subfolders: `high` and `low`

### Step 2: Running the Experiments

Once the files are sorted, run the `runExperiments.sh` script in the `homework3` directory, which queues up all tasks for both high and low dimension files. This will take between 10 and 30 minutes. 

This script automatically outputs the task results to the `/workspaces/ezr/tmp` directory, which will contain two subfolders: `high` and `low`, storing all the experiment data

### Step 3: Generating Final Tables

After the experiments are complete, you can generate the final tables by running the `run_rqsh.sh` script:

1.  run the script `run_rqsh.sh` inside the `/workspaces/ezr/homework3` directory 
2.  select opption for either high or low


## Results

### Low Dimensional


### High Dimensional


## Discussion