import random
import argparse
import math
from ezr import the, DATA, csv, stats
from time import time

"""
HW 3 Testing an Research Hypothesis

Sanjit Verma (skverma), Arul Sharma (asharm52), Sarvesh Soma (ssomasu)

Usage: 
    
"""

scoring_policies = [
    ('exploit', lambda B, R: B - R),
    ('explore', lambda B, R: (math.exp(B) + math.exp(R)) / (1e-30 + abs(math.exp(B) - math.exp(R))))
]

def parse_arguments():
    """
    this methoid will parse command-line arguments
    """
    parser = argparse.ArgumentParser(description="Experiments on Low and High Dimensional Datasets")
    parser.add_argument('-t', '--train', required=True, help="Path to the dataset (CSV)")
    return parser.parse_args()


def is_low_dimension(dataset, threshold=6):
    """
    determines if the dataset is low-dimensional
    """
    return len(dataset.cols.x) < threshold


def load_dataset(dataset_path):
    """
    loads the dataset from the CSV 
    """
    try:
        dataset = DATA().adds(csv(dataset_path))
        return dataset
    except Exception as e:
        raise


def dumb_method(num_rows, dataset):
    """
    Select num rows randomly and sorts by Chebyshev distance
    """
    sampled_rows = random.choices(dataset.rows, k=num_rows)
    sorted_rows = sorted(sampled_rows, key=dataset.chebyshev)

    # Test: Top item should have min Chebyshev distance
    assert abs(dataset.chebyshev(sorted_rows[0]) - min(dataset.chebyshev(row) for row in sampled_rows)) < 1e-6
    return sorted_rows


def smart_method(num_rows, dataset, score_func):
    """
    use active learning to select num row based on the score function.
    """
    selected_rows = dataset.shuffle().activeLearning(score=score_func)

    if len(selected_rows) > num_rows:
        selected_rows = selected_rows[:num_rows] 
    if len(selected_rows) < num_rows:
        selected_rows += random.choices(dataset.rows, k=num_rows - len(selected_rows))

    # Test: Ensure correct number of rows selected
    assert len(selected_rows) == num_rows, "Active learning selection returned incorrect number of rows"
    return selected_rows[:num_rows]


def run_trials(method, num_rows, dataset, trials=20, score_func=None):
    """
    Runs trials for  the dumb or smart method
    """
    results = []
    for _ in range(trials):
        if method == 'dumb':
            selected = dumb_method(num_rows, dataset)
        elif method == 'smart':
            selected = smart_method(num_rows, dataset, score_func)
        results.append(dataset.chebyshev(selected[0]))

    # Test: Ensure number of trials
    assert len(results) == trials, f"{method.capitalize()} method did not run {trials} trials."
    return results


def execute_experiment(dataset, dimension_label):
    """
    Runs experiment for different N values using both methods
    """
    results = []
    trials = 20
    N_values = [20, 30, 40, 50]

    for N in N_values:

        # dumb
        dumb_chebyshev = run_trials('dumb', N, dataset, trials)
        results.append(stats.SOME(dumb_chebyshev, f"{dimension_label}_dumb,{N}"))

        # smart
        for policy_name, score_func in scoring_policies:
            smart_chebyshev = run_trials('smart', N, dataset, trials, score_func=score_func)
            results.append(stats.SOME(smart_chebyshev, f"{dimension_label}_smart_{policy_name},{N}"))

    # Test: Ensure shuffle changes order
    original_order = dataset.rows[:]
    dataset.shuffle()
    assert original_order != dataset.rows, "Shuffle did not change the order of the dataset"

    stats.report(results, 0.01)


def main():
    args = parse_arguments()

    # load data
    dataset = load_dataset(args.train)

    # check dim of data
    dimension_label = "low_dimension" if is_low_dimension(dataset) else "high_dimension"
    execute_experiment(dataset, dimension_label)

if __name__ == "__main__":
    main()
