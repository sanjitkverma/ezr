import random
import argparse
from ezr import the, DATA, csv, stats
from time import time

"""
HW 3 Testing an Research Hypothesis

Sanjit Verma (skverma), Arul Sharma (asharm52), Sarvesh Soma (ssomasu)

Usage:
    python3.13 extend_hw3.py -t <path_to_csv>
"""

def random_selection(num_rows, dataset):
    sampled_rows = random.choices(dataset.rows, k=num_rows)
    sorted_rows = sorted(sampled_rows, key=lambda row: dataset.chebyshev(row))
    
    top_chebyshev_distance = dataset.chebyshev(sorted_rows[0])
    min_chebyshev_distance = min(dataset.chebyshev(row) for row in sampled_rows)

    # Test 1: Check if chebyshevs().rows[0] has the minimum Chebyshev distance
    assert abs(top_chebyshev_distance - min_chebyshev_distance) < 1e-6, "Top item in chebyshevs() sorting is incorrect"

    return sorted_rows

def execute_experiment(dataset_file, dimension_label):
    the.train = dataset_file
    dataset = DATA().adds(csv(the.train))

    results = []

    for num in [20, 30, 40, 50]:
        random_guesses = [random_selection(num, dataset) for _ in range(20)]
        
        # Test 2: Check if random_guesses length matches num
        assert all(len(guess) == num for guess in random_guesses), "Random guesses have wrong length"
        
        random_chebyshev = [dataset.chebyshev(guess[0]) for guess in random_guesses]

        the.Last = num
        active_learning_guesses = [dataset.shuffle().activeLearning() for _ in range(20)]
        
        # Test 3: Check if active learning guesses length match num
        assert all(len(guess) == num for guess in active_learning_guesses), "Active learning guesses have wrong length"
        
        active_chebyshev = [dataset.chebyshev(guess[0]) for guess in active_learning_guesses]

        # Test 4: Check if 20 iteration are run
        assert len(random_guesses) == 20, "Dumb guesses not run 20 times"
        assert len(active_learning_guesses) == 20, "Smart guesses not run 20 times"

        # Test 5: ensure shuffle changes the order
        original_order = dataset.rows[:]
        dataset.shuffle()
        assert original_order != dataset.rows, "Shuffle did not change the order of the dataset"
        
        results += [
            stats.SOME(random_chebyshev, f"{dimension_label}_random,{num}"),
            stats.SOME(active_chebyshev, f"{dimension_label}_active,{num}")
        ]

    stats.report(results, 0.01)

def is_low_dimension(dataset):
    return len(dataset.cols.x) < 6

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiments on low and high dim datasets")
    parser.add_argument('-t', '--train', required=True, help="Path to the dataset (CSV)")
    args = parser.parse_args()

    random.seed(the.seed)
    dataset = DATA().adds(csv(args.train))

    if is_low_dimension(dataset):
        print(f"running experiment for low dim dataset: {args.train}")
        execute_experiment(args.train, dimension_label="low_dimension")
    else:
        print(f"running experiment for high dim dataset: {args.train}")
        execute_experiment(args.train, dimension_label="high_dimension")
