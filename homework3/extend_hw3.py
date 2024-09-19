import random
import argparse
import math
import statistics
from ezr import the, DATA, csv, stats
from time import time

"""
HW 3 Testing an Research Hypothesis

Sanjit Verma (skverma), Arul Sharma (asharm52), Sarvesh Soma (ssomasu)

single file use from ezr directory: python3.13 -B homework3/extend_hw3.py -t data/optimize/{folder name}/{file}
    
"""

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', required=True, help="path to datga")
    return parser.parse_args()

def load_dataset(file_path):
    try:
        return DATA().adds(csv(file_path))
    except Exception as e:
        raise RuntimeError(f"failed to load data: {file_path}: {e}")

def run_smart(dataset, scoringP, repeats=20):
    somes = []
    for what, how in scoringP:
        for N in [20, 30, 40, 50]:
            start = time()
            result = []
            runs = 0

            for _ in range(repeats):
                original_rows = dataset.rows[:]

                tmp = dataset.shuffle().activeLearning(score=how)

                # TEST: Does d.shuffle() really jiggle the order of the data?
                shuffled_rows = dataset.rows[:]
                is_shuffled = original_rows != shuffled_rows
                assert is_shuffled, "Shuffle did not change the order of the data."

                # TEST: Are the results of smart method the right length (i.e., N)?
                tmp = tmp[:N]

                runs += len(tmp)
                result.append(dataset.chebyshev(tmp[0]))

            elapsedt = (time() - start) / repeats
            tag = f"smart,{N}"
            print(f"{tag} : {elapsedt:.2f} secs")
            somes.append(stats.SOME(result, tag))

            # TEST: Does you code really run some experimental treatment 20 times for statistical validity?
            assert len(result) == repeats, f"Expected {repeats} results for smart, got {len(result)}."

    return somes

def run_dumb(dataset, repeats=20):
    somes = []
    for N in [20, 30, 40, 50]:
        start = time()
        result = []
        runs = 0

        for _ in range(repeats):
            tmp = random.choices(dataset.rows, k=N)

            # TEST: Are dumb lists the right length (i.e., N)?
            assert len(tmp) == N, f"Dumb method not the expected rows."

            tmp_sorted = sorted(tmp, key=lambda row: dataset.chebyshev(row))

            # TEST: Does chebyshevs().rows[0] return the top item in that sort?
            assert tmp_sorted[0] == sorted(tmp, key=lambda row: dataset.chebyshev(row))[0], \
                "Chebyshev sorting failed."

            runs += len(tmp_sorted)
            result.append(dataset.chebyshev(tmp_sorted[0]))

        elapsedt = (time() - start) / repeats
        tag = f"dumb,{N}"
        print(f"{tag} : {elapsedt:.2f} secs")
        somes.append(stats.SOME(result, tag))

        # TEST: Does you code really run some experimental treatment 20 times for statistical validity?
        assert len(result) == repeats, f"Expected {repeats} results for dumb, got {len(result)}."

    return somes

def run_experiment(dataset):
    b4 = [dataset.chebyshev(row) for row in dataset.rows]

    asIs = statistics.median(b4) if b4 else 0
    div = statistics.stdev(b4) if len(b4) > 1 else 0
    print(f"asIs\t: {asIs:.3f}")
    print(f"div\t: {div:.3f}")
    print(f"rows\t: {len(dataset.rows)}")
    print(f"xcols\t: {len(dataset.cols.x)}")
    print(f"ycols\t: {len(dataset.cols.y)}\n")

    if len(dataset.cols.x) < 6:  
        print("low_dimension")
    else:
        print("high_dimension")
        
    somes = [stats.SOME(b4, f"asIs,{len(dataset.rows)}")]

    scoringP = [
        ('smart', lambda B, R: B - R),
        ('smart', lambda B, R: (math.exp(B) + math.exp(R)) / (1e-30 + abs(math.exp(B) - math.exp(R))))
    ]

    somes += run_smart(dataset, scoringP)
    somes += run_dumb(dataset)
    stats.report(somes, 0.01)

def main():
    args = parse_arguments()
    dataset = load_dataset(args.train)
    run_experiment(dataset)

if __name__ == "__main__":
    main()
