import sys
import random
from ezr import the, DATA, csv, dot, COL, NUM, SYM

def hybrid_acquisition(train_file):
    """
    Hybrid acquisition function that combines uncertainty, diversity, and error-reduction techniques.
    This function processes the training data file, and applies the hybrid method to select data points
    for labeling, balancing between uncertainty, diversity, and error-reduction.

    execute like: python3 hybridLabels.py data/misc/****.csv > /Users/sanjitverma/git/ezr/output/*****.txt
    """
    # Load data
    d = DATA().adds(csv(train_file))
    
    # Step 2: Implement uncertainty sampling (using Monte Carlo Dropout or similar)
    uncertain_rows = uncertainty_sampling(d)

    # Step 3: Implement diversity sampling (using clustering techniques)
    diverse_rows = diversity_sampling(d, uncertain_rows)
    
    # Step 4: Apply error-reduction strategies to prioritize samples
    prioritized_rows = error_reduction_sampling(d, diverse_rows)
    
    # Step 5: Select samples for labeling
    selected_samples = select_samples(prioritized_rows)
    
    # Return the samples chosen by the hybrid method
    return selected_samples

def uncertainty_sampling(data):
    """
    Perform uncertainty sampling by calculating the uncertainty of each row using Monte Carlo Dropout.
    """
    uncertain_rows = []
    for row in data.rows:
        # Compute uncertainty (e.g., by variance in prediction or Monte Carlo Dropout)
        uncertainty_score = compute_uncertainty(data, row)
        uncertain_rows.append((row, uncertainty_score))
    
    # Sort by uncertainty score (higher uncertainty prioritized)
    uncertain_rows.sort(key=lambda x: x[1], reverse=True)
    
    return [row[0] for row in uncertain_rows]

def compute_uncertainty(data, row):
    """
    Compute the uncertainty for a given row.
    You can implement Monte Carlo Dropout or use variance-based uncertainty.
    """
    # For simplicity, let's assume we have a function that computes uncertainty
    # Placeholder logic here
    return random.random()

def diversity_sampling(data, uncertain_rows):
    """
    Perform diversity sampling by clustering the uncertain rows and ensuring selected rows
    represent diverse regions of the data space.
    """
    diverse_rows = []
    
    # Cluster uncertain rows using hierarchical agglomerative clustering (HAC)
    cluster = data.cluster(uncertain_rows, sortp=True)  # This returns a single cluster object
    
    # Traverse the cluster and gather leaf nodes
    for node, is_leaf in cluster.nodes():  # .nodes() returns both nodes and a flag if it's a leaf
        if is_leaf:
            diverse_rows.append(select_diverse_sample(node))
    
    return diverse_rows

def select_diverse_sample(cluster_node):
    """
    Select a diverse sample from the given cluster node.
    """
    return cluster_node.mid

def error_reduction_sampling(data, diverse_rows):
    """
    Prioritize samples based on the expected error reduction.
    """
    prioritized_rows = []
    for row in diverse_rows:
        error_reduction_score = compute_error_reduction(data, row)
        prioritized_rows.append((row, error_reduction_score))
    
    # Sort by error reduction score (higher reduction prioritized)
    prioritized_rows.sort(key=lambda x: x[1], reverse=True)
    
    return [row[0] for row in prioritized_rows]

def compute_error_reduction(data, row):
    """
    Compute the expected error reduction for a given row.
    """
    # Placeholder logic for error reduction calculation
    return random.random()

def select_samples(prioritized_rows, num_samples=10):
    """
    Select a set number of samples from the prioritized rows.
    """
    return prioritized_rows[:num_samples]

if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python3 hybridLabels.py <train_file>")
        sys.exit(1)
    
    train_file = sys.argv[1]
    
    # Run hybrid acquisition on the training file
    selected_samples = hybrid_acquisition(train_file)
    
    # Output the selected samples
    for sample in selected_samples:
        print(sample)
