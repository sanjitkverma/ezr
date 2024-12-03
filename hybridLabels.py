import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from stats import SOME, report

from ezr import DATA, csv  

'''
Sample Command: python3 hybridLabels.py data/config/SS-C.csv > ~/hybrid_method_output.txt
'''

def hybrid_acquisition(df, labeled_indices, num_samples=5, n_clusters=3, samples_per_cluster=2):
    """
    Hybrid acquisition function that combines uncertainty, diversity, and error-reduction techniques.
    Selects a specified number of samples for labeling.
    
    Parameters:
    - df: pandas DataFrame with all data.
    - labeled_indices: set of currently labeled indices.
    - num_samples: total number of samples to select.
    - n_clusters: number of clusters for diversity sampling.
    - samples_per_cluster: number of samples to select from each cluster.
    
    Returns:
    - selected_samples: list of selected data rows.
    """
    # Uncertainty Sampling
    uncertain_indices = uncertainty_sampling(df, k=5, labeled_indices=labeled_indices)
    
    # Diversity Sampling
    diverse_indices = diversity_sampling(df, uncertain_indices, n_clusters=n_clusters, samples_per_cluster=samples_per_cluster)
    
    # Error Reduction Sampling
    prioritized_indices = error_reduction_sampling(df, diverse_indices)
    
    # Select top samples
    selected_samples = select_samples(df, prioritized_indices, num_samples=num_samples)
    
    return selected_samples

def preprocess_data(df):
    """
    Preprocess the DataFrame by encoding categorical variables and handling missing values.
    
    Parameters:
    - df: pandas DataFrame containing the dataset.
    
    Returns:
    - df_processed: Preprocessed pandas DataFrame with all numeric features.
    """
    # Handle special characters by treating them as missing values
    df = df.replace('?', np.nan)
    df = df.infer_objects(copy=False)
    
    # Identify non-numeric columns
    non_numeric_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Non-numeric columns: {non_numeric_cols}")
    
    # Encode categorical features
    if non_numeric_cols:
        # Label Encoding (for ordinal data)
        label_encoders = {}
        for col in non_numeric_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"Encoded column '{col}' with Label Encoding.")
    
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        print("Handled missing values with mean imputation.")
    
    return df

def uncertainty_sampling(df, k=5, labeled_indices=None):
    """
    Select top uncertain samples excluding already labeled data.
    
    Parameters:
    - df: pandas DataFrame with all numeric features.
    - k: Number of neighbors for density estimation.
    - labeled_indices: set of currently labeled indices.
    
    Returns:
    - uncertain_indices: list of top uncertain sample indices.
    """
    if labeled_indices is None:
        labeled_indices = set()
    uncertain_rows = []
    target_col = df.columns[-1]
    for index, row in df.iterrows():
        if index in labeled_indices:
            continue
        uncertainty_score = compute_uncertainty_density(df, row, k, target_col)
        uncertain_rows.append((index, uncertainty_score))
    uncertain_rows.sort(key=lambda x: x[1], reverse=True)
    sorted_indices = [idx for idx, score in uncertain_rows]
    top_uncertain = sorted_indices[:10]  # Select top 10 uncertain
    print(f"Top 10 most uncertain samples: {top_uncertain}")
    return top_uncertain

def compute_uncertainty_density(df, row, k=5, target_col=None):
    """
    Compute uncertainty based on density using k-NN.
    
    Parameters:
    - df: pandas DataFrame with all numeric features.
    - row: pandas Series representing the data row.
    - k: Number of neighbors to consider.
    - target_col: Name of the target column.
    
    Returns:
    - uncertainty_score: Average distance to k-nearest neighbors.
    """
    if target_col is None:
        target_col = df.columns[-1]
    features = df.drop(columns=target_col).values
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(features)
    distances, indices = nbrs.kneighbors([row.drop(target_col).values])
    avg_distance = np.mean(distances[0][1:]) 
    return avg_distance

def diversity_sampling(df, uncertain_indices, n_clusters=3, samples_per_cluster=2):
    """
    Perform diversity sampling by clustering uncertain samples and selecting multiple samples per cluster.
    
    Parameters:
    - df: pandas DataFrame with all numeric features.
    - uncertain_indices: list of row indices sorted by uncertainty.
    - n_clusters: number of clusters for diversity sampling.
    - samples_per_cluster: number of samples to select from each cluster.
    
    Returns:
    - diverse_indices: list of diverse row indices selected from each cluster.
    """
    diverse_indices = []
    features = df.loc[uncertain_indices].drop(columns=df.columns[-1]).values
    max_clusters = len(uncertain_indices) // samples_per_cluster
    n_clusters = min(n_clusters, max_clusters) if max_clusters > 0 else 1
    if n_clusters == 0:
        n_clusters = 1
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features)
    labels = kmeans.labels_
    for cluster_idx in range(n_clusters):
        cluster_member_indices = np.where(labels == cluster_idx)[0]
        for i in range(min(samples_per_cluster, len(cluster_member_indices))):
            most_uncertain_idx = cluster_member_indices[i]
            diverse_indices.append(uncertain_indices[most_uncertain_idx])
    print(f"Selected {len(diverse_indices)} diverse samples from clusters.")
    return diverse_indices

def error_reduction_sampling(df, diverse_indices, num_simulations=10):
    """
    Prioritize samples based on expected error reduction.
    
    Parameters:
    - df: pandas DataFrame with all numeric features.
    - diverse_indices: list of diverse row indices selected from clustering.
    - num_simulations: number of simulations to run.
    
    Returns:
    - prioritized_indices: list of row indices sorted by error reduction score.
    """
    prioritized_rows = []
    for idx in diverse_indices:
        error_reduction_score = compute_error_reduction(df, idx, num_simulations)
        prioritized_rows.append((idx, error_reduction_score))
    prioritized_rows.sort(key=lambda x: x[1], reverse=True)
    prioritized_indices = [idx for idx, score in prioritized_rows]
    print(f"Prioritized {len(prioritized_indices)} samples based on error reduction.")
    return prioritized_indices

def compute_error_reduction(df, idx, num_simulations=10):
    """
    Simulate error reduction for a given sample.
    
    Parameters:
    - df: pandas DataFrame with all numeric features.
    - idx: index of the data row.
    - num_simulations: number of simulations to run.
    
    Returns:
    - error_reduction_score: average simulated error reduction.
    """
    row = df.loc[idx]
    target_col = df.columns[-1]
    initial_uncertainty = compute_uncertainty_density(df, row, k=5, target_col=target_col)
    expected_gradient = compute_expected_gradient(df, row)
    error_reduction_estimates = []
    for i in range(num_simulations):
        simulated_error = initial_uncertainty - (expected_gradient + 0.01 * i)
        error_reduction_estimates.append(simulated_error)
    error_reduction_score = np.mean(error_reduction_estimates)
    return max(0, error_reduction_score)

def compute_expected_gradient(df, row):
    """
    Compute a deterministic expected gradient norm based on feature values.
    
    Parameters:
    - df: pandas DataFrame with all numeric features.
    - row: pandas Series representing the data row.
    
    Returns:
    - gradient_norm: a normalized gradient norm.
    """
    gradient_norm = np.linalg.norm(row.drop(df.columns[-1]).values) / 10 
    return gradient_norm

def select_samples(df, prioritized_indices, num_samples=5):
    """
    Select the top 'num_samples' based on error reduction scores.
    
    Parameters:
    - df: pandas DataFrame with all numeric features.
    - prioritized_indices: list of row indices sorted by error reduction score.
    - num_samples: total number of samples to select.
    
    Returns:
    - selected_samples: list of selected data rows.
    """
    selected_indices = prioritized_indices[:num_samples]
    selected_samples = df.loc[selected_indices].values.tolist()
    print(f"Selected top {len(selected_samples)} samples for labeling.")
    return selected_samples

def adaptive_sample_selection(current_r2, previous_r2, base_samples=5, max_samples=15, min_samples=2):
    """
    Adjust the number of samples to select based on model performance improvement.
    
    Parameters:
    - current_r2: Current R² score of the model.
    - previous_r2: Previous R² score of the model.
    - base_samples: Base number of samples to select.
    - max_samples: Maximum number of samples to select.
    - min_samples: Minimum number of samples to select.
    
    Returns:
    - adjusted_num_samples: Adjusted number of samples to select.
    """
    improvement = current_r2 - previous_r2
    if improvement < 0.01:
        # If improvement is low, select fewer samples
        adjusted_num_samples = max(min_samples, base_samples - 1)
    elif improvement > 0.05:
        # If improvement is high, select more samples
        adjusted_num_samples = min(max_samples, base_samples + 1)
    else:
        adjusted_num_samples = base_samples
    return adjusted_num_samples

def active_learning_loop_with_model(train_file, 
                                    performance_threshold=0.85, 
                                    max_iterations=20,  
                                    num_samples_per_iteration=20, 
                                    n_clusters=3, 
                                    samples_per_cluster=5,
                                    improvement_threshold=0.01, 
                                    patience=3 
                                   ):
    """
    Active learning loop that iteratively selects samples for labeling.
    Stops when performance_threshold is met, or no improvement is observed over 'patience' iterations,
    or max_iterations are reached.
    
    Parameters:
    - train_file: Path to the training CSV file.
    - performance_threshold: Desired model R2 score to stop labeling.
    - max_iterations: Maximum number of active learning iterations.
    - num_samples_per_iteration: Initial number of samples to select in each iteration.
    - n_clusters: Number of clusters for diversity sampling.
    - samples_per_cluster: Number of samples to select from each cluster.
    - improvement_threshold: Minimum required improvement in R2 to consider as progress.
    - patience: Number of consecutive iterations allowed without improvement before stopping.
    
    Returns:
    - all_selected_samples: List of all selected samples across iterations.
    """
    # Load and preprocess data
    d = DATA().adds(csv(train_file))
    column_names = [col.txt for col in d.cols.all]
    df = pd.DataFrame(d.rows, columns=column_names)
    df = preprocess_data(df)
    
    # Determine target column
    target_col = df.columns[-1]
    print(f"Target column: {target_col}")
    
    # Initialize labeled and unlabeled sets
    labeled_indices = set()
    all_indices = set(df.index)
    initial_labels = 25  # Start with 5 labeled samples
    initial_labeled = np.random.choice(list(all_indices), size=initial_labels, replace=False)
    labeled_indices.update(initial_labeled)
    print(f"Initial labeled indices: {initial_labeled}")
    
    # Split into training and validation
    train_df = df.loc[list(labeled_indices)]  # Convert set to list
    unlabeled_indices = all_indices - labeled_indices
    val_df = df.drop(labels=labeled_indices).sample(frac=0.2, random_state=42)
    print(f"Validation set size: {len(val_df)}")
    
    # Train initial model
    model = RandomForestRegressor(random_state=42)
    model.fit(train_df.drop(columns=target_col), train_df[target_col])
    
    # Evaluate initial performance
    predictions = model.predict(val_df.drop(columns=target_col))
    r2 = r2_score(val_df[target_col], predictions)
    mse = mean_squared_error(val_df[target_col], predictions)
    print(f"Initial Model R2: {r2:.2f}, MSE: {mse:.2f}")
    
    all_selected_samples = []

    mse_stats = SOME(txt='MSE')
    r2_stats = SOME(txt='R2')
    iteration_stats = []
    
    # Initialize variables for early stopping
    best_r2 = r2
    no_improve_count = 0
    
    previous_r2 = r2  # To track improvement for adaptive sample selection
    
    for iteration in range(max_iterations):
        print(f"\n--- Active Learning Iteration {iteration + 1} ---")
        print(f"Current R2: {r2:.2f}, Previous R2: {previous_r2:.2f}")
        
        # Adaptive sample selection based on improvement
        if iteration > 0:
            num_samples_per_iteration = adaptive_sample_selection(r2, previous_r2, 
                                                                   base_samples=5, 
                                                                   max_samples=15, 
                                                                   min_samples=2)
            print(f"Adaptive number of samples for this iteration: {num_samples_per_iteration}")
        
        selected_samples = hybrid_acquisition(df, labeled_indices, num_samples=num_samples_per_iteration, 
                                             n_clusters=n_clusters, samples_per_cluster=samples_per_cluster)
        # Find indices of selected samples
        selected_indices = []
        for sample in selected_samples:
            # Find the index of the sample
            # Note: To handle floating point precision, use np.allclose or similar if exact match fails
            match = df[(df.drop(columns=target_col) == sample[:-1]).all(axis=1)]
            if not match.empty:
                selected_idx = match.index[0]
                selected_indices.append(selected_idx)
            else:
                print(f"Sample {sample} not found in DataFrame.")
        
        all_selected_samples.extend(selected_samples)
        labeled_indices.update(selected_indices)
        print(f"Selected indices for labeling: {selected_indices}")
        print(f"Total labeled samples: {len(labeled_indices)}")
        
        if len(labeled_indices) == len(df):
            print("All samples have been labeled. Stopping active learning.")
            break
        
        # Retrain model with new labels
        train_df = df.loc[list(labeled_indices)]  # Convert set to list
        model.fit(train_df.drop(columns=target_col), train_df[target_col])
        
        # Evaluate model performance
        predictions = model.predict(val_df.drop(columns=target_col))
        r2 = r2_score(val_df[target_col], predictions)
        mse = mean_squared_error(val_df[target_col], predictions)
        print(f"Iteration {iteration + 1} Model R2: {r2:.2f}, MSE: {mse:.2f}")

        # Add to statistical summaries
        mse_stats.add(mse)
        r2_stats.add(r2)
        iteration_stats.append((iteration + 1, mse, r2))
        
        # Check for statistical significance
        if iteration > 0:
            previous_r2_stat = SOME([iteration_stats[-2][2]])
            current_r2_stat = SOME([r2])
            if not previous_r2_stat.bootstrap(current_r2_stat):
                print("The improvement in R² is statistically significant.")
                no_improve_count = 0
            else:
                print("No significant improvement in R².")
                no_improve_count += 1
                if no_improve_count >= patience:
                    print("No significant improvement over several iterations. Stopping active learning.")
                    break
        else:
            no_improve_count = 0
        
        # Update previous_r2 for adaptive sample selection
        previous_r2 = r2
        
        # Check performance threshold
        if r2 >= performance_threshold:
            print(f"Desired performance threshold of {performance_threshold*100}% R2 reached. Stopping active learning.")
            break
    
    somes = [mse_stats, r2_stats]
    report(somes)
    print(f"\nTotal Labels Selected: {len(all_selected_samples)}")
    return all_selected_samples

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 hybridLabels.py <train_file>")
        sys.exit(1)
    
    train_file = sys.argv[1]
    
    performance_threshold = 0.90
    max_iterations = 20 
    num_samples_per_iteration = 5 
    n_clusters = 3
    samples_per_cluster = 2
    improvement_threshold = 0.01  # Minimum required improvement in R2
    patience = 2 
    
    # Run active learning loop with model integration
    all_selected_samples = active_learning_loop_with_model(
        train_file,
        performance_threshold=performance_threshold,
        max_iterations=max_iterations,
        num_samples_per_iteration=num_samples_per_iteration,
        n_clusters=n_clusters,
        samples_per_cluster=samples_per_cluster,
        improvement_threshold=improvement_threshold,
        patience=patience
    )
    
    print("\nAll Selected Samples:")
    for sample in all_selected_samples:
        print(sample)
