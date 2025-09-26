import numpy as np


def bootstrap_confidence_interval(data, n_bootstrap=1000, confidence=0.95):
    data = np.array(data)
    mean = np.mean(data)

    # Generate bootstrap samples
    bootstrap_samples = np.random.choice(data, (n_bootstrap, len(data)), replace=True)     # n_boostrap x num_samples
    bootstrap_means = np.mean(bootstrap_samples, axis=1)                                        # n_boostrap x 1

    # Calculate percentiles
    lower_bound = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper_bound = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)

    return mean, lower_bound, upper_bound


def bootstrap_ci_across_seeds(data, num_bootstrap=200, confidence=0.9):
    data = np.array(data)
    mean = np.mean(data)
    num_seeds = len(data)
    num_samples = len(data[0])

    # Initialize array to hold all bootstrap means
    bootstrap_samples = np.zeros((num_bootstrap, num_seeds, num_samples))

    for i in range(num_seeds):
        # Generate bootstrap samples for each seed
        bootstrap_samples[:, i, :] = np.random.choice(data[i], (num_bootstrap, num_samples), replace=True)

    # Average the bootstrap samples across seeds
    bootstrap_means = np.mean(bootstrap_samples, axis=(1, 2))

    # Calculate percentiles
    lower_bound = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper_bound = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)

    return mean, lower_bound, upper_bound


if __name__ == '__main__':
    # Single seed data
    single_seed_data = [0.8, 0.82, 0.78, 0.81, 0.79]
    mean, lower_bound, upper_bound = bootstrap_confidence_interval(single_seed_data, n_bootstrap=1000, confidence=0.9)
    print(f'Mean: {mean}, 90% CI: ({lower_bound:.4f}, {upper_bound:.4f})')
    
    # Data across multiple seeds
    data = [[0.8, 0.82, 0.78, 0.81, 0.79], [0.84, 0.12, 0.38, 0.89, 0.29]]
    mean, lower_bound, upper_bound = bootstrap_ci_across_seeds(data, num_bootstrap=200, confidence=0.9)
    print(f'Mean: {mean}, 90% CI: ({lower_bound:.4f}, {upper_bound:.4f})')
