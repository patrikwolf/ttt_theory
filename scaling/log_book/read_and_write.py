import os
import math
import pandas as pd

from datetime import datetime
from tabulate import tabulate


def log_to_csv(results, filename, mnist=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)
    if mnist:
        assert filename.startswith('mnist_'), 'Filename must start with mnist_ for MNIST logs.'

    # Add date and time
    results['date'] = datetime.now().strftime('%Y-%m-%d')
    results['time'] = datetime.now().strftime('%H:%M:%S')

    # Convert new row to DataFrame
    df_new = pd.DataFrame([results])

    # Check if file exists
    if os.path.exists(path):
        try:
            df_existing = pd.read_csv(path, sep=';')

            # Combine columns
            all_columns = list(df_existing.columns.union(df_new.columns))

            # Reorder: date, time, then sorted remaining
            remaining_cols = [col for col in all_columns if col not in ['date', 'time']]
            ordered_columns = ['date', 'time'] + remaining_cols

            # Ensure both frames have the same columns
            df_existing = df_existing.reindex(columns=ordered_columns)
            df_new = df_new.reindex(columns=ordered_columns)

            # Append new row
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(path, index=False, sep=';')

        except Exception as e:
            print(f"Failed to update existing CSV: {e}")
    else:
        # Create new CSV
        df_new.to_csv(path, index=False, sep=';')


def read_csv(filename, cluster=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if cluster:
        log_dir = os.path.join(base_dir, 'logs_cluster')
    else:
        log_dir = os.path.join(base_dir, 'logs')
    path = os.path.join(log_dir, filename)

    if not os.path.exists(path):
        print(f"File {path} does not exist")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, sep=';')

        # Move 'date' and 'time' to first positions if they exist
        for col in ['date', 'time']:
            if col in df.columns:
                df.insert(0 if col == 'date' else 1, col, df.pop(col))

        # Reverse row order
        df = df.iloc[::-1]

        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return pd.DataFrame()


def print_tabulated(filename, cols=None, head=None, cluster=False, filters=None, sort_by=None, ascending=False, truncate_d=5):
    # Read the CSV file
    df = read_csv(filename, cluster=cluster)
    all_cols = df.columns.tolist()

    # Subsample specific columns if provided
    if cols is not None:
        df = df[cols]

    # Filter columns
    if filters is not None:
        for col, value in filters.items():
            if col in df.columns:
                if isinstance(value, list):
                    df = df[df[col].isin(value)]
                else:
                    df = df[df[col] == value]
            else:
                print(f"Warning: Column '{col}' not found in DataFrame! Skipping filter.")

    # Sort columns
    if sort_by is not None:
        if isinstance(sort_by, str):
            sort_by = [sort_by]
        df = df.sort_values(by=sort_by, ascending=ascending)

    # Truncate all float columns
    float_cols = df.select_dtypes(include='float').columns
    df[float_cols] = df[float_cols].map(lambda v: round_float(v, n=truncate_d))

    # Print the DataFrame
    if head is not None:
        print(tabulate(df.head(head), headers='keys', tablefmt='pretty', showindex=False))
    else:
        print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))

    return all_cols, df


def round_float(x, n=5):
    if not isinstance(x, float):
        return x
    if math.isnan(x):
        return x
    return round(x, n)


if __name__ == '__main__':
    # Example usage
    results = {
        'date': '05.08.2025',
        'time': '08:00:00',
        'model_name': 'example_model',
        #'accuracy': 0.95,
        'epoch': 1,
        'loss': 0.05,
        'tv_distance': 0.1,
        'jsd_distance': 0.2,
    }

    # Log to CSV
    log_to_csv(results, 'test.csv')

    # Print tabulated results
    all_cols, df = print_tabulated(filename='test.csv', head=10)
    print(all_cols)

    print('*')

    cols = ['date', 'time', 'epoch']
    print_tabulated(filename='test.csv', cols=cols, head=3)