import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os
import json
from multiprocessing import Pool


class ThresholdDiscovery:
    def __init__(self, config_path, data_dir):
        self.config = self.load_config(config_path)
        self.data_dir = data_dir

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return json.load(file)

    def load_data(self, file_path):
        try:
            data = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
            data.set_index('Date', inplace=True)
            return data
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            return pd.DataFrame()  # Return an empty DataFrame if loading fails

    def discover_outliers(self, asset_class, data, threshold_range):
        outlier_counts = []

        for threshold in threshold_range:
            total_outliers = 0
            for column in data.columns:
                clean_data = self.clean_data(data[column])
                if clean_data.empty:
                    continue

                model = IsolationForest(contamination=threshold)
                clean_data = clean_data.dropna()
                reshaped_data = clean_data.values.reshape(-1, 1)
                model.fit(reshaped_data)

                outlier_flags = model.predict(reshaped_data)
                total_outliers += (outlier_flags == -1).sum()

            outlier_counts.append((threshold, total_outliers))

        return outlier_counts

    def discover_missing_data(self, data, threshold_range):
        missing_counts = []

        for threshold in threshold_range:
            total_missing = 0
            for column in data.columns:
                rolling_count = data[column].isna().rolling(window=threshold).sum()
                total_missing += (rolling_count >= 1).sum()

            missing_counts.append((threshold, total_missing))

        return missing_counts

    def discover_staleness(self, data, threshold_range):
        staleness_counts = []

        for threshold in threshold_range:
            total_stale = 0
            for column in data.columns:
                clean_data = self.clean_data(data[column])
                stale_count = 0
                last_value = None
                days_stale = 0

                for value in clean_data:
                    if pd.isna(value):
                        continue
                    if last_value is not None and value == last_value:
                        days_stale += 1
                    else:
                        if days_stale >= threshold:
                            stale_count += 1
                        days_stale = 0
                    last_value = value

                total_stale += stale_count

            staleness_counts.append((threshold, total_stale))

        return staleness_counts

    def clean_data(self, series):
        if '%' in str(series.iloc[0]):
            return series.str.rstrip('%').astype(float)
        return series.astype(float)

    def save_results(self, results, asset_class, metric, processor_id):
        output_dir = os.path.join(self.data_dir, 'Output', asset_class)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{metric}_threshold_discovery_proc_{processor_id}.csv')
        df = pd.DataFrame(results, columns=['Threshold', f'Number of {metric}'])
        df.to_csv(output_file, index=False)

    def process_file(self, asset_class, file_path, processor_id):
        try:
            data = self.load_data(file_path)

            if data.empty:
                raise ValueError("Loaded data is empty.")

            # Outlier threshold discovery
            outlier_range = np.arange(*self.config[asset_class]['outlier'])
            outlier_results = self.discover_outliers(asset_class, data, outlier_range)
            self.save_results(outlier_results, asset_class, 'outliers', processor_id)

            # Missing data threshold discovery
            missing_range = range(*self.config[asset_class]['missing_data'])
            missing_results = self.discover_missing_data(data, missing_range)
            self.save_results(missing_results, asset_class, 'missing_data', processor_id)

            # Staleness threshold discovery
            staleness_range = range(*self.config[asset_class]['staleness'])
            staleness_results = self.discover_staleness(data, staleness_range)
            self.save_results(staleness_results, asset_class, 'staleness', processor_id)

            # Move file to Processed folder
            self.move_file(file_path, os.path.join(os.path.dirname(file_path), 'Processed'))
        except Exception as e:
            print(f"Failed processing {file_path}: {e}")
            # Move file to Failed folder if processing fails
            self.move_file(file_path, os.path.join(os.path.dirname(file_path), 'Failed'))

    def move_file(self, file_path, destination):
        os.makedirs(destination, exist_ok=True)
        try:
            os.rename(file_path, os.path.join(destination, os.path.basename(file_path)))
        except Exception as e:
            print(f"Error moving file {file_path} to {destination}: {e}")

    def combine_results(self, asset_class):
        output_dir = os.path.join(self.data_dir, 'Output', asset_class)
        combined_results = {'outliers': [], 'missing_data': [], 'staleness': []}

        for metric in combined_results.keys():
            combined_df = pd.DataFrame(columns=['Threshold', f'Number of {metric}'])
            for file in os.listdir(output_dir):
                if file.startswith(f'{metric}_threshold_discovery_proc_'):
                    file_path = os.path.join(output_dir, file)
                    df = pd.read_csv(file_path)
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
                    os.remove(file_path)  # Remove partial results file after combining

            combined_df.to_csv(os.path.join(output_dir, f'{metric}_threshold_discovery_combined.csv'), index=False)

    def run_discovery(self):
        with Pool(processes=8) as pool:
            for asset_class in self.config:
                asset_class_path = os.path.join(self.data_dir, asset_class)
                files = [os.path.join(asset_class_path, file) for file in os.listdir(asset_class_path) if
                         file.endswith('.csv')]

                # Split files into 8 chunks
                chunk_size = max(1, len(files) // 8)
                file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]

                # Assign chunks to processors
                for processor_id, chunk in enumerate(file_chunks):
                    pool.apply_async(self.process_files_chunk, args=(asset_class, chunk, processor_id))

            pool.close()
            pool.join()

        # Combine results after all processing is done
        for asset_class in self.config:
            self.combine_results(asset_class)

    def process_files_chunk(self, asset_class, files_chunk, processor_id):
        for file_path in files_chunk:
            self.process_file(asset_class, file_path, processor_id)


if __name__ == "__main__":
    config_path = "path/to/threshold_config.json"  # Path to your threshold configuration JSON file
    data_dir = "path/to/MarketData"  # Path to your MarketData folder

    discovery = ThresholdDiscovery(config_path, data_dir)
    discovery.run_discovery()
