import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os
import matplotlib.pyplot as plt
import json


class MarketDataProcessor:
    def __init__(self, asset_class, file_path, thresholds):
        self.asset_class = asset_class
        self.file_path = file_path
        self.data = self.load_data()
        self.results = {'outliers': [], 'missing_data': [], 'staleness': []}
        self.thresholds = thresholds  # Accept thresholds as a parameter
        self.process_success = True

    def load_data(self):
        try:
            data = pd.read_csv(self.file_path, parse_dates=['Date'], dayfirst=True)
            data.set_index('Date', inplace=True)
            return data
        except Exception as e:
            print(f"Error loading data for {self.file_path}: {e}")
            self.process_success = False
            return pd.DataFrame()  # Return empty DataFrame if loading fails

    def identify_outliers(self):
        if self.data.empty:
            return

        try:
            threshold = self.thresholds['outlier']
            for column in self.data.columns:
                clean_data = self.clean_data(self.data[column])
                model = IsolationForest(contamination=threshold)
                clean_data = clean_data.dropna()  # Drop NaN values before fitting
                reshaped_data = clean_data.values.reshape(-1, 1)
                model.fit(reshaped_data)

                # Predict outliers (-1 means outlier)
                outlier_flags = model.predict(reshaped_data)
                outliers = clean_data[outlier_flags == -1].index.tolist()
                self.results['outliers'].extend([(index, column) for index in outliers])
        except Exception as e:
            print(f"Error identifying outliers in {self.file_path}: {e}")
            self.process_success = False

    def clean_data(self, series):
        if '%' in str(series.iloc[0]):
            return series.str.rstrip('%').astype(float)
        return series.astype(float)

    def identify_missing_data(self):
        threshold = self.thresholds['missing_data']
        for column in self.data.columns:
            rolling_count = self.data[column].isna().rolling(window=threshold).sum()
            missing_dates = rolling_count[rolling_count >= 1].index.tolist()
            self.results['missing_data'].extend([(index, column) for index in missing_dates])

    def identify_staleness(self):
        threshold = self.thresholds['staleness']
        for column in self.data.columns:
            clean_data = self.clean_data(self.data[column])
            staleness_dates = []
            last_value = None
            stale_count = 0

            for date, value in clean_data.iteritems():
                if pd.isna(value):
                    continue

                if last_value is not None and value == last_value:
                    stale_count += 1
                else:
                    if stale_count >= threshold:
                        staleness_dates.append(date)
                    stale_count = 0
                last_value = value

            self.results['staleness'].extend([(index, column) for index in staleness_dates])

    def save_plot(self):
        # Plot the data and highlight outliers, missing data, and staleness
        num_plots = len(self.data.columns)
        cols = 6
        rows = (num_plots // cols) + (1 if num_plots % cols != 0 else 0)
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        axes = axes.flatten()

        for i, column in enumerate(self.data.columns):
            ax = axes[i]
            clean_data = self.clean_data(self.data[column])
            ax.plot(clean_data, label=f'{column} Data', color='blue')

            # Plot outliers
            outlier_dates = [date for date, col in self.results['outliers'] if col == column]
            ax.scatter(outlier_dates, clean_data.loc[outlier_dates], color='red', label='Outliers')

            # Plot missing data
            missing_dates = [date for date, col in self.results['missing_data'] if col == column]
            for date in missing_dates:
                ax.axvspan(date, date, color='grey', alpha=0.5, label='Missing Data')

            # Plot staleness data
            staleness_dates = [date for date, col in self.results['staleness'] if col == column]
            for date in staleness_dates:
                ax.axvspan(date, date, color='orange', alpha=0.3, label='Staleness')

            ax.set_title(f'{column}')
            ax.grid(True)
            ax.legend()

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plot_path = os.path.join(os.path.dirname(self.file_path), 'Processed',
                                 f"{os.path.basename(self.file_path)}_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    def process_data(self):
        # Main method to process the data
        self.identify_outliers()
        self.identify_missing_data()
        self.identify_staleness()
        if self.process_success:
            self.save_plot()
        return self.results, self.process_success


class AssetClassAnalyzer:
    def __init__(self, base_directory, thresholds):
        self.base_directory = base_directory
        self.asset_classes = self.get_asset_class_folders()
        self.thresholds = thresholds  # Accept thresholds as a parameter
        self.all_results = {'outliers': [], 'missing_data': [], 'staleness': []}

    def get_asset_class_folders(self):
        # Get subfolders for each asset class
        return [f.path for f in os.scandir(self.base_directory) if f.is_dir()]

    def create_directories(self, asset_class_folder):
        # Create Processed and Failed directories for each asset class
        processed_path = os.path.join(asset_class_folder, 'Processed')
        failed_path = os.path.join(asset_class_folder, 'Failed')
        os.makedirs(processed_path, exist_ok=True)
        os.makedirs(failed_path, exist_ok=True)
        return processed_path, failed_path

    def move_file(self, file, destination):
        try:
            os.rename(file, os.path.join(destination, os.path.basename(file)))
        except Exception as e:
            print(f"Error moving file {file} to {destination}: {e}")

    def save_combined_results(self, asset_class_name):
        # Save combined results for each asset class
        output_folder = os.path.join(self.base_directory, 'Output', asset_class_name)
        os.makedirs(output_folder, exist_ok=True)

        for key in self.all_results:
            df = pd.DataFrame(self.all_results[key], columns=['Date', 'Curve', 'File'])
            output_path = os.path.join(output_folder, f"{key}_results.csv")
            df.to_csv(output_path, index=False)

    def analyze(self):
        for asset_class_folder in self.asset_classes:
            processed_path, failed_path = self.create_directories(asset_class_folder)
            asset_class_name = os.path.basename(asset_class_folder)

            if asset_class_name not in self.thresholds:
                print(f"No thresholds found for asset class {asset_class_name}, skipping.")
                continue

            for file in os.listdir(asset_class_folder):
                file_path = os.path.join(asset_class_folder, file)
                if not file_path.endswith('.csv'):
                    continue

                # Instantiate the processor with identified thresholds
                processor = MarketDataProcessor(asset_class_name, file_path, self.thresholds[asset_class_name])
                results, success = processor.process_data()

                # Append results to the combined results list with the file name
                for key in results:
                    for date, curve in results[key]:
                        self.all_results[key].append((date, curve, os.path.basename(file_path)))

                # Move file to appropriate folder based on success
                if success:
                    self.move_file(file_path, processed_path)
                else:
                    self.move_file(file_path, failed_path)

            # Save the combined results for each asset class
            self.save_combined_results(asset_class_name)


if __name__ == "__main__":
    # Example setup, adjust the paths to your actual file locations
    base_directory = "path/to/MarketData"  # Your MarketData folder

    # Load identified thresholds from JSON or another configuration file after discovery
    identified_thresholds_path = "path/to/identified_thresholds.json"
    with open(identified_thresholds_path, 'r') as f:
        identified_thresholds = json.load(f)

    analyzer = AssetClassAnalyzer(base_directory, identified_thresholds)
    analyzer.analyze()
