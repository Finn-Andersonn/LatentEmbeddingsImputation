# --------------------------------------
# After analyzing map.py its clear some tanks have not been recorded as often.
# Need to answer how often are they recorded on average? Is there a pattern in recording?
#
# For the former it seems a median amount of time between images taken for each tank is 21 days.
# For the latter, it seems the missingness occurs heavily in the summer.
# Could be an interesting question here regarding cause of it -> EIA reporting? Sunlight?
# Equally interesting, we will investigate if we can find similarity amongst the tanks to improve imputation
# --------------------------------------


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer:
    '''
    
    '''
    def __init__(self, data_path, output_dir, n_permutations=1000):
        self.data_path = data_path
        self.output_dir = output_dir
        self.n_permutations = n_permutations

    def load_data(self):
        df = pd.read_csv(self.data_path)
        # Ensure we don't have duplicate columns (like tank_id appearing twice)
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    
    def analyze_recording_intervals(self, df=None):
        """
        Description:
            Compute and analyze the average time interval (in days) between recordings for each tank.
            Returns a Series with the average interval per tank.

        Args:
            df: pandas dataframe

        Returns:
            Histogram
        """
        if df is None:
            df = self.load_data()
        
        if 'imaging_time' not in df.columns:
            raise ValueError("Column 'imaging_time' not found in the dataset.")
        df['imaging_time'] = pd.to_datetime(df['imaging_time'], errors='coerce')
        
        df_sorted = df.sort_values(['tank_id', 'imaging_time'])
        df_sorted['time_diff'] = df_sorted.groupby('tank_id')['imaging_time'].diff()
        df_sorted['time_diff_days'] = df_sorted['time_diff'].dt.total_seconds() / (24 * 3600)
        
        avg_time_diff = df_sorted.groupby('tank_id')['time_diff_days'].mean()
        print("Descriptive statistics for average recording interval (in days):")
        print(avg_time_diff.describe())
        
        plt.figure(figsize=(12, 6))
        sns.histplot(avg_time_diff, bins=30, kde=True)
        plt.title("Distribution of Average Recording Intervals (in days) per Tank")
        plt.xlabel("Average Interval (days)")
        plt.ylabel("Number of Tanks")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "avg_recording_interval_hist.png"))
        plt.close()
        
        return avg_time_diff

    def analyze_recording_frequency(self, df=None, freq='D'):
        """
        Description:
            Analyze the frequency of recordings per tank_id.
            The monthly panel (36,036 rows) captures this seasonality by ensuring all months are represented, 
            even those with fewer images (e.g., July).
                -> This helps highlight why we chose monthly (closer to mean) than daily occurences
                -> Monthly results in ~= 51% missingness, while daily results in 99% (becomes to noisy and essentially random guessing)
                -> Monthly aggregation sidesteps this issue, focusing on broader trends.

        Args:
            df: pandas dataframe
            freq: 'D' for daily, 'M' for monthly.

        Returns:
            dataframe
        """
        if df is None:
            df = self.load_data()
        
        if 'imaging_time' in df.columns:
            df['imaging_time'] = pd.to_datetime(df['imaging_time'], errors='coerce')
        
        counts = df.groupby('tank_id').size()
        print("\nRecording counts per tank_id:")
        print(counts.describe())
        
        if freq == 'D':
            df['period'] = df['imaging_time'].dt.to_period('D')
        elif freq == 'M':
            df['period'] = df['imaging_time'].dt.to_period('M')
        else:
            raise ValueError("freq must be 'D' (daily) or 'M' (monthly)")
        
        period_counts = df.groupby('tank_id')['period'].nunique()
        print(f"\nNumber of unique {freq} periods per tank_id:")
        print(period_counts.describe())
        
        return df

    def restructure_panel(self, df=None, freq='D'):
        '''
        Description:
            This is where we restructure the dataset to have rows for all the tanks for each month from 2014 to end of dataset (2018)
            We include known static values for those NaN rows -> max volume, farm, region, Location, farm_type
        
            Args:
                df: original pandas dataframe
                freq: daily (too high missingness to make meaningful contribution) so called as monthly

            Returns:
                df: restructured pandas dataframe
        '''
        if df is None:
            df = self.load_data()
            if 'imaging_time' in df.columns:
                df['imaging_time'] = pd.to_datetime(df['imaging_time'], errors='coerce')
        
        # dropping duplicate tank_id column if it exists
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        
        if freq == 'D':
            df['period'] = df['imaging_time'].dt.to_period('D')
        elif freq == 'M':
            df['period'] = df['imaging_time'].dt.to_period('M')
        else:
            raise ValueError("freq must be 'D' or 'M'")
        
        overall_start = df['period'].min().to_timestamp()
        overall_end = df['period'].max().to_timestamp()
        if freq == 'D':
            full_range = pd.date_range(start=overall_start, end=overall_end, freq='D')
        elif freq == 'M':
            full_range = pd.date_range(start=overall_start, end=overall_end, freq='MS')
        
        full_periods = pd.PeriodIndex(full_range, freq=freq)
        tank_ids = df['tank_id'].unique()
        panel_index = pd.MultiIndex.from_product([tank_ids, full_periods], names=['tank_id', 'period'])
        
        # a mapping of tank_id to static features
        static_features = ['max_vol', 'tank farm', 'region', 'Location', 'farm_type']
        tank_static_info = df.groupby('tank_id')[static_features].first().reset_index()
        
        print(f"Rows before aggregation: {df.shape[0]}")
        df_panel = df.groupby(['tank_id', 'period']).agg({
            'tank volume': 'mean',
            'fill_pct': 'mean',
            'max_vol': 'first',
            'imaging_time': 'first',
            'tank farm': 'first',
            'region': 'first',
            'Location': 'first',
            'farm_type': 'first'
        }).reset_index()
        print(f"Rows after aggregation: {df_panel.shape[0]}")
        df_complete = df_panel.set_index(['tank_id', 'period']).reindex(panel_index).reset_index()
        
        # merge static features back into df_complete
        df_complete = df_complete.merge(
            tank_static_info,
            on='tank_id',
            how='left',
            suffixes=('', '_static')
        )
        
        # replace NaN values in static columns with the values from the mapping
        for col in static_features:
            df_complete[col] = df_complete[col].fillna(df_complete[f'{col}_static'])
            df_complete = df_complete.drop(columns=f'{col}_static')
        
        # sorted by period, then tank_id
        df_complete = df_complete.sort_values(['period', 'tank_id']).reset_index(drop=True)
        
        output_path = os.path.join(self.output_dir, 'complete_panel.csv')
        df_complete.to_csv(output_path, index=False)
        print(f"Saved complete panel to {output_path}")
        
        return df_complete

    def analyze_missingness_panel(self, df_panel):
        '''
        Description:
            Computes missingness statistics on the complete panel.
        
        Args:
            df_panel: restructured pandas dataframe

        Returns:
            Histogram: Number of tanks per missing percentage over complete panel
            Time series: mean missing percentage for tank volume over time in the restructured dataframe
        '''
        features_to_check = ['tank volume', 'fill_pct']
        available_features = [col for col in features_to_check if col in df_panel.columns]
        
        missing_counts = df_panel[available_features].isna().sum()
        missing_percentages = (missing_counts / len(df_panel)) * 100
        overall_missing = pd.DataFrame({
            'Missing Count': missing_counts,
            'Missing Percentage': missing_percentages
        })
        print("\nOverall Missingness in the Complete Panel:")
        print(overall_missing)
        
        missing_by_tank = df_panel.groupby('tank_id')['tank volume'].apply(lambda x: x.isna().mean() * 100) # mean & convert to percent from decimal
        print("\nMissingness by tank_id (tank volume):")
        print(missing_by_tank.describe())
        
        plt.figure(figsize=(12, 6))
        sns.histplot(missing_by_tank, bins=30)
        plt.title('Distribution of Missingness Percentage for tank volume by tank_id')
        plt.xlabel('Missing Percentage (%)')
        plt.ylabel('Number of Tanks')
        plt.savefig(os.path.join(self.output_dir, "missingness_by_tank_id_panel.png"))
        plt.close()
        
        missing_by_time = df_panel.groupby('period')['tank volume'].apply(lambda x: x.isna().mean() * 100)
        plt.figure(figsize=(12, 6))
        missing_by_time.plot(kind='line', marker='o')
        plt.title('Missingness Percentage for tank volume Over Time in the Complete Panel')
        plt.xlabel('Period')
        plt.ylabel('Missing Percentage (%)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "missingness_by_time_panel.png"))
        plt.close()
        
        return overall_missing

    def run_analysis(self):
        ''''
        Calling it all. 
        '''
        # 1: Load raw pandas datafrane
        raw_df = self.load_data()
        print(f"Original df shape: {raw_df.shape}")
        
        # 2: inform frequency choice
        avg_intervals = self.analyze_recording_intervals(raw_df)
        median_interval = avg_intervals.median()
        print(f"\nMedian recording interval: {median_interval:.2f} days")
        
        # Decide frequency based on median interval:
        # For example, if median is >= 7 days, use weekly (or monthly) frequency.
        if median_interval >= 7:
            chosen_freq = 'M'  # you might choose 'W' for weekly if desired
        else:
            chosen_freq = 'D'
        print(f"Chosen frequency for panel restructuring: {chosen_freq}")
        
        # 3: recording frequency using the chosen frequency.
        freq_df = self.analyze_recording_frequency(raw_df, freq=chosen_freq)
        
        # 4: Restructure the data into a complete panel.
        complete_panel = self.restructure_panel(freq_df, freq=chosen_freq)
        print(f"Complete panel shape: {complete_panel.shape}")
        
        # 5: missingness on the complete panel.
        self.analyze_missingness_panel(complete_panel)
        
        

# Example usage:
analyzer = DataAnalyzer(
    data_path='RawLouisiana.csv',
    output_dir='/Users/finn/Desktop/UCL Masters/Data Science/Project/graphs/missingness/'
)
analyzer.run_analysis()


# the specific day within the month (and thus the day_of_week) is less relevant because:
# The median recording interval is 20.85 days, meaning tanks are imaged about once per month (MORE SPECIFIC in paper).
# The monthly panel structure already captures the month of observation via the period column (e.g., 2014-01).
#
# The "Satellite Imaging by Hour" histogram shows that imaging is almost always between 10:00 and 12:00. 
# This lack of variation means imaging_hour (computed in corr_utilities.py) won’t help predict fill_pct.
# Additionally, the time of day the image was taken is unlikely to affect the actual fill level of the tank 
# (which is determined by operational decisions, not the satellite’s schedule).