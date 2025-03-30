# -------------------------------------------------------
# Current features cover spatial, temporal, and volumetric aspects, which are great for the imputation task. 
# However, to improve imputation and similarity learning (especially for RAG), we can consider adding features 
# that provide more context about the missing data patterns and enhance the embeddings’ ability to capture similarity
# Data to add to improve imputation:
#   1) Weather data DONE
#      -> temperature
#      -> cloud cover
#      -> precipitation
#      -> Enhances similarity learning by capturing environmental context (e.g., tanks imaged under similar weather conditions might have correlated fill_pct).
#   2) EIA Reporting dates
#      -> Adds a temporal context that could explain imaging patterns, improving the embeddings’ ability to group tanks by operational cycles.
#   3) Oil Prices NOT USING
#      -> Regional oil prices can capture economic drivers of tank fill levels, which might correlate with missing data patterns 
#         (e.g., higher summer production might explain missing fill_pct)
#   4) Temporal Features DONE
#      -> Day of week
#      -> Season
#      -> can help model periodic patterns in imaging and fill levels, addressing the pre-noon/non-summer bias directly.
#   5) Spatial Features DONE
#      -> Distance to nearest port
#      -> Population Density
#      -> Enhances the graph construction (e.g., weighting edges by port distance) and embeddings (e.g., tanks near ports might have similar operational patterns).
# NOTE: Adding too many features runs risk of curse of dimensionality (mentioned and monitored in causality.py)
#
# EIA 810 survey data, AIS ship tracking (vessel arrivals at nearby docks), EIA inventory releases, 
# EIA refiner utilization data, Year-end inventory reporting/audits (EIA-810, EPA GHG reports)
#
# EPA's GHG Reporting Rule expanded in 2016 for petroleum facilities
#
# EIA increased PADD3 monitoring frequency
# -------------------------------------------------------
import pandas as pd
from datetime import datetime, timedelta
from meteostat import Stations, Hourly
from geopy.distance import geodesic
import numpy as np
import ssl
from typing import List, Dict, Optional
import requests
import os
import json
ssl._create_default_https_context = ssl._create_unverified_context

class WeatherData:
    def __init__(self, df):
        self.df = df.copy()
        if self.df['imaging_time'].dtype == 'O':
            self.df['imaging_time'] = pd.to_datetime(self.df['imaging_time'])
    
    def parse_location(self, loc_str):
        loc_str = loc_str.replace("POINT(", "").replace(")", "")
        lon_str, lat_str = loc_str.split()
        return float(lat_str), float(lon_str)
    
    def get_unique_locations(self):
        self.df['lat'], self.df['lon'] = zip(*self.df['Location'].apply(self.parse_location))
        return self.df[['lat', 'lon']].drop_duplicates().reset_index(drop=True)
    
    def get_period_of_interest(self):
        start_date = self.df['imaging_time'].min().to_pydatetime()
        end_date = self.df['imaging_time'].max().to_pydatetime()
        return start_date, end_date
    
    def get_stations(self, start_date, end_date):
        stations = Stations()
        stations = stations.bounds((33.0, -94.0), (28.9, -88.8))
        stations = stations.inventory('hourly', (start_date, end_date)).fetch()
        if stations.empty:
            raise ValueError("No weather stations found in the region for the period.")
        return stations
    
    def download_weather_data(self, unique_locations, start_date, end_date):
        uloc = unique_locations.copy()
        uloc['station_id'] = None
        uloc['station_dist'] = float('inf')
        for idx, row in uloc.iterrows():
            stations_sorted = self.stations.copy()
            stations_sorted['distance'] = stations_sorted.apply(
                lambda x: geodesic((row['lat'], row['lon']), (x['latitude'], x['longitude'])).kilometers,
                axis=1
            )
            stations_sorted = stations_sorted.sort_values('distance')
            for _, station in stations_sorted.head(3).iterrows():
                try:
                    hourly = Hourly(station.name, start_date, end_date)
                    df_weather = hourly.fetch()
                    if not df_weather.empty:
                        uloc.at[idx, 'station_id'] = station.name
                        uloc.at[idx, 'station_dist'] = station['distance']
                        break
                except Exception as e:
                    print(f"Error fetching {station.name}: {str(e)}")
                    continue
        weather_dfs = []
        for station in uloc['station_id'].unique():
            if pd.isna(station):
                continue
            print(f"Downloading for station {station}...")
            try:
                hourly = Hourly(station, start_date, end_date)
                df_weather = hourly.fetch()
                if not df_weather.empty:
                    df_weather = df_weather.reset_index()
                    df_weather['station_id'] = station
                    weather_dfs.append(df_weather)
            except Exception as e:
                print(f"Failed to get data for {station}: {str(e)}")
        return uloc, weather_dfs
    
    def combine_data(self, unique_locations, weather_dfs):
        df_weather_all = pd.concat(weather_dfs, ignore_index=True)
        df_tanks = self.df.copy()
        df_tanks = df_tanks.merge(unique_locations, on=['lat', 'lon'], how='left')
        df_tanks = df_tanks.sort_values('imaging_time')
        df_weather_all = df_weather_all.sort_values('time')
        df_weather_all.rename(columns={'time': 'weather_time'}, inplace=True)
        df_merged = pd.merge_asof(
            df_tanks,
            df_weather_all,
            left_on='imaging_time',
            right_on='weather_time',
            by='station_id',
            direction='nearest',
            tolerance=pd.Timedelta('1h')
        )
        df_merged.rename(columns={
            'temp': 'temperature',
            'coco': 'cloud_cover',
            'prcp': 'precipitation'
        }, inplace=True)
        return df_merged
    
    def run(self):
        unique_locations = self.get_unique_locations()
        start_date, end_date = self.get_period_of_interest()
        print("Period:", start_date, "to", end_date)
        self.stations = self.get_stations(start_date, end_date)
        unique_locations, weather_dfs = self.download_weather_data(unique_locations, start_date, end_date)
        df_merged = self.combine_data(unique_locations, weather_dfs)
        df_merged.to_csv("fulldata_with_weather.csv", index=False)
        print("Data saved to fulldata_with_weather.csv")
        return df_merged
    
    def check_missing_points(self, df):
        print("Shape:", df.shape)
        print("Missing (NaN) values per column:")
        print(df.isna().sum())
        print("\nZero values per numeric column:")
        for col in df.select_dtypes(include=['number']).columns:
            zero_count = (df[col] == 0).sum()
            print(f"{col}: {zero_count}")
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        print(f"\nTotal data points (cells): {total_cells}")
        print(f"Total missing data points (NaN): {missing_cells}")
        print(f"Percentage missing: {missing_cells/total_cells*100:.2f}%")

class EnhancedAdditionalFeatures:
    """
    Enhanced version with:
    - Predefined Louisiana ports as fallback
    - Multiple port data source options
    - Additional spatial features
    - Better error handling
    """
    # Predefined major ports in Louisiana (lat, lon)
    LOUISIANA_PORTS = [
        {'port': 'Port of New Orleans', 'lat': 29.933, 'lon': -90.080},
        {'port': 'Port of Baton Rouge', 'lat': 30.457, 'lon': -91.187},
        {'port': 'Port of South Louisiana', 'lat': 30.069, 'lon': -90.484},
        {'port': 'Port of Lake Charles', 'lat': 30.226, 'lon': -93.217},
        {'port': 'Port of Plaquemines', 'lat': 29.356, 'lon': -89.487}
    ]

    def __init__(self, df):
        self.df = df.copy()
        if self.df['imaging_time'].dtype == 'O':
            self.df['imaging_time'] = pd.to_datetime(self.df['imaging_time'])
        
        # Ensure we have coordinates
        if 'lat' not in self.df or 'lon' not in self.df:
            self._add_coordinates()

    def _add_coordinates(self):
        """Parse coordinates from Location column if not already present"""
        def parse_loc(loc_str):
            loc_str = loc_str.replace("POINT(", "").replace(")", "")
            lon, lat = loc_str.split()
            return float(lat), float(lon)
        
        self.df['lat'], self.df['lon'] = zip(*self.df['Location'].apply(parse_loc))

    def get_ports(self, source: str = 'predefined', bbox: tuple = (28.0, -95.0, 33.0, -88.0)) -> List[Dict]:
        """
        Get port locations from specified source.
        Options: 'predefined', 'osm', or provide custom list.
        """
        if source == 'predefined':
            return self.LOUISIANA_PORTS
        elif source == 'osm':
            return self._fetch_ports_from_osm(bbox)
        else:
            return source  # Assume it's a custom list

    def _fetch_ports_from_osm(self, bbox: tuple) -> List[Dict]:
        """
        More robust OSM query with error handling and expanded search criteria
        """
        query = f"""
        [out:json];
        (
          node["harbour"="yes"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
          node["seamark:type"="harbour"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
          node["port"="yes"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
        );
        out;
        """
        try:
            response = requests.get(
                "https://overpass-api.de/api/interpreter",
                params={"data": query},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return [
                {
                    "port": element.get("tags", {}).get("name", f"Port_{i}"),
                    "lat": element["lat"],
                    "lon": element["lon"]
                }
                for i, element in enumerate(data.get("elements", []))
            ]
        except Exception as e:
            print(f"OSM query failed: {str(e)}. Using predefined ports.")
            return self.LOUISIANA_PORTS
        
    def add_eia_report_features(self):
        """
        Add features related to EIA reporting cycles:
        - days_since_last_eia_report (weekly/monthly)
        - days_until_next_eia_report
        - is_eia_report_week (bool)
        """
        print("Adding EIA report features...")
        
        # Generate all EIA report dates in the dataset's time range
        start_date = self.df['imaging_time'].min()
        end_date = self.df['imaging_time'].max() + timedelta(days=7)
        all_report_dates = self._generate_eia_report_dates(start_date, end_date)
        
        # For each row, find nearest report date
        self.df['days_since_eia_report'] = self.df['imaging_time'].apply(
            lambda x: self._days_since_nearest_report(x, all_report_dates)
        )
        
        # Additional derived features
        self.df['is_eia_report_week'] = self.df['days_since_eia_report'] <= 7
        self.df['eia_report_cycle'] = self.df['days_since_eia_report'] % 7  # Weekly cycle position
        
        print("EIA report features added successfully")

    def _generate_eia_report_dates(self, start_date, end_date):
        """Generate all EIA report dates between start and end dates"""
        dates = []
        current_date = start_date
        
        # Weekly reports (every Wednesday)
        while current_date <= end_date:
            if current_date.weekday() == 2:  # Wednesday
                dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Monthly reports (1st of each month)
        current_date = start_date.replace(day=1)
        while current_date <= end_date:
            dates.append(current_date)
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year+1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month+1)
        
        return sorted(list(set(dates)))  # Remove duplicates and sort

    def _days_since_nearest_report(self, target_date, report_dates):
        """Calculate days since most recent EIA report"""
        report_dates = pd.to_datetime(report_dates)
        past_reports = report_dates[report_dates <= target_date]
        
        if len(past_reports) > 0:
            last_report = past_reports.max()
            return (target_date - last_report).days
        else:
            # If no past reports, use days until next report (as negative value)
            future_reports = report_dates[report_dates > target_date]
            if len(future_reports) > 0:
                next_report = future_reports.min()
                return (target_date - next_report).days  # Negative value
            else:
                return np.nan

    def add_spatial_features(self, ports: Optional[List[Dict]] = None):
        """
        Add multiple spatial features:
        - Distance to nearest port (using best available source)
        - Distance to nearest major city
        - Coastal proximity flag
        """
        ports = ports or self.get_ports()
        
        # 1. Distance to nearest port
        if ports:
            self._add_distance_to_feature(ports, 'port', 'distance_to_nearest_port')
        else:
            self.df['distance_to_nearest_port'] = np.nan
            print("No ports available - skipping port distance calculation")

        # 2. Distance to major cities (New Orleans, Baton Rouge, etc.)
        major_cities = [
            {'name': 'New Orleans', 'lat': 29.951, 'lon': -90.072},
            {'name': 'Baton Rouge', 'lat': 30.451, 'lon': -91.187},
            {'name': 'Lafayette', 'lat': 30.224, 'lon': -92.020}
        ]
        self._add_distance_to_feature(major_cities, 'city', 'distance_to_major_city')

        # 3. Coastal proximity flag (within 50km of coastline)
        coastline = [
            {'lat': 29.0, 'lon': -89.0},  # Approximate coastal points
            {'lat': 29.5, 'lon': -89.5},
            {'lat': 30.0, 'lon': -90.0}
        ]
        self._add_distance_to_feature(coastline, 'coast', 'distance_to_coast')
        self.df['is_coastal'] = self.df['distance_to_coast'] <= 50

    def _add_distance_to_feature(self, points: List[Dict], feature_type: str, col_name: str):
        """Helper method to calculate distance to nearest feature"""
        unique_locs = self.df[['lat', 'lon']].drop_duplicates()
        
        def min_distance(row):
            return min(
                geodesic((row['lat'], row['lon']), (p['lat'], p['lon'])).km
                for p in points
            ) if points else np.nan
        
        unique_locs[col_name] = unique_locs.apply(min_distance, axis=1)
        self.df = self.df.merge(unique_locs, on=['lat', 'lon'], how='left')
        print(f"Added {col_name} based on {len(points)} {feature_type} locations")

    def add_temporal_features(self):
        """Enhanced temporal features with cyclical encoding"""
        # Basic features
        self.df['day_of_week'] = self.df['imaging_time'].dt.day_name()
        self.df['month'] = self.df['imaging_time'].dt.month
        
        self.df['season'] = self.df['imaging_time'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })

    def add_crude_oil_prices(self, excel_path: str, sheet_name: str = "Data 1"):
        """
        Loads daily WTI and Brent prices from an Excel file and merges them 
        into the main DataFrame by matching dates.

        :param excel_path: Path to the local XLS(X) file with daily WTI/Brent data.
        :param sheet_name: Name of the sheet in the Excel file containing the data.
        """
        print(f"Reading crude oil prices from Excel: {excel_path} ...")

        # 1) Load Excel data
        df_prices = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=2)
        print(df_prices.columns)

        # 2) Clean up and rename columns
        #    Adjust skiprows or header if needed to handle extra rows.
        #    The example below assumes columns are something like:
        #      "Date", "Cushing, OK WTI Spot Price FOB (Dollars per Barrel)",
        #      "Europe Brent Spot Price FOB (Dollars per Barrel)"
        df_prices.rename(
            columns={
                "Date": "price_date",
                "Cushing, OK WTI Spot Price FOB (Dollars per Barrel)": "wti_price",
                "Europe Brent Spot Price FOB (Dollars per Barrel)": "brent_price"
            },
            inplace=True
        )

        # 3) Convert date column to datetime
        df_prices['price_date'] = pd.to_datetime(df_prices['price_date'])

        # 4) Optionally, remove rows where price is missing
        df_prices.dropna(subset=['wti_price', 'brent_price'], how='any', inplace=True)

        # 5) Calculate the WTI-Brent spread
        df_prices['wti_brent_spread'] = df_prices['wti_price'] - df_prices['brent_price']

        # 6) Sort both DataFrames by date for asof-merge
        df_prices.sort_values('price_date', inplace=True)
        self.df['date_only'] = self.df['imaging_time'].dt.normalize()
        self.df.sort_values('date_only', inplace=True)

        # 7) Merge using nearest date
        #    - If you want an exact match on day (i.e., no "nearest" logic),
        #      you could do a simple pd.merge(on='date_only', how='left').
        #    - For "nearest" daily match, use merge_asof.
        merged = pd.merge_asof(
            self.df,
            df_prices,
            left_on='date_only',
            right_on='price_date',
            direction='nearest'
        )

        # 8) Cleanup
        merged.drop(columns=['date_only', 'price_date'], inplace=True)

        # 9) Save the result
        self.df = merged
        print("Crude oil prices merged successfully!")
        print("New columns:", [col for col in self.df.columns if 'price' in col.lower()])

        return self.df
        

    def run(self, port_source: str = 'predefined'):
        """Complete feature engineering pipeline"""
        print("Starting feature engineering...")
        
        # 1. Get ports from specified source
        #ports = self.get_ports(port_source)
        #print(f"Using {len(ports)} port locations from {port_source}")
        
        # 2. Add spatial features
        #self.add_spatial_features(ports)
        
        # 3. Add temporal features
        #self.add_temporal_features()

        #self.add_eia_report_features()
        print("Gathering oil prices...")
        self.add_crude_oil_prices()
        
        print("Feature engineering completed successfully!")
        return self.df

if __name__ == '__main__':
    #df_tanks = pd.read_csv("RawLouisiana.csv")
    #df_tanks['imaging_time'] = pd.to_datetime(df_tanks['imaging_time'])
    #wd = WeatherData(df_tanks)
    #df_result = wd.run()
    #wd.check_missing_points(df_result)
    # drop columns with significant missing weather data
    # lets drop cloud_cover, precipitation, snow, wdir, wpgt, pres, tsun
    # ==> bunch of missing data even in this -> thats ok! Thats why were doing this
    # Step 1 is about adding new features (e.g., weather, EIA dates) to the **existing rows**, not restructuring the dataset.
    # ===========================================================================
    #df_result = pd.read_csv("fulldata_with_weather.csv")
    #df_result.drop(columns=['cloud_cover', 'precipitation', 'snow', 'wdir', 'wpgt', 'pres', 'tsun', 'weather_time', 'station_dist', 'station_id'], inplace=True)
    # remaining columns
    #print("Remaining columns:")
    #print(df_result.columns)
    #df_result.to_csv("fulldata_with_weather_cleaned.csv", index=False)
    # ===========================================================================
    #df = pd.read_csv("fulldata_with_weather_cleaned.csv")    
    #feature_adder = EnhancedAdditionalFeatures(df)
    
    # Try OSM first, fallback to predefined ports
    #df_enhanced = feature_adder.run(port_source='osm')      
    #df_enhanced.to_csv("fulldata_with_additional_features.csv", index=False)
    #print("Saved enhanced features to fulldata_with_additional_features.csv")
    # ===========================================================================
    df = pd.read_csv("fulldata_with_additional_features.csv")
    feature_adder = EnhancedAdditionalFeatures(df)
    df_enhanced = feature_adder.add_crude_oil_prices(
        excel_path="/Users/finn/Desktop/UCL Masters/Data Science/Project/PET_PRI_SPT_S1_D.xls", 
        sheet_name="Data 1"
    )
    df_enhanced.to_csv("full_data.csv", index=False)
    print("Saved enhanced features to full_data.csv")