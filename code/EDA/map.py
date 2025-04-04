# ----------------------------------------
# Want to identify where each tank is, visual proximity (values to come)
# Importantly, we also notice after plotting the tank volume by farm over time we clearly identify the infrequent
# recording we previously expected.
# 
# ----------------------------------------


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import geopandas as gpd
import contextily as cx
from shapely.geometry import Point
import matplotlib
from matplotlib import patheffects
from matplotlib.patches import Circle, Wedge
from adjustText import adjust_text
df = pd.read_csv('RawLouisiana.csv')


class fill_vol():
    def __init__(self, df):
        self.df = df
        self.tank_farm = df['tank_farm'] if 'tank_farm' in df.columns else None
        self.df['imaging_time'] = pd.to_datetime(self.df['imaging_time'])

    def tank_farm_counts(self):
        return self.df['tank farm'].value_counts().to_dict()

    def mean_volume_by_time(self, column='tank volume', freq='M', save_path='mean_volume_by_time.png', ncols=3):
        '''
        Description:
            As each recording is taken sporadically for each farm---inconsistently across months---plot the mean
            volume over time. This is for each tank farm.

        Args:
            df.column: Tank volume
            freq: Month (variable)
        Returns:
            Time series: Mean volume for each tank farm over dataframe lenth
        '''
        df = self.df.copy()
        df['period'] = df['imaging_time'].dt.to_period(freq)        
        month_year_dict = {p: {} for p in df['period'].unique()}
        for period in month_year_dict:
            for farm in df['tank farm'].unique():
                month_year_dict[period][farm] = []

        for _, row in df.iterrows():
            period = row['imaging_time'].to_period(freq)
            farm = row['tank farm']
            if farm in month_year_dict[period]:
                month_year_dict[period][farm].append(row[column])

        month_years = sorted(month_year_dict.keys(), key=lambda x: x.start_time)
        tank_farms = list(next(iter(month_year_dict.values())).keys())
        
        n_farms = len(tank_farms)
        nrows = int(np.ceil(n_farms / ncols))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 3), squeeze=False)
        
        for idx, farm in enumerate(tank_farms):
            means = []
            months = []
            for i, period in enumerate(month_years):
                volumes = month_year_dict[period].get(farm, [])
                means.append(np.mean(volumes) if volumes else np.nan)
                months.append(str(period))
            
            ax = axs[idx // ncols][idx % ncols]
            ax.plot(months, means, marker='o', label=f'{column} - {farm}')
            ax.set_title(f'{column.capitalize()} for {farm}')
            ax.set_xlabel('Period')
            ax.set_ylabel(f'Mean {column}')
            
            step = max(1, len(months) // 10)
            xticks = np.arange(0, len(months), step)
            ax.set_xticks(xticks)
            ax.set_xticklabels([months[i] for i in xticks], rotation=45)
            ax.legend()

        for i in range(n_farms, nrows * ncols):
            fig.delaxes(axs[i // ncols][i % ncols])

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Saved plot to {os.path.abspath(save_path)}")


    def tank_volume_histograms(self, save_path='tank_volume_histograms.png', ncols=3):
        '''
        Description:
            See how often each tank farm reaches max capacities -> distribution of each farm.
            We also clearly see the undefined tank farm.
        Args:
            df.column: Tank farm
            df.colum: Tank volume
        
        Returns:
            Histogram: frequency of tank volume occurences
        '''
        tank_farms = self.df['tank farm'].unique()
        n_farms = len(tank_farms)
        nrows = int(np.ceil(n_farms / ncols))
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 3), squeeze=False)

        for idx, farm in enumerate(tank_farms):
            ax = axs[idx // ncols][idx % ncols]
            subset = self.df[self.df['tank farm'] == farm]
            sns.histplot(subset['tank volume'], bins=20, kde=True, ax=ax, color='blue')
            ax.set_title(f'Tank Volume Distribution - {farm}')
            ax.set_xlabel('Tank Volume')
            ax.set_ylabel('Frequency')

        # hide unused subplots
        for i in range(n_farms, nrows * ncols):
            fig.delaxes(axs[i // ncols][i % ncols])

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Saved histogram to {os.path.abspath(save_path)}")


    def tank_farm_correlation(self, column='tank volume', freq='M', save_path='tank_correlation_heatmap.png'):
        '''
        Description:
            Any significant correlation amongst tank farms? This is difficult as doesnt consider time lags here
                -> properly addressed in corr.py

        Args:
            df.column: Tank volume
            freq: Month (variable)
        
        Returns:
            Correlation heatmap: correlation amongst tank farms in one period
        '''
        df = self.df.copy()
        df['period'] = df['imaging_time'].dt.to_period(freq)
        month_years = sorted(df['period'].unique(), key=lambda x: x.start_time)
        tank_farms = df['tank farm'].unique()
        mean_volumes = {(p, farm): [] for p in month_years for farm in tank_farms}
        for _, row in df.iterrows():
            key = (row['imaging_time'].to_period(freq), row['tank farm'])
            mean_volumes[key].append(row[column])
        data_dict = {}
        for p in month_years:
            row = []
            for farm in tank_farms:
                values = mean_volumes.get((p, farm), [])
                row.append(np.mean(values) if values else np.nan)
            data_dict[p] = row
        heat_df = pd.DataFrame(data_dict, index=tank_farms).T
        correlation_matrix = heat_df.corr()
        plt.figure(figsize=(10, 8))
        plt.title("Correlation Matrix between Tank Farms")
        plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
        plt.colorbar()
        plt.xticks(np.arange(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
        plt.yticks(np.arange(len(correlation_matrix.index)), correlation_matrix.index)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Saved correlation heatmap to {os.path.abspath(save_path)}")

    def freq_tank_farm_map(self, save_path='freq_tank_farms_map.png', with_basemap=True):
        ''''
        Description:
            Most difficult plot by far. 
            -> The rank (#1, #2, etc.) is based on the maximum tank volume (max_capacity_per_farm) of each tank farm.
            -> The number in parentheses ((123)) represents how many recordings were made at that tank far (ax.annotate(f"#{row['Rank']}\n({row['recording_count']})")
            -> The red wedge fills a portion of the circle based on the number of recordings relative to the tank farm with the most recordings (max_recording).
            Spatial Clustering: We can see clear clustering in location -> inspires the graph-based similarity approach for imputation.
        Args:
            save_path (str): Path to save the plot to
            with_basemap (bool): Whether to include a basemap (OpenStreetMap)
        Returns:
            None
        '''
        plt.style.use('default')
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        
        df = self.df.copy()

        # --- Coordinate Processing ---
        coords = df['Location'].str.extract(
            r'POINT\(\s*(?P<lon>[-+]?\d*\.\d+)\s+(?P<lat>[-+]?\d*\.\d+)\s*\)'
        ).astype(float)
        df['lon'] = coords['lon']
        df['lat'] = coords['lat']

        # --- Data Processing ---
        # recordings per tank farm
        recordings = df.groupby('tank farm').size().reset_index(name='recording_count')
        
        # maximum tank volume per tank
        max_per_tank = df.groupby('tank_id')['tank volume'].max().reset_index()
        tank_farm_map = df[['tank_id', 'tank farm', 'lon', 'lat']].drop_duplicates(subset='tank_id')
        max_per_tank = max_per_tank.merge(tank_farm_map, on='tank_id', how='left')
        
        # aggregate per tank farm: maximum tank volume and average location
        max_capacity_per_farm = max_per_tank.groupby('tank farm').agg({
            'tank volume': 'max',
            'lat': 'mean',
            'lon': 'mean'
        }).sort_values(by='tank volume', ascending=False).reset_index()
        max_capacity_per_farm = max_capacity_per_farm.merge(recordings, on='tank farm', how='left')
        
        # compute rank for labeling (by max volume)
        max_capacity_per_farm['Rank'] = range(1, len(max_capacity_per_farm) + 1)
        
        # Ahh -> Use a constant circle radius for all farms
        constant_radius = 8000  # adjust as needed (in EPSG:3857 meters)
        max_capacity_per_farm['radius'] = constant_radius
        
        # wedge fill fraction (recording_count relative to max)
        max_recording = max_capacity_per_farm['recording_count'].max()
        max_capacity_per_farm['fraction'] = max_capacity_per_farm['recording_count'] / max_recording

        # --- Create GeoDataFrame & Project ---
        geometry = [Point(xy) for xy in zip(max_capacity_per_farm['lon'], max_capacity_per_farm['lat'])]
        gdf = gpd.GeoDataFrame(max_capacity_per_farm, geometry=geometry, crs='EPSG:4326')
        gdf = gdf.to_crs(epsg=3857)
        
        fig, ax = plt.subplots(figsize=(16, 16))
        
        # circles and wedges for each tank farm
        for idx, row in gdf.iterrows():
            center = (row.geometry.x, row.geometry.y)
            radius = row['radius']
            frac = row['fraction']
            
            # Full circle outline
            circle_outline = Circle(center, radius, facecolor='none', edgecolor='darkred', linewidth=2, zorder=2)
            ax.add_patch(circle_outline)
            # Red wedge indicating fraction
            if frac > 0:
                wedge = Wedge(center, radius, theta1=90, theta2=90 - frac * 360,
                            facecolor='red', edgecolor='none', zorder=3)
                ax.add_patch(wedge)
        
        # Compute the median x and y of all centers
        median_x = gdf.geometry.x.median()
        median_y = gdf.geometry.y.median()
        fixed_offset = 30000  # adjust as needed

        label_positions = []
        circle_positions = []
        min_label_distance = 40000
        label_positions = []
        circle_positions = []  # Store circle centers and radii
        min_label_distance = 40000  # Minimum distance between labels

        # 1: collect all circle positions
        for idx, row in gdf.iterrows():
            center = (row.geometry.x, row.geometry.y)
            radius = row['radius']
            circle_positions.append((center, radius))

        # 2: by rank for consistent placement
        gdf_sorted = gdf.sort_values('Rank')

        # 3: annotate each tank farm with improved smart positioning
        for idx, row in gdf_sorted.iterrows():
            center = (row.geometry.x, row.geometry.y)
            
            # graph cluttered so trying multiple angles for placement
            best_pos = None
            best_score = -float('inf')
            angles = np.linspace(0, 2*np.pi, 24, endpoint=False)  # 24 possible positions for more options
            
            for angle in angles:
                # different distances from center
                for distance_factor in [1.0, 1.3, 1.6]:  # attempting different distances
                    dx = fixed_offset * distance_factor * np.cos(angle)
                    dy = fixed_offset * distance_factor * np.sin(angle)
                    candidate_pos = (center[0] + dx, center[1] + dy)
                    
                    # Calculate score based on:
                    # 4. Distance to other labels (higher is better)
                    # 5. Distance to circle centers (higher is better)
                    # 6. Whether the label overlaps any circle (avoid)
                    
                    # Start with a base score
                    score = 0
                    
                    # 4. Check distance to other labels
                    min_dist_to_label = float('inf')
                    if label_positions:
                        for existing_pos in label_positions:
                            dist = np.sqrt((existing_pos[0] - candidate_pos[0])**2 + 
                                        (existing_pos[1] - candidate_pos[1])**2)
                            min_dist_to_label = min(min_dist_to_label, dist)
                        
                        # Higher distance to labels = better score
                        score += min_dist_to_label / 10000  # Scale factor <- stay inside page (roughly)
                    else:
                        # If no labels yet, give a good score
                        score += 10
                    
                    # 5. Check distance to circle centers (other than this circle)
                    overlaps_any_circle = False
                    for circ_center, circ_radius in circle_positions:
                        # skip checking the current circle
                        if circ_center == center:
                            continue
                            
                        # Label position to circle center distance
                        dist_to_circle_center = np.sqrt((circ_center[0] - candidate_pos[0])**2 + 
                                                    (circ_center[1] - candidate_pos[1])**2)
                        
                        # Check: does label overlaps this circle?
                        if dist_to_circle_center < circ_radius + 15000:  # Add buffer for label size
                            overlaps_any_circle = True
                            break
                    
                    # heavily penalize positions that overlap circles
                    if overlaps_any_circle:
                        score -= 1000
                        
                    # Choose position with maximum score
                    if score > best_score:
                        best_score = score
                        best_pos = candidate_pos
            
            # If we couldn't find a good position, increase the offset and try again
            if best_score < -500:  # If all positions overlap circles
                # Try with a larger offset
                for extra_factor in [1.8, 2.0, 2.2]:
                    for angle in angles:
                        dx = fixed_offset * extra_factor * np.cos(angle)
                        dy = fixed_offset * extra_factor * np.sin(angle)
                        candidate_pos = (center[0] + dx, center[1] + dy)
                        
                        # Check if overlaps any circle
                        overlaps_any_circle = False
                        for circ_center, circ_radius in circle_positions:
                            if circ_center == center:
                                continue
                            dist_to_circle_center = np.sqrt((circ_center[0] - candidate_pos[0])**2 + 
                                                        (circ_center[1] - candidate_pos[1])**2)
                            if dist_to_circle_center < circ_radius + 15000:
                                overlaps_any_circle = True
                                break
                        
                        if not overlaps_any_circle:
                            best_pos = candidate_pos
                            break
                    
                    if best_pos is not None:
                        break
            
            # last resort
            if best_pos is None:
                # Place it very far in a random direction
                angle = np.random.random() * 2 * np.pi
                best_pos = (center[0] + fixed_offset * 2.5 * np.cos(angle), 
                        center[1] + fixed_offset * 2.5 * np.sin(angle))
            
            # Add this label position to our list
            label_positions.append(best_pos)
            
            # annotation
            ax.annotate(f"#{row['Rank']}\n({row['recording_count']})",
                        xy=center,
                        xytext=best_pos,
                        arrowprops=dict(arrowstyle='->', color='black', lw=1),
                        fontsize=12, ha='center', va='center',
                        bbox=dict(boxstyle='circle,pad=0.3', fc='black', ec='none'),
                        alpha=0.5, color='white', fontweight='bold',
                        zorder=4)
        
        # Set axis limits based on data bounds (with margin)
        xmin, ymin, xmax, ymax = gdf.total_bounds
        x_margin = (xmax - xmin) * 0.1
        y_margin = (ymax - ymin) * 0.1
        ax.set_xlim(xmin - x_margin, xmax + x_margin)
        ax.set_ylim(ymin - y_margin, ymax + y_margin)
        
        # --- Add Basemap ---
        if with_basemap:
            cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, zorder=0)
        
        ax.set_title('Tank Farms: Constant Circles with Wedge Fill and Fixed Offset Labels', fontsize=16)
        ax.set_axis_off()
        plt.savefig(save_path, dpi=350)
        print(f"Saved final map to {save_path}")
        plt.close()

save_path = '/Users/finn/Desktop/UCL Masters/Data Science/Project/graphs/fill_vol/'
fill_vol = fill_vol(df)
#fill_vol.mean_volume_by_time(save_path = save_path + 'mean_volume_by_time.png')
#fill_vol.tank_volume_histograms(save_path = save_path + 'tank_volume_histograms.png')
#fill_vol.tank_farm_correlation(save_path = save_path + 'tank_correlation_heatmap.png')
#fill_vol.max_tank_capacity_per_farm(save_path = save_path + 'max_tank_capacity_per_farm.png')
#fill_vol.debug_ranked_tank_farm_map()
#fill_vol.ranked_tank_farm_map(save_path = save_path + 'ranked_tank_farms_map.png')
fill_vol.freq_tank_farm_map(save_path = save_path + 'freq_tank_farms_map.png')