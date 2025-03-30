from weather import Weather
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import webbrowser
import os

df = pd.read_csv('RawLouisiana.csv')

class Location(Weather):
    def __init__(self, df):
        super().__init__(df)
        self.df = df
        self.tank_farm = df['tank_farm'] if 'tank_farm' in df.columns else None

    def better_satellites(self, df, save_path=None, freq='Y'):
        freq_dict = {'H': 'hour', 'D': 'day', 'W': 'weekday', 'M': 'month', 'Y': 'year'}
        label_dict = {'H': 'Hour (24h)', 'D': 'Day', 'W': 'Day of Week', 'M': 'Month', 'Y': 'Year'}
        if freq not in freq_dict:
            raise ValueError(f"Frequency '{freq}' not supported. Use one of: {list(freq_dict.keys())}")

        if df is None:
            df = self.df
        if 'imaging_time' in df.columns:
            df['imaging_time'] = pd.to_datetime(df['imaging_time'])
            time_attr = freq_dict[freq]
            df['time_bin'] = getattr(df['imaging_time'].dt, time_attr)
            fig, ax = self._setup_plot()
            satellite_trend = df.groupby(['time_bin', 'scene_source_id']).size().unstack(fill_value=0)
            satellite_trend.plot(kind='bar', stacked=True, ax=ax)
            bars = ax.patches
            n_bins = len(satellite_trend.index)
            n_satellites = len(satellite_trend.columns)
            for i, rect in enumerate(bars):
                bin_idx = i % n_bins
                satellite_idx = i // n_bins
                if satellite_idx >= n_satellites or bin_idx >= n_bins:
                    continue
                height = satellite_trend.iloc[bin_idx, satellite_idx]
                if height < 5:
                    continue
                x = rect.get_x() + rect.get_width() / 2
                prev_heights = satellite_trend.iloc[bin_idx, :satellite_idx].sum()
                y = prev_heights + height / 2
                ax.text(x, y, str(int(height)), ha='center', va='center', color='white', fontweight='bold')
            self._finalize_plot(fig, ax, title=f'Satellite Image Sources by {label_dict[freq]}', xlabel=label_dict[freq], ylabel='Number of Images', save_path=save_path)
        return
    
    def website_marker(self, df):
        df[['lon', 'lat']] = df['Location'].str.extract(r'POINT\(\s*(?P<lon>[-+]?\d*\.\d+)\s+(?P<lat>[-+]?\d*\.\d+)\s*\)').astype(float) # vectorized

        m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=13)

        marker_cluster = MarkerCluster().add_to(m)

        for idx, row in df.iterrows():
            folium.Marker(
                location=[row['lat'], row['lon']], 
                popup=row['tank_id']
            ).add_to(marker_cluster)
            if idx % 10000 == 0:
                print(f"Added {idx} markers")

        m.save('map.html')

        map_path = os.path.abspath('map.html')
        print(f"Map saved to: {map_path}")  # Verify path
        webbrowser.open(f'file://{map_path}', new=2)


    
if __name__ == '__main__':
    location = Location(df)
    location.show = True
    save_path = '/Users/finn/Desktop/UCL Masters/Data Science/Project/graphs/location/'
    location.better_satellites(df, save_path=save_path + 'satellite_M_sources.png', freq='M')
    location.better_satellites(df, save_path=save_path + 'satellite_Y_sources.png', freq='Y')