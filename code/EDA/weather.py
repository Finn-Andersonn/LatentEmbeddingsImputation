# ----------------------------------------
# Understanding when images have been taken overtime
# Identifying skew in recordings
# Starting to seem that more images are taken in winter, attributing to our missingness -> perhaps sunlight is issue
# ----------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests

df = pd.read_csv('RawLouisiana.csv')
    
class Weather:
    '''
    Description:
        Plots the occurence of satellite imaging:
            -> Day
            -> Month
            -> Year
    Args:
        df: Pandas dataframe

    Returns:
        Histogram: Isolating days/weeks/months of occurence 
        Time series: Daily image acquisition volume over time frame
    '''
    def __init__(self, df):
        self.df = df
        self.show = False
        self.figsize = (14, 6)
        self.dpi = 150
        plt.style.use('seaborn-whitegrid')
        
    def _setup_plot(self):
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, facecolor='#f5f5f5')        
        ax.tick_params(axis='both', which='major', labelsize=12, colors='gray')
        ax.grid(True, linestyle='--', color='gray', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        return fig, ax
    
    def _finalize_plot(self, fig, ax, title, xlabel, ylabel, save_path=None):
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel(xlabel, fontsize=14, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=14, labelpad=10)        
        if ax.get_xaxis().get_scale() != 'log' and isinstance(ax.get_xaxis().get_major_formatter(), mdates.DateFormatter):
            fig.autofmt_xdate()
        else:
            plt.xticks(rotation=45)            
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5),
                     frameon=True, edgecolor='black', framealpha=0.9)
        
        plt.tight_layout()        
        if save_path:
            plt.savefig(save_path)
        if self.show:
            plt.show()
        else:
            plt.close()
    
    def satellite_series(self, df=None, save_path=None):
        if df is None:
            df = self.df            
        df = df.copy()
        if 'imaging_time' in df.columns:
            df['imaging_time'] = pd.to_datetime(df['imaging_time'])
            df['date'] = df['imaging_time'].dt.date            
            fig, ax = self._setup_plot()            
            daily_counts = df.groupby('date').size()
            ax.plot(daily_counts.index, daily_counts.values, marker='o', 
                   linestyle='-', linewidth=2.5, color='blue')            
            self._finalize_plot(fig, ax, title='Daily Image Acquisition Volume', xlabel='Date', ylabel='Images per day', save_path=save_path)
        return
    
    def satellite_freq(self, df=None, save_path=None, freq='H'):
        freq_dict = {'H': 'hour', 'D': 'day', 'W': 'weekday', 'M': 'month', 'Y': 'year'}
        label_dict = {'H': 'Hour (24h)', 'D': 'Day', 'W': 'Day of Week', 'M': 'Month', 'Y': 'Year'}
        if freq not in freq_dict:
            raise ValueError(f"Frequency '{freq}' not supported. Use one of: {list(freq_dict.keys())}")
        dt_attr = freq_dict[freq]
        x_label = label_dict[freq]
        if df is None:
            df = self.df            
        df = df.copy()
        if 'imaging_time' in df.columns:
            df['imaging_time'] = pd.to_datetime(df['imaging_time'])
            df['time_unit'] = getattr(df['imaging_time'].dt, dt_attr)        
            fig, ax = self._setup_plot()            
            time_counts = df['time_unit'].value_counts().sort_index()
            ax.bar(time_counts.index, time_counts.values, color='teal', alpha=0.8)
            self._finalize_plot(fig, ax, title=f'Satellite Imaging by {x_label}', xlabel=x_label, ylabel='Number of Images',save_path=save_path)
            return

    

if __name__ == '__main__':
    weather = Weather(df)
    weather.show = True

    # frequency of satellite imaging
    save_path = '/Users/finn/Desktop/UCL Masters/Data Science/Project/graphs/weather/freq/'
    freq = 'H'
    weather.satellite_freq(save_path=save_path + 'image_'+ freq + '_freq.png', freq= freq)
    weather.satellite_freq(save_path=save_path + 'image_D_freq.png', freq='D')
    weather.satellite_freq(save_path=save_path + 'image_M_freq.png', freq='M')
    weather.satellite_freq(save_path=save_path + 'image_Y_freq.png', freq='Y')

    # daily satellite imaging volume
    save_path = '/Users/finn/Desktop/UCL Masters/Data Science/Project/graphs/weather/series/'
    weather.satellite_series(save_path=save_path + 'daily_image_volume.png')