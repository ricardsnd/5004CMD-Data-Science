# Project: 5004CMD
# Ricards N-D

import pandas as pandas_lib
import dask.dataframe as dask_eng
import matplotlib.pyplot as plotter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as nump
import time
import os

# Configuration & File Setup
trips_distance_data = '5004 trips data.csv' 
scale_for_graph = 1_000_000 # To make our graphs readable (Millions)

def run_sequential():
    """ Runs the data filtering using standard, single-core Pandas. """
    start = time.time()
    mobility_data = pandas_lib.read_csv(trips_distance_data)
    mobility_data.columns = mobility_data.columns.str.strip() 
    national_data = mobility_data[mobility_data['Level'] == 'National']
    return  national_data, time.time() - start

def run_parallel(partitions):
    """ Runs the data filtering using Dask, spreading the work across CPU cores. """
    start = time.time()
    # assume_missing=True saves the Dask ValueError with dtypes
    raw_data = dask_eng.read_csv(trips_distance_data, dtype={'County Name': 'object', 'Level': 'object'}, assume_missing=True)
    raw_data = raw_data.repartition(npartitions=partitions)
    raw_data.columns = raw_data.columns.str.strip()
    
    national_data = raw_data[raw_data['Level'] == 'National'].compute()
    return  national_data, time.time() - start

if __name__ == "__main__":
    if not os.path.exists(trips_distance_data):
        print(f"Couldn't find {trips_distance_data}.")
    else:
        print("Starting the analysis")

        # QUESTION E: Benchmarking Sequential vs Parallel
        print("Running Performance Benchmarks")
        _, seq_time = run_sequential()
        print(f"Sequential Processing Time: {seq_time:.4f} seconds")
        
        _, par_10_time = run_parallel(10)
        print(f"Parallel Processing (10 cores) Time: {par_10_time:.4f} seconds")
        
        national_data, par_20_time = run_parallel(20)
        print(f"Parallel Processing (20 cores) Time: {par_20_time:.4f} seconds")

        # Pre-processing for the visuals
        national_data['Date'] = pandas_lib.to_datetime(national_data['Date'])
        national_data = national_data.sort_values('Date')

        # DATA VISUALIZATION SECTION:

        # QUESTION A: Staying Home vs Traveling
        print("Plot A: Mobility Trends")
        plotter.figure(figsize=(12, 6)) # Made slightly wider for better time viewing
        
        plotter.plot(national_data['Date'], national_data['Population Staying at Home']/scale_for_graph, 
                 label='Staying at Home', color='#2980B9', linewidth=2.5, zorder=3)
        plotter.plot(national_data['Date'], national_data['Population Not Staying at Home']/scale_for_graph, 
                 label='Traveling', color='#D35400', linewidth=2.5, zorder=3)
        
        plotter.title('Figure 1: National Mobility', fontsize=15, fontweight='bold', pad=15)
        plotter.ylabel('Population (Millions)', fontsize=12)
        plotter.xlabel('Date', fontsize=12)
        plotter.legend(fontsize=11, loc='best')
        plotter.grid(True, linestyle='--', alpha=0.6, zorder=0)
        plotter.tight_layout()
        plotter.savefig('Plot_A_Mobility.png', dpi=300) # High-res save

        # QUESTION B: High Volume Scatterplot (>10M)
        print("Plot B: High Volume Comparison")
        threshold = 10_000_000
        set_10_25 = national_data[national_data['Number of Trips 10-25'] > threshold]
        set_50_100 = national_data[national_data['Number of Trips 50-100'] > threshold]
        
        fig, (ax1, ax2) = plotter.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Figure 2: High Volume Travel Days (>10M Trips)', fontsize=16, fontweight='bold', y =0.98)
        
        # Subplot 1
        ax1.scatter(set_10_25['Date'], set_10_25['Number of Trips 10-25']/scale_for_graph, 
                    color='#27AE60', alpha=0.7, s=60, edgecolor='black', zorder=3)
        ax1.set_title('10-25 Miles', fontsize=13)
        ax1.set_ylabel('Trips (Millions)', fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.6, zorder=0)
        
        # Subplot 2
        ax2.scatter(set_50_100['Date'], set_50_100['Number of Trips 50-100']/scale_for_graph, 
                    color='#C0392B', alpha=0.7, s=60, edgecolor='black', zorder=3)
        ax2.set_title('50-100 Miles', fontsize=13)
        ax2.set_ylabel('Trips (Millions)', fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.6, zorder=0)
        
        fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust to fit the main title
        plotter.savefig('Plot_B_Scatters.png', dpi=300)

        # QUESTION C: Predictive Modelling
        print("Machine Learning Model")
        model_data = national_data[['Number of Trips', 'Number of Trips 5-10']].dropna()
        num_trips = model_data[['Number of Trips']].values
        num_trips_510 = model_data['Number of Trips 5-10'].values
        
        linear_model = LinearRegression().fit(num_trips, num_trips_510)
        predictions = linear_model.predict(num_trips)
        r2 = r2_score(num_trips_510, predictions)
        rmse = nump.sqrt(mean_squared_error(num_trips_510, predictions))
        
        print(f"R2: {r2:.4f} , RMSE: {rmse:.2f}")

        plotter.figure(figsize=(10, 6))
        plotter.scatter(num_trips/scale_for_graph, num_trips_510/scale_for_graph, alpha=0.4, color='#7F8C8D', s=40, zorder=2, label='Actual Data')
        plotter.plot(num_trips/scale_for_graph, predictions/scale_for_graph, color='#E74C3C', linewidth=3, zorder=3, label=f'Regression Line ($R^2$: {r2:.2f})')
        
        plotter.title('Figure 3: Predicting 5-10 Mile Trips', fontsize=15, fontweight='bold', pad=15)
        plotter.xlabel('Total Trips (Millions)', fontsize=12)
        plotter.ylabel('5-10 Mile Trips (Millions)', fontsize=12)
        plotter.legend(fontsize=11)
        plotter.grid(True, linestyle='--', alpha=0.6, zorder=0)
        plotter.tight_layout()
        plotter.savefig('Plot_C_Regression.png', dpi=300)

        # QUESTION D: Trip Distribution
        print("Trip Distance Distribution")
        dist_cols = ['Number of Trips <1', 'Number of Trips 1-3', 'Number of Trips 3-5', 'Number of Trips 5-10', 'Number of Trips 10-25']
        averages = national_data[dist_cols].mean() / scale_for_graph
        
        plotter.figure(figsize=(10, 6))
        # Custom colours for the bars
        colors = ['#34495E', '#2E4053', '#283747', '#212F3D', '#1B2631']
        bars = averages.plot(kind='bar', color=colors, edgecolor='black', zorder=3)
        
        plotter.title('Figure 4: Average Number of Travelers by Distance', fontsize=15, fontweight='bold', pad=15)
        plotter.ylabel('Average Daily Trips (Millions)', fontsize=12)
        plotter.xlabel('Distance Brackets (Miles)', fontsize=12)
        
        # Make the x-axis labels horizontal or slightly angled for easier reading
        plotter.xticks(rotation=15, fontsize=11) 
        plotter.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        plotter.tight_layout()
        plotter.savefig('Plot_D_Distribution.png', dpi=300)

        print("Done!")