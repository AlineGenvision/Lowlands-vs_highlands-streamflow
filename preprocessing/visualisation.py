import folium
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdt
import matplotlib.ticker as mtk

from apollo import osgconv as osg
from preprocessing import surface_interpolation as si


def plot_spatial_distribution(domain_data, catchment_polygon, date=(1979 ,1 ,1), value_column='Rain', resolution=0.25,
                     max_value=None, bbox=None, crop=True):
    '''
       Plots the spatial data using Folium with intensity based on values.

       Parameters
       ----------
       domain_data : xarray.Dataset
           The dataset containing spatial data with coordinates and time dimension.
       catchment_polygon : shapely.geometry.Polygon
           Polygon representing the catchment area.
       date : tuple, optional
           A tuple specifying the date (year, month, day) for selecting the data. Default is (1979, 1, 1).
       value_column : str, optional
           The name of the column containing the values for intensity. Default is 'Rain'.
       resolution : float, optional
           The resolution of the grid cells for plotting. Default is 0.25.
       max_value : float, optional
           The maximum value for the color scale. If None, the maximum value from
           the data will be used. Default is None.
       bbox : list of tuples, optional
           A list of (latitude, longitude) tuples defining the bounding box to highlight
           on the map. Default is None.

       Returns
       -------
       folium.Map
           A Folium map object with the spatial data plotted.
       '''

    points = catchment_polygon.centroid
    lat, lon = osg.BNG_2_latlon(points.x[0] ,points.y[0])

    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = bbox
        if crop is True:
            # For plotting purposes: don't include the last rectangles in both directions
            domain_data = domain_data.sel(latitude=slice(lat_max, lat_min+resolution),
                                          longitude=slice(lon_min, lon_max-resolution))

    data = domain_data.sel(time=dt.datetime(date[0], date[1], date[2]))
    df = data.to_dataframe().reset_index()

    m = folium.Map(location=[lat, lon], zoom_start=8)

    gdf = catchment_polygon.to_crs("EPSG:4326")
    layer = folium.GeoJson(gdf, name='catchment').add_to(m)

    if max_value is None:
        max_value = df[value_column].max()
    colormap = folium.LinearColormap(colors=['blue', 'green', 'yellow', 'orange', 'red'],
                                     vmin=0,
                                     vmax=max_value)
    colormap.caption = value_column
    colormap.add_to(m)

    df = df.dropna()

    # Add grid cells to the map
    for _, row in df.iterrows():
        lat, lon = row['latitude'], row['longitude']
        value = row[value_column]
        color = colormap(value)
        folium.Rectangle(
            bounds=[(lat, lon), (lat -resolution, lon +resolution)],
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.2,
            popup=f"{value_column}: {value}"
        ).add_to(m)

    if bbox is not None:

        folium.Polygon(
            locations=[(lat_min, lon_min), (lat_max, lon_min), (lat_max, lon_max), (lat_min, lon_max)],
            color='blue',
            weight=2,
            fill=False,
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def plot_centroid_interpolation(catchment_boundary, df_data, resolution, labels=False):

    points = catchment_boundary.centroid
    lat_c, lon_c = osg.BNG_2_latlon(points.x[0], points.y[0])
    lat_raster = si.round_to_nearest(lat_c, base=resolution, buffer=0)
    lon_raster = si.round_to_nearest(lon_c, max=False, base=resolution, buffer=0)

    m = folium.Map(location=[lat_c, lon_c], zoom_start=8)
    folium.TileLayer('CartoDB Positron No Labels').add_to(m)
    folium.Marker([lat_c, lon_c], popup='Centroid').add_to(m)

    gdf = catchment_boundary.to_crs("EPSG:4326")
    layer = folium.GeoJson(gdf, name='catchment', color='steelblue', opacity='0.95').add_to(m)

    for _, row in df_data.iterrows():
        lat, lon = row['latitude'], row['longitude']
        folium.Rectangle(
            bounds=[(lat, lon), (lat - resolution, lon + resolution)],
            color='lightgray',
            fill=False,
        ).add_to(m)

    if labels is True:

        offset_x = resolution * 0.15
        offset_y = resolution * 0.35

        # Add labels to the corners
        corners = {
            'Q11': (lat_raster + offset_x, lon_raster - offset_y),
            'Q12': (lat_raster + offset_x, lon_raster + resolution + 0.5*offset_y),
            'Q21': (lat_raster - resolution - offset_x, lon_raster - offset_y),
            'Q22': (lat_raster - resolution - offset_x, lon_raster + resolution + 0.5*offset_y),
            'P': (lat_c - offset_x, lon_c + 0.3*offset_y)
        }

        for label, (lat, lon) in corners.items():
            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(html=f'<div style="font-size: 12px; color: black;">{label}</div>')
            ).add_to(m)

    folium.Rectangle(
        bounds=[(lat_raster, lon_raster), (lat_raster - resolution, lon_raster + resolution)],
        color='orangered',
        fill=False,
        fill_opacity=0.2,
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def compare_precipitation_and_flow(year, dfs_precipitation, colors, labels, plot_snow=False, df_separate=None,
                                   label_separate=None, save_path=None, figsize=None):

    if figsize is None:
        figsize = (16, 5)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.set_xlim([dt.date(year, 1, 1), dt.date(year, 12, 31)])
    ax1.set_xlabel(year, fontweight='bold', fontsize=12)

    ax1.xaxis.set_major_locator(mdt.MonthLocator())
    ax1.xaxis.set_major_formatter(mdt.DateFormatter('%b'))
    ax1.set_ylim(0, 1.1*max([df[df['Date'].dt.year == year]['Rain'].max() for df in dfs_precipitation]))
    ax1.set_ylabel('Precipitation (mm)', fontweight='bold', fontsize=12)
    ax1.yaxis.set_major_locator(mtk.MaxNLocator(5))

    ax1.grid(c='black', ls='dotted', lw=0.5)

    for i, df in enumerate(dfs_precipitation):
        ax1.plot(df['Date'], df['Rain'], colors[i], lw=2.0, ls='-', label=f"precipitation {labels[i]}")

    if plot_snow is True:
        df_snow = dfs_precipitation[0]
        ax1.plot(df_snow['Date'],df_snow['Snow Melt'], 'orangered', lw=2.1, ls='-', label='snow melt')
        ax1.spines['top'].set_visible(False)
        ax1.tick_params(axis='both', labelsize=12)

    if df_separate is not None:

        if label_separate is None:
            label_separate = 'Flow'

        ax2 = ax1.twinx()
        ax2.set_ylabel(f"{label_separate} (m³/s)", fontweight='bold', fontsize=12)
        ax2.set_ylim(df_separate[label_separate].max()*2,0)
        ax2.tick_params(axis='both', labelsize=12)

        ax2.plot(df_separate['Date'], df_separate[label_separate], 'black', lw=1.5, ls='-', label='riverflow')

        handles, labels = ax1.get_legend_handles_labels()
        if df_separate is not None:
            handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2

        ax1.legend(handles, labels, loc='center left', bbox_to_anchor=(0.0, 0.70), fontsize=12)
    else:
        ax1.legend(loc='upper left', fontsize=12)

    ax1.spines['bottom'].set_linewidth(1)
    ax1.spines['left'].set_linewidth(1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.tick_params(axis='both', labelsize=12)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_hourly_rainfall(df_rain, year, month, days, save_path=None):
    hours = list(range(24))
    features_needed = ['Date'] + [f"Rain_-{i + 1}" for i in hours]
    df_filtered = df_rain[(pd.to_datetime(df_rain['Date']).dt.year == year) &
                          (pd.to_datetime(df_rain['Date']).dt.month == month)][features_needed]

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 6))
    axes = axes.flatten()

    for i, day in enumerate(days):
        ax = axes[i]

        # Plot the mean
        mean = df_filtered[day:day + 1][[f"Rain_-{i + 1}" for i in range(24)]].sum(axis=1) / 24
        mean_value = mean.iloc[0]
        ax.axhline(y=mean_value, color='black', ls='--', lw='1.5', label='Average Rain')

        # Calculate and plot parameterizations
        row = df_filtered.iloc[day, 1:]
        x_max = abs(int(row.idxmax().split('_')[1]) + 1)
        y_max = row[row.idxmax()]
        ax.plot([x_max, x_max], [y_max, mean_value], color='orangered', ls='-', lw=1.5, label='Parametrisation peak')
        ax.plot([x_max, hours[-1]], [mean_value, mean_value], color='orangered', ls='-', lw=1.5)

        # Plot the hourly rainfall values
        for index, row in df_filtered[day:day + 1].iterrows():
            ax.plot(hours, row[1:], label=row['Date'], color='Teal')

        ax.set_xlabel(f"{df_filtered.iloc[day]['Date']} [h]", fontweight='bold', fontsize=12)
        ax.set_ylim(-0.2, 2.5)
        if i == 0 or i == 2:
            ax.set_ylabel('Rainfall [mm]', fontweight='bold', fontsize=12)
        else:
            ax.set_yticklabels([])
        if i == len(days) - 1:
            ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, c='grey', ls='dotted', lw=0.5)
        ax.set_xticks(hours)
        ax.set_xticklabels([str((i + 9) % 24) for i in range(24)])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()