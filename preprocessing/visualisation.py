import folium
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdt
import matplotlib.ticker as mtk

from apollo import osgconv as osg


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


def compare_precipitation_and_flow(dfs_precipitation, colors, labels, df_flow, year):

    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax1.set_xlim([dt.date(year, 1, 1), dt.date(year, 12, 31)])
    ax1.set_xlabel('Date')

    ax1.xaxis.set_major_locator(mdt.MonthLocator())
    ax1.xaxis.set_major_formatter(mdt.DateFormatter('%b'))
    ax1.set_ylim(0, 0.8*max(df['Rain'].max() for df in dfs_precipitation))
    ax1.set_ylabel('Precipitation (mm)')
    ax1.yaxis.set_major_locator(mtk.MaxNLocator(5))

    ax1.grid(c='black', ls='dotted', lw=0.5)

    for i, df in enumerate(dfs_precipitation):
        ax1.plot(df['Date'], df['Rain'], colors[i], lw=2.5, ls='-', label=f"precipitation_{labels[i]}")

    ax2 = ax1.twinx()
    ax2.set_ylabel('Flow (m'+r'$^3$'+'s'+r'$^{-1}$'+')')
    ax2.set_ylim(4 * df_flow['Flow'].max(),0)

    #ax2.plot(rf_9to9['Date'],rf_9to9['Snow Melt']*2, 'green', lw=2.5, ls='-', label='precipitation 9h to 9h')
    ax2.plot(df_flow['Date'], df_flow['Flow'], 'black', lw=2.8, ls='-', label='Flow')

    ax1.legend(loc=0, bbox_to_anchor=(0.25,0.8))
    plt.show()