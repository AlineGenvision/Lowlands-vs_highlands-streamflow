import scipy
import folium
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from apollo import osgconv as osg


def round_to_nearest(value, max=True, base=0.25, buffer=1):
    '''
       Return the nearest value in terms of the data resolution.

       Parameters
       ----------
       value : float
           The value to be rounded.
       max : bool, optional
           If True, rounds up to the nearest base and adds the buffer. (Default is True)
           If False, rounds down to the nearest base and subtracts the buffer.
       base : float, optional
           The base to which the value should be rounded. For example, if base is 0.25,
           the value will be rounded to the nearest multiple of 0.25. Default is 0.25.
       buffer : int, optional
           The number of base units to add or subtract to/from the rounded value. Default is 1.

       Returns
       -------
       float
           The value rounded to the nearest base, adjusted by the buffer.
       '''
    if max:
        return np.ceil(value/base) * base + buffer*base
    else:
        return np.floor(value / base) * base - buffer*base


def extract_raster_to_interpolate(catchment_boundary, resolution, buffer=1):
    '''
        Return the nearest value in terms of the data resolution.

        Parameters
        ----------
        catchment_path : String
            Maximum flow for scaling the y axis
        resolution : Boolean
            Pandas dataframe with observations, predictions, and date columns

        Returns
        -------
        The four corners of the bounding box of the given shape, with a certain buffer around and in lat-lon coordinates.

        '''
    minx, miny, maxx, maxy = catchment_boundary.total_bounds

    lat_min, lon_min = osg.BNG_2_latlon(minx, miny)
    lat_max, lon_max = osg.BNG_2_latlon(maxx, maxy)

    lat_min = round_to_nearest(lat_min, max=False, base=resolution, buffer=buffer)
    lon_min = round_to_nearest(lon_min, max=False, base=resolution, buffer=buffer)
    lat_max = round_to_nearest(lat_max, base=resolution, buffer=buffer)
    lon_max = round_to_nearest(lon_max, base=resolution, buffer=buffer)

    return lat_min, lat_max, lon_min, lon_max


def interpolate_surface(dataset, var_name='tp', multiplicator=1000*24, plot=False):
    '''
    Interpolates spatial data using RegularGridInterpolator for each time step in the dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset containing spatial data with coordinates and time dimension.
    var_name : str, optional
        The name of the variable to interpolate. Default is 'tp'.
    multiplicator: int, optional
        Muliplication factor to display the final interpolations in the correct units (e.g. *24 for daily total).
    plot : bool, optional
        If True, plots the original and interpolated data for the first few time steps. Default is False.

    Returns
    -------
    interpolated_functions : dict
        A dictionary where keys are time steps and values are the corresponding interpolator functions.
    '''
    lats, lons = dataset.latitude.values, dataset.longitude.values
    times = dataset.time.values

    interpolated_functions = {}

    for i, time in enumerate(times):

        data = dataset[var_name].sel(time=time)
        interpolator = scipy.interpolate.RegularGridInterpolator((lats, lons), data.values * multiplicator,
                                                                 method='cubic')
        interpolated_functions[time] = interpolator

        if i < 5 and plot:
            print(time)
            plot_interpolation(lats, lons, interpolator)
    return interpolated_functions


def evaluate_interpolator_on_grid(x1_surface, x2_surface, interpolator):
    '''
    Evaluates the interpolator function on a grid.

    Parameters
    ----------
    x1_surface : ndarray
        Grid of x1 values (e.g., latitudes).
    x2_surface : ndarray
        Grid of x2 values (e.g., longitudes).
    interpolator : function
        Interpolator function that takes an array of points and returns interpolated values.

    Returns
    -------
    ndarray
        Interpolated values on the provided grid.
    '''
    points = np.vstack([x1_surface.ravel(), x2_surface.ravel()]).T
    return interpolator(points).reshape(x1_surface.shape)


def plot_interpolation(x1, x2, f):
    '''
        Plots the interpolated surface using a 3D plot.

        Parameters
        ----------
        x1 : array_like
            Array of x1 values (e.g., latitudes).
        x2 : array_like
            Array of x2 values (e.g., longitudes).
        f : function
            Interpolator function that takes an array of points and returns interpolated values.

        Returns
        -------
        None
        '''
    x1_surface, x2_surface = np.meshgrid(np.linspace(x1.min(), x1.max(), len(x1)), np.linspace(x2.min(), x2.max(), len(x2)), indexing='ij')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1_surface, x2_surface, evaluate_interpolator_on_grid(x1_surface, x2_surface, f), alpha=0.5)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    plt.show()


def convert_polygon(polygon, conversion_func):
    '''
        Converts the coordinates of a given polygon using a specified conversion function.

        Parameters
        ----------
        polygon : shapely.geometry.Polygon
            The polygon whose coordinates are to be converted.
        conversion_func : function
            A function that takes two arguments (x, y) and returns a tuple (new_x, new_y)
            representing the converted coordinates.

        Returns
        -------
        shapely.geometry.Polygon
            A new polygon with coordinates converted by the provided conversion function.
        '''
    new_coords = [conversion_func(*coord) for coord in polygon.exterior.coords]
    return Polygon(new_coords)


# TODO : Cleanup function
def plot_integration(polygon, grid_resolution=0.05):
    latlon_polygon = convert_polygon(polygon.geometry.iloc[0], osg.BNG_2_latlon)
    lat_min, lon_min, lat_max, lon_max = latlon_polygon.bounds
    x = np.arange(lat_min, lat_max, grid_resolution)
    y = np.arange(lon_min, lon_max, grid_resolution)
    grid_x, grid_y = np.meshgrid(x, y)

    all_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    prepared_polygon = prep(latlon_polygon)
    mask = np.array([prepared_polygon.contains(Point(p)) for p in all_points])
    inside_points = all_points[mask]

    points = polygon.centroid
    lat, lon = osg.BNG_2_latlon(points.x[0], points.y[0])
    m = folium.Map(location=[lat, lon], zoom_start=8)
    layer = folium.GeoJson(polygon.to_crs("EPSG:4326"), name='catchment').add_to(m)

    for point in all_points:
        folium.CircleMarker(location=[point[0], point[1]], color='green', radius=0.5).add_to(m)

    for point in inside_points:
        folium.CircleMarker(location=[point[0], point[1]], color='red', radius=0.5).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def integrate_rainfall_over_polygon(polygon, interp_func, grid_resolution=0.05):
    '''
    Integrates interpolated rainfall values over a given polygon. This polygon is first convert to lat-lon coordinates.
    This function approximates the integration by evaluating the interpolation function over a grid within the polygon,
    the grid has a predefined resolution. The surface-interpolated precipitation is then the sum of the precipitation
     at all the points within the polygon, divided by the number of points.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The polygon over which to integrate rainfall.
    interp_func : function
        Interpolator function that takes an array of points and returns interpolated values.
    resolution : float, optional
        Resolution of the grid for the integration. Default is 0.05.

    Returns
    -------
    float
        The integrated rainfall over the polygon.
    '''

    latlon_polygon = convert_polygon(polygon.geometry.iloc[0], osg.BNG_2_latlon)
    lat_min, lon_min, lat_max, lon_max = latlon_polygon.bounds
    x = np.arange(lat_min, lat_max, grid_resolution)
    y = np.arange(lon_min, lon_max, grid_resolution)
    grid_x, grid_y = np.meshgrid(x, y)

    points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    prepared_polygon = prep(latlon_polygon)
    mask = np.array([prepared_polygon.contains(Point(p)) for p in points])
    inside_points = points[mask]

    rainfall_values = interp_func(inside_points)
    return np.sum(rainfall_values) / len(inside_points)
