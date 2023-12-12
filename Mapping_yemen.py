# this script finds a safe, short route through a conflict zone

# load packages
import numpy as np
import pandas as pd
import geopandas as gpd
import heapq
import matplotlib.pyplot as plt
import seaborn as sns

# import os
import os

# set wd and check
os.chdir("/Users/tomversluys/yemen_analysis")
os.getcwd()

# load data
conflict_df = pd.read_csv("filtered_df.csv")
conflict_df = conflict_df[['id', 'longitude', 'latitude']]

# convert df to geo df
conflict_geo = gpd.GeoDataFrame(
    conflict_df,
    geometry=gpd.points_from_xy(conflict_df.longitude, conflict_df.latitude)
)

# geo geometry and round coordinates to make more usable
real_data = conflict_geo[['geometry']]
real_data = gpd.GeoDataFrame(real_data)
real_data['rounded_longitude'] = real_data.geometry.x.round()
real_data['rounded_latitude'] = real_data.geometry.y.round()

#################################### get conflicts per unit of area
real_conflicts_per_square_unit = real_data.groupby(['rounded_latitude', 'rounded_longitude']).size().reset_index(name='conflicts')

# create the complete conflict data for the real dataset
unique_longs = real_data['rounded_longitude'].unique()
unique_lats = real_data['rounded_latitude'].unique()
long_grid, lat_grid = np.meshgrid(unique_longs, unique_lats)
real_location_combinations = pd.DataFrame({
    'latitude': lat_grid.ravel(),
    'longitude': long_grid.ravel()
})

real_complete_conflict_data = pd.merge(real_location_combinations, real_conflicts_per_square_unit,
                                       how='left',
                                       left_on=['latitude', 'longitude'],
                                       right_on=['rounded_latitude', 'rounded_longitude'])
real_complete_conflict_data['conflicts'] = real_complete_conflict_data['conflicts'].fillna(0)

# create function containing dijkstra's algorithm to get shortest conditional path
def dijkstra_with_path(grid, start, end, conflict_data, weight, avoid_coords):
    distances = {vertex: float('infinity') for vertex in grid}
    distances[start] = 0
    predecessor = {vertex: None for vertex in grid}
    pq = [(0, start)]

    while pq:
        current_distance, current_vertex = heapq.heappop(pq)

        if current_vertex == end:
            break

        for neighbor in grid.get(current_vertex, []):
            if neighbor in avoid_coords:
                continue  # Skip this neighbor if it's in the avoidance list

            conflict_level = conflict_data.get(neighbor, 0)
            new_distance = current_distance + 10 + (conflict_level * weight)

            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                predecessor[neighbor] = current_vertex
                heapq.heappush(pq, (new_distance, neighbor))

    # backtrack to find the path
    path = []
    current = end
    while current is not None:
        path.insert(0, current)
        current = predecessor[current]

    return path, distances[end]

# create a grid representation for dijkstra's algorithm
def create_grid_graph(conflict_data):
    grid = {}
    for index, row in conflict_data.iterrows():
        lon, lat = index
        neighbors = [
            (lon + dx, lat + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]
            if (dx != 0 or dy != 0) and (lon + dx, lat + dy) in conflict_data.index
        ]
        grid[(lon, lat)] = neighbors
        if (lon, lat) in [(44, 17), (48, 13)]:  # Example nodes for inspection
            print(f"Neighbors of ({lon}, {lat}): {neighbors}")
    return grid

# creating the complete grid
unique_longs = np.arange(real_data['rounded_longitude'].min(), real_data['rounded_longitude'].max() + 1, 1)
unique_lats = np.arange(real_data['rounded_latitude'].min(), real_data['rounded_latitude'].max() + 1, 1)

complete_grid = pd.MultiIndex.from_product([unique_longs, unique_lats], names=['longitude', 'latitude'])
complete_grid_df = pd.DataFrame(index=complete_grid).reset_index()

# merging with existing conflict data and filling NaNs
conflict_data_complete = complete_grid_df.merge(real_conflicts_per_square_unit, how='left', left_on=['longitude', 'latitude'], right_on=['rounded_longitude', 'rounded_latitude'])
# conflict_data_complete['conflicts'].fillna(0, inplace=True)
nominal_conflict_value = 0  # This is an arbitrary small value
conflict_data_complete['conflicts'].fillna(nominal_conflict_value, inplace=True)
conflict_data_complete['rounded_longitude'] = conflict_data_complete['longitude']
conflict_data_complete['rounded_latitude'] = conflict_data_complete['latitude']

# make sure the index is sorted
conflict_data_complete.set_index(['longitude', 'latitude'], inplace=True)
conflict_data_complete.sort_index(inplace=True)

# now create the grid graph
grid_graph = create_grid_graph(conflict_data_complete)

# convert conflict data to a suitable format for Dijkstra
conflict_data_dict = conflict_data_complete['conflicts'].to_dict()

# avoid_coords = []  # Clearing avoid list
weight = 0.5
avoid_coords = [(46, 16), (47, 17), (48, 13), (46, 17)]  # List of coordinates to avoid
start = (44, 18)  # Assuming this is a valid node in your grid
end = (46, 13)    # Assuming this is also a valid node

path, path_cost = dijkstra_with_path(grid_graph, start, end, conflict_data_dict, weight, avoid_coords)
print(path)

# use haverstine to find distance across path
def haversine(coord1, coord2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Coordinates in radians
    lat1 = np.radians(coord1[1])
    lon1 = np.radians(coord1[0])
    lat2 = np.radians(coord2[1])
    lon2 = np.radians(coord2[0])

    # Differences in coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance

def calculate_route_length_haversine(route):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += haversine(route[i], route[i + 1])
    return total_distance

#  usage
route_length_km = calculate_route_length_haversine(path)
print(f"Route Length: {route_length_km} kilometers")

# create heatmap of conflict with path
plt.figure(figsize=(10, 8))
ax = sns.kdeplot(
    x=real_data.geometry.x,
    y=real_data.geometry.y,
    cmap="Reds",
    shade=True,
    bw_adjust=0.5
)

# Plotting the shortest path on the heatmap
for i in range(len(path) - 1):
    start_point = path[i]
    end_point = path[i + 1]
    ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], marker='', color='black', linestyle='dotted', linewidth=2, label="Safe path" if i == 0 else "")

# Plotting avoidance coordinates as large stars and adding a dummy plot for the legend
for coord in avoid_coords:
    ax.scatter(*coord, color='black', marker='*', s=200)
ax.scatter([], [], color='black', marker='*', s=200, label='Enemy Camps')  # Dummy scatter for legend

# Plotting and labeling start and end points
start_point, end_point = path[0], path[-1]
ax.scatter(*start_point, color='green', s=70, zorder=5, label='Drop off')  # Start point
ax.scatter(*end_point, color='red', s=70, zorder=5, label='Target')  # End point

# Adding route length label
mid_index = len(path) // 2
mid_point = path[mid_index]
label_x, label_y = mid_point[0], mid_point[1]
ax.text(label_x + 0.3, label_y, f'Route Length: {route_length_km:.2f} km', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

# Adding titles and labels
ax.set_title('Shortest safe path through a conflict zone in Yemen')
ax.set_xlabel('Longitude', fontsize=16)
ax.set_ylabel('Latitude', fontsize=16)

# Add the color scale legend for the shaded conflict areas
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('Conflict intensity', fontsize=16)

# Show legend for plot elements (path, enemy camps, start/end points)
legend = ax.legend(loc='upper right', frameon=True)

# Reformat the axis to the original CRS
ax.set_xlabel('Longitude', fontsize=16)
ax.set_ylabel('Latitude', fontsize=16)

plt.savefig('safest_route_heatmap.png', dpi = 500)
plt.show()


