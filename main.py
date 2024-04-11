#Basic Libraries

import pandas as pd
import streamlit as st
from PIL import Image
import base64
import time



#Distance Libraries
import osrm
import streamlit_folium
import folium
import polyline

#GoogleOR
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

#API Description
import requests
import warnings
warnings.filterwarnings("ignore")

#==============>GoogleOR Tools CVRP<================
#"""Capacited Vehicles Routing Problem (CVRP)."""


def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = distance_matrix
    data['demands'] = demand
    data['vehicle_capacities'] = vehicle_capacities
    data['num_vehicles'] = len(vehicle_capacities)
    data['depot'] = 0  # index of depo
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    st.markdown('# Optimized Routes')
    st.text(f'Objective: {solution.ObjectiveValue()}')
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {0} with capacity {1}:\n'.format(vehicle_id,data['vehicle_capacities'][vehicle_id])
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(data_fnl["Location"].iloc[manager.IndexToNode(index)],
                                                       route_load)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} Load({1})\n'.format(data_fnl["Location"].iloc[manager.IndexToNode(index)], route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance / 10)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        st.text(plan_output)
        total_distance += route_distance
        total_load += route_load
    st.text('Total distance of all routes: {}m'.format(total_distance / 10))
    st.text('Total load of all routes: {}'.format(total_load))


def save_solution(data, manager, routing, solution):
    total_distance = 0
    total_load = 0
    vehicle_optimized_route = pd.DataFrame(
        columns=["VehicleID", "Optimized_route", "Optimized_route_index", "Total_load", "Total_distance"])
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route_distance = 0
        routes_fnl = []
        routes_index = []
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            routes_fnl.append(data_fnl["Location"].iloc[manager.IndexToNode(index)])
            routes_index.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        routes_fnl.append(data_fnl["Location"].iloc[manager.IndexToNode(index)])
        routes_index.append(manager.IndexToNode(index))
        vehicle_optimized_route = vehicle_optimized_route.append({"VehicleID": vehicle_id, \
                                                                  "Optimized_route": routes_fnl, \
                                                                  "Optimized_route_index": routes_index, \
                                                                  "Total_load": route_load, \
                                                                  "Total_distance": route_distance}, ignore_index=True)
    return pd.DataFrame(vehicle_optimized_route)


def main():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    global display, summary_df
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance Constraint
    routing.AddDimension(transit_callback_index,
                         0,  # null capacity slack
                         800000,  # vehicle maximum distance ==> 80km*10 ==> 10 is scaling factor
                         True,  # start cumul to zero
                         'Distance')

    distance_dimension = routing.GetDimensionOrDie('Distance')
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)  # to escape local minima
    search_parameters.time_limit.FromSeconds(10)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        display = print_solution(data, manager, routing, solution)
        summary_df = save_solution(data, manager, routing, solution)

    return display, summary_df

#======================>Utility Functions<=======================

def get_coordinated_for_selected_locs(Optimized_Route_Df):

    li_coord=[]

    for route in Optimized_Route_Df['Optimized_route_index']:
        coord=[]
        for loc in route:
            if(loc)==0:
                coord.append([data_fnl.Longitude.iloc[0],data_fnl.Latitude.iloc[0]])
            else:
                coord.append([data_fnl.Longitude.iloc[loc],data_fnl.Latitude.iloc[loc]])
        li_coord.append(coord)
    return li_coord


def get_route(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat):
    loc = "{},{};{},{}".format(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)
    url = "http://router.project-osrm.org/route/v1/driving/"
    r = requests.get(url + loc)
    if r.status_code != 200:
        return {}

    res = r.json()
    routes = polyline.decode(res['routes'][0]['geometry'])
    start_point = [res['waypoints'][0]['location'][1], res['waypoints'][0]['location'][0]]
    end_point = [res['waypoints'][1]['location'][1], res['waypoints'][1]['location'][0]]
    distance = res['routes'][0]['distance']

    out = {'route': routes,
           'start_point': start_point,
           'end_point': end_point,
           'distance': distance
           }

    return out

def routing_mapping(x):
    routing_list = []
    individual_routes = []
    for z in range(len(x)):
        w = x.iloc[z]
        test = get_route(w.pickup_lon,w.pickup_lat,w.dropoff_lon,w.dropoff_lat)
        y = test.get('route')
        individual_routes.append(y)
        if z != (len(x)-1):
            for n in range(len(y)-1):
                routing_list.append(y[n])
        else:
            for n in range(len(y)):
                routing_list.append(y[n])
    return routing_list, individual_routes

# Define function to scale distance Matrix and covert float to int values
def scale_integer_func(Q):
    return [[int(10 * x) for x in L] for L in Q]

#======================>BASIC UI<=======================

st.set_page_config(layout="centered")
img2 = Image.open("img (2).jpg")
st.image(img2, width=200)
st.title("My grocery-store")
st.write("My grocery-store is a SaaS platform that empowers all Small to Medium business owners that require delivery "
         "route optimization to maximize profits.This helps you to manage customer subscriptions and deliveries with "
         "unrivalled efficiency. My-Grocery-store digitizes your delivery and optimizes the routes "
         "with its tech-enabled"
         "end-to-end solutions.")
file_ = open("route-optimisation.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()



st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    unsafe_allow_html=True,
)

#======================>Taking Inputs from User<=======================
col1, col2 = st.beta_columns([4,4])
with col1:
    depo_lat =  st.text_input("Enter Latitude coordinates of your Home Depo Base")
with col2:
    depo_long = st.text_input("Enter Longitude coordinates of your Home Depo Base")

vehicle_capacities = st.text_input("Max Capacity of Vehicle", help='Add Space between vehicle Capacities')
#Upload A Excel Sheet
uploaded_files = st.file_uploader("Choose a Excel file", type = 'xlsx')

img2 = Image.open("img_1.png")
st.image(img2)

#======================>Application Data Process<=======================
if uploaded_files and depo_lat and depo_long and vehicle_capacities is not None:
    if st.button("Submit"):
        #Depo cords
        depo_coordinates = f'{depo_lat}, {depo_long}'
        #Vehicle Capacity
        vehicle_capacities = list(map(float, vehicle_capacities.split()))
        #Dataset
        dataset = pd.read_excel(uploaded_files,engine='openpyxl',sheet_name=0)
        #Add depo to dataset
        depo_data = []
        depo_data.insert(0, {'Location': "Depo", 'Coordinates': depo_coordinates, 'Demand':0})
        data_fnl = pd.concat([pd.DataFrame(depo_data),dataset], ignore_index = True)
        data_fnl["Latitude"] = data_fnl.Coordinates.apply(lambda x :  x.split(",")[1].strip())
        data_fnl["Longitude"] = data_fnl.Coordinates.apply(lambda x :  x.split(",")[0])
        # Total Demand of Customers
        demand = data_fnl.Demand.to_list()
        temp = []
        for x in range(len(data_fnl)):
            temp.append(f'{data_fnl.Latitude.iloc[x]},{data_fnl.Longitude.iloc[x]}')
        locations = ";".join(temp)
        # OSRM API to get Distance Matrix
        url = "http://router.project-osrm.org/table/v1/driving/"
        url2 = f"?annotations=distance"
        r = requests.get(url + locations + url2)
        response = r.json()
        if response["code"] == 'Ok':
            distance_mx = response["distances"]
            distance_matrix = scale_integer_func(distance_mx)  # 99x99 Matrix

            result_display, result_save = main()
            st.dataframe(result_save, 1600,1200)


            # Define a folium Map
            m = folium.Map(location=[data_fnl.Longitude.iloc[0], data_fnl.Latitude.iloc[0]], zoom_start=14)

            # plot Depo
            folium.Marker(location=[data_fnl.Longitude.iloc[0], data_fnl.Latitude.iloc[0]],
                          icon=folium.Icon(icon='home', color='red')).add_to(m)

            # Colors for folium route + Marker
            color_route = ['blue', 'red', 'black', 'purple', 'green', 'orange', 'pink', 'lightblue', 'lightgreen',
                           'gray']


            with st.spinner('Loading Optimized Map....'):
                for x in range(len(result_save)):
                    if len(result_save.Optimized_route_index.iloc[x]) > 1:
                        routeX = pd.DataFrame(
                            columns=['pickup', 'dropoff', 'pickup_lon', 'pickup_lat', 'dropoff_lon', 'dropoff_lat'])
                        for i in range(len(result_save.Optimized_route_index.iloc[x]) - 1):
                            routeX.loc[i] = [result_save.Optimized_route.iloc[x][i]] + [
                                result_save.Optimized_route.iloc[x][i + 1]] + [
                                                get_coordinated_for_selected_locs(result_save)[x][i][1]] + [
                                                get_coordinated_for_selected_locs(result_save)[x][i][0]] + [
                                                get_coordinated_for_selected_locs(result_save)[x][i + 1][1]] + [
                                                get_coordinated_for_selected_locs(result_save)[x][i + 1][0]]
                        routing_list, individual_routes = routing_mapping(routeX)
                        folium.PolyLine(routing_list, weight=8, color=color_route[x], opacity=0.9).add_to(m)
                        for z in range(1, len(routeX) - 1):
                            folium.Marker(location=[routeX.pickup_lat.iloc[z], routeX.pickup_lon.iloc[z]],
                                          popup=f'{z}. {routeX.pickup.iloc[z]}',
                                          icon=folium.Icon(icon='play', color=color_route[x])).add_to(m)
                streamlit_folium.folium_static(m)
            st.success('Done!')








