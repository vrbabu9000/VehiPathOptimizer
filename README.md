# VehiPathOptimizer

VehiPathOptimizer is an advanced, open-source fleet optimization tool designed to streamline the routing process for a fleet of vehicles. This Python-based solution efficiently addresses the Generic Vehicle Routing Fleet Optimization problem by integrating the power of Google OR-tools with the Open Source Routing Machine (OSRM) for precise distance calculations. The tool's uniqueness lies in its capability to leverage locally hosted mapping data through OSRM, deployed on a Docker image, for generating an accurate distance matrix essential for route optimization. Specifically tailored for operations in Kollam, Kerala, India, this tool facilitates the upload of delivery points via a .csv file, offering users an optimized set of routes along with a visual mapping for easy navigation and understanding. 

**I last worked on this in Dec 2021 and most of the modules used here maybe very outdated and would require a lot of code refactoring to accurately work.**

## Key Features

- **User-Friendly Data Input**: Accepts coordinates of delivery points through a .csv file upload.
- **Optimized Route Calculation**: Utilizes a local OSRM instance hosted on Docker for accurate distance matrix generation and Google OR-tools for optimization, focusing on minimizing travel distance and cost.
- **Visual Route Mapping**: Provides a visual representation of the optimized routes, enhancing clarity and usability.
- **Local Focus with Global Application**: While designed with a focus on Kollam, Kerala, the tool's generic nature allows for adaptation to various geographical locations.

## Getting Started

### Prerequisites

- Python 3.x
- Docker (for hosting the local OSRM instance)
- Google OR-tools
- Additional Python libraries: Pandas, Numpy, Matplotlib, Seaborn, Folium

### Installation

1. **Clone the repository** to get started with VehiPathOptimizer:

   ```bash
   git clone https://github.com/vrbabu9000/VehiPathOptimizer.git
2. Set up the OSRM Docker instance for your local mapping data. Detailed instructions can be found in the OSRM documentation.

3. Install required Python libraries using the provided requirements.txt:
    ```bash
   pip install -r requirements.txt
### Usage
1. **Prepare Your Dataset:** Format your delivery dataset as per the given template and save it as a .csv file. The dataset should include coordinates and demand for each delivery point.

2. **Upload Your Dataset:** Follow the instructions in the notebook to upload your .csv file.

3. **Run the Notebook:** Execute the Jupyter notebook to process the uploaded data. The notebook will generate optimized routes and a visual map showcasing these routes for easy understanding and navigation.

4. **View the Optimized Routes:** The final output includes a set of optimized routes with visual mapping, making it easier for users to plan their logistics and delivery schedules efficiently.

### Contributing
I last worked on this module in 2021 and most of the modules are quite outdated. A lot more improvements can be made. Your contributions can help make VehiPathOptimizer even better. Feel free to fork the repository, make improvements, and submit pull requests. We're excited to see your innovative ideas and enhancements!

### License
This project is licensed under the MIT License. For more details, see the LICENSE file in the repository.
