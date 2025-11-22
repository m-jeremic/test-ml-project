##Average temperature prediction - the linear regression project

-We would like to be able to predict average temperature of cities based on their latitude.
-We already have data for temperatures and latitudes for some European cities.

# Linear Regression Project

Implementation of linear regression with one variable to predict average temperature based on city latitude.

## Project Structure
- `data/`: Contains the dataset file
- `src/`: Core implementation modules
- `notebooks/`: Jupyter notebook with analysis

## Setup
1. Clone the repository ://github.com/yourusername/test-ml-project.git
   - cd supermarket-profit-prediction
2. Create virtual environment: `python -m venv venv`
3. Activate environment:
   - Windows: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`

## Running the Analysis
### Using Jupyter Notebook
1. Make sure your virtual environment is activated:
   - Windows: `venv\Scripts\activate`
2. Start Jupyter Notebook from terminal:
   jupyter notebook
# Run from command line with parameters
python main.py --target-column "avg temp" --feature-column "latitude"