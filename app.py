# Import necessary modules from Flask and Python standard library
from flask import Flask, render_template, request, redirect, url_for
import os                            # For creating folders like 'output'
from vrp_solver import solve_vrp     # Import the custom vehicle routing function

app = Flask(__name__)                # Initialize Flask application
@app.route('/', methods=['GET'])     # Route for the homepage - only responds to GET requests
def index():
    return render_template('index.html')       # Render the main page (index.html)

# Route to handle form submission and run the solver - responds to POST requests
@app.route('/run', methods=['POST'])
def run():
    result = None                               # Initialize result variable
    try:
        # ----------- FORM DATA PARSING -----------
        num_customers = int(request.form['num_customers'])     # Read number of customers entered in the form
        customer_coords_raw = request.form['customer_coords'].strip().split('\n')        # Read customer coordinates input (multiline text area), split by newline
        num_vehicles = int(request.form['num_vehicles'])        # Read number of vehicles entered in the form
        depot_raw = request.form['depot']                       # Read depot coordinates from form input

        # ----------- INPUT CLEANUP -----------

        # Convert customer coordinates to list of [x, y] integers
        customers = [list(map(int, line.split())) for line in customer_coords_raw]
        depot = list(map(int, depot_raw.strip().split()))        # Convert depot input string into list of integers [x, y]

        # ----------- CALL THE SOLVER FUNCTION -----------
        result = solve_vrp(num_customers, customers, num_vehicles, depot)    # Call the Genetic Algorithm function to solve VRP

    except Exception as e:
        result = f"Error: {e}"                  # If there's any error (e.g. invalid input), show it as output
    return render_template('index.html', result=result)     # Re-render the index page and pass result back to the template

# Start the Flask app
if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)    # Ensure the output folder exists for saving the plot image
    app.run(debug=True)                        # Run the app in debug mode for live reloading and error display

# test