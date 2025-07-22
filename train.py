import csv
import json

def read_data(filename):
    """Read data from CSV file"""
    mileages = []
    prices = []
    
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            mileages.append(float(row['km']))
            prices.append(float(row['price']))
    
    return mileages, prices

def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def train_model(mileages, prices, learning_rate=0.01, iterations=8000):
    """Train model using gradient descent"""
    # Initial values
    theta0 = 0.0
    theta1 = 0.0
    m = len(mileages)
    print(f"Number of data points: {m}")
    
    # Normalize data (to stabilize learning)
    max_mileage = max(mileages)
    normalized_mileages = [x / max_mileage for x in mileages]
    
    for i in range(iterations):
        # Temporary variables for gradient calculation
        sum_errors_theta0 = 0
        sum_errors_theta1 = 0
        
        # Calculate error for all data points
        for j in range(m):
            predicted_price = estimate_price(normalized_mileages[j], theta0, theta1)
            error = predicted_price - prices[j]
            
            sum_errors_theta0 += error
            sum_errors_theta1 += error * normalized_mileages[j]
        
        # Calculate gradient according to PDF formula
        tmp_theta0 = learning_rate * (1/m) * sum_errors_theta0
        tmp_theta1 = learning_rate * (1/m) * sum_errors_theta1
        
        # Simultaneous update
        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1
    
    # Adjust theta1 to normalize
    theta1 = theta1 / max_mileage
    
    return theta0, theta1, max_mileage

def save_parameters(theta0, theta1, max_mileage):
    """Save learned parameters"""
    params = {
        'theta0': theta0,
        'theta1': theta1,
        'max_mileage': max_mileage
    }
    
    with open('model.json', 'w') as file:
        json.dump(params, file)
    
    print(f"Model parameters saved:")
    print(f"θ0 = {theta0:.6f}")
    print(f"θ1 = {theta1:.6f}")

def main():
    try:
        # Read data
        mileages, prices = read_data('data.csv')
        print(f"Data read: {len(mileages)} items")
        
        # Train model
        print("Training model...")
        theta0, theta1, max_mileage = train_model(mileages, prices)
        
        # Save parameters
        save_parameters(theta0, theta1, max_mileage)
        print("Training completed!")
        
    except FileNotFoundError:
        print("Error: data.csv file not found")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()