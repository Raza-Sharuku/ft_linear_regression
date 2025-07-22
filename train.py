import csv
import json

def read_data(filename):
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

def train_model(mileages, prices, learning_rate=0.2, iterations=10):
    theta0 = 0.0
    theta1 = 0.0
    m = len(mileages)
    print(f"Number of data points: {m}")
    
    max_mileage = max(mileages)
    normalized_mileages = [x / max_mileage for x in mileages]
    
    for i in range(iterations):
        sum_errors_theta0 = 0
        sum_errors_theta1 = 0
        
        # Calculate error for all data points(right side of the equation)
        for j in range(m):
            predicted_price = estimate_price(normalized_mileages[j], theta0, theta1)
            error = predicted_price - prices[j]
            
            sum_errors_theta0 += error
            sum_errors_theta1 += error * normalized_mileages[j]
        
        # Calculate gradient according to formula of subject(left side of the equation)
        tmp_theta0 = learning_rate * (1/m) * sum_errors_theta0
        tmp_theta1 = learning_rate * (1/m) * sum_errors_theta1
        # print(f"tmp_theta0: {round(tmp_theta0, 6)}, tmp_theta1: {round(tmp_theta1, 6)}")
        
        # update theta0 and theta1
        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1
        # print(f"theta0: {round(theta0, 6)}, theta1: {round(theta1, 6)}")
    
    # Adjust theta1 to normalize
    theta1 = theta1 / max_mileage
    
    return theta0, theta1

def save_parameters(theta0, theta1):
    params = {
        'theta0': theta0,
        'theta1': theta1,
    }
    
    with open('model.json', 'w') as file:
        json.dump(params, file)
    
    print(f"theta0 = {theta0:.6f}")
    print(f"theta1 = {theta1:.6f}")

def main():
    try:
        # Read data
        mileages, prices = read_data('data.csv')
        print(f"Data read: {len(mileages)} items")
        
        # Train model
        print("Training model...")
        theta0, theta1 = train_model(mileages, prices)
        
        # Save parameters
        save_parameters(theta0, theta1)
        print("Training completed!")
        
    except FileNotFoundError:
        print("Error: data.csv file not found")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()