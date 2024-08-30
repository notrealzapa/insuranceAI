from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import joblib
from main3 import predict_and_calculate  # Importing backend functions

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load the pre-trained model
best_model = joblib.load('best_model.pkl')

# Define the SQLAlchemy model for storing customer data
class CustomerData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    risk_factor = db.Column(db.Float)
    insurance_price = db.Column(db.Float)

# Create the database
with app.app_context():
    db.create_all()

# Flask route to render the HTML frontend
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle CSV uploads and process the data
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'name' not in request.form:
        return "File or name missing", 400

    file = request.files['file']
    dataset_name = request.form['name']
    
    if file.filename == '':
        return "No selected file", 400

    if file:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file)

        # Process each row in the DataFrame
        results = []
        for _, row in df.iterrows():
            customer_data = row.to_dict()
            # Use the provided name for the dataset entry
            predicted_risk_factor, insurance_price = predict_and_calculate(customer_data)

            # Save to the SQL database
            new_entry = CustomerData(name=dataset_name, risk_factor=predicted_risk_factor, insurance_price=insurance_price)
            db.session.add(new_entry)
            db.session.commit()

            results.append({
                'name': dataset_name,
                'risk_factor': predicted_risk_factor,
                'insurance_price': insurance_price
            })

        return jsonify(results)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
