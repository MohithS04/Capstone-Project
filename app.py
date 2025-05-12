from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session

# Use try/except to handle missing model files gracefully
try:
    model_path = r"/Users/mohithreddy/Downloads/ASH/kmeans_model.pkl"
    scaler_path = r"/Users/mohithreddy/Downloads/ASH/standard_scaler.pkl"
    features_path = r"/Users/mohithreddy/Downloads/ASH/selected_features.pkl"
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    selected_features = joblib.load(features_path)
    model_loaded = True
except Exception as e:
    print(f"Error loading models: {e}")
    model_loaded = False

cluster_labels = {
    0: "Low Need",
    1: "Medium Need",
    2: "High Need"
}

allocation_map = {
    0: 10,
    1: 30,
    2: 60
}

# New allocation amount data
allocation_amounts = {
    0: 23255.81,  # Low Need cluster
    1: 28571.43,  # Medium Need cluster
    2: 127659.57  # High Need cluster
}

# Allocation strategy data
allocation_strategy = {
    "total_budget": 10000000,
    "clusters": [
        {
            "cluster": 2,
            "category": "High Need",
            "countries": 47,
            "weight": 3,
            "weighted_need": 141,
            "allocation_percent": 60,
            "total_allocation": 6000000,
            "per_country": 127659.57
        },
        {
            "cluster": 1,
            "category": "Medium Need",
            "countries": 105,
            "weight": 2,
            "weighted_need": 210,
            "allocation_percent": 30,
            "total_allocation": 3000000,
            "per_country": 28571.43
        },
        {
            "cluster": 0,
            "category": "Low Need",
            "countries": 43,
            "weight": 1,
            "weighted_need": 43,
            "allocation_percent": 10,
            "total_allocation": 1000000,
            "per_country": 23255.81
        }
    ]
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return render_template('index.html', 
                               error_message="Error: Models not loaded correctly")
    
    try:
        # Capture country name
        country_name = request.form['country']
        
        input_data = [
            float(request.form['health']),
            float(request.form['income']),
            float(request.form['life_expec']),
            float(request.form['total_fer']),
            float(request.form['health_per_capita'])
        ]
        
        input_df = pd.DataFrame([input_data], columns=selected_features)
        scaled_input = scaler.transform(input_df)
        cluster = model.predict(scaled_input)[0]
        label = cluster_labels.get(cluster, f"Cluster {cluster}")
        allocation = allocation_map[cluster]
        
        # Get allocation amount for the cluster
        allocation_amount = allocation_amounts[cluster]
        
        # Store prediction results in session
        session['prediction'] = {
            'country_name': country_name,
            'cluster': int(cluster),
            'label': label,
            'allocation': allocation,
            'allocation_amount': allocation_amount,
            'input_data': {
                'health': input_data[0],
                'income': input_data[1],
                'life_expec': input_data[2],
                'total_fer': input_data[3],
                'health_per_capita': input_data[4]
            }
        }
        
        # Redirect to results page
        return redirect(url_for('results'))
    except Exception as e:
        return render_template('index.html', 
                               error_message=f'Error: {str(e)}')

@app.route('/results')
def results():
    # Get prediction from session
    prediction = session.get('prediction')
    if not prediction:
        return redirect(url_for('home'))
    
    return render_template('results.html', 
                          cluster=prediction['cluster'],
                          label=prediction['label'],
                          allocation=prediction['allocation'],
                          allocation_amount=prediction.get('allocation_amount', 0),
                          input_data=prediction['input_data'],
                          country_name=prediction['country_name'],
                          allocation_strategy=allocation_strategy)

@app.route('/aid_strategy')
def aid_strategy():
    cluster_table = [
        {"cluster": 0, "category": "Low Need", "description": "Countries with strong economic indicators, well-developed healthcare systems, higher life expectancy, and lower fertility rates."},
        {"cluster": 1, "category": "Medium Need", "description": "Countries with moderate development, average income levels, and improving but still developing health infrastructure."},
        {"cluster": 2, "category": "High Need", "description": "Countries facing significant challenges with low GDP, poor health indicators, higher fertility rates, and lower life expectancy."}
    ]
    return render_template('aid_strategy.html', cluster_table=cluster_table)

@app.route('/cluster_details/<int:cluster_id>')
def cluster_details(cluster_id):
    cluster_details = {
        0: {
            "name": "Low Need",
            "allocation": 10,
            "allocation_amount": allocation_amounts[0],
            "description": "Countries with strong economic indicators, well-developed healthcare systems, higher life expectancy, and lower fertility rates.",
            "characteristics": [
                "High GDP per capita (typically over $20,000)",
                "Life expectancy above 75 years",
                "Low fertility rates (typically below 2.1)",
                "Strong healthcare infrastructure",
                "Lower rates of preventable diseases"
            ],
            "example_countries": ["United States", "Canada", "Japan", "Germany", "United Kingdom", "Australia"],
            "recommended_aid": [
                "Technical assistance and knowledge transfer",
                "Research and development partnerships",
                "Support for specific vulnerable populations",
                "Environmental sustainability initiatives",
                "Natural disaster relief when needed"
            ]
        },
        1: {
            "name": "Medium Need",
            "allocation": 30,
            "allocation_amount": allocation_amounts[1],
            "description": "Countries with moderate development, average income levels, and improving but still developing health infrastructure.",
            "characteristics": [
                "Moderate GDP per capita (typically $5,000-$20,000)",
                "Life expectancy between 65-75 years",
                "Fertility rates between 2.1-3.5",
                "Developing healthcare systems with some gaps",
                "Moderate rates of preventable diseases"
            ],
            "example_countries": ["Brazil", "Mexico", "Turkey", "Thailand", "South Africa", "Argentina"],
            "recommended_aid": [
                "Healthcare system strengthening programs",
                "Education and vocational training",
                "Infrastructure development projects",
                "Small business and entrepreneurship support",
                "Water and sanitation improvements"
            ]
        },
        2: {
            "name": "High Need",
            "allocation": 60,
            "allocation_amount": allocation_amounts[2],
            "description": "Countries facing significant challenges with low GDP, poor health indicators, higher fertility rates, and lower life expectancy.",
            "characteristics": [
                "Low GDP per capita (typically below $5,000)",
                "Life expectancy below 65 years",
                "Higher fertility rates (typically above 3.5)",
                "Limited healthcare infrastructure",
                "Higher rates of preventable diseases and malnutrition"
            ],
            "example_countries": ["Niger", "Democratic Republic of Congo", "Mali", "Afghanistan", "Somalia", "Yemen"],
            "recommended_aid": [
                "Essential healthcare services and vaccination programs",
                "Food security and nutrition programs",
                "Basic education access improvements",
                "Clean water access programs",
                "Maternal and child health initiatives",
                "Disease prevention and treatment"
            ]
        }
    }
    
    if cluster_id not in cluster_details:
        return redirect(url_for('aid_strategy'))
    
    return render_template('cluster_details.html', details=cluster_details[cluster_id])

if __name__ == "__main__":
    app.run(debug=True, port=5050)