from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json

app = Flask(__name__)
CORS(app)

def generate_dataset(dataset_type, n_samples, noise_level, seed=42):
    np.random.seed(seed)
    X = np.linspace(-3, 3, n_samples)
    noise = np.random.randn(n_samples) * noise_level

    if dataset_type == "sine":
        y = np.sin(X) * 3 + noise
    elif dataset_type == "quadratic":
        y = X**2 - 2*X + noise
    elif dataset_type == "cubic":
        y = X**3 - 3*X + noise
    elif dataset_type == "linear":
        y = 2*X + 1 + noise
    else:
        y = np.sin(X) * 3 + noise

    return X, y

@app.route('/api/train', methods=['POST'])
def train_models():
    data = request.json
    dataset_type = data.get('datasetType', 'sine')
    n_samples = int(data.get('nSamples', 100))
    noise_level = float(data.get('noiseLevel', 0.5))
    poly_degree = int(data.get('polyDegree', 3))
    svr_c = float(data.get('svrC', 1.0))
    svr_epsilon = float(data.get('svrEpsilon', 0.1))
    svr_kernel = data.get('svrKernel', 'rbf')
    custom_x = data.get('customX', None)
    custom_y = data.get('customY', None)

    # Generate or use custom data
    if custom_x and custom_y:
        X = np.array(custom_x)
        y = np.array(custom_y)
    else:
        X, y = generate_dataset(dataset_type, n_samples, noise_level)

    X_reshape = X.reshape(-1, 1)

    # --- Linear Regression ---
    lr_model = LinearRegression()
    lr_model.fit(X_reshape, y)
    y_pred_lr = lr_model.predict(X_reshape)

    # --- Polynomial Regression ---
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=poly_degree)),
        ('linear', LinearRegression())
    ])
    poly_pipeline.fit(X_reshape, y)
    y_pred_poly = poly_pipeline.predict(X_reshape)

    # --- SVR ---
    svr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel=svr_kernel, C=svr_c, epsilon=svr_epsilon))
    ])
    svr_pipeline.fit(X_reshape, y)
    y_pred_svr = svr_pipeline.predict(X_reshape)

    def metrics(y_true, y_pred, name):
        mse = mean_squared_error(y_true, y_pred)
        return {
            "model": name,
            "r2": round(r2_score(y_true, y_pred), 4),
            "mse": round(mse, 4),
            "rmse": round(np.sqrt(mse), 4),
            "mae": round(mean_absolute_error(y_true, y_pred), 4)
        }

    # Sort by X for clean lines
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx].tolist()

    response = {
        "xValues": X_sorted,
        "yActual": y[sort_idx].tolist(),
        "yLinear": y_pred_lr[sort_idx].tolist(),
        "yPoly": y_pred_poly[sort_idx].tolist(),
        "ySVR": y_pred_svr[sort_idx].tolist(),
        "metrics": [
            metrics(y, y_pred_lr, "Linear Regression"),
            metrics(y, y_pred_poly, f"Polynomial (deg {poly_degree})"),
            metrics(y, y_pred_svr, f"SVR ({svr_kernel})")
        ],
        "nSamples": n_samples,
        "datasetType": dataset_type
    }

    return jsonify(response)

@app.route('/api/predict', methods=['POST'])
def predict_single():
    """Predict a single X value using all three models"""
    data = request.json
    x_input = float(data.get('xValue'))
    dataset_type = data.get('datasetType', 'sine')
    n_samples = int(data.get('nSamples', 100))
    noise_level = float(data.get('noiseLevel', 0.5))
    poly_degree = int(data.get('polyDegree', 3))
    svr_c = float(data.get('svrC', 1.0))
    svr_epsilon = float(data.get('svrEpsilon', 0.1))
    svr_kernel = data.get('svrKernel', 'rbf')

    X, y = generate_dataset(dataset_type, n_samples, noise_level)
    X_reshape = X.reshape(-1, 1)
    x_test = np.array([[x_input]])

    # Linear
    lr_model = LinearRegression()
    lr_model.fit(X_reshape, y)

    # Poly
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=poly_degree)),
        ('linear', LinearRegression())
    ])
    poly_pipeline.fit(X_reshape, y)

    # SVR
    svr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel=svr_kernel, C=svr_c, epsilon=svr_epsilon))
    ])
    svr_pipeline.fit(X_reshape, y)

    return jsonify({
        "xInput": x_input,
        "linearPred": round(float(lr_model.predict(x_test)[0]), 4),
        "polyPred": round(float(poly_pipeline.predict(x_test)[0]), 4),
        "svrPred": round(float(svr_pipeline.predict(x_test)[0]), 4)
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "ML Comparison API running!"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
