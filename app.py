from flask import Flask, render_template, request, jsonify
import numpy as np
from kmeans import KMeans

app = Flask(__name__)

kmeans = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_dataset')
def generate_dataset():
    np.random.seed(42)
    X = np.random.rand(100, 2)
    return jsonify(X.tolist())

@app.route('/kmeans_step/<method>', methods=['POST'])
def kmeans_step(method):
    global kmeans
    data = request.json
    X = np.array(data['dataset'])

    if kmeans is None:
        kmeans = KMeans(k=3, init_method=method)
        
    result = kmeans.step_through(X)
    
    return jsonify(result)

@app.route('/kmeans_complete/<method>', methods=['POST'])
def kmeans_complete(method):
    global kmeans
    data = request.json
    X = np.array(data['dataset'])
    
    kmeans = KMeans(k=3, init_method=method)
    clusters, centroids = kmeans.fit(X)
    
    return jsonify({'clusters': clusters.tolist(), 'centroids': centroids.tolist(), 'converged': True})

@app.route('/reset_kmeans')
def reset_kmeans():
    global kmeans
    kmeans = None
    return "KMeans reset successfully"

if __name__ == '__main__':
    app.run(debug=True)
