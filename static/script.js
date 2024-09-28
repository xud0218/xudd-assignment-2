let dataset = [];
let isConverged = false;
const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

function generateDataset() {
    fetch('/generate_dataset')
        .then(response => response.json())
        .then(data => {
            dataset = data;
            drawPlot(data);
        });
}

function stepThroughKMeans() {
    if (isConverged) {
        alert("The algorithm has already converged!");
        return;
    }

    const method = document.getElementById('init-method').value;
    fetch(`/kmeans_step/${method}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ dataset: dataset })
    })
    .then(response => response.json())
    .then(result => {
        if (result.converged) {
            isConverged = true;
            document.getElementById("convergence-warning").style.display = "block";
            alert("The dataset has converged! No further steps can be made.");
        }
        drawPlot(dataset);
        drawCentroids(result.centroids);
        highlightClusters(result.clusters);
    });
}

function completeKMeans() {
    const method = document.getElementById('init-method').value;
    fetch(`/kmeans_complete/${method}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ dataset: dataset })
    })
    .then(response => response.json())
    .then(result => {
        drawPlot(dataset);
        drawCentroids(result.centroids);
        highlightClusters(result.clusters);
    });
}

function resetKMeans() {
    isConverged = false;
    dataset = [];
    d3.select('#plot').html('');
    document.getElementById("convergence-warning").style.display = "none";
    
    fetch('/reset_kmeans')
    .then(() => console.log("KMeans reset on server"));
}

function drawPlot(data) {
    const svg = d3.select('#plot').html('').append('svg').attr('width', 500).attr('height', 500);
    svg.selectAll('circle').data(data).enter().append('circle')
       .attr('cx', d => d[0] * 500)
       .attr('cy', d => d[1] * 500)
       .attr('r', 5)
       .attr('fill', 'gray');  // Default color before cluster assignment
}

function highlightClusters(clusters) {
    const svg = d3.select('svg');

    svg.selectAll('circle')  // Color points by their cluster
        .data(clusters)
        .transition()
        .duration(500)
        .attr('fill', d => colorScale(d));
}

function drawCentroids(centroids) {
    const svg = d3.select('svg');
    svg.selectAll('rect').data(centroids).enter().append('rect')
       .attr('x', d => d[0] * 500 - 5)
       .attr('y', d => d[1] * 500 - 5)
       .attr('width', 10)
       .attr('height', 10)
       .attr('fill', 'red');
}
