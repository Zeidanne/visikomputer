/**
 * KNN Page JavaScript
 * Handles KNN model training, visualization, and prediction
 */

document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const knnForm = document.getElementById('knnForm');
    const predictionForm = document.getElementById('predictionForm');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');
    const errorMessage = document.getElementById('errorMessage');
    const kValueSlider = document.getElementById('kValue');
    const kValueDisplay = document.getElementById('kValueDisplay');
    const predictBtn = document.getElementById('predictBtn');
    const predictHint = document.getElementById('predictHint');
    
    // Chart instance
    let scatterChart = null;
    
    // Model state
    let modelTrained = false;
    let datasetInfo = null;

    // Initialize
    loadDatasetPreview();
    initKSlider();
    initAxisSelectors();

    /**
     * Load dataset preview
     */
    async function loadDatasetPreview() {
        try {
            const response = await fetch('/knn/dataset-info');
            const data = await response.json();
            
            if (response.ok) {
                datasetInfo = data;
                displayDatasetPreview(data);
            }
        } catch (error) {
            console.error('Error loading dataset:', error);
        }
    }

    /**
     * Display dataset preview in table
     */
    function displayDatasetPreview(data) {
        const tbody = document.getElementById('datasetBody');
        if (!tbody || !data.preview) return;

        const classNames = ['Setosa', 'Versicolor', 'Virginica'];
        const classStyles = ['class-setosa', 'class-versicolor', 'class-virginica'];

        let html = '';
        data.preview.forEach((row, index) => {
            const classIdx = row.target;
            html += `
                <tr>
                    <td>${index + 1}</td>
                    <td>${row.sepal_length.toFixed(1)}</td>
                    <td>${row.sepal_width.toFixed(1)}</td>
                    <td>${row.petal_length.toFixed(1)}</td>
                    <td>${row.petal_width.toFixed(1)}</td>
                    <td><span class="class-label ${classStyles[classIdx]}">${classNames[classIdx]}</span></td>
                </tr>
            `;
        });
        tbody.innerHTML = html;
    }

    /**
     * Initialize K value slider
     */
    function initKSlider() {
        if (kValueSlider && kValueDisplay) {
            kValueSlider.addEventListener('input', function() {
                kValueDisplay.textContent = this.value;
            });
        }
    }

    /**
     * Initialize axis selectors for scatter plot
     */
    function initAxisSelectors() {
        const xAxis = document.getElementById('xAxis');
        const yAxis = document.getElementById('yAxis');

        if (xAxis && yAxis) {
            xAxis.addEventListener('change', updateChart);
            yAxis.addEventListener('change', updateChart);
        }
    }

    /**
     * Train model form submission
     */
    if (knnForm) {
        knnForm.addEventListener('submit', async function(e) {
            e.preventDefault();

            // Get selected features
            const featureCheckboxes = document.querySelectorAll('input[name="features"]:checked');
            if (featureCheckboxes.length < 2) {
                showError('Pilih minimal 2 fitur untuk training');
                return;
            }

            const features = Array.from(featureCheckboxes).map(cb => parseInt(cb.value));

            // Get form data
            const formData = {
                features: features,
                k: parseInt(document.getElementById('kValue').value),
                test_size: parseFloat(document.getElementById('testSize').value),
                metric: document.getElementById('distanceMetric').value
            };

            // Show loading
            loadingIndicator.classList.remove('hidden');
            resultsSection.classList.add('hidden');
            errorSection.classList.add('hidden');

            try {
                const response = await fetch('/knn/train', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(formData)
                });

                const data = await response.json();
                loadingIndicator.classList.add('hidden');

                if (!response.ok) {
                    throw new Error(data.error || 'Terjadi kesalahan saat training');
                }

                modelTrained = true;
                displayResults(data);
                enablePrediction();
            } catch (error) {
                loadingIndicator.classList.add('hidden');
                showError(error.message);
            }
        });
    }

    /**
     * Display training results
     */
    function displayResults(data) {
        resultsSection.classList.remove('hidden');

        // Metrics
        document.getElementById('accuracyValue').textContent = formatPercent(data.accuracy);
        document.getElementById('precisionValue').textContent = formatPercent(data.precision);
        document.getElementById('recallValue').textContent = formatPercent(data.recall);

        // Confusion Matrix
        displayConfusionMatrix(data.confusion_matrix);

        // Classification Report
        displayClassificationReport(data.classification_report);

        // Scatter Plot
        createScatterChart(data.visualization_data);
    }

    /**
     * Display confusion matrix
     */
    function displayConfusionMatrix(matrix) {
        const container = document.getElementById('confusionMatrix');
        if (!container || !matrix) return;

        const labels = ['Set', 'Ver', 'Vir'];
        
        let html = `
            <div class="confusion-cell header"></div>
            <div class="confusion-cell header">Set</div>
            <div class="confusion-cell header">Ver</div>
            <div class="confusion-cell header">Vir</div>
        `;

        matrix.forEach((row, i) => {
            html += `<div class="confusion-cell row-label">${labels[i]}</div>`;
            row.forEach((val, j) => {
                const cellClass = i === j ? 'diagonal' : 'off-diagonal';
                html += `<div class="confusion-cell ${cellClass}">${val}</div>`;
            });
        });

        container.innerHTML = html;
    }

    /**
     * Display classification report
     */
    function displayClassificationReport(report) {
        const tbody = document.getElementById('reportBody');
        if (!tbody || !report) return;

        const classNames = ['Setosa', 'Versicolor', 'Virginica'];
        const classStyles = ['class-setosa', 'class-versicolor', 'class-virginica'];

        let html = '';
        report.forEach((row, index) => {
            html += `
                <tr>
                    <td><span class="class-label ${classStyles[index]}">${classNames[index]}</span></td>
                    <td>${row.precision.toFixed(2)}</td>
                    <td>${row.recall.toFixed(2)}</td>
                    <td>${row.f1_score.toFixed(2)}</td>
                    <td>${row.support}</td>
                </tr>
            `;
        });
        tbody.innerHTML = html;
    }

    /**
     * Create scatter chart
     */
    function createScatterChart(data) {
        const ctx = document.getElementById('scatterChart');
        if (!ctx || !data) return;

        const colors = ['#f87171', '#34d399', '#818cf8'];
        const classNames = ['Setosa', 'Versicolor', 'Virginica'];
        
        const xIdx = parseInt(document.getElementById('xAxis').value);
        const yIdx = parseInt(document.getElementById('yAxis').value);

        // Group data by class
        const datasets = [0, 1, 2].map(classIdx => {
            const points = data.filter(d => d.target === classIdx);
            return {
                label: classNames[classIdx],
                data: points.map(p => ({
                    x: p.features[xIdx],
                    y: p.features[yIdx]
                })),
                backgroundColor: colors[classIdx],
                borderColor: colors[classIdx],
                pointRadius: 6,
                pointHoverRadius: 8
            };
        });

        if (scatterChart) {
            scatterChart.destroy();
        }

        scatterChart = new Chart(ctx, {
            type: 'scatter',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: (${context.parsed.x.toFixed(1)}, ${context.parsed.y.toFixed(1)})`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: getFeatureName(xIdx),
                            color: '#94a3b8'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#94a3b8'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: getFeatureName(yIdx),
                            color: '#94a3b8'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#94a3b8'
                        }
                    }
                }
            }
        });
        
        // Store data for updates
        window.visualizationData = data;
    }

    /**
     * Update chart when axis changes
     */
    function updateChart() {
        if (window.visualizationData) {
            createScatterChart(window.visualizationData);
        }
    }

    /**
     * Get feature name by index
     */
    function getFeatureName(idx) {
        const names = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)'];
        return names[idx] || '';
    }

    /**
     * Enable prediction after training
     */
    function enablePrediction() {
        if (predictBtn) {
            predictBtn.disabled = false;
        }
        if (predictHint) {
            predictHint.textContent = 'Masukkan nilai fitur dan klik Prediksi';
        }
    }

    /**
     * Prediction form submission
     */
    if (predictionForm) {
        predictionForm.addEventListener('submit', async function(e) {
            e.preventDefault();

            if (!modelTrained) {
                showError('Train model terlebih dahulu');
                return;
            }

            const features = [
                parseFloat(document.getElementById('sepalLength').value),
                parseFloat(document.getElementById('sepalWidth').value),
                parseFloat(document.getElementById('petalLength').value),
                parseFloat(document.getElementById('petalWidth').value)
            ];

            try {
                const response = await fetch('/knn/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ features })
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || 'Terjadi kesalahan saat prediksi');
                }

                displayPrediction(data);
            } catch (error) {
                showError(error.message);
            }
        });
    }

    /**
     * Display prediction result
     */
    function displayPrediction(data) {
        const resultSection = document.getElementById('predictionResult');
        const predictedClass = document.getElementById('predictedClass');
        const confidenceFill = document.getElementById('confidenceFill');
        const confidenceValue = document.getElementById('confidenceValue');
        const neighborsList = document.getElementById('neighborsList');

        if (!resultSection) return;

        resultSection.classList.remove('hidden');
        resultSection.classList.add('success');

        // Class name with color
        const classNames = ['Setosa', 'Versicolor', 'Virginica'];
        const classColors = ['#dc2626', '#059669', '#2563eb'];
        
        predictedClass.innerHTML = `
            <span style="color: ${classColors[data.predicted_class]}">
                ${classNames[data.predicted_class]}
            </span>
        `;

        // Confidence
        const confidence = data.confidence * 100;
        confidenceFill.style.width = confidence + '%';
        confidenceValue.textContent = confidence.toFixed(1) + '%';

        // Neighbors
        if (data.neighbors && neighborsList) {
            let neighborsHTML = '';
            data.neighbors.forEach((n, i) => {
                neighborsHTML += `
                    <div class="neighbor-item">
                        <span>#${i + 1}</span>
                        <span class="class-label ${getClassStyle(n.class)}">${classNames[n.class]}</span>
                        <span class="neighbor-distance">distance: ${n.distance.toFixed(4)}</span>
                    </div>
                `;
            });
            neighborsList.innerHTML = neighborsHTML;
        }

        // Reset animation
        setTimeout(() => {
            resultSection.classList.remove('success');
        }, 500);
    }

    /**
     * Get class style
     */
    function getClassStyle(classIdx) {
        const styles = ['class-setosa', 'class-versicolor', 'class-virginica'];
        return styles[classIdx] || '';
    }

    /**
     * Show error
     */
    function showError(message) {
        if (errorSection && errorMessage) {
            errorMessage.textContent = message;
            errorSection.classList.remove('hidden');
        }
    }

    /**
     * Format percentage
     */
    function formatPercent(value) {
        return (value * 100).toFixed(1) + '%';
    }
});
