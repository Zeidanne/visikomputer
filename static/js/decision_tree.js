document.addEventListener('DOMContentLoaded', function() {
    loadDataset();
    setupPrediction();
    setupCalculation();
    setupBuildTree();
});

function loadDataset() {
    fetch('/dt/dataset')
        .then(response => response.json())
        .then(data => {
            const tbody = document.getElementById('datasetBody');
            tbody.innerHTML = '';
            
            data.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${row.Outlook}</td>
                    <td>${row.Temperature}</td>
                    <td>${row.Humidity}</td>
                    <td>${row.Windy}</td>
                    <td>
                        <span class="badge ${row.PlayGolf === 'Yes' ? 'badge-success' : 'badge-error'}">
                            ${row.PlayGolf}
                        </span>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        })
        .catch(error => console.error('Error loading dataset:', error));
}

function setupBuildTree() {
    document.getElementById('buildTreeBtn').addEventListener('click', function() {
        const container = document.getElementById('treeContainer');
        const treeVis = document.getElementById('treeVisualization');
        
        // Show loading state
        this.disabled = true;
        this.innerHTML = '<div class="spinner" style="width: 20px; height: 20px; border-width: 2px;"></div> Building...';
        
        fetch('/dt/build')
            .then(response => response.json())
            .then(treeStructure => {
                container.classList.remove('hidden');
                treeVis.innerHTML = ''; // Clear previous
                
                // Render tree recursively
                const treeHTML = renderTreeRecursive(treeStructure);
                treeVis.innerHTML = treeHTML;
                
                // Reset button
                this.disabled = false;
                this.innerHTML = 'Rebuild Tree';
            })
            .catch(error => {
                console.error('Error building tree:', error);
                this.disabled = false;
                this.innerHTML = 'Build Tree';
            });
    });
}

function renderTreeRecursive(node, label = 'Root') {
    // Leaf node (Prediction Result)
    if (typeof node === 'string') {
        const leafClass = node === 'Yes' ? 'leaf-yes' : 'leaf-no';
        return `
            <div class="node-group">
                <div class="connector-label">${label}</div>
                <div class="node node-leaf ${leafClass}">
                    ${node}
                </div>
            </div>
        `;
    }
    
    // Internal node (Attribute)
    const attribute = Object.keys(node)[0];
    const branches = node[attribute];
    
    let branchesHTML = '<div class="tree-level">';
    for (const [value, subtree] of Object.entries(branches)) {
        branchesHTML += renderTreeRecursive(subtree, value);
    }
    branchesHTML += '</div>';
    
    // If it's the root call (label is Root), we don't need connector-label for the root node itself
    // But our recursive structure assumes connector-label is the incoming branch.
    // For the very top input, we can handle it.
    
    const nodeHTML = `
        <div class="node-group">
            ${label !== 'Root' ? `<div class="connector-label">${label}</div>` : ''}
            <div class="node node-attribute mb-3">
                ${attribute}?
            </div>
            ${branchesHTML}
        </div>
    `;
    
    return nodeHTML;
}

function setupPrediction() {
    const form = document.getElementById('dtPredictionForm');
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => data[key] = value);
        
        fetch('/dt/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(result => {
            const resultDiv = document.getElementById('dtPredictionResult');
            const valueDiv = document.getElementById('dtResultValue');
            const pathDiv = document.getElementById('dtPathSteps');
            
            resultDiv.classList.remove('hidden');
            resultDiv.classList.add('success'); // Add animation class
            
            // Show result
            const predClass = result.prediction === 'Yes' ? 'text-success' : 'text-error';
            valueDiv.innerHTML = `<span class="${predClass}" style="font-size: 2rem; font-weight: bold;">${result.prediction}</span>`;
            
            // Show path
            pathDiv.innerHTML = '';
            if (result.path) {
                result.path.forEach((step, index) => {
                    const stepEl = document.createElement('div');
                    stepEl.className = 'path-step';
                    stepEl.innerHTML = `
                        <span class="badge badge-primary">${step.node}</span>
                        <span class="path-arrow">→</span>
                        <strong>${step.value}</strong>
                    `;
                    pathDiv.appendChild(stepEl);
                });
            }
        })
        .catch(error => console.error('Error predicting:', error));
    });
}

function setupCalculation() {
    document.getElementById('startCalculationBtn').addEventListener('click', function() {
        calculateStep({});
    });
}

function calculateStep(filters) {
    const container = document.getElementById('calculationSteps');
    if (Object.keys(filters).length === 0) container.innerHTML = '';
    
    fetch('/dt/calculate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filters: filters })
    })
    .then(response => response.json())
    .then(data => {
        if (!data.gains || Object.keys(data.gains).length === 0) return;
        
        const stepId = 'step-' + Object.keys(filters).length;
        const stepCard = document.createElement('div');
        stepCard.className = 'step-card';
        
        // Find best gain
        let maxGain = -1;
        let bestAttr = '';
        Object.entries(data.gains).forEach(([attr, gain]) => {
            if (gain > maxGain) {
                maxGain = gain;
                bestAttr = attr;
            }
        });
        
        let gainsHTML = '<div class="grid grid-2">';
        Object.entries(data.gains).forEach(([attr, gain]) => {
            const isBest = attr === bestAttr;
            gainsHTML += `
                <div class="p-2 border rounded ${isBest ? 'bg-primary-50 border-primary' : ''}">
                    <div class="font-bold ${isBest ? 'text-primary' : ''}">${attr}</div>
                    <div class="text-sm">Gain: ${gain.toFixed(3)}</div>
                </div>
            `;
        });
        gainsHTML += '</div>';

        stepCard.innerHTML = `
            <div class="step-header">
                <span class="step-title">Node Level ${Object.keys(filters).length + 1}</span>
                <span class="entropy-badge">Entropy: ${data.current_entropy.toFixed(3)}</span>
            </div>
            <div class="step-body">
                <p><strong>Current Context:</strong> ${Object.keys(filters).length === 0 ? 'Root' : JSON.stringify(filters)}</p>
                <div class="mb-3">
                    <h5 class="mb-2">Candidates Information Gain:</h5>
                    ${gainsHTML}
                </div>
                <div class="alert alert-success">
                    <strong>Selected:</strong> ${bestAttr} (Gain: ${maxGain.toFixed(3)})
                </div>
            </div>
        `;
        
        container.appendChild(stepCard);
    });
}
