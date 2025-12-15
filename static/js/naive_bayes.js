document.addEventListener('DOMContentLoaded', function() {
    loadDataset();
    setupPrediction();
});

function loadDataset() {
    fetch('/nb/dataset')
        .then(response => response.json())
        .then(data => {
            const tbody = document.getElementById('datasetBody');
            tbody.innerHTML = '';
            
            data.forEach((row, index) => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${index + 1}</td>
                    <td>${row.Penghasilan}</td>
                    <td>${row.Pekerjaan}</td>
                    <td>${row.Promo}</td>
                    <td>
                        <span class="badge ${row.Beli === 'Ya' ? 'badge-success' : 'badge-error'}">
                            ${row.Beli}
                        </span>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        })
        .catch(error => console.error('Error loading dataset:', error));
}

function setupPrediction() {
    const form = document.getElementById('nbPredictionForm');
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => data[key] = value);
        
        fetch('/nb/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(result => {
            displayResults(result);
        })
        .catch(error => console.error('Error predicting:', error));
    });
}

function displayResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.classList.remove('hidden');
    
    // Display Prior
    const priorTable = document.getElementById('priorTable');
    priorTable.innerHTML = '';
    for (const [kelas, details] of Object.entries(result.prior)) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${kelas}</strong></td>
            <td>${details.count}</td>
            <td>${details.total}</td>
            <td class="prob-value">${details.probability}</td>
        `;
        priorTable.appendChild(tr);
    }
    
    // Display Likelihood
    const likelihoodDetails = document.getElementById('likelihoodDetails');
    likelihoodDetails.innerHTML = '';
    
    for (const [kelas, features] of Object.entries(result.likelihood)) {
        const card = document.createElement('div');
        card.className = 'calc-step';
        
        let featuresHTML = '';
        for (const [feature, details] of Object.entries(features)) {
            featuresHTML += `
                <div class="calc-step-formula">
                    P(${feature}=${details.value} | Beli=${kelas}) = ${details.count}/${details.total} = <strong>${details.probability}</strong>
                </div>
            `;
        }
        
        card.innerHTML = `
            <div class="calc-step-title">Beli = ${kelas}</div>
            ${featuresHTML}
        `;
        likelihoodDetails.appendChild(card);
    }
    
    // Display Posterior Bars
    const posteriorYa = result.posterior_normalized['Ya'] || 0;
    const posteriorTidak = result.posterior_normalized['Tidak'] || 0;
    
    document.getElementById('barYes').style.height = (posteriorYa * 100) + '%';
    document.getElementById('barNo').style.height = (posteriorTidak * 100) + '%';
    document.getElementById('valueYes').textContent = (posteriorYa * 100).toFixed(1) + '%';
    document.getElementById('valueNo').textContent = (posteriorTidak * 100).toFixed(1) + '%';
    
    // Display Final Prediction
    const finalPrediction = document.getElementById('finalPrediction');
    finalPrediction.textContent = result.prediction === 'Ya' ? 'BELI' : 'TIDAK BELI';
    finalPrediction.className = 'final-prediction ' + (result.prediction === 'Ya' ? 'yes' : 'no');
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
