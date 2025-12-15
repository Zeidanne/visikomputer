/**
 * GLCM Page JavaScript
 * Handles image upload and GLCM feature extraction
 */

document.addEventListener('DOMContentLoaded', function() {
    const glcmForm = document.getElementById('glcmForm');
    const imageInput = document.getElementById('imageInput');
    const uploadArea = document.getElementById('uploadArea');
    const previewContainer = document.getElementById('preview-container');
    const preview = document.getElementById('preview');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');
    const errorMessage = document.getElementById('errorMessage');

    // Drag and drop handling
    if (uploadArea) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.remove('dragover');
            }, false);
        });

        uploadArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length) {
                imageInput.files = files;
                handleImagePreview(files[0]);
            }
        }, false);
    }

    // Image preview handler
    if (imageInput) {
        imageInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                handleImagePreview(file);
            }
        });
    }

    function handleImagePreview(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            previewContainer.classList.remove('hidden');
            
            // Update collapsible height after image loads
            preview.onload = function() {
                setTimeout(() => updateCollapsibleHeight(previewContainer), 100);
            };
        };
        reader.readAsDataURL(file);
    }

    // Form submission
    if (glcmForm) {
        glcmForm.addEventListener('submit', async function(e) {
            e.preventDefault();

            // Hide previous results/errors
            resultsSection.classList.add('hidden');
            errorSection.classList.add('hidden');
            loadingIndicator.classList.remove('hidden');

            const formData = new FormData(glcmForm);

            try {
                const response = await fetch('/extract', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                loadingIndicator.classList.add('hidden');

                if (!response.ok) {
                    throw new Error(data.error || 'Terjadi kesalahan');
                }

                displayResults(data);
            } catch (error) {
                loadingIndicator.classList.add('hidden');
                errorMessage.textContent = error.message;
                errorSection.classList.remove('hidden');
            }
        });
    }

    function displayResults(data) {
        resultsSection.classList.remove('hidden');

        // Display parameters
        const parametersDisplay = document.getElementById('parametersDisplay');
        parametersDisplay.innerHTML = `
            <div class="param-item">
                <div>
                    <div class="param-label">Distances</div>
                    <div class="param-value">${data.parameters.distances.join(', ')}</div>
                </div>
            </div>
            <div class="param-item">
                <div>
                    <div class="param-label">Angles</div>
                    <div class="param-value">${data.parameters.angles.join('°, ')}°</div>
                </div>
            </div>
            <div class="param-item">
                <div>
                    <div class="param-label">Gray Levels</div>
                    <div class="param-value">${data.parameters.levels}</div>
                </div>
            </div>
            <div class="param-item">
                <div>
                    <div class="param-label">Image Size</div>
                    <div class="param-value">${data.image_shape[0]} × ${data.image_shape[1]} px</div>
                </div>
            </div>
        `;

        // Display features
        const featuresResults = document.getElementById('featuresResults');
        const featureNames = {
            'contrast_avg': { name: 'Contrast', desc: 'Perbedaan intensitas' },
            'dissimilarity_avg': { name: 'Dissimilarity', desc: 'Ketidaksamaan piksel' },
            'homogeneity_avg': { name: 'Homogeneity', desc: 'Keseragaman tekstur' },
            'energy_avg': { name: 'Energy', desc: 'Keseragaman distribusi' },
            'correlation_avg': { name: 'Correlation', desc: 'Korelasi linear' }
        };

        let featuresHTML = '';
        for (const [key, info] of Object.entries(featureNames)) {
            const value = data.features_avg[key];
            featuresHTML += `
                <div class="feature-result-card">
                    <div class="feature-name">
                        ${info.name}
                        <span class="text-muted" style="font-size: 0.8em; margin-left: 0.5rem;">${info.desc}</span>
                    </div>
                    <div class="feature-value">${formatNumber(value, 6)}</div>
                </div>
            `;
        }
        featuresResults.innerHTML = featuresHTML;

        // Update collapsible height
        setTimeout(() => updateCollapsibleHeight(resultsSection), 100);
    }
});
