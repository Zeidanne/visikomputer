/**
 * Common JavaScript Utilities
 * Shared functionality across all pages
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all collapsibles
    initCollapsibles();
});

/**
 * Initialize collapsible sections
 */
function initCollapsibles() {
    const collapsibles = document.querySelectorAll('.collapsible-header');
    
    collapsibles.forEach(header => {
        header.addEventListener('click', function() {
            const parent = this.parentElement;
            const content = this.nextElementSibling;
            
            // Toggle active class
            parent.classList.toggle('active');
            
            // Toggle content height
            if (parent.classList.contains('active')) {
                // Set a large max-height to allow content to expand fully
                content.style.maxHeight = content.scrollHeight + 500 + 'px';
            } else {
                content.style.maxHeight = '0px';
            }
        });
    });
    
    // Initialize active collapsibles
    document.querySelectorAll('.collapsible.active').forEach(collapsible => {
        const content = collapsible.querySelector('.collapsible-content');
        if (content) {
            content.style.maxHeight = content.scrollHeight + 500 + 'px';
        }
    });
}

/**
 * Update collapsible height (useful after dynamic content changes)
 */
function updateCollapsibleHeight(element) {
    const activeCollapsible = element ? 
        element.closest('.collapsible.active') : 
        document.querySelector('.collapsible.active');
    
    if (activeCollapsible) {
        const content = activeCollapsible.querySelector('.collapsible-content');
        if (content) {
            content.style.maxHeight = content.scrollHeight + 'px';
        }
    }
}

/**
 * Show loading state
 */
function showLoading(loadingElement) {
    if (loadingElement) {
        loadingElement.classList.remove('hidden');
    }
}

/**
 * Hide loading state
 */
function hideLoading(loadingElement) {
    if (loadingElement) {
        loadingElement.classList.add('hidden');
    }
}

/**
 * Show error message
 */
function showError(errorSection, errorMessage, message) {
    if (errorSection && errorMessage) {
        errorMessage.textContent = message;
        errorSection.classList.remove('hidden');
    }
}

/**
 * Hide error message
 */
function hideError(errorSection) {
    if (errorSection) {
        errorSection.classList.add('hidden');
    }
}

/**
 * Format number with precision
 */
function formatNumber(num, precision = 4) {
    return Number(num).toFixed(precision);
}

/**
 * Format percentage
 */
function formatPercent(num, precision = 1) {
    return (num * 100).toFixed(precision) + '%';
}

/**
 * Debounce function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Generate random ID
 */
function generateId() {
    return Math.random().toString(36).substring(2, 9);
}
