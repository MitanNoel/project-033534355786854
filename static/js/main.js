// Main JavaScript file for Indonesian Emotion Detection App

// Document ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips if Bootstrap tooltips are used
    initTooltips();

    // File upload validation
    setupFileUpload();

    // Form validations
    setupFormValidations();

    // Auto-dismiss alerts after 5 seconds
    autoDismissAlerts();
});

// Initialize Bootstrap tooltips
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// File upload validation
function setupFileUpload() {
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                // Check file size (max 16MB)
                const maxSize = 16 * 1024 * 1024;
                if (file.size > maxSize) {
                    alert('File terlalu besar! Maksimal ukuran file adalah 16MB.');
                    fileInput.value = '';
                    return;
                }

                // Check file extension
                const fileName = file.name.toLowerCase();
                if (!fileName.endsWith('.csv')) {
                    alert('Format file tidak valid! Silakan upload file CSV.');
                    fileInput.value = '';
                    return;
                }

                // Show file name
                console.log('File selected:', file.name);
            }
        });
    }
}

// Form validations
function setupFormValidations() {
    // Upload form validation
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            const fileInput = document.getElementById('file');
            if (!fileInput.files || fileInput.files.length === 0) {
                e.preventDefault();
                alert('Silakan pilih file CSV untuk diupload!');
                return false;
            }
        });
    }

    // Training form validation
    const trainingForm = document.getElementById('trainingForm');
    if (trainingForm) {
        trainingForm.addEventListener('submit', function(e) {
            const checkboxes = document.querySelectorAll('input[name="scenarios"]:checked');
            if (checkboxes.length === 0) {
                e.preventDefault();
                alert('Silakan pilih minimal satu skenario untuk dilatih!');
                return false;
            }

            // Show loading state
            const submitBtn = trainingForm.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Training sedang berjalan...';
            }
        });
    }

    // Prediction form validation
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            let hasInput = false;
            for (let i = 1; i <= 5; i++) {
                const textInput = document.getElementById(`text_${i}`);
                if (textInput && textInput.value.trim() !== '') {
                    hasInput = true;
                    break;
                }
            }

            if (!hasInput) {
                e.preventDefault();
                alert('Silakan masukkan minimal satu teks untuk diprediksi!');
                return false;
            }
        });
    }
}

// Auto-dismiss alerts
function autoDismissAlerts() {
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    alerts.forEach(function(alert) {
        // Only auto-dismiss success and info alerts
        if (alert.classList.contains('alert-success') || alert.classList.contains('alert-info')) {
            setTimeout(function() {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }, 5000);
        }
    });
}

// Utility function to format numbers
function formatNumber(num, decimals = 4) {
    return parseFloat(num).toFixed(decimals);
}

// Utility function to format percentage
function formatPercentage(num, decimals = 2) {
    return (parseFloat(num) * 100).toFixed(decimals) + '%';
}

// Show/hide loading spinner
function showLoading(element, message = 'Loading...') {
    if (element) {
        element.innerHTML = `
            <div class="text-center py-5">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-3">${message}</p>
            </div>
        `;
    }
}

// Hide loading spinner
function hideLoading(element, content = '') {
    if (element) {
        element.innerHTML = content;
    }
}

// Smooth scroll to element
function smoothScrollTo(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

// Copy text to clipboard
function copyToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text)
            .then(() => {
                showToast('Teks berhasil disalin!', 'success');
            })
            .catch(err => {
                console.error('Failed to copy:', err);
                fallbackCopyToClipboard(text);
            });
    } else {
        fallbackCopyToClipboard(text);
    }
}

// Fallback copy to clipboard for older browsers
function fallbackCopyToClipboard(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.top = '0';
    textArea.style.left = '0';
    textArea.style.opacity = '0';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();

    try {
        document.execCommand('copy');
        showToast('Teks berhasil disalin!', 'success');
    } catch (err) {
        console.error('Fallback copy failed:', err);
        showToast('Gagal menyalin teks', 'danger');
    }

    document.body.removeChild(textArea);
}

// Show toast notification
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        // Create toast container if it doesn't exist
        const container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(container);
    }

    const toastId = 'toast-' + Date.now();
    const toastHTML = `
        <div id="${toastId}" class="toast align-items-center text-white bg-${type} border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;

    document.getElementById('toastContainer').insertAdjacentHTML('beforeend', toastHTML);

    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, {
        autohide: true,
        delay: 3000
    });
    toast.show();

    // Remove toast after it's hidden
    toastElement.addEventListener('hidden.bs.toast', function() {
        toastElement.remove();
    });
}

// Export functions for use in other scripts
window.appUtils = {
    formatNumber,
    formatPercentage,
    showLoading,
    hideLoading,
    smoothScrollTo,
    copyToClipboard,
    showToast
};
