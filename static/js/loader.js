function showLoader() {
    document.getElementById("loading-screen").style.display = "block";  // Show loader
    document.getElementById("submit-btn").disabled = true;  // Disable button
}

// Hide loader when result image is loaded
window.onload = function () {
    if (document.getElementById("result-image")) {
        document.getElementById("loading-screen").style.display = "none";
        document.getElementById("submit-btn").disabled = false;
    }
};