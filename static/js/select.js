// Function to show preview of uploaded images
function showPreview(event, previewId) {
    const input = event.target;
    const preview = document.getElementById(previewId);

    if (input.files && input.files[0]) {
        const reader = new FileReader();

        reader.onload = function (e) {
            preview.src = e.target.result;
            preview.style.display = 'block';
        };

        reader.readAsDataURL(input.files[0]);
    }
}

// Script for Updating Selected Style Image 
    // Map predefined styles to their corresponding image URLs
    const predefinedStyles = {
        "style1": ".././predefined/abstract.jfif",
        "style2": ".././predefined/abstract.jfif",
        "style3": ".././predefined/abstract.jfif",
        "style4": ".././predefined/abstract.jfif",
    };

    // Function to update the selected style image preview
    function updateSelectedStyleImage() {
        const selectElement = document.getElementById("predefined-style-select");
        const selectedStyle = selectElement.value;
        const selectedStyleImageDiv = document.getElementById("selected-style-image");
        const selectedStylePreview = document.getElementById("selected-style-preview");

        if (selectedStyle && predefinedStyles[selectedStyle]) {
            // Show the selected style image
            selectedStylePreview.src = predefinedStyles[selectedStyle];
            selectedStyleImageDiv.classList.remove("hidden");
        } else {
            // Hide the selected style image if no valid option is selected
            selectedStyleImageDiv.classList.add("hidden");
        }
    }

    // Ensure selection persists on page reload
document.addEventListener("DOMContentLoaded", updateSelectedStyleImage);