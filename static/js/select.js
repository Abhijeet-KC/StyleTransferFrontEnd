// Script for Updating Selected Style Image 
    // Map predefined styles to their corresponding image URLs
    const predefinedStyles = {
        "style1": "https://via.placeholder.com/300?text=Impressionist",
        "style2": "https://via.placeholder.com/300?text=Modern+Art",
        "style3": "https://via.placeholder.com/300?text=Abstract",
        "style4": "https://via.placeholder.com/300?text=Watercolor"
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
