# Style Transfer Using Transformer

This project allows you to upload content and style images, and apply style transfer using transformer. The backend is built with Flask, and the frontend is styled using Tailwind CSS for a clean, responsive user interface.

## Features

- Upload a content image and a style image.
- View a live preview of both images before submission.
- Apply style transfer and view the result.
- Backend handles image uploads and processes them for style transfer.

## Technologies Used

- **Backend:** Flask (Python)
- **Frontend:** HTML, Tailwind CSS
- **Image Processing:** Custom style transfer model
- **Other Libraries:** Werkzeug

## How It Works

1. **Upload Content and Style Images:**
   - Navigate to the homepage.
   - Select both a content image and a style image from your local device.
   - The images will be previewed on the page as you upload them.

2. **Apply Style Transfer:**
   - After submitting, the Flask backend processes the images and applies the style transfer.
   - The result will be displayed on the page.

3. **View Result:**
   - The result image is shown immediately after the style transfer is completed.

## Dependencies

Make sure to install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

The `requirements.txt` should include:

```
Flask
Werkzeug
```

## Contributing

Feel free to open an issue or submit a pull request if you find any bugs or want to suggest improvements. Contributions are always welcome!

