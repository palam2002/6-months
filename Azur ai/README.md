# **Azure AI - OCR & Face Detection** 📸🧑‍💻

This project integrates multiple Azure Cognitive Services to perform Optical Character Recognition (OCR) and Face Detection on images. Additionally, it allows users to upload and manage images in Azure Blob Storage. The application is built using Streamlit for a simple and user-friendly interface, and it utilizes various Azure services such as Computer Vision, Face API, and Blob Storage.

---

## Key Features

- **OCR (Optical Character Recognition)**: Extract text from images using Azure's Computer Vision service.
- **Face Detection**: Detect faces in images using Azure's Face API.
- **Azure Blob Storage Integration**: Upload images to an existing or newly created Azure Blob Storage container. The application also checks for the existence of an image in the selected container before uploading.
- **User-Friendly Interface**: Built with Streamlit to provide a seamless experience for users to interact with the application.
  
---

## Prerequisites

Before running this project, ensure you have the following:

- Python 3.7 or above
- An active Azure subscription with the following services enabled:
  - **Azure Computer Vision**
  - **Azure Face API**
  - **Azure Blob Storage**
- Required API keys and endpoints for Azure services
- Streamlit installed for the front-end interface

---

## Installation

### Step 1: Install Required Libraries

Install the necessary libraries using pip:

```bash
pip install streamlit azure-cognitiveservices-vision-computervision azure-storage-blob azure-cognitiveservices-vision-face Pillow msrest
```

### Step 2: Set Up Azure Credentials
In the script, replace the following placeholders with your actual Azure credentials:

- `AZURE_REGION` - Region for Azure Computer Vision service
- `AZURE_KEY` - Subscription key for Azure Computer Vision
- `AZURE_ENDPOINT` - Endpoint URL for Azure Computer Vision
- `FACE_API_KEY` - Subscription key for Azure Face API
- `FACE_API_ENDPOINT` - Endpoint URL for Azure Face API
- `STORAGE_CONNECTION_STRING` - Connection string for Azure Blob Storage

## Usage
Select or Create a Blob Storage Container:
- The user can select an existing container or create a new one in Azure Blob Storage.

Upload an Image:
- Upload an image in JPG, PNG, or JPEG format.

Perform OCR:
- Click the "Perform OCR" button to extract text from the uploaded image using Azure's OCR capabilities.

Face Detection:
- Click the "Detect Faces" button to identify and extract faces from the uploaded image.

Upload to Azure Blob Storage:
- Click the "Upload to Azure Blob Storage" button to upload the image to the selected Azure Blob Storage container.

## Running the Application
To run the application locally, open a terminal and execute the following command:
```
streamlit run app.py
```
Replace app.py with the name of your Streamlit Python file.

## Screenshot
![screenshot](https://github.com/user-attachments/assets/071de696-1aa5-4336-ad76-2e5e03d2867f)



