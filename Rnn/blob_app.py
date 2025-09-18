# 1. Set Up Azure Blob Client

from azure.storage.blob import BlobServiceClient

def get_blob_service_client(connection_string):
    return BlobServiceClient.from_connection_string(connection_string)

#2. Create Blob Container

def create_blob_container(blob_service_client, container_name):
    try:
        container_client = blob_service_client.create_container(container_name)
        return container_client
    except Exception as e:
        return f"Error: {e}"


#3. Upload Images

def upload_images_to_blob(blob_service_client, container_name, file_list):
    try:
        container_client = blob_service_client.get_container_client(container_name)
        for file in file_list:
            blob_client = container_client.get_blob_client(file.name)
            blob_client.upload_blob(file, overwrite=True)
        return f"Uploaded {len(file_list)} images successfully."
    except Exception as e:
        return f"Error: {e}"


#Step 4: Extract Information from Images

import easyocr

def extract_text_from_images(blob_service_client, container_name):
    reader = easyocr.Reader(['en'])
    container_client = blob_service_client.get_container_client(container_name)
    extracted_data = {}

    for blob in container_client.list_blobs():
        blob_client = container_client.get_blob_client(blob.name)
        image_data = blob_client.download_blob().readall()

        # Perform OCR
        results = reader.readtext(image_data)
        extracted_text = " ".join([res[1] for res in results])
        extracted_data[blob.name] = extracted_text

    return extracted_data

#Step 5: Build the Streamlit App

import streamlit as st
from PIL import Image
def main():
    st.title("Azure Blob Storage Manager")
    st.sidebar.title("Navigation")
    options = ["Create Container", "Upload Images", "Extract Image Information", "Show Container", "Delete Container"]
    choice = st.sidebar.selectbox("Choose Action", options)
    
    DefaultEndpointsProtocol='DefaultEndpointsProtocol=https;' 
    EndpointSuffix='EndpointSuffix=core.windows.net'
    AccountName = st.text_input("Account Name")
    if AccountName.strip() == "":
        AccountName='AccountName=blobstorageaccountml;'
    AccountKey = st.text_input("Account Key") 
    if AccountKey.strip() == "":
        AccountKey='AccountKey=AP4/n72bNnLwraJbZdCqaEZ5WsXIcvvF+n1a0oXSaILOP9sNdyF8XVJtKmXYi+IpCm/VsojveV2Q+AStV12+fQ==;'
    connection_string = DefaultEndpointsProtocol+AccountName+AccountKey+EndpointSuffix  
    
    if choice == "Create Container": 
        container_name = st.text_input("Enter Container Name")
        if st.button("Create"):
            blob_service_client = get_blob_service_client(connection_string)
            result = create_blob_container(blob_service_client, container_name)
            st.success(result)

    elif choice == "Upload Images":
        container_name = st.text_input("Enter Container Name")
        uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])
        if st.button("Upload"):
            blob_service_client = get_blob_service_client(connection_string)
            result = upload_images_to_blob(blob_service_client, container_name, uploaded_files)
            st.success(result)

    elif choice == "Extract Image Information":
        container_name = st.text_input("Enter Container Name")
        if st.button("Extract"):
            blob_service_client = get_blob_service_client(connection_string)
            data = extract_text_from_images(blob_service_client, container_name)
            st.write(data)
    
    elif choice == "Show Container": 
        if st.button("Show"):
            blob_service_client = get_blob_service_client(connection_string)
            containers = blob_service_client.list_containers()
            for container in containers:
                st.success(container.name)   
                                
    elif choice == "Delete Container":  
        container_name = st.text_input("Enter Container Name")
        if st.button("Delete"):
            blob_service_client = get_blob_service_client(connection_string)
            result = blob_service_client.delete_container(container_name)
            st.success("Deleted container.")             

if __name__ == "__main__":
    main()
