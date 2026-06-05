# Use an official light python image 
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3


#Stablish the working directory inside the container
WORKDIR /app

RUN mkdir -p /app/${MODELS_FOLDER_NAME} && mkdir -p /app/${DATA_FOLDER_NAME} && mkdir -p /app/${SAVE_DATA_FOLDER_NAME}

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

#Command to run FastAPI 
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]