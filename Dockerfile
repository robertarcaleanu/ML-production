# 
FROM python:3.10

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./app /code/app

# 
# CMD ["fastapi", "run", "app/run_deployment.py", "--port", "80"]

# Expose the port that the FastAPI application will run on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "app/run_deployment:app", "--host", "0.0.0.0", "--port", "8000"]