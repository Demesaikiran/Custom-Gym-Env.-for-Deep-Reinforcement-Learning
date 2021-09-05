FROM python:3.8.0-buster

# Make a directory for our application
WORKDIR /1D_MassControlDRL

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all the data
COPY . .

# Run the application
CMD ["python", "dummy_run.py", "--render", "False"]