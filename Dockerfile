## Base image
FROM python:3.8-slim-buster

## Upgrade pip (optional)
RUN pip install --upgrade pip 

## Copiying all the files to source directory
COPY . /src

## Set working directory to previously added source directory
WORKDIR /src

## Giving permissions to python file
RUN chmod +x loan_prediction.py

## Install required python dependencies from requirements file
RUN pip install -r requirements.txt

## Run loan_prediction.py file
ENTRYPOINT ["python"]

CMD ["loan_prediction.py"]
