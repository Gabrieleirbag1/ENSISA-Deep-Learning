FROM tensorflow/tensorflow:2.8.0
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY . /app
CMD ["python", "main.py"]

