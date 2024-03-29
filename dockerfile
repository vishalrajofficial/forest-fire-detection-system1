FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY . /app

EXPOSE 5000

CMD ["python", "app.py"]
