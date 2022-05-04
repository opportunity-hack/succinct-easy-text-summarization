# Ref https://runnable.com/docker/python/dockerize-your-flask-application
FROM ubuntu:16.04
RUN apt-get -y update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip python3-dev
ENV PYTHONIOENCODING=utf-8
RUN pip3 install --upgrade pip

COPY app/ /app
WORKDIR /app
RUN pip3 install -r requirements.txt

COPY run_app.py /
WORKDIR /
ENTRYPOINT ["python3"]
CMD ["run_app.py"]
