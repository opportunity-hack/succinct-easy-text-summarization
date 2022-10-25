# Ref https://runnable.com/docker/python/dockerize-your-flask-application
FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
        software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
    python3.7 \
    python3-pip
RUN python3.7 -m pip install pip
RUN apt-get update && apt-get install -y \
    python3-distutils \
    python3-setuptools
RUN python3.7 -m pip install pip --upgrade pip
RUN apt-get install -y python3.7-dev
RUN apt-get install -y bash
RUN apt-get install -y systemd

RUN pip3 install flask
RUN pip3 install flask-appconfig
RUN pip3 install flask-bootstrap
RUN pip3 install flask-nav
RUN pip3 install flask-debug
RUN pip3 install flask-wtf
RUN pip3 install Flask-Reuploaded
RUN pip3 install email_validator
RUN pip3 install sklearn
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install nltk
RUN pip3 install seaborn
RUN pip3 install spacy
RUN pip3 install gensim
RUN pip3 install pytextrank
RUN python3.7 -m spacy download en_core_web_sm
#RUN python3.7 -m nltk download omw-1.4

RUN pip3 install huey

COPY app/ /app
WORKDIR /app
#RUN pip3 install -r requirements.txt

COPY run.sh /
RUN chmod +x /run.sh

COPY run_app.py /
WORKDIR /
CMD ["/run.sh"]
#CMD ["run_app.py"]
#CMD ["run.sh"]
