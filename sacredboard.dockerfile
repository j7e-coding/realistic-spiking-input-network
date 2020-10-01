FROM python

RUN apt-get update
RUN apt-get -y install build-essential libpython3-dev
RUN pip install sacredboard

CMD ["sacredboard", "-m", "sacred"]

