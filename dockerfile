#Deriving the latest base image
FROM python:latest


#Labels as key value pair
LABEL Maintainer="omer benhaim"


# Any working directory can be chosen as per choice like '/' or '/home' etc
# i have chosen /usr/app/src
WORKDIR /usr/app/src

#to COPY the remote file at working directory in container
COPY ./ ./
# Now the structure looks like this '/usr/app/src/test.py'
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt

#CMD instruction should be used to run the software
#contained by your image, along with any arguments.

CMD [ "python", "main.py"]