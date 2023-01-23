FROM osrf/ros:noetic-desktop

RUN apt-get update
RUN apt-get install -y git && apt-get install -y python3-pip && apt-get install -y vim
RUN apt-get install -y python3.8-tk


RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

WORKDIR /catkin_ws

COPY requirements.txt requirements.txt

RUN pip install numpy --upgrade && pip install -r requirements.txt 
