FROM python:3.8-slim-buster
MAINTAINER HiteshKotte "hiteshkotte@gmail.com"
ENV PYTORCH_NO_CUDA_MEMORY_CACHING 1

# Install all required packages in a single RUN command
RUN apt-get update && \
    apt-get install -y \
    gcc \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-tk \
    libgtk2.0-dev \
    pkg-config \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app

# Install Python requirements
RUN pip install -r requirements.txt

ENV QT_PLUGIN_PATH=/usr/local/lib/python3.8/site-packages/cv2/qt/plugins/platforms

EXPOSE 5000

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]







#FROM python:3.8-slim-buster
#MAINTAINER HiteshKotte "hiteshkotte@gmail.com"
#ENV PYTORCH_NO_CUDA_MEMORY_CACHING 1

#RUN apt-get update && apt-get install -qy gcc libxcb-xinerama0 libxkbcommon-x11-0 libgl1-mesa-glx libglib2.0-0
#RUN apt-get install -y python3-tk
#RUN apt-get update && \
    #apt-get install -y libgtk2.0-dev pkg-config
#RUN apt-get update && \
    #apt-get install -y ffmpeg



#COPY . /app
#WORKDIR /app
#RUN pip install -r requirements.txt
#ENV QT_PLUGIN_PATH=/usr/local/lib/python3.8/site-packages/cv2/qt/plugins/platforms
#EXPOSE 5000
#CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]

