FROM python:3.7.7

RUN apt-get update -y
RUN apt-get install -y libopenblas-base libopenblas-dev gfortran rustc
RUN mkdir /txt2onto
WORKDIR /txt2onto
COPY . .
RUN pip install -r requirements.txt

CMD ["/bin/sh","-c","bash"]

