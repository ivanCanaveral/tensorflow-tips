FROM ubuntu

WORKDIR /home

RUN apt-get update && apt-get install -y python3-pip curl

RUN pip3 install jupyter tensorflow tensorflow-datasets requests

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
