FROM python:3.7-stretch@sha256:ba2b519dbdacc440dd66a797d3dfcfda6b107124fa946119d45b93fc8f8a8d77

WORKDIR /app

RUN apt-get clean \
    && apt-get -y update \
    && apt-get install -y wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir pytest

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p exp/Ubuntu/ \
    && cd exp/Ubuntu/ \
    && wget https://github.com/HephaestusProject/pytorch-ReCoSa/releases/download/v0.2.3-nightly/v0.2.3.tar.gz \
    && tar zxvf v0.2.3.tar.gz \
    && rm -rf v0.2.3.tar.gz \
    && cd ../..

COPY . .

ENV LANG C.UTF-8

CMD [ "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]