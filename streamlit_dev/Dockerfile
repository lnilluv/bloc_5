FROM continuumio/miniconda3

WORKDIR /app

COPY . /app/

RUN pip install -r requirements.txt

CMD streamlit run --server.port $PORT app.py