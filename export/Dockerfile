FROM python:3.11-bullseye

RUN pip3 install onnxruntime==1.20.0 pandas==2.2.3 tokenizers==0.20.3
COPY model /model
COPY predict.py /predict.py

CMD [ "python", "/predict.py" ]