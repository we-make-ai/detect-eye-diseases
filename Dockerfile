FROM python:3.6-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

# Install pytorch and fastai
RUN pip install torch torchvision
RUN pip install fastai

# Install starlette and uvicorn
RUN pip install starlette uvicorn python-multipart aiohttp

ADD eye_disease.py eye_disease.py
ADD export.pkl export.pkl
ADD fnames.csv fnames.csv
# ADD usa-inaturalist-cats.pth usa-inaturalist-cats.pth

# Run it once to trigger resnet download
RUN python eye_disease.py

EXPOSE 8008

# Start the server
CMD ["python", "eye_disease.py", "serve"]
