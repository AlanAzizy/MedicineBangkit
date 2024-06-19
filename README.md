# VoiceHealth Guide

this project is provided by bangkit capstone team C241-PS524
1. Dzakirozaan Uzlahwasata - Cloud Computing - Institut Teknologi Sepuluh November
2. Wisnuyasha Faizal - Cloud Computing - Institut Teknologi Sepuluh November
3. Fadhlan Nazhif Azizy - Machine Learning - Institut Teknologi Bandung
4. Imam Rusydi Ibrahim - Machine Learning - Institut Teknologi Bandung
5. Agung Muhammad Sholeh - Machine Learning - Universitas Brawijaya
6. Aloysius Bataona Manullang - Mobile Development - Institut Teknologi Sepuluh November

## How to use

### Using Trained Model
- Open Your Docker desktop
- Go to [DockerHub Image](https://hub.docker.com/r/alansti/medicine)
- Copy pull command or `docker pull alansti/medicine`
- Open a terminal inside directory you want to save project
- Run command
- Go Inside Directory MedicineBangkit
- Run Command `docker run -d -p 8000:8000 alansti/medicine:latest`
- Yoi can change the second `8000` with your available port
- Open browser and insert `0.0.0.0:8000/docs` in your browser bar or `127.0.0.1:8000/docs`
- Open the predict tab
- Modify keluhan inside request body with your keluhan
- It will show list of drug recommended to you

### Using Untrained Model
- Clone this repository
- Go Inside directory MedicineBangkit
- Activate virtual environment
- install all dependency with `pip install -r requirements.txt`
- Prepores data with running `python 'App/preprocessing.py'`
- Train model with running `python 'App/modelling & training cnn.py'`
- Wait until the training process is finished
- Run Model with `uvicorn 'App.main:app' --host 0.0.0.0 --port 8000'
- Open browser and insert `0.0.0.0:8000/docs` in your browser bar or `127.0.0.1:8000/docs`
- Open the predict tab
- Modify keluhan inside request body with your keluhan
- It will show list of drug recommended to you
