FROM python:3.9

# назначим рабочую папку
WORKDIR /app

# скопируем необходимые для работы файлы
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY cloudpickle_pipeline_for_deployment.pkl cloudpickle_pipeline_for_deployment.pkl

# установим нужные пакеты python
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# запустим uvicorn с нужными нам параметрами
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]



