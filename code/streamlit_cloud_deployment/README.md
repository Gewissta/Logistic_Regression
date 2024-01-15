# Развертывание сложного конвейера scikit-learn с помощью Streamlit

## Запускаем локально:

#### устанавливаем зависимости
``` pip3 install -r requirements.txt ```   

#### запускаем  
``` streamlit run app.py --server.port=8501 --server.address=0.0.0.0 ```    
   
------------------------------------------------------------------  

## Запускаем в Docker:

#### создаем образ Docker

``` docker build -t streamlit . ```  
  
#### запускаем поименованный контейнер в daemon-режиме, используя порт 8501

``` docker run -rm -p 8501:8501 --name streamlit -d streamlit ``` 

