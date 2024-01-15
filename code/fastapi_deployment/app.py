# импортируем библиотеки numpy, pandas и cloudpickle
import numpy as np
import pandas as pd
import cloudpickle

# импортируем класс FastAPI, обеспечивающий все функции API
from fastapi import FastAPI
# из библиотеки pydantic импортируем класс BaseModel,
# он будет нужен для проверки корректности входных данных
from pydantic import BaseModel
# импортируем класс Request из FastAPI, чтобы затем передать 
# его в качестве параметра basic_predict
from fastapi import Request

# функция предварительной подготовки
def preprocessing(df):
    # значения переменной age меньше 18 заменяем
    # минимально допустимым значением возраста
    df['age'] = np.where(df['age'] < 18, 18, df['age'])

    # создаем переменную Ratio - отношение количества
    # просрочек 90+ к общему количеству просрочек
    sum_of_delinq = (df['NumberOfTimes90DaysLate'] +
                     df['NumberOfTime30-59DaysPastDueNotWorse'] +
                     df['NumberOfTime60-89DaysPastDueNotWorse'])

    cond = (df['NumberOfTimes90DaysLate'] == 0) | (sum_of_delinq == 0)
    df['Ratio'] = np.where(
        cond, 0, df['NumberOfTimes90DaysLate'] / sum_of_delinq)

    # создаем индикатор нулевых значений переменной
    # NumberOfOpenCreditLinesAndLoans
    df['NumberOfOpenCreditLinesAndLoans_is_0'] = np.where(
        df['NumberOfOpenCreditLinesAndLoans'] == 0, 'T', 'F')

    # создаем индикатор нулевых значений переменной
    # NumberRealEstateLoansOrLines
    df['NumberRealEstateLoansOrLines_is_0'] = np.where(
        df['NumberRealEstateLoansOrLines'] == 0, 'T', 'F')

    # создаем индикатор нулевых значений переменной
    # RevolvingUtilizationOfUnsecuredLines
    df['RevolvingUtilizationOfUnsecuredLines_is_0'] = np.where(
        df['RevolvingUtilizationOfUnsecuredLines'] == 0, 'T', 'F')

    # преобразовываем переменные в категориальные, применив
    # биннинг и перевод в единый строковый формат
    for col in ['NumberOfTime30-59DaysPastDueNotWorse',
                'NumberOfTime60-89DaysPastDueNotWorse',
                'NumberOfTimes90DaysLate']:
        df.loc[df[col] > 3, col] = 4
        df[col] = df[col].apply(lambda x: f"cat_{x}")

    # создаем список списков - список 2-факторных взаимодействий
    lst = [
        ['NumberOfDependents',
         'NumberOfTime30-59DaysPastDueNotWorse'],
        ['NumberOfTime60-89DaysPastDueNotWorse',
         'NumberOfTimes90DaysLate'],
        ['NumberOfTime30-59DaysPastDueNotWorse',
         'NumberOfTime60-89DaysPastDueNotWorse'],
        ['NumberRealEstateLoansOrLines_is_0',
         'NumberOfTimes90DaysLate'],
        ['NumberOfOpenCreditLinesAndLoans_is_0',
         'NumberOfTimes90DaysLate']
    ]

    # создаем взаимодействия
    for i in lst:
        f1 = i[0]
        f2 = i[1]
        df[f1 + ' + ' + f2 + '_interact'] = (df[f1].astype(str) + ' + '
                                             + df[f2].astype(str))
    return df

# загрузим сохраненный ранее конвейер
with open('cloudpickle_pipeline_for_deployment.pkl', 'rb') as file:
    pipe = cloudpickle.load(file)

# создаем экземпляр класса FastAPI, назвав его app
app = FastAPI()

@app.get("/")
async def root():
    return {"Message": "This is a test path for a general FastAPI health check."}

# определение пути для прогнозирования без проверки данных
@app.post('/basic_predict')
async def basic_predict(request: Request):

    # получаем JSON из тела запроса
    input_data = await request.json()

    print(type(input_data))

    # преобразовываем JSON в pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # выполняем предварительную обработку новых данных
    input_df = preprocessing(input_df)

    # вычисляем вероятности для новых данных
    output = pipe.predict_proba(input_df)[:, 1][0]

    # возвращаем вывод пользователю
    return output

"""
# Для проверки этого пути в bash-консоли можно набрать такой POST-запрос:
curl -X 'POST' \
   'http://127.0.0.1:8000/basic_predict' \
   -H 'accept: application/json' \
   -d '{"RevolvingUtilizationOfUnsecuredLines": 0.88551908, "age": 43.0, 
   "NumberOfTime30-59DaysPastDueNotWorse": 0.0, "DebtRatio": 0.177512717, 
   "MonthlyIncome": 5700.0, "NumberOfOpenCreditLinesAndLoans": 4.0, 
   "NumberOfTimes90DaysLate": 0.0, "NumberRealEstateLoansOrLines": 0.0, 
   "NumberOfTime60-89DaysPastDueNotWorse": 0.0, "NumberOfDependents": 0.0}'
   
# output: 0.48133564201075385
"""

class InputData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: int

# определение пути для прогнозирования с проверкой данных
@app.post('/predict')
async def predict(data: InputData):
    # преобразовываем входные данные в датафрейм pandas
    input_df = pd.DataFrame([data.dict()])

    # осуществляем замену символов в названиях столбцов
    # датафрейма, чтобы соответствовать входным 
    # данным, которые ожидает модель
    input_df.columns = input_df.columns.str.replace('_', '-')

    # выполняем предварительную обработку новых данных
    input_df = preprocessing(input_df)

    # вычисляем вероятности для новых данных
    output = pipe.predict_proba(input_df)[:, 1][0]

    # возвращаемое значение
    return output

"""
# Для проверки этого пути в bash-консоли можно набрать такой POST-запрос:

curl -X 'POST' 'http://127.0.0.1:8000/predict' -H 'Content-Type: application/json' \
  -d '{"RevolvingUtilizationOfUnsecuredLines": 0.5, "age": 40,
  "NumberOfTime30_59DaysPastDueNotWorse": 2, "DebtRatio": 0.9,
  "MonthlyIncome": 7000, "NumberOfOpenCreditLinesAndLoans": 2,
  "NumberOfTimes90DaysLate": 2, "NumberRealEstateLoansOrLines": 2,
  "NumberOfTime60_89DaysPastDueNotWorse": 2, "NumberOfDependents": 2 }'

# output: 0.6146162030411374
"""