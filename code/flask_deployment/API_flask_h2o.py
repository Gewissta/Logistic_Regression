# импортируем библиотеки и классы
import pandas as pd
import numpy as np
from flask import Flask
from flask_restful import Resource, Api, reqparse
import h2o

# создаем наше приложение REST API
app = Flask(__name__)
api = Api(app)

# инициализируем H2O
h2o.init()

# загрузим нашу обученную модель
uploaded_model = h2o.load_model('logreg')

# создадим объект парсера Flask
parser = reqparse.RequestParser(bundle_errors=True)
# парсим аргументы
parser.add_argument('RevolvingUtilizationOfUnsecuredLines')
parser.add_argument('age')
parser.add_argument('NumberOfTime30_59DaysPastDueNotWorse')
parser.add_argument('DebtRatio')
parser.add_argument('MonthlyIncome')
parser.add_argument('NumberOfOpenCreditLinesAndLoans')
parser.add_argument('NumberOfTimes90DaysLate')
parser.add_argument('NumberRealEstateLoansOrLines')
parser.add_argument('NumberOfTime60_89DaysPastDueNotWorse')
parser.add_argument('NumberOfDependents')

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

# создадим словарь col_dict с описанием типов столбцов в датафрейме h2o
# категориальные столбцы - enum, числовые столбцы - real и int
col_dict = {
    'RevolvingUtilizationOfUnsecuredLines': 'real',
    'age': 'int',
    'NumberOfTime30-59DaysPastDueNotWorse': 'enum',
    'DebtRatio': 'real',
    'MonthlyIncome': 'int',
    'NumberOfOpenCreditLinesAndLoans': 'int',
    'NumberOfTimes90DaysLate': 'enum',
    'NumberRealEstateLoansOrLines': 'int',
    'NumberOfTime60-89DaysPastDueNotWorse': 'enum',
    'NumberOfDependents': 'int',
    'Ratio': 'real',
    'NumberOfOpenCreditLinesAndLoans_is_0': 'enum',
    'NumberRealEstateLoansOrLines_is_0': 'enum',
    'RevolvingUtilizationOfUnsecuredLines_is_0': 'enum',
    'NumberOfDependents + NumberOfTime30-59DaysPastDueNotWorse_interact': 'enum',
    'NumberOfTime60-89DaysPastDueNotWorse + NumberOfTimes90DaysLate_interact': 'enum',
    'NumberOfTime30-59DaysPastDueNotWorse + NumberOfTime60-89DaysPastDueNotWorse_interact': 'enum',
    'NumberRealEstateLoansOrLines_is_0 + NumberOfTimes90DaysLate_interact': 'enum',
    'NumberOfOpenCreditLinesAndLoans_is_0 + NumberOfTimes90DaysLate_interact': 'enum'
}


# создадим пустой словарь output для выходных данных
output = {}

# создадим класс ресурса Flask
class LoanArrearsForecast(Resource):
    # создадим метод get, он будет обрабатывать GET-запрос и
    # возвращать прогнозный результат, выданный нашей моделью
    def get(self):

        # получаем входные аргументы с помощью объекта парсера
        args = parser.parse_args()
        # извлекаем значения соответствующих переменных
        RevolvingUtilizationOfUnsecuredLines = float(
            args['RevolvingUtilizationOfUnsecuredLines'])
        age = int(args['age'])
        NumberOfTime30_59DaysPastDueNotWorse = int(
            args['NumberOfTime30_59DaysPastDueNotWorse'])
        DebtRatio = float(args['DebtRatio'])
        MonthlyIncome = float(args['MonthlyIncome'])
        NumberOfOpenCreditLinesAndLoans = int(
            args['NumberOfOpenCreditLinesAndLoans'])
        NumberOfTimes90DaysLate = int(
            args['NumberOfTimes90DaysLate'])
        NumberRealEstateLoansOrLines = int(
            args['NumberRealEstateLoansOrLines'])
        NumberOfTime60_89DaysPastDueNotWorse = int(
            args['NumberOfTime60_89DaysPastDueNotWorse'])
        NumberOfDependents = float(args['NumberOfDependents'])

        # создаем словарь из полученных значений
        data = {
            'RevolvingUtilizationOfUnsecuredLines': RevolvingUtilizationOfUnsecuredLines,
            'age': age,
            'NumberOfTime30-59DaysPastDueNotWorse': NumberOfTime30_59DaysPastDueNotWorse,
            'DebtRatio': DebtRatio,
            'MonthlyIncome': MonthlyIncome,
            'NumberOfOpenCreditLinesAndLoans': NumberOfOpenCreditLinesAndLoans,
            'NumberOfTimes90DaysLate': NumberOfTimes90DaysLate,
            'NumberRealEstateLoansOrLines': NumberRealEstateLoansOrLines,
            'NumberOfTime60-89DaysPastDueNotWorse': NumberOfTime60_89DaysPastDueNotWorse,
            'NumberOfDependents': NumberOfDependents
        }

        # и преобразуем его в датафрейм pandas
        input_df = pd.DataFrame([data])

        # выполняем предварительную обработку новых данных
        input_df = preprocessing(input_df)

        # преобразуем данные в датафрейм h2o, используя 
        # словарь c описанием типов столбцов
        input_h2o_df = h2o.H2OFrame(input_df, 
                                    column_types=col_dict)

        # подаем данные на вход нашей модели, получаем 
        # прогноз и преобразуем его в датафрейм pandas
        predictions = uploaded_model.predict(
            input_h2o_df).as_data_frame()

        # извлекаем прогнозные значения из получившегося датафрейма
        # и заполняем соответствующие поля словаря выходных данных
        output['predict'] = str(predictions['predict'].values[0])
        output['p0'] = str(predictions['p0'].values[0])
        output['p1'] = str(predictions['p1'].values[0])

        # возвращаем этот словарь
        return output

# зададим путь для нашего GET-запроса
# мы говорим API соотнести наш класс с путем в адресной строке - '/'
api.add_resource(LoanArrearsForecast, '/')

# если этот файл запущен не как модуль
if __name__ == '__main__':
    # то запуск приложения Flask с включенным режимом отладки
    # на порту 8080 и слушая все адреса
    # (при окончательном запуске в промышленной 
    # среде режим отладки нужно отключить)
    app.run(debug=True, port=8080, host='0.0.0.0')


"""
# Для проверки работоспособности нашего REST API мы можем в bash консоли набрать следующий запрос:

curl -X 'GET' 'http://127.0.0.1:8080/' -H 'Content-Type: application/json' \
  -d '{"RevolvingUtilizationOfUnsecuredLines":0.88551908,"age":43, 
          "NumberOfTime30_59DaysPastDueNotWorse":0,"DebtRatio":0.177512717, 
          "MonthlyIncome":5700.0,"NumberOfOpenCreditLinesAndLoans":4, 
          "NumberOfTimes90DaysLate":0, "NumberRealEstateLoansOrLines":0, 
          "NumberOfTime60_89DaysPastDueNotWorse":0, "NumberOfDependents":0.0}'
  
# Ответ от REST API будет: 
{
    "predict": "0",
    "p0": "0.9679559934931616",
    "p1": "0.0320440065068383"
}
"""