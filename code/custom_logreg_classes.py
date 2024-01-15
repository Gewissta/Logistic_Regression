import numpy as np
import scipy.optimize
from scipy.optimize.linesearch import (line_search_wolfe1, 
                                       line_search_wolfe2, 
                                       LineSearchWarning)
import warnings
warnings.simplefilter('ignore', LineSearchWarning)

# пишем собственный класс, строящий логистическую регрессию
# с помощью метода градиентного спуска с 
# ранней остановкой по допуску сходимости
class LogisticRegression_GD:
    """
    Класс, строящий логистическую регрессию
    с помощью метода градиентного спуска с 
    ранней остановкой по допуску сходимости.
    
    Параметры
    ----------
    init_method: string, по умолчанию 'zero'
        Метод инициализации весов.
        Можно выбрать 'random' или 'zero'.
    lr: float, по умолчанию 0.01
        Темп обучения.
    tol: float, по умолчанию 1e-5
        Допуск сходимости.
    num_iter: int, по умолчанию 10000
        Количество итераций градиентного спуска.
    fit_intercept: bool, по умолчанию True
        Добавление константы.
    verbose: bool, по умолчанию True
        Печать результатов оптимизации.
        
    Атрибуты
    ----------
    loss_by_iter_: ndarray of shape (num_iter, )
        Список значений функции потерь по итерациям.       
    """
    def __init__(self, init_method='zero', lr=0.01, 
                 tol=1e-5, num_iter=10000,        
                 fit_intercept=True, verbose=True):

        if init_method not in ['random', 'zero']:     
            raise ValueError(
                "init_method must be one of {'random', 'zero'}, "
                "got '%s' instead" % init_method
            )
        
        # метод инициализации весов
        self.init_method = init_method    
        # темп обучения
        self.lr = lr
        # допуск сходимости
        self.tol = tol
        # количество итераций градиентного спуска
        self.num_iter = num_iter
        # добавление константы
        self.fit_intercept = fit_intercept
        # печать результатов оптимизации
        self.verbose = verbose
        # список, в котором будем хранить значения
        # функции потерь
        self.loss_by_iter_ = []  

    # частный метод __add_intercept добавляет константу
    def __add_intercept(self, X):
        # создаем массив из единиц, количество единиц
        # равно количеству наблюдений
        intercept = np.ones((X.shape[0], 1))
        # конкатенируем массив из единиц с массивом 
        # признаков по оси столбцов
        return np.concatenate((intercept, X), axis=1)

    # частный метод __sigmoid вычисляет значение сигмоиды
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z)) 
    
    # частный метод __loss вычисляет значение 
    # логистической функции потерь
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    # метод .fit() выполняет обучение 
    def fit(self, X, y):
        # если задан fit_intercept=True, добавляем константу
        if self.fit_intercept:
            X = self.__add_intercept(X)
            
        # инициализируем значение функции потерь на предыдущей
        # итерации бесконечно большим значением
        l_prev = np.inf

        # инициализируем веса
        self.theta = np.zeros(X.shape[1])

        # выполняем градиентный спуск
        for i in range(self.num_iter):
            # вычисляем скалярное произведение 
            # матрицы предикторов и вектора весов
            z = np.dot(X, self.theta)
            # к полученному результату применяем сигмоиду, по сути
            # получаем вероятности положительного класса
            h = self.__sigmoid(z)
            
            # вычисляем значение функции потерь
            loss = self.__loss(h, y)
            # добавляем значение функции потерь в список
            self.loss_by_iter_.append(loss)
            
            # если разница между предыдущим значением и
            # текущим значением функции потерь меньше 
            # заданного порога (tol), то прерываем цикл, 
            # т.е. реализована ранняя остановка, которая 
            # ограничивает количество итераций (num_iter)
            if l_prev - loss < self.tol:
                return
            # присваиваем функции потерь на предыдущей
            # итерации текущее значение
            l_prev = loss
            
            # получаем вектор градиента, для этого делим 
            # скалярное произведение транспонированной 
            # матрицы предикторов и вектора разностей между 
            # вероятностями и фактическими значениями зависимой 
            # переменной на количество наблюдений
            gradient = np.dot(X.T, (h - y)) / y.size
            # обновляем веса, вычитаем из текущего приближения 
            # вектора весов вектор градиента, умноженный 
            # на некоторый темп обучения
            self.theta -= self.lr * gradient

            # если задано verbose=True, печатаем номер
            # итерации, значение функции потерь и веса                   
            if self.verbose:
                print(f"Итерация: {i}\n", 
                      f"Функция потерь: {round(loss, 3)}\n", 
                      f"Коэфф-ты: {np.round(self.theta, 3)}")

    # метод .loss_visualize() отрисовывает 
    # кривую функции потерь
    def loss_visualize(self):  
        plt.plot(range(len(self.loss_by_iter_)), self.loss_by_iter_)
        plt.xticks(np.arange(0, self.num_iter, step=self.num_iter / 5))
        plt.xlabel("Количество итераций")
        plt.ylabel("Логистическая функция потерь")
        plt.show()

    # метод .predict_proba() вычисляет вероятности
    def predict_proba(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))
    
    # метод .predict() вычисляет прогнозы
    def predict(self, X, threshold):
        return (self.predict_proba(X) >= threshold).astype(int)
    

# пишем собственный класс, строящий логистическую регрессию
# с помощью метода наивного координатного спуска с 
# ранней остановкой по допуску сходимости
class LogisticRegression_CD:
    """
    Класс, строящий логистическую регрессию с помощью
    метода наивного координатного спуска с 
    ранней остановкой по допуску сходимости.
    
    Параметры
    ----------
    init_method: string, по умолчанию 'zero'
        Метод инициализации весов.
        Можно выбрать 'random' или 'zero'.
    lr: float, по умолчанию 0.01
        Темп обучения.
    tol: float, по умолчанию 1e-5
        Допуск сходимости.
    num_iter: int, по умолчанию 10000
        Количество итераций градиентного спуска.
    fit_intercept: bool, по умолчанию True
        Добавление константы.
    verbose: bool, по умолчанию True
        Печать результатов оптимизации.   
        
    Атрибуты
    ----------
    loss_by_iter_: ndarray of shape (num_iter, )
        Список значений функции потерь по итерациям.
    """
    def __init__(self, init_method='zero', lr=0.01, 
                 tol=1e-5, num_iter=10000,
                 fit_intercept=True, verbose=True):

        if init_method not in ['random', 'zero']:     
            raise ValueError(
                "init_method must be one of {'random', 'zero'}, "
                "got '%s' instead" % init_method
            )

        # метод инициализации весов
        self.init_method = init_method    
        # темп обучения
        self.lr = lr
        # допуск сходимости
        self.tol = tol
        # количество итераций градиентного спуска
        self.num_iter = num_iter
        # добавление константы
        self.fit_intercept = fit_intercept
        # печать результатов градиентного спуска
        self.verbose = verbose
        # список, в котором будем хранить значения
        # функции потерь
        self.loss_by_iter_ = []  

    # частный метод __add_intercept добавляет константу
    def __add_intercept(self, X):
        # создаем массив из единиц, количество единиц
        # равно количеству наблюдений
        intercept = np.ones((X.shape[0], 1))
        # конкатенируем массив из единиц с массивом 
        # предикторов по оси столбцов
        return np.concatenate((intercept, X), axis=1)

    # частный метод __sigmoid вычисляет значение сигмоиды
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z)) 
    
    # частный метод __loss вычисляет значение 
    # логистической функции потерь
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    # метод .fit() выполняет обучение 
    def fit(self, X, y):
        # если задан fit_intercept=True, добавляем константу
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # инициализируем значение функции потерь на предыдущей
        # итерации бесконечно большим значением
        l_prev = np.inf

        # инициализируем веса
        self.theta = np.zeros(X.shape[1])

        # выполняем градиентный спуск
        for i in range(self.num_iter):
            # вычисляем скалярное произведение 
            # матрицы предикторов и вектора весов
            z = np.dot(X, self.theta)
            # к полученному результату применяем сигмоиду, по сути
            # получаем вероятности положительного класса
            h = self.__sigmoid(z)

            # вычисляем значение функции потерь
            loss = self.__loss(h, y)
            # добавляем значение функции потерь в список
            self.loss_by_iter_.append(loss)
            
            # если разница между предыдущим значением и
            # текущим значением функции потерь меньше 
            # заданного порога (tol), то прерываем цикл, 
            # т.е. реализована ранняя остановка, которая 
            # ограничивает количество итераций (num_iter)
            if l_prev - loss < self.tol:
                return
            # присваиваем функции потерь на предыдущей
            # итерации текущее значение
            l_prev = loss
            
            # записываем количество предикторов
            m = X.shape[1]
            # перебираем координаты
            for j in range(m):
                X_j = X[:,j].reshape(-1,1)
                # обновляем веса
                self.theta[j] -= self.lr * (X_j.T.dot(h - y)) / y.size

            # если задано verbose=True, печатаем номер
            # итерации, значение функции потерь и веса                   
            if self.verbose:
                print(f"Итерация: {i}\n", 
                      f"Функция потерь: {round(loss, 3)}\n", 
                      f"Коэфф-ты: {np.round(self.theta, 3)}")

    # метод .loss_visualize() отрисовывает 
    # кривую функции потерь
    def loss_visualize(self):              
        plt.plot(range(len(self.loss_by_iter_)), self.loss_by_iter_)
        plt.xticks(np.arange(0, self.num_iter, step=self.num_iter / 5))
        plt.xlabel("Количество итераций")
        plt.ylabel("Логистическая функция потерь")
        plt.show()

    # метод .predict_proba() вычисляет вероятности
    def predict_proba(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))
    
    # метод .predict() вычисляет прогнозы
    def predict(self, X, threshold):
        return (self.predict_proba(X) >= threshold).astype(int)


# пишем собственный класс, строящий логистическую 
# регрессию с помощью метода Ньютона
class LogisticRegression_Newton:
    """
    Класс, строящий логистическую регрессию
    с помощью метода Ньютона.
    
    Параметры
    ----------
    init_method: string, по умолчанию 'zero'
        Метод инициализации весов.
        Можно выбрать 'random' или 'zero'.
    num_iter: int, по умолчанию 5
        Количество итераций ньютоновской оптимизации.
    fit_intercept: bool, по умолчанию True
        Добавление константы.
    verbose: bool, по умолчанию True
        Печать результатов оптимизации.
        
    Атрибуты
    ----------
    loss_by_iter_: ndarray of shape (num_iter, )
        Список значений функции потерь по итерациям.       
    """
    def __init__(self, init_method='zero', num_iter=5, 
                 fit_intercept=True, verbose=True):
        
        if init_method not in ['random', 'zero']:     
            raise ValueError(
                "init_method must be one of {'random', 'zero'}, "
                "got '%s' instead" % init_method
            )

        # количество итераций ньютоновской оптимизации
        self.num_iter = num_iter
        # добавление константы
        self.fit_intercept = fit_intercept
        # печать результатов оптимизации
        self.verbose = verbose
        # список, в котором будем хранить значения
        # функции потерь
        self.loss_by_iter_ = []
    
    # частный метод __add_intercept добавляет константу
    def __add_intercept(self, X):
        # создаем массив из единиц, количество единиц
        # равно количеству наблюдений
        intercept = np.ones((X.shape[0], 1))
        # конкатенируем массив из единиц с массивом 
        # предикторов по оси столбцов
        return np.concatenate((intercept, X), axis=1)
    
    # частный метод __sigmoid вычисляет значение сигмоиды
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # частный метод __get_loss вычисляет значение 
    # логистической функции потерь
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    # метод .fit() выполняет обучение
    def fit(self, X, y):
        # если задан fit_intercept=True, добавляем константу
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # инициализируем веса признаков нулями
        self.theta = np.zeros(X.shape[1])
        
        # выполняем ньютоновскую оптимизацию
        for i in range(self.num_iter):
            # вычисляем скалярное произведение 
            # матрицы предикторов и вектора весов
            z = np.dot(X, self.theta)
            # к полученному результату применяем сигмоиду, по сути
            # получаем вероятности положительного класса
            h = self.__sigmoid(z) 
            
            # вычисляем градиент
            grad = np.dot(h - y, X)
            # вычисляем гессиан
            X_ = (h * (1 - h))[:, np.newaxis] * X
            hess = np.dot(X_.T, X)
            # получаем обратный гессиан
            inv_hess = np.linalg.pinv(hess)
            # умножаем градиент на обратный гессиан
            grad = np.dot(inv_hess, grad)
            # выполняем обновление весов
            self.theta -= grad

            # если задано verbose=True, то вычисляем значение функции
            # потерь и добавляем значение функции потерь в список для
            # отрисовки с помощью метода .loss_visualize(), затем
            # печатаем номер итерации, значение функции потерь и веса
            if self.verbose:
                loss = self.__loss(h, y)
                self.loss_by_iter_.append(loss)
                print(f"Итерация: {i}\n", 
                      f"Функция потерь: {round(loss, 3)}\n", 
                      f"Коэфф-ты: {np.round(self.theta, 3)}")
         
    # метод .loss_visualize() отрисовывает 
    # кривую функции потерь
    def loss_visualize(self):  
        plt.plot(range(len(self.loss_by_iter_)), self.loss_by_iter_)
        plt.xticks(np.arange(0, self.num_iter, step=self.num_iter / 5))
        plt.xlabel("Количество итераций")
        plt.ylabel("Логистическая функция потерь")
        plt.show()
                
    # метод .predict_proba() вычисляет вероятности
    def predict_proba(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))
    
    # метод .predict() вычисляет прогнозы
    def predict(self, X, threshold):
        return (self.predict_proba(X) >= threshold).astype(int)
    
    
# пишем собственный класс, строящий логистическую
# регрессию с регуляризацией и использованием
# градиентного спуска
class RegularizedLogisticRegression_GD:
    """
    Класс, строящий логистическую регрессию
    с регуляризацией и использованием
    градиентного спуска.
    
    Параметры
    ----------
    penalty: string, по умолчанию 'l2'
        Метод регуляризации. 
        Можно выбрать 'l1' или 'l2'.
    lr: float, по умолчанию 0.1
        Темп обучения.
    tol: float, по умолчанию 1e-5
        Допуск сходимости.
    max_iter: int, по умолчанию 1e7
        Максимальное количество итераций 
        градиентного спуска.
    lambda_: float, по умолчанию 0.001
        Сила регуляризации.
    fit_intercept: bool, по умолчанию True
        Добавление константы.   
    """
    def __init__(self, penalty='l2', lr=0.1, tol=1e-5, max_iter=1e7,
                 lambda_=0.001, fit_intercept=True):
        
        # проверяем параметр penalty, задающий регуляризацию,
        # на соответствие значениям 'l1' или 'l2'
        if penalty not in ['l2', 'l1']:
            raise ValueError(
                "penalty must be 'l1' or 'l2' "
                "got '%s' instead" % penalty
            )
        
        # тип регуляризации (должен быть либо 'l1' , либо 'l2')
        self.penalty = penalty
        # темп обучения
        self.lr = lr
        # допуск сходимости
        self.tol = tol
        # максимальное количество итераций
        self.max_iter = max_iter
        #  штрафной коэффициент, т.е. вводим штраф за слишком
        # большие оценки коэффициентов регрессии
        self.lambda_ = lambda_
        # добавление константы
        self.fit_intercept = fit_intercept

    # метод .fit() выполняет обучение
    def fit(self, X, y):
        # если задан параметр fit_intercept=True
        if self.fit_intercept:
            # то добавляем константу, т.е. добавляем
            # первый столбец из единиц
            X = np.c_[np.ones(X.shape[0]), X]

        # инициализируем значение функции потерь на предыдущей
        # итерации бесконечно большим значением
        l_prev = np.inf
        # инициализируем веса признаков нулями
        self.beta = np.zeros(X.shape[1])

        # выполняем градиентный спуск
        for _ in range(int(self.max_iter)):
            # применяем сигмоид-преобразование к скалярному произведению
            # массива предикторов и вектора весов, получаем
            # вероятности положительного класса
            y_pred = self._sigmoid(np.dot(X, self.beta))
            # вычисляем значение логистической функции потерь
            loss = self._NLL(X, y, y_pred)
            # если разница между предыдущим значением и текущим значением 
            # функции потерь меньше заданного порога (tol), то прерываем 
            # цикл, т.е. реализована ранняя остановка, которая ограничивает 
            # количество итераций (max_iter)
            if l_prev - loss < self.tol:
                return
            # присваиваем функции потерь на предыдущей 
            # итерации текущее значение
            l_prev = loss
            # обновляем веса, вычитаем из текущего приближения вектора 
            # весов вектор градиента, умноженный на некоторый 
            # темп обучения
            self.beta -= self.lr * self._NLL_grad(X, y, y_pred)

    # метод _sigmoid вычисляет значение сигмоиды
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # метод _NLL вычисляет значение логистической функции потерь
    def _NLL(self, X, y, y_pred):
        # вычисляем значение логистической функции потерь без штрафа
        nll = -np.log(np.where(y == 1, y_pred, 1 - y_pred)).sum()
        
        # вычисляем штрафное слагаемое, представляющее собой
        # произведение штрафного коэффициента и L2-нормы весов 
        # (регрессионных коэффициентов), если order=2, возвращаем 
        # np.sum(np.abs(x)**2)**(1./2), т.е. квадратный корень из 
        # суммы квадратов модулей регрессионных коэффициентов и 
        # возводим в квадрат, чтобы получить сумму квадратов 
        # регрессионных коэффициентов
        if self.penalty == 'l2': 
            # если первый элемент вектора весов - это константа,
            # регуляризацию применяем ко всем элементам вектора
            # весов, кроме первого
            if self.fit_intercept:
                penalty = (self.lambda_ / 2) * np.linalg.norm(
                    self.beta[1:], ord=2) ** 2             
            # если вектор весов не содержит константу, применяем
            # регуляризацию ко всем элементам вектора весов
            else:
                penalty = (self.lambda_ / 2) * np.linalg.norm(
                    self.beta, ord=2) ** 2
        
        # вычисляем штрафное слагаемое, представляющее собой 
        # произведение штрафного коэффициента и L1-нормы весов 
        # (регрессионных коэффициентов), если order=1, возвращаем 
        # np.sum(np.abs(x)), т.е. сумму модулей регрессионных
        # коэффициентов
        if self.penalty == 'l1':
            # если первый элемент вектора весов - это константа,
            # регуляризацию применяем ко всем элементам вектора
            # весов, кроме первого
            if self.fit_intercept:
                penalty = self.lambda_ * np.linalg.norm(
                    self.beta[1:], ord=1)
            # если вектор весов не содержит константу, применяем
            # регуляризацию ко всем элементам вектора весов
            else:
                penalty = self.lambda_ * np.linalg.norm(
                    self.beta, ord=1)
            
        # вычисляем итоговое значение логистической функции потерь, 
        # прибавив к значению логистической функции потерь штрафное 
        # слагаемое, полученную сумму делим на количество наблюдений
        return (penalty + nll) / X.shape[0]

    # метод _NLL_grad вычисляет вектор градиента
    def _NLL_grad(self, X, y, y_pred):
        # если тип регуляризации l2
        if self.penalty == 'l2':
            # если первый элемент вектора весов - это константа,
            # регуляризацию применяем ко всем элементам вектора
            # весов, кроме первого
            if self.fit_intercept:
                # штрафуем все веса, кроме первого (константы)
                d_penalty = self.lambda_ * self.beta[1:]
                # подставляем константу в вектор оштрафованных весов
                d_penalty = np.r_[self.beta[0], d_penalty]
            # если вектор весов не содержит константу, применяем
            # регуляризацию ко всем элементам вектора весов
            else:
                d_penalty = self.lambda_ * self.beta

        # если тип регуляризации l1
        if self.penalty == 'l1':
            # если первый элемент вектора весов - это константа,
            # регуляризацию применяем ко всем элементам вектора
            # весов, кроме первого
            if self.fit_intercept:
                # штрафуем все веса, кроме первого (константы)
                d_penalty = self.lambda_ * np.sign(self.beta[1:])
                # подставляем константу в вектор оштрафованных весов
                d_penalty = np.r_[self.beta[0], d_penalty]
            # если вектор весов не содержит константу, применяем
            # регуляризацию ко всем элементам вектора весов
            else:
                d_penalty = self.lambda_ * np.sign(self.beta)
                
        # получаем вектор градиента, для этого вычисляем скалярное 
        # произведение матрицы предикторов и вектора разностей между 
        # фактическими значениями зависимой переменной и вероятностями 
        # положительного класса, прибавляем к полученным результатам 
        # оштрафованные веса, берем итоги с обратным знаком и делим 
        # на количество наблюдений
        return -(np.dot(y - y_pred, X) + d_penalty) / X.shape[0]

    # метод .predict_proba() вычисляет вероятности
    def predict_proba(self, X):
        # если задано fit_intercept=True
        if self.fit_intercept:
            # добавляем константу
            X = np.c_[np.ones(X.shape[0]), X]
        return self._sigmoid(np.dot(X, self.beta))

    # метод .predict() вычисляет прогнозы
    def predict(self, X, threshold):
        # получаем прогнозы в зависимости от установленного порога
        return (self.predict_proba(X) >= threshold).astype(int)

    
# пишем класс RegularizedLogisticRegression_Newton, 
# строим логистическую регрессию с L2-регуляризацией и
# использованием ньютоновского метода
class RegularizedLogisticRegression_Newton():
    """
    Класс, строящий логистическую регрессию
    с L2-регуляризацией и использованием
    метода Ньютона.
    
    Параметры
    ----------
    tol: float, по умолчанию 1e-5
        Допуск сходимости.
    max_iter: int, по умолчанию 10
        Максимальное количество итераций 
        ньютоновской оптимизации.
    lambda_: float, по умолчанию 0.001
        Сила регуляризации.
    fit_intercept: bool, по умолчанию True
        Добавление константы.   
    """
    def __init__(self, tol=1e-5, max_iter=10, lambda_=0.001, 
                 fit_intercept=True):
        # максимальное количество итераций
        self.max_iter = max_iter
        # допуск сходимости
        self.tol = tol
        #  штрафной коэффициент, т.е. вводим штраф за слишком
        # большие оценки коэффициентов регресии
        self.lambda_ = lambda_
        # добавление константы
        self.fit_intercept = fit_intercept

    # метод .fit() выполняет обучение
    def fit(self, X, y):
        # если задан параметр fit_intercept=True
        if self.fit_intercept:
            # то добавляем константу, т.е. добавляем
            # первый столбец из единиц
            X = np.c_[np.ones(X.shape[0]), X]
            
        # инициализируем значение функции потерь на предыдущей
        # итерации бесконечно большим значением
        l_prev = np.inf
        # инициализируем веса признаков нулями
        self.beta = np.zeros(X.shape[1])

        # выполняем ньютоновскую оптимизацию
        for _ in range(int(self.max_iter)):
            # применяем сигмоид-преобразование к скалярному произведению
            # массива предикторов и вектора весов, по сути получаем
            # вероятности положительного класса
            y_pred = self._sigmoid(np.dot(X, self.beta))
            
            # вычисляем значение логистической функции потерь
            loss = self._NLL(X, y, y_pred)
            # если разница между предыдущим значением и текущим значением 
            # функции потерь меньше заданного порога (tol), то прерываем 
            # цикл, т.е. реализована ранняя остановка, которая ограничивает 
            # количество итераций (max_iter)
            if l_prev - loss < self.tol:
                return
            # присваиваем функции потерь на предыдущей 
            # итерации текущее значение
            l_prev = loss 
            # обновляем веса, вычитаем из текущего приближения вектора 
            # весов вектор градиента, умноженный на некоторый 
            # темп обучения
            self.beta -= self._NLL_grad(X, y, y_pred)

    # метод _sigmoid вычисляет значение сигмоиды
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # метод _NLL вычисляет значение логистической функции потерь
    def _NLL(self, X, y, y_pred):
        # вычисляем значение логистической функции потерь без штрафа
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        nll = -np.log(np.where(y == 1, y_pred, 1 - y_pred)).sum()
        
        # вычисляем штрафное слагаемое, представляющее собой
        # произведение штрафного коэффициента и L2-нормы весов 
        # (регрессионных коэффициентов), если order=2, возвращаем 
        # np.sum(np.abs(x)**2)**(1./2), т.е. квадратный корень из 
        # суммы квадратов модулей регрессионных коэффициентов и 
        # возводим в квадрат, чтобы получить сумму квадратов 
        # регрессионных коэффициентов
       
        # если первый элемент вектора весов - это константа,
        # регуляризацию применяем ко всем элементам вектора
        # весов, кроме первого
        if self.fit_intercept:
            penalty = (self.lambda_ / 2) * np.linalg.norm(
                self.beta[1:], ord=2) ** 2             
        # если вектор весов не содержит константу, применяем
        # регуляризацию ко всем элементам вектора весов
        else:
            penalty = (self.lambda_ / 2) * np.linalg.norm(
                self.beta, ord=2) ** 2
            
        # вычисляем итоговое значение логистической функции потерь, 
        # прибавив к значению логистической функции потерь штрафное 
        # слагаемое, полученную сумму делим на количество наблюдений
        return (penalty + nll) / X.shape[0]
    
    # метод _NLL_grad вычисляет вектор градиента
    def _NLL_grad(self, X, y, y_pred):      
        # если первый элемент вектора весов - это константа,
        # регуляризацию применяем ко всем элементам вектора
        # весов, кроме первого
        if self.fit_intercept:
            d_penalty = np.r_[self.beta[0], self.lambda_ * self.beta[1:]]
            lambda_diag = np.diag(np.r_[0, [self.lambda_] * len(
                self.beta[1:])])
        # если вектор весов не содержит константу, применяем
        # регуляризацию ко всем элементам вектора весов
        else:
            d_penalty = self.lambda_ * self.beta
            lambda_diag = np.diag(np.ones_like(self.beta) * self.lambda_)
        
        # получаем вектор градиента, для этого вычисляем скалярное 
        # произведение матрицы предикторов и вектора разностей между 
        # вероятностями положительного класса и фактическими значениями 
        # зависимой переменной, прибавляем к полученным результатам 
        # оштрафованные веса, берем итоги и делим на количество наблюдений
        grad = (np.dot(X.T, y_pred - y) + d_penalty) / X.shape[0]
        # вычисляем гессиан
        X_ = (y_pred * (1 - y_pred))[:, np.newaxis] * X
        hess = np.dot(X_.T, X) / X.shape[0] + lambda_diag  
        # получаем обратный гессиан
        inv_hess = np.linalg.pinv(hess)
        # возвращаем произведение градиента и обратного гессиана
        return np.dot(inv_hess, grad)
    
    # метод .predict_proba() вычисляет вероятности
    def predict_proba(self, X):
        # если задано fit_intercept=True
        if self.fit_intercept:
            # добавляем константу
            X = np.c_[np.ones(X.shape[0]), X]
        return self._sigmoid(np.dot(X, self.beta))

    # метод .predict() вычисляет прогнозы
    def predict(self, X, threshold):
        # получаем прогнозы в зависимости от установленного порога
        return (self.predict_proba(X) >= threshold).astype(int)
    
    
# пишем собственный класс, строящий логистическую
# регрессию с L1-регуляризацией и использованием
# проксимального градиентного спуска
class RegularizedLogisticRegression_PGD:
    """
    Класс, строящий логистическую регрессию
    с L1-регуляризацией и использованием
    проксимального градиентного спуска.
    
    Параметры
    ----------
    lr: float, по умолчанию 0.15
        Темп обучения.
    lambda_: float, по умолчанию 0.001
        Сила регуляризации.
    max_iter: int, по умолчанию 1000
        Максимальное количество итераций 
        проксимального градиентного спуска.
    tol: float, по умолчанию 1e-4
        Допуск сходимости. 
    fit_intercept: bool, по умолчанию True
        Добавление константы.
    """
    def __init__(self, lr=0.15, lambda_=0.001, 
                 max_iter=1000, tol=1e-4, fit_intercept=True):     
        self.lr = lr
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        # добавление константы
        self.fit_intercept = fit_intercept
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # частный метод __add_intercept добавляет константу
    def __add_intercept(self, X):
        # создаем массив из единиц, количество единиц
        # равно количеству наблюдений
        intercept = np.ones((X.shape[0], 1))
        # конкатенируем массив из единиц с массивом 
        # признаков по оси столбцов
        return np.concatenate((intercept, X), axis=1)

    def _proximal(self, x, lambda_):
        # если первый элемент вектора весов - это константа,
        # регуляризацию применяем ко всем элементам вектора
        # весов, кроме первого
        if self.fit_intercept:
            new_w = np.sign(x[1:]) * np.maximum(
                np.abs(x[1:]) - np.full(len(x[1:]), lambda_), 0.0)
            new_w = np.r_[x[0], new_w]
        # если вектор весов не содержит константу, применяем
        # регуляризацию ко всем элементам вектора весов
        else:
            new_w = np.sign(x) * np.maximum(
            np.abs(x) - np.full(len(x), lambda_), 0.0)
        return new_w 
    
    
    def fit(self, X, y):  
        # если задан fit_intercept=True, добавляем константу
        if self.fit_intercept:
            X = self.__add_intercept(X)
            
        # инициализируем веса признаков нулями    
        self.w = np.zeros(X.shape[1])    
        # выполняем градиентный спуск
        for _ in range(self.max_iter):
            # получаем вектор градиента
            grad = -1 / X.shape[0] * np.dot(
                X.T, (y - self._sigmoid(np.dot(X, self.w))))
            # обновляем веса
            new_w = self._proximal(
                self.w - self.lr * grad, self.lr * self.lambda_)
            
            if np.all(np.abs(new_w - self.w) < self.tol):
                return
            self.w = new_w
                    
        return self
    
    def predict_proba(self, X):
        # если задан fit_intercept=True, добавляем константу
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self._sigmoid(np.dot(X, self.w))

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)
    
    
# пишем класс RegularizedLogisticRegression_BFGS, 
# строим логистическую регрессию с L2-регуляризацией и
# использованием метода BFGS
class RegularizedLogisticRegression_BFGS():
    """
    Класс, строящий логистическую регрессию
    с L2-регуляризацией и использованием
    метода BFGS.
    
    Параметры
    ----------
    tol: float, по умолчанию 1e-5
        Допуск сходимости.
    max_iter: int, по умолчанию 10
        Максимальное количество итераций 
        квазиньютоновской оптимизации.
    lambda_: float, по умолчанию 0.001
        Сила регуляризации.
    fit_intercept: bool, по умолчанию True
        Добавление константы.   
    """
    def __init__(self, eps_loss=1e-5, max_iter=60, 
                 lambda_=0.001, fit_intercept=True):
        # допуск сходимости
        self.eps_loss = eps_loss
        # максимальное количество итераций
        self.max_iter = max_iter
        #  штрафной коэффициент, т.е. вводим штраф за слишком
        # большие оценки коэффициентов регресии
        self.lambda_ = lambda_
        # добавление константы
        self.fit_intercept = fit_intercept

    # метод _sigmoid вычисляет значение сигмоиды
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # метод _loss вычисляет значение логистической функции потерь
    def _loss(self, beta, X, y):
        # применяем сигмоид-преобразование к скалярному произведению
        # массива признаков и вектора весов, по сути получаем
        # вероятности положительного класса
        y_pred = self._sigmoid(np.dot(X, beta))
        # вычисляем значение логистической функции потерь без штрафа
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)).mean()
        
        # вычисляем штрафное слагаемое, представляющее собой
        # произведение штрафного коэффициента и L2-нормы весов 
        # (регрессионных коэффициентов), если order=2, возвращаем 
        # np.sum(np.abs(x)**2)**(1./2), т.е. квадратный корень из 
        # суммы квадратов модулей регрессионных коэффициентов и 
        # возводим в квадрат, чтобы получить сумму квадратов 
        # регрессионных коэффициентов
        
        # если первый элемент вектора весов - это константа,
        # регуляризацию применяем ко всем элементам вектора
        # весов, кроме первого
        if self.fit_intercept:
            penalty = (self.lambda_ / 2) * np.linalg.norm(beta[1:], ord=2) ** 2
        # если вектор весов не содержит константу, применяем
        # регуляризацию ко всем элементам вектора весов
        else:
            penalty = (self.lambda_ / 2) * np.linalg.norm(beta, ord=2) ** 2
            
        # вычисляем итоговое значение логистической функции потерь, прибавив к
        # значению логистической функции потерь штрафное слагаемое, полученную
        # сумму делим на количество наблюдений
        return loss + penalty / X.shape[0]
        """
        в sklearn реализация математики иная
        return loss + penalty
        """

    # метод _grad вычисляет вектор градиента
    def _grad(self, beta, X, y):
        # применяем сигмоид-преобразование к скалярному произведению
        # массива признаков и вектора весов, по сути получаем
        # вероятности положительного класса
        y_pred = self._sigmoid(np.dot(X, beta))
        # если первый элемент вектора весов - это константа,
        # регуляризацию применяем ко всем элементам вектора
        # весов, кроме первого
        if self.fit_intercept:
            # штрафуем все веса, кроме первого (константы)
            penalty = self.lambda_ * beta[1:]
            # подставляем константу в вектор оштрафованных весов
            penalty = np.r_[beta[0], penalty]
        # если вектор весов не содержит константу, применяем
        # регуляризацию ко всем элементам вектора весов
        else:
            penalty = self.lambda_ * beta
        # получаем вектор градиента, для этого вычисляем скалярное 
        # произведение матрицы признаков и вектора разностей между 
        # вероятностями положительного класса и фактическими значениями 
        # зависимой переменной, прибавляем к полученным результатам 
        # оштрафованные веса, берем итоги и делим на количество наблюдений   
        return (np.dot(y_pred - y, X) + penalty) / X.shape[0]
        """
        в sklearn реализация математики иная
        return np.dot(y_pred - y, X) / X.shape[0] + penalty
        """

    # функция линейного поиска шага alpha
    def _line_search_wolfe1_or_wolfe2(self, f_loss, f_grad, beta, pk, grad_0, 
                                      loss_0, old_loss, args=(), **kwargs):
        
        # алгоритм линейного поиска шага alpha
        ret = line_search_wolfe1(f_loss, f_grad, beta, pk, grad_0,
                                loss_0, old_loss, args,
                                **kwargs)

        # если алгоритм line_search_wolfe1 не обнаружил шаг alpha,
        # то воспользуемся алгоритмом line_search_wolfe
        if ret[0] is None:
            kwargs2 = {}
            kwargs2['amax'] = kwargs['amax']
            ret = line_search_wolfe2(f_loss, f_grad, beta, pk, grad_0,
                                    loss_0, old_loss, args,
                                    extra_condition=None, **kwargs2)
        return ret

    # метод .fit() выполняет обучение
    def fit(self, X, y):
        # если задан параметр fit_intercept=True
        if self.fit_intercept:
            # то добавляем константу, т.е. добавляем
            # первый столбец из единиц
            X = np.c_[np.ones(X.shape[0]), X]
        N = X.shape[1]
        if self.max_iter is None:
            self.max_iter = N * 200
        # инициализируем начальную точку
        # инициализируем веса признаков нулями
        self.beta = np.zeros(N)
        k = 0
        # вычисляем значение логистической функции потерь в начальной точке
        loss = self._loss(self.beta, X, y)
        # значение градиента в начальной точке
        grad = self._grad(self.beta, X, y)
        # инициализируем единичную матрицу
        I = np.eye(N, dtype=int)
        # инициализируем матрицу Гессе
        H_k = I
        # инициализируем значение loss на предыдущей итерации
        old_loss = loss + np.linalg.norm(grad) / 2
        
        # выполняем поиск оптимальной матрицы признаков
        while (np.linalg.norm(grad, ord=1) > self.eps_loss) and (k < self.max_iter):
            # ищем направление убывания
            p_k = -np.dot(H_k, grad)
            # коэффициент alpha_k (размер шага) находим, используя линейный поиск 
            # (line search) такой, чтобы alpha_k удовлетворял условиям Вольфе
            # line_search возвращает также значения loss, old_loss и 
            # grad_new - значение градиента на следующей итерации
            alpha_k, fc, gc, loss, old_loss, grad_new = \
            self._line_search_wolfe1_or_wolfe2(
                self._loss, self._grad, 
                self.beta, p_k, grad, loss, 
                old_loss, (X, y), 
                amin=1e-100, amax=1e100)
            if alpha_k is None:
                break

            s_k = alpha_k * p_k
            
            # обновляем веса признаков
            self.beta += s_k

            if grad_new is None:
                grad_new = self._grad(self.beta, X, y)
            y_k = grad_new - grad
            grad = grad_new
        
            k += 1
            if np.linalg.norm(grad, ord=1) <= self.eps_loss:
                break
            
            # определяем новые значения матрицы Гессе
            try:
                ro_k = 1.0 / (np.dot(y_k, s_k))
            except ZeroDivisionError:
                ro_k = 1000.0
            if np.isinf(ro_k):  
                ro_k = 1000.0
            # A1 = I - np.dot(s_k[:,np.newaxis],y_k[:,np.newaxis].T)* ro_k
            A1 = I - ro_k * s_k[:, np.newaxis] * y_k[np.newaxis, :]
            # A2 = I - np.dot(y_k[:,np.newaxis],s_k[:,np.newaxis].T)* ro_k
            A2 = I - ro_k * y_k[:, np.newaxis] * s_k[np.newaxis, :]
            # H_k = np.dot(A1, np.dot(H_k, A2)) + (ro_k * np.dot(
            # s_k[:,np.newaxis], s_k[:,np.newaxis].T))
            H_k = np.dot(A1, np.dot(H_k, A2)) + (
                ro_k * s_k[:, np.newaxis] * s_k[np.newaxis, :])

        return self

    # метод .predict_proba() вычисляет вероятности
    def predict_proba(self, X):
        # если задано fit_intercept=True
        if self.fit_intercept:
            # добавляем константу
            X = np.c_[np.ones(X.shape[0]), X]
        return self._sigmoid(np.dot(X, self.beta))

    # метод .predict() вычисляет прогнозы
    def predict(self, X, threshold):
        # получаем прогнозы в зависимости от установленного порога
        return (self.predict_proba(X) >= threshold).astype(int)