"""
Метод Нелдера-Мида (метод деформирующегося многогранника, симплекс-метод).
Одна из возможных реализаций с использованием модуля functions

Классы:
    NelderMead - класс, реализующий метод Нелдера-Мида
"""
import numpy as np
from functions import BaseFunction


class NelderMead:
    """Класс, реализующий метод Нелдера-Мида.
    Параметры алгоритма задаются при создании экземпляра класса

    Свойства:
        params - словарь аргументов метода
        function - оптимизироваемая функция в виде sympy.Expr
        simplex - текущий симплекс в виде двумерного списка

    Методы:
        fit - задание функции и начального симплекса
        run - запуск алгоритма
    """

    # pylint: disable=too-many-instance-attributes
    # Nine is reasonable in this case :)
    def __init__(self, *,
                 alpha: float = 1, betta: float = 0.5, gamma: float = 2,
                 max_steps: int = 1000, eps0: float = 0.001,
                 max_blank: int = 10, eps1: float = 0.001):
        """Инициализатор метода

        :param alpha: коэффициент отражения
        :param betta: коэффициент сжатия
        :param gamma: коэффициент растяжения
        :param max_steps: максимальной количество итераций
        :param eps0: предельная дисперсия точек
        :param max_blank: максимальное число "бесполезных" итераций
        :param eps1: минимальная разница между "полезными" итерациями
        """
        self.__alpha = alpha
        self.__betta = betta
        self.__gamma = gamma
        self.__max_steps = max_steps
        self.__eps0 = eps0
        self.__eps1 = eps1
        self.__max_blank = max_blank
        self.__simplex = []
        self.__function = None
        self.__check_args()

    @property
    def params(self) -> dict:
        """Параметры метода

        :return: словарь с названиями и значения параметров
        """
        return {"alpha": self.__alpha,
                "betta": self.__betta,
                "gamma": self.__gamma,
                "max_steps": self.__max_steps,
                "eps0": self.__eps0,
                "eps1": self.__eps1,
                "max_blank": self.__max_blank}

    @property
    def function(self) -> BaseFunction:
        """Оптимизируемая функция

        :return: экземпляр класса BaseFunction
        """
        return self.__function

    @property
    def simplex(self) -> list:
        """Текущий симплекс

        :return: двумерный список точек
        """
        return [list(point) for (point, value) in self.__simplex]

    def fit(self, function: BaseFunction, simplex: list = None) -> None:
        """Инициализация оптимизируемой функции и начального симплекса.
        При отсутствии начального симплекса, создаются случайный симплекс

        :param function: оптимизируемая функция как экземпляр BaseFunction
        :param simplex: начальный симплекс как двумерный список точек

        :exception: AttributeError
        """
        # Проверки входных данных
        if not isinstance(function, BaseFunction):
            raise AttributeError("function not an instance of BaseFunction")
        self.__function = function
        if simplex is None:
            # Создание случайного начального симплекса
            self.__create_simplex()
        else:
            # Проверка и приведение симплекса к списку массивов numpy.array
            self.__simplex = simplex
            self.__check_simp()
            self.__simplex_modify()
        # Вычисление значения функции от каждой точки
        # и создание итогового вида симплекса в виде списка кортежей
        self.__simplex = self.__make_points()

    def run(self, *, action=None) -> float:
        """Запуск метода

        :param action: опциональное действие в конце каждой итерации.
            Должен быть callable-объект, принимающий экземпляр данного класса

        :return: найденный минимум
        :exception: AttributeError
        """
        # Проверка наличия функции
        if not isinstance(self.__function, BaseFunction):
            raise AttributeError("No function in class, use fit method")
        # Основной цикл метода
        iteration = 0
        while iteration <= self.__max_steps:
            # Сортировка точек по значению функции
            self.__simplex.sort(key=lambda x: x[1])
            points = [point for (point, value) in self.__simplex]

            # Вычисление необходимых точек
            centroid = 1 / len(points[:-1]) * sum(points[:-1])
            best, good = self.__simplex[0], self.__simplex[-2]
            worst = self.__simplex[-1]
            # Отражение
            reflected = self.__reflection(centroid)
            # Выбор замены
            if reflected[1] < best[1]:
                # Растяжение
                self.__expansion(centroid, reflected)
            elif reflected[1] < good[1]:
                self.__simplex[-1] = reflected
            else:
                if reflected[1] < worst[1]:
                    self.__simplex[-1] = reflected
                # Сжатие
                self.__contraction(centroid)
            # Опциональное действие
            if callable(action):
                action(self)
            # Условие останова
            if self.__stop():
                break
            iteration += 1
        self.__simplex.sort(key=lambda x: x[1])
        return self.__simplex[0][1]

    def __check_simp(self) -> None:
        """ Проверка симплекса на корректность

        :exception: AttributeError
        """
        if not isinstance(self.__simplex, list):
            raise AttributeError("Simplex is not list")
        if len(self.__simplex) != self.__function.dimension + 1:
            raise AttributeError("Simplex length less than dimension + 1")
        for point in self.__simplex:
            if len(point) != self.__function.dimension:
                raise AttributeError(f"{point} length less than dimension")

    def __create_simplex(self) -> None:
        """Создание случайного симплекса"""
        # Нулевая точка
        self.__simplex = [np.array([0]*self.__function.dimension)]
        for point in range(self.__function.dimension):
            prev_point = self.__simplex[point]
            new_point = prev_point
            while np.equal(prev_point, new_point).all():
                new_point = np.random.randint(low=(min(prev_point) - 1),
                                              high=(max(prev_point) + 1),
                                              size=self.__function.dimension)
            self.__simplex.append(new_point)

    def __simplex_modify(self) -> None:
        """Перевод каждой точки из списка в numpy.ndarray"""
        for index, value in enumerate(self.__simplex):
            self.__simplex[index] = np.array(value)

    def __make_points(self) -> list:
        """Вычисление значения функции от точек симплекса

        :return: список кортежей из точек и их значения
        """
        points_with_values = []
        for point in self.__simplex:
            value = self.__function.calculate(list(point))
            points_with_values.append((point, value))
        return points_with_values

    def __reflection(self, centroid) -> tuple:
        """Операция отражения

        :param centroid: центр масс

        :return: кортеж из отражённой точки и значения функции
        """
        worst = self.__simplex[-1]
        reflected = (1 + self.__alpha) * centroid - self.__alpha * worst[0]
        return reflected, self.__function.calculate(list(reflected))

    def __expansion(self, centroid, reflected) -> None:
        """Операция растяжения. Автоматически заменяет худшую точку

        :param centroid: центр масс

        :param reflected: отражённая точка
        """
        expanded = (1-self.__gamma) * centroid + self.__gamma * reflected[0]
        value = self.__function.calculate(list(expanded))
        if value <= reflected[1]:
            self.__simplex[-1] = (expanded, value)
        else:
            self.__simplex[-1] = reflected

    def __contraction(self, centroid) -> None:
        """Операция сжатия. Заменяет худшую точку.
        Если необходимо, начинает глобальное сжатие

        :param centroid: центр масс
        """
        worst = self.__simplex[-1]
        condensed = (1-self.__betta) * centroid + self.__betta * worst[0]
        value = self.__function.calculate(list(condensed))
        if value < worst[1]:
            self.__simplex[-1] = (condensed, value)
        else:
            self.__global_contraction()

    def __global_contraction(self) -> None:
        """Операция глобального сжатия"""
        best = self.__simplex[0][0]
        for index, (point, _) in enumerate(self.__simplex[1:]):
            new_point = best + (point - best)/2
            new_value = self.__function.calculate(list(new_point))
            self.__simplex[index] = (new_point, new_value)

    def __stop(self) -> bool:
        """Условия останова

        :return: булево значение
        """
        points = [point for (point, value) in self.__simplex]
        dispersion = np.var(points)
        return dispersion < self.__eps0

    def __check_args(self):
        """Проверка параметров метода"""
        if self.__alpha < 0:
            raise AttributeError("alpha should be >= 0")
        if self.__betta < 0:
            raise AttributeError("betta should be >= 0")
        if self.__gamma < 0:
            raise AttributeError("gamma should be >= 0")
        if not isinstance(self.__max_steps, int):
            raise AttributeError("max_steps should be integer")
        if self.__eps0 < 0:
            raise AttributeError("eps0 should be > 0")

    def __str__(self) -> str:
        """Строковое представление текущего состояния метода

        :return: строка с информацией о параметрах, функции и симплексе
        """
        return f"Params: alpha = {self.__alpha}, betta = {self.__betta}," \
               f" gamma = {self.__gamma}\n" \
               f"steps = {self.__max_steps}, accuracy = {self.__eps0}\n" \
               f"Function: {self.__function.expr}\n" \
               f"Simplex: {self.__simplex}"
