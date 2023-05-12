"""
Поиск оптимальных параметров для оптимизационных методов

Классы:
    SearchMethodParams - работа с произвольными методами
    SearchNelderMeadParams - работа с методом Нелдера-Мида
    SearchConditional - работа с условным методом Нелдера-Мида

Пример:
    import numpy as np
    from scripts.functions import Rosenbroke
    from scripts.point import Point
    searcher = SearchNelderMeadParams(
    {
        "alpha": np.linspace(0.5, 1.5, 3),
        "betta": np.linspace(0, 1, 3),
        "gamma": np.linspace(1.5, 2.5, 3)
    },
        Rosenbroke(), [Point.zero(2)]
    )
    res = searcher.run(max_optimal_count=20)
    print("TOP-20:")
    number = 1
    for value, params in res:
        print(f"\t#{number}:")
        print(f"\tValue: {value}")
        print(f"\tParams: {params}")
        number += 1
"""
from itertools import product
from scripts.nelder_mead import NelderMead
from scripts.functions import BaseFunction
from scripts.point import Point
from scripts.conditional_nm import ConditionalNelderMead


class SearchMethodParams:
    """Поиск оптимальных параметров для произвольного оптимизационного метода.
    Класс, реализующий метод, должен иметь методы fit и run.

    Методы:
        run(int)
    """
    def __init__(self, method, params_to_search: dict,
                 data_to_fit: list, data_to_run: list):
        """Конструктор класса

        Args:
            method: класс, реализующий оптимизационный метод
            params_to_search: словарь параметров метода и с набором их значений
            data_to_fit: список данных, передаваемых в ``method().fit``
            data_to_run: список данных, передаваемых в ``method().run``
        """
        self.__method = method
        self.__params_to_search = params_to_search.copy()
        self._fit = data_to_fit.copy()
        self._run = data_to_run.copy()
        self._optimal_params = []

    def run(self, max_optimal_count: int = 1):
        """Запуск поиска оптимальных параметров

        Args:
            max_optimal_count: максимальное количество оптимальных наборов

        Returns:
            отсортированный список из оптимальных параметров и значений, полученных при них

        """
        keys = self.__params_to_search.keys()
        values = self.__params_to_search.values()

        self._optimal_params = []

        for params in product(*values):
            current_params = dict(zip(keys, params))
            current_value = self._step(current_params)
            element = [current_value, current_params]
            self._save_params(element, max_optimal_count)

        return self._optimal_params

    def _step(self, params: dict) -> float:
        """Шаг поиска

        Args:
            params: текущий набор параметров

        Returns:
            результат работы метода
        """
        method = self.__method(**params)
        method.fit(*self._fit)
        return method.run(*self._run)

    def _save_params(self, element, max_count):
        """Сохранение набора параметров и значения.
        Выполняет проверку на количество хранимых наборов

        Args:
            element: конкретный набор параметров и значение
            max_count: максимальное допустимое количество наборов
        """
        self._optimal_params.append(element)
        self._optimal_params.sort(key=lambda x: x[0])
        if len(self._optimal_params) > max_count:
            self._optimal_params.pop()


class SearchNelderMeadParams(SearchMethodParams):
    """Класс для поиска оптимальных параметров метода Нелдера-Мида"""
    def __init__(self, nm_params: dict, function: BaseFunction, points: list):
        """Конструктор класса

        Args:
            nm_params: словарь параметров со списками их значений
            function: целевая функция
            points: начальный симплекс
        """
        super().__init__(NelderMead, nm_params, [function, *points], [])


class SearchConditional(SearchMethodParams):
    """Поиск оптимальных параметров условного метода Нелдера-Мида"""
    def __init__(self, conditional_params: dict, nm_params: dict,
                 function: BaseFunction, conditions: list,
                 start_point: Point):
        """Конструктор класса

        Args:
            conditional_params: параметры условного метода
            nm_params: параметры самого метода Нелдера-Мида
            function: целевая функция
            conditions: список ограничений
            start_point: начальная точка для условного метода
        """
        super().__init__(ConditionalNelderMead, conditional_params,
                         [NelderMead(), function, *conditions],
                         [start_point])
        self.__nm_params = nm_params

    def run(self, max_optimal_count: int = 5) -> list:
        keys = self.__nm_params.keys()
        values = self.__nm_params.values()

        true_optimal_params = []
        for params in product(*values):
            current_params = dict(zip(keys, params))
            nm_method = NelderMead(**current_params)
            self._fit[0] = nm_method
            element: list = super().run(1)[0]
            element.insert(1, current_params)
            self._optimal_params = true_optimal_params
            self._save_params(element, max_optimal_count)
            true_optimal_params = self._optimal_params.copy()

        return true_optimal_params

    def _step(self, params: dict) -> float:
        result = super()._step(params)
        return result[1]
