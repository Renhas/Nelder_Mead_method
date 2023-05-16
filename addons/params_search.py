"""
Поиск оптимальных параметров для оптимизационных методов
"""
from typing import Sequence, Tuple
from itertools import product
from nelder_mead.nelder_mead import NelderMead
from utilities.functions import BaseFunction
from utilities.point import Point
from utilities.constraints import Constraint
from nelder_mead.conditional_nm import ConditionalNelderMead


# pylint: disable=too-few-public-methods
class SearchMethodParams:
    """Поиск оптимальных параметров для произвольного оптимизационного метода.
    Класс, реализующий метод, должен иметь методы fit и run.
    """
    def __init__(self, method, params_to_search: dict,
                 data_to_fit: list, data_to_run: list):
        """Инициализатор класса

        Args:
            method: объект, реализующий оптимизационный метод
            params_to_search: словарь параметров метода с набором их значений
            data_to_fit: список данных, передаваемых в method().fit
            data_to_run: список данных, передаваемых в method().run
        """
        self.__method = method
        self.__params_to_search = params_to_search.copy()
        self._fit = data_to_fit.copy()
        self._run = data_to_run.copy()
        self._optimal_params = []

    def run(self, max_optimal_count: int = 1) \
            -> Sequence[Tuple[float, dict]]:
        """Запуск поиска оптимальных параметров

        Args:
            max_optimal_count: максимальное количество оптимальных наборов

        Returns:
            Отсортированный список из оптимальных параметров и значений,
             полученных при этих параметрах
        """
        keys = self.__params_to_search.keys()
        values = self.__params_to_search.values()

        self._optimal_params = []

        for params in product(*values):
            current_params = dict(zip(keys, params))
            current_value = self._step(current_params)
            element = (current_value, current_params)
            self._save_params(element, max_optimal_count)

        return self._optimal_params

    def _step(self, params: dict) -> float:
        """Шаг поиска

        Args:
            params: текущий набор параметров

        Returns:
            Результат работы метода
        """
        method = self.__method(**params)
        method.fit(*self._fit)
        return method.run(*self._run)

    def _save_params(self, element: Tuple[float, dict],
                     max_count: int):
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


# pylint: disable=too-few-public-methods
class SearchNelderMeadParams(SearchMethodParams):
    """Класс для поиска оптимальных параметров метода Нелдера-Мида"""
    def __init__(self, nm_params: dict, function: BaseFunction, points: list):
        """Инициализатор класса

        Args:
            nm_params: словарь параметров со списками их значений
            function: целевая функция
            points: начальный симплекс
        """
        super().__init__(NelderMead, nm_params, [function, *points], [])


# pylint: disable=too-few-public-methods
class SearchConditional(SearchMethodParams):
    """Поиск оптимальных параметров условного метода Нелдера-Мида"""

    # pylint: disable=too-many-arguments
    def __init__(self, conditional_params: dict, nm_params: dict,
                 function: BaseFunction, conditions: Sequence[Constraint],
                 start_point: Point):
        """Инициализатор класса

        Args:
            conditional_params: параметры условного метода Нелдера-Мида
            nm_params: параметры самого метода Нелдера-Мида
            function: целевая функция
            conditions: список ограничений
            start_point: начальная точка для условного метода
        """
        super().__init__(ConditionalNelderMead, conditional_params,
                         [NelderMead(), function, *conditions],
                         [start_point])
        self.__nm_params = nm_params

    def run(self, max_optimal_count: int = 5) \
            -> Sequence[Tuple[float, dict]]:
        keys = self.__nm_params.keys()
        values = self.__nm_params.values()

        true_optimal_params = []
        for params in product(*values):
            current_params = dict(zip(keys, params))
            nm_method = NelderMead(**current_params)
            self._fit[0] = nm_method
            element = super().run(1)[0]
            params_dict = {"nm_params": current_params,
                           "cnm_params": element[1]}
            element = (element[0], params_dict)
            self._optimal_params = true_optimal_params
            self._save_params(element, max_optimal_count)
            true_optimal_params = self._optimal_params.copy()

        return true_optimal_params

    def _step(self, params: dict) -> float:
        result = super()._step(params)
        return result[1]
