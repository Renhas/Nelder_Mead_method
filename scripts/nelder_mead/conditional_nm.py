"""
Модуль, адаптирующий Нелдера-Мида под задачу условной оптимизации

Классы:
    ConditionalNelderMead - класс, применяющий Нелдера-Мида для решения задачи условной оптимизации
"""
import typing
from scripts.utilities.point import Point
from scripts.nelder_mead.nelder_mead import NelderMead
from scripts.utilities.constraints import Constraint
from scripts.utilities.functions import BaseFunction


class ConditionalNelderMead:
    """
    Класс, адаптирующий метод Нелдера-Мида под задачу условной оптимизации.
        Достигается это путём сведения задачи условной оптимизации к задаче безусловной оптимизации
        по принципу барьерных (штрафных) функций

    Свойства:
        parameters - параметры барьерного метода
        nm_method - метод Нелдера-Мида
        function - целевая функция
        constraints - кортеж ограничений
    Методы:
        fit - инициализация метода Нелдера-Мида, целевой функции и ограничений
        run - старт алгоритма из некой стартовой точки
    """
    def __init__(self, *, eps: float = 0.0001, betta: float = 1.5,
                 start_weight: float = 1.0, max_steps: int = 1000):
        """Конструктор класса

        :param eps: предельная величина ошибки (штрафа)
        :param betta: коэффициент увеличения штрафного коэффициента
        :param start_weight: начальный штрафной коэффициент
        :param max_steps: максимальное количество шагов
        """
        self.__eps = eps
        self.__betta = betta
        self.__start_weight = start_weight
        self.__max_steps = max_steps
        self.__check_init_args()
        self.__nm_method: NelderMead = None
        self.__func: BaseFunction = None
        self.__constraints = None

    def fit(self, nm_method: NelderMead, func: BaseFunction, *args):
        """Инициализация метода Нелдера-Мида, целевой функции и ограничений

        :param nm_method: метод Нелдера-Мида,
            использующийся для решения задачи безусловной оптимизации
        :param func: целевая функция
        :param args: кортеж ограничений
        """
        self.__nm_method = nm_method
        self.__func = func
        self.__constraints = args
        self.__check_fit_args()

    def run(self, start_point: Point, nm_action: typing.Callable = None,
            action: typing.Callable = None):
        """Запуск алгоритма из начальной точки

        :param start_point: начальная точка алгоритма
        :param nm_action: опциональное действие для метода Нелдера-Мида
        :param action: опциональное действие для конца каждой итерации
        :return: решение и значение функции
        """
        if self.__nm_method is None:
            raise AttributeError("There is no NelderMead, use fit method!")
        error_weight = self.__start_weight
        solution = start_point
        error = self.__eps + 1
        step = 0
        while not self.__stop(error):
            step += 1
            error_func = [constr.error_func for constr in self.__constraints]
            error_func = error_weight * sum(error_func)
            new_func = self.__func + error_func
            self.__nm_method.fit(new_func, solution)
            self.__nm_method.run(action=nm_action)
            solution = self.__nm_method.simplex.best[0]
            if action is not None:
                action(self)
            error = error_func.calculate(solution)
            error_weight *= self.__betta
            if step >= self.__max_steps:
                break
        return solution, self.__func.calculate(solution)

    def __stop(self, error) -> bool:
        """Условия останова

        :param error: штраф
        :return: булево значение
        """
        return error < self.__eps

    @property
    def parameters(self) -> dict:
        """Параметры алгоритма

        :return: словарь с параметрами
        """
        return {"eps": self.__eps,
                "betta": self.__betta,
                "max_steps": self.__max_steps,
                "start_weight": self.__start_weight}

    @property
    def nm_method(self):
        """Метод Нелдера-Мида

        :return: копия экземпляра метода
        """
        return NelderMead(**self.__nm_method.params)

    @property
    def function(self):
        """Целевая функция

        :return: копия целевой функции
        """
        func = self.__func
        return BaseFunction(func.expr, func.variables)

    @property
    def constraints(self):
        """Ограничения

        :return: кортеж ограничений
        """
        return self.__constraints

    def __check_fit_args(self):
        """Проверка параметров метода fit"""
        if not isinstance(self.__nm_method, NelderMead):
            raise AttributeError("nm_method must be a NelderMead")
        if not isinstance(self.__func, BaseFunction):
            raise AttributeError("func must be a BaseFunction")
        dim = self.__func.dimension
        for element in self.__constraints:
            if not isinstance(element, Constraint):
                raise AttributeError(f"{element} must be a Constraint")
            if element.function.dimension != dim:
                raise AttributeError(f"{element} dimension must be {dim}")

    def __check_init_args(self):
        """Проверка параметров алгоритма"""
        if not isinstance(self.__eps, float):
            raise AttributeError("epx must be a float")
        if not isinstance(self.__betta, float):
            raise AttributeError("betta must be a float")
        if not isinstance(self.__max_steps, int):
            raise AttributeError("max_steps must be a integer")
        if not isinstance(self.__start_weight, float):
            raise AttributeError("start_weight must be a float")
        if self.__eps < 0:
            raise AttributeError("eps must be >= 0")
        if self.__betta <= 1:
            raise AttributeError("betta must be > 1")
        if self.__max_steps < 1:
            raise AttributeError("max_steps must be >= 1")
        if self.__start_weight <= 0:
            raise AttributeError("start_weight must be > 0")
