"""
Модуль с функционалом визуализации метода Нелдера-Мида

Классы:
    PlotSettings - управление графиком
    NelderMead2DAnimation - анимация работы метода
"""
import typing

import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from sympy.utilities.lambdify import lambdify
from scripts.functions import BaseFunction
from scripts.nelder_mead import NelderMead, Point


class PlotSettings:
    """Класс для работы с графиком

    Методы класса:
        plot_point(Point, str)
        show()
        set_style(str)

    Методы экземпляра:
        contour_init(BaseFunction, float, float, int)

    Свойства:
        fig - фигура, на которой рисуется график
        axes - оси графика
        line - линия для точек симплекса
        text - дополнительный текст в левом верхнем углу
        title - название графика
    """
    def __init__(self, title: str, x_limits: tuple, y_limits: tuple,
                 line_width: float = 1):
        """Конструктор класса

        :param title: название графика
        :param x_limits: диапазон значений по x
        :param y_limits: диапазон значений по y
        :param line_width: размер линии, соединяющей точки
        """
        if not isinstance(x_limits, tuple):
            raise AttributeError("x_limits must be a tuple")
        if not isinstance(y_limits, tuple):
            raise AttributeError("y_limits must be a tuple")
        self.__x_limits = x_limits
        self.__y_limits = y_limits
        self.__title = title
        self.__fig = plt.figure()
        self.__ax = plt.axes(xlim=x_limits, ylim=y_limits)
        self.__line,  = self.__ax.plot([], [], lw=line_width)
        self.__text_on_plot = plt.text(0, 0, "None")
        plt.title(title)

    @staticmethod
    def plot_point(true_point: Point, fmt: str = "ro"):
        """Отрисовка одной точки

        :param true_point: точка
        :param fmt: стиль точки
        """
        plt.plot(true_point.values[0], true_point.values[1], fmt)

    @staticmethod
    def show():
        """Отобразить график в интерактивном режиме"""
        plt.show()

    @staticmethod
    def set_style(style: str):
        """Установить стиль графика

        :param style: название стиля
        """
        plt.style.use(style)

    def contour_init(self, function: BaseFunction,
                     x_step: float = 0.05, y_step: float = 0.05,
                     levels: int = 10):
        """Отрисовка линий уровня функции с двумя(!) переменными

        :param function: функция
        :param x_step: шаг, с которым генерируются точки по x
        :param y_step: шаг, с которым генерируются точки по y
        :param levels: количество уровней
        """
        x_limits, y_limits = self.__x_limits, self.__y_limits
        x_data = np.arange(x_limits[0], x_limits[1], x_step)
        y_data = np.arange(y_limits[0], y_limits[1], y_step)
        nm_func = lambdify(function.variables, function.expr, "numpy")
        xgrid, ygrid = np.meshgrid(x_data, y_data)
        zgrid = nm_func(xgrid, ygrid)
        self.__ax.contour(xgrid, ygrid, zgrid, levels=levels)

    @property
    def fig(self) -> matplotlib.figure.Figure:
        """Фигура, на которой рисуется график

        :return: фигура
        """
        return self.__fig

    @property
    def axes(self) -> matplotlib.pyplot.axes:
        """Оси графика

        :return: оси
        """
        return self.__ax

    @property
    def line(self) -> matplotlib.pyplot.Line2D:
        """Линия, проходящая по точкам

        :return: линия
        """
        return self.__line

    @property
    def text(self) -> matplotlib.pyplot.Text:
        """Текст в левом верхнем углу графика

        :return: текст
        """
        return self.__text_on_plot

    @property
    def title(self) -> str:
        """Название графика

        :return: название
        """
        return self.__title


class NelderMead2DAnimation:
    """Класс для анимации работы метода Нелдера-Мида

    Методы:
        animate(int, bool, *, Callable)
        save(str)
    """
    def __init__(self, method: NelderMead, plot: PlotSettings):
        """Конструктор класса

        :param method: метод Нелдера-Мида в начальном состоянии
        :param plot: класс, управляющий графиком
        """
        if not isinstance(method, NelderMead):
            raise AttributeError("method must be a NelderMean")
        if not isinstance(plot, PlotSettings):
            raise AttributeError("plot must be a PlotSettings")
        self.__plot = plot
        self.__action = None
        self.__anim = None
        self.__method = method
        self.__all_x = []
        self.__all_y = []

    def __save_points(self, method: NelderMead):
        """Сохранение всех точек симплекса

        :param method: метод Нелдера-Мида
        """
        sim = method.simplex
        best, good = sim.best[0].values, sim.good[0].values
        worst = sim.worst[0].values
        self.__all_x.append([best[0], good[0], worst[0], best[0]])
        self.__all_y.append([best[1], good[1], worst[1], best[1]])

    def __init_animation(self) -> tuple:
        """Инициализация первоначальных данных для анимации

        :return: кортеж из одного элемента - линии на графике
        """
        self.__plot.line.set_data([], [])
        return (self.__plot.line,)

    def __animation(self, frame: int) -> tuple:
        """Функция анимации, вызывается каждый кадр

        :param frame: кадр анимации

        :return: кортеж из одного элемента - линии на графике
        """

        index = frame
        if index > len(self.__all_x) - 1:
            index = len(self.__all_x) - 1
        if self.__action is not None:
            self.__action(frame)
        self.__plot.line.set_data(self.__all_x[index], self.__all_y[index])
        return (self.__plot.line,)

    def animate(self, interval: int = 300, *, blit: bool = False,
                action: typing.Callable = None):
        """Анимация работы метода Нелдера-Мида

        :param interval: интервал между кадрами
        :param blit: True - будут перерисовываться
         только сильно меняющиеся объекты. False - перерисовывается всё.
        :param action: действие в конце каждой итерации.
         Callable-объект, принимающий номер кадра
        """
        self.__action = action
        self.__save_points(self.__method)
        self.__method.run(action=self.__save_points)

        self.__anim = FuncAnimation(self.__plot.fig, self.__animation,
                                    init_func=self.__init_animation,
                                    frames=len(self.__all_x),
                                    interval=interval, blit=blit)

    def save(self, path: str):
        """Сохранение анимации в файл

        :param path: путь к файлу
        """
        self.__anim.save(path, writer="imagemagick")
