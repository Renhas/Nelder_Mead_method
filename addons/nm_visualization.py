"""
Визуализация работы метода Нелдера-Мида
"""
from typing import Union, Tuple, Callable, Sequence
import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from sympy.utilities.lambdify import lambdify
from utilities.functions import BaseFunction
from nelder_mead.nelder_mead import NelderMead, Point


class PlotSettings:
    """Работа с графиком"""
    def __init__(self, title: str, x_limits: Tuple[float, float],
                 y_limits: Sequence[float], line_width: float = 1):
        """Инициализатор класса

        Args:
            title: название графика
            x_limits: диапазон значений по x
            y_limits: диапазон значений по y
            line_width: размер линии, соединяющей точки

        Raises:
            AttributeError - если переданные параметры некорректны
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
        self.__text_on_plot = plt.text(0, 0, "")
        plt.title(title)

    @staticmethod
    def plot_point(point: Point, fmt: str = "ro"):
        """Отрисовка одной точки

        Args:
            point: точка
            fmt: стиль точки в формате :mod:`matplotlib`
        """
        plt.plot(point.values[0], point.values[1], fmt)

    @staticmethod
    def show():
        """Отобразить график в интерактивном режиме"""
        plt.show()

    @staticmethod
    def set_style(style: str):
        """Установить стиль графика

        Args:
            style: название стиля из списка стилей :mod:`matplotlib`
        """
        plt.style.use(style)

    @property
    def fig(self) -> matplotlib.figure.Figure:
        """Фигура, на которой рисуется график"""
        return self.__fig

    @property
    def axes(self) -> matplotlib.pyplot.axes:
        """Оси графика"""
        return self.__ax

    @property
    def line(self) -> matplotlib.pyplot.Line2D:
        """Линия, проходящая по точкам"""
        return self.__line

    @property
    def text(self) -> matplotlib.pyplot.Text:
        """Текст в левом верхнем углу графика"""
        return self.__text_on_plot

    @property
    def title(self) -> str:
        """Название графика"""
        return self.__title

    def contour_init(self, function: BaseFunction,
                     x_step: float = 0.05, y_step: float = 0.05,
                     levels: Union[int, Sequence[float]] = 10):
        """Отрисовка линий уровня функции с двумя(!) переменными

        Args:
            function: функция
            x_step: шаг генерации точек по x
            y_step: шаг генерации точек по y
            levels: количество уровней или список значений функции
        """
        x_limits, y_limits = self.__x_limits, self.__y_limits
        x_data = np.arange(x_limits[0], x_limits[1], x_step)
        y_data = np.arange(y_limits[0], y_limits[1], y_step)
        nm_func = lambdify(function.variables, function.expr, "numpy")
        xgrid, ygrid = np.meshgrid(x_data, y_data)
        zgrid = nm_func(xgrid, ygrid)
        self.__ax.contour(xgrid, ygrid, zgrid, levels=levels)


class NelderMead2DAnimation:
    """Анимация работы метода Нелдера-Мида"""
    def __init__(self, method: NelderMead, plot: PlotSettings):
        """Инициализатор класса

        Args:
            method: метод Нелдера-Мида в начальном состоянии
            plot: объект для управления графиком

        Raises:
            AttributeError - если переданные параметры некорректны
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

    def animate(self, interval: int = 300, *, blit: bool = False,
                action: Callable = None):
        """Анимация работы метода Нелдера-Мида

        Args:
            interval: интервал между кадрами
            blit: True - будут перерисовываться
             только сильно меняющиеся объекты. False - перерисовывается всё.
            action: действие в конце каждой итерации. Функция должна принимать значение кадра
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

        Args:
            path: путь к файлу
        """
        self.__anim.save(path, writer="imagemagick")

    def __save_points(self, method: NelderMead):
        """Сохранение трёх точек симплекса"""
        sim = method.simplex
        best, good = sim.best[0].values, sim.good[0].values
        worst = sim.worst[0].values
        self.__all_x.append([best[0], good[0], worst[0], best[0]])
        self.__all_y.append([best[1], good[1], worst[1], best[1]])

    def __init_animation(self) -> Tuple[matplotlib.pyplot.Line2D]:
        """Инициализация первоначальных данных для анимации

        Returns:
            Линия на графике
        """
        self.__plot.line.set_data([], [])
        return (self.__plot.line,)

    def __animation(self, frame: int) -> Tuple[matplotlib.pyplot.Line2D]:
        """Функция анимации, вызывается каждый кадр

        Args:
            frame: кадр анимации

        Returns:
            Линия на графике
        """
        index = frame
        if index > len(self.__all_x) - 1:
            index = len(self.__all_x) - 1
        if self.__action is not None:
            self.__action(frame)
        self.__plot.line.set_data(self.__all_x[index], self.__all_y[index])
        return (self.__plot.line,)
