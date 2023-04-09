"""
Модуль, демонстрирующий работу с классом NelderMead
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from sympy.utilities.lambdify import lambdify
from scripts.functions import Rosenbroke
from scripts.nelder_mead import NelderMead, Point


class NelderMead2DAnimation:
    def __init__(self, method: NelderMead):
        if not isinstance(method, NelderMead):
            raise AttributeError("method must be a NelderMean")
        self.__anim = None
        self.__y_limits = None
        self.__x_limits = None
        self.__text_on_plot = None
        self.__ln = None
        self.__ax = None
        self.__fig = None
        self.__method = method
        self.__all_x = []
        self.__all_y = []

    def __save_points(self, method: NelderMead):
        """Сохранение всех точек симплекса"""
        sim = method.simplex
        best, good = sim.best[0].values, sim.good[0].values
        worst = sim.worst[0].values
        self.__all_x.append([best[0], good[0], worst[0], best[0]])
        self.__all_y.append([best[1], good[1], worst[1], best[1]])

    def __init_animation(self) -> tuple:
        """Инициализация первоначальных данных для анимации

        :return: кортеж из одного элемента - линии на графике
        """
        self.__ln.set_data([], [])
        return (self.__ln,)

    def __animation(self, frame: int) -> tuple:
        """Функция анимации, вызывается каждый кадр

        :param frame: кадр анимации

        :return: кортеж из одного элемента - линии на графике
        """

        index = frame
        if index > len(self.__all_x) - 1:
            index = len(self.__all_x) - 1
        self.__text_on_plot.set_text(f"Итерация: {index + 1}")
        self.__ln.set_data(self.__all_x[index], self.__all_y[index])
        return (self.__ln,)

    def plot_init(self, title: str, true_point: Point, x_limits: tuple, y_limits: tuple,
                  line_width: float = 1, fmt: str = "ro"):
        if not isinstance(x_limits, tuple):
            raise AttributeError("x_limits must be a tuple")
        if not isinstance(y_limits, tuple):
            raise AttributeError("y_limits must be a tuple")
        self.__fig = plt.figure()
        self.__ax = plt.axes(xlim=x_limits, ylim=y_limits)
        self.__ln, = self.__ax.plot([], [], lw=line_width)
        self.__x_limits = x_limits
        self.__y_limits = y_limits
        plt.title(title)
        plt.plot(true_point.values[0], true_point.values[1], fmt)

    def text_on_plot_init(self, x_coor: float, y_coor: float,
                          font_size: int = 13):
        self.__text_on_plot = plt.text(x_coor, y_coor, "None",
                                       fontsize=font_size)

    def contour_init(self,  x_step: float = 0.05, y_step: float = 0.05,
                     levels=10):
        xgrid, ygrid, zgrid = self.__plot_contour(x_step, y_step)
        self.__ax.contour(xgrid, ygrid, zgrid, levels=levels)

    def __plot_contour(self, step_x: float, step_y: float):
        """Отрисовка линий уровня функции с двумя(!) переменными

        :param step_x: шаг, с которым генерируются точки по x
        :param step_y: шаг, с которым генерируются точки по y
        :return: кортеж из numpy.ndarray, используемых в contour
        """
        x_limits, y_limits = self.__x_limits, self.__y_limits
        function = self.__method.function
        x_data = np.arange(x_limits[0], x_limits[1], step_x)
        y_data = np.arange(y_limits[0], y_limits[1], step_y)
        nm_func = lambdify(function.variables, function.expr, "numpy")
        xgrid, ygrid = np.meshgrid(x_data, y_data)
        zgrid = nm_func(xgrid, ygrid)
        return xgrid, ygrid, zgrid

    def animate(self, interval: int = 300, blit: bool = False):
        self.__save_points(self.__method)
        self.__method.run(action=self.__save_points)
        self.__anim = FuncAnimation(self.__fig, self.__animation,
                                    init_func=self.__init_animation,
                                    frames=len(self.__all_x),
                                    interval=interval, blit=blit)

    def save(self, path: str):
        self.__anim.save(path, writer="imagemagick")


def main():
    """Основная функция, запускающая метод и анимации"""
    method = NelderMead(max_steps=100, max_blank=10, eps1=0.001)
    method.fit(Rosenbroke(), Point(10, 2), Point(3, 5), Point(5, 8))
    title = "Симплекс для функции Розенброка"
    anim = NelderMead2DAnimation(method)
    anim.plot_init(title, Point(1, 1), (-15, 15), (-1, 15), fmt="ro")
    anim.contour_init()
    anim.animate()
    anim.text_on_plot_init(-15, 16.5)
    plt.show()


if __name__ == "__main__":
    main()
