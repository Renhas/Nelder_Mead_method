"""
Модуль, демонстрирующий работу с классом NelderMead
"""
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.animation import FuncAnimation
import numpy as np
from sympy.utilities.lambdify import lambdify
from functions import Rosenbroke, BaseFunction
from nelder_mead import NelderMead

plt.style.use("dark_background")
x_limits, y_limits = (-15, 15), (-1, 15)
fig, ax = plt.figure(), plt.axes(xlim=x_limits, ylim=y_limits)
ln, = ax.plot([], [], lw=1)
all_x, all_y = [], []
text_on_plot: Text = plt.text(-15, 16.5, "33", fontsize=13)


def save_points(method: NelderMead):
    """Сохранение всех точек симплекса

    :param method: экземпляр класса NelderMead
    """
    points = method.simplex
    best, good, worst = points[0], points[-2], points[-1]
    all_x.append([best[0], good[0], worst[0], best[0]])
    all_y.append([best[1], good[1], worst[1], best[1]])


def init_animation() -> tuple:
    """Инициализация первоначальных данных для анимации

    :return: кортеж из одного элемента - линии на графике
    """
    ln.set_data([], [])
    return (ln,)


def animation(frame: int) -> tuple:
    """Функция анимации, вызывается каждый кадр

    :param frame: кадр анимации

    :return: кортеж из одного элемента - линии на графике
    """

    index = frame
    if index > len(all_x) - 1:
        index = len(all_x) - 1
    text_on_plot.set_text(f"Итерация: {index+1}")
    ln.set_data(all_x[index], all_y[index])
    return (ln,)


def plot_contour(function: BaseFunction):
    """Отрисовка линий уровня функции с двумя(!) переменными

    :param function: функция для анализа
    :return: кортеж из numpy.ndarray, используемых в contour
    """
    x_data = np.arange(x_limits[0], x_limits[1], 0.05)
    y_data = np.arange(y_limits[0], y_limits[1], 0.05)
    nm_func = lambdify(function.variables, function.expr, "numpy")
    xgrid, ygrid = np.meshgrid(x_data, y_data)
    zgrid = nm_func(xgrid, ygrid)
    return xgrid, ygrid, zgrid


def main():
    """Основная функция, запускающая метод и анимации"""
    method = NelderMead(max_steps=100, max_blank=10, eps1=0.001)
    method.fit(Rosenbroke(), [[10, 2], [3, 5], [5, 8]])
    method.run(action=save_points)
    plt.title("Симплекс для функции Розенброка")
    plt.plot(1, 1, "ro")
    xgrid, ygrid, zgrid = plot_contour(Rosenbroke())
    ax.contour(xgrid, ygrid, zgrid, levels=10)
    all_x.append([10, 3, 5, 10])
    all_y.append([2, 5, 8, 2])

    anim = FuncAnimation(fig, animation, init_func=init_animation,
                         frames=len(all_x), interval=300, blit=False)
    anim.resume()
    # Для сохранения анимации в файл, уберите комментарии ниже :)
    # file = r"gifs/animation_100_dark.gif"
    # anim.save(file, writer="imagemagick")
    plt.show()


if __name__ == "__main__":
    main()
