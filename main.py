"""
Модуль, демонстрирующий работу с классом NelderMead
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functions import Rosenbroke
from nelder_mead import NelderMead

fig, ax = plt.figure(), plt.axes(xlim=(-5, 11), ylim=(0, 14))
ln, = ax.plot([], [], lw=1)
all_x, all_y = [], []


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
    ln.set_data(all_x[index], all_y[index])
    return (ln,)


def main():
    """Основная функция, запускающая метод и анимации"""
    plt.title("Симплекс для функции Розенброка, 10 шагов")
    plt.plot(1, 1, "ro")
    all_x.append([10, 3, 5, 10])
    all_y.append([2, 5, 8, 2])
    method = NelderMead(max_steps=10)
    method.fit(Rosenbroke(), [[10, 2], [3, 5], [5, 8]])
    method.run(action=save_points)
    anim = FuncAnimation(fig, animation, init_func=init_animation, frames=20,
                         interval=300, blit=True)
    anim.resume()
    # Для сохранения анимации в файл, уберите комментарии ниже :)
    # f = r"c://Users/Hasan/Desktop/animation_10.gif"
    # anim.save(f, writer="imagemagick")
    plt.show()


if __name__ == "__main__":
    main()
