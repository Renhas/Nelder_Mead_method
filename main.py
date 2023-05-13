"""
Модуль, демонстрирующий работу с классом NelderMead
"""
from scripts.utilities.functions import Rosenbroke
from scripts.utilities.point import Point
from scripts.nelder_mead import NelderMead
from scripts.addons.nm_visualization import PlotSettings, NelderMead2DAnimation


def main():
    """Основная функция, запускающая метод и анимации"""
    method = NelderMead(max_steps=100, max_blank=10, eps1=0.001)
    method.fit(Rosenbroke(), Point(10, 2), Point(3, 5), Point(5, 8))
    title = "Симплекс для функции Розенброка"
    plot = PlotSettings(title, (-15, 15), (-1, 15))
    plot.contour_init(Rosenbroke())
    plot.plot_point(Point(1, 1), fmt="ro")
    plot.text.set_position((-15, 16.5))
    plot.text.set_size(13)
    anim = NelderMead2DAnimation(method, plot)

    def text_setter(index: int):
        plot.text.set_text(f"Итерация: {index + 1}")

    anim.animate(action=text_setter)
    #anim.save("gifs/animation_100.gif")
    plot.show()


if __name__ == "__main__":
    main()
