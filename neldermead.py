import numpy as np
from functions import BaseFunction


class NelderMead:
    def __init__(self, *, alpha=1, betta=0.5, gamma=2, max_steps=1000,
                 eps0=0.001, max_blank=10, eps1=0.001):
        self.__alpha = alpha
        self.__betta = betta
        self.__gamma = gamma
        self.__max_steps = max_steps
        self.__eps0 = eps0
        self.__eps1 = eps1
        self.__max_blank = max_blank
        self.__simplex = list()
        self.__function = None

    @property
    def params(self):
        return {"alpha": self.__alpha,
                "betta": self.__betta,
                "gamma": self.__gamma,
                "max_steps": self.__max_steps,
                "eps0": self.__eps0,
                "eps1": self.__eps1,
                "max_blank": self.__max_blank}

    @property
    def function(self):
        return self.__function

    @property
    def simplex(self):
        return [list(point) for (point, value) in self.__simplex]

    def fit(self, function: BaseFunction, simplex: list = None):
        if not isinstance(function, BaseFunction):
            raise AttributeError("function not an instance of BaseFunction")
        self.__function = function
        if simplex is None:
            self.__create_simplex()
        else:
            self.__simplex = simplex
            self.__check_simp()
            self.__simplex_modify()
        self.__simplex = self.__make_points()

    def run(self, *, action=None):
        if not isinstance(self.__function, BaseFunction):
            raise AttributeError("No function in class, use fit method")
        iteration = 0
        while iteration <= self.__max_steps:
            self.__simplex.sort(key=lambda x: x[1])
            points = [point for (point, value) in self.__simplex]

            centroid = 1 / len(points[:-1]) * sum(points[:-1])
            best, good = self.__simplex[0], self.__simplex[-2]
            worst = self.__simplex[-1]
            reflected = self.__reflection(centroid)

            if reflected[1] < best[1]:
                self.__expansion(centroid, reflected)
            elif reflected[1] < good[1]:
                self.__simplex[-1] = reflected
            else:
                if reflected[1] < worst[1]:
                    self.__simplex[-1] = reflected
                self.__contraction(centroid)
            if callable(action):
                action(self)
            if self.__stop():
                break
            iteration += 1
        return self.__simplex[0][1]

    def __check_simp(self):
        if not isinstance(self.__simplex, list):
            raise AttributeError("Simplex is not list")
        elif len(self.__simplex) != self.__function.dimension + 1:
            raise AttributeError("Simplex length less than dimension + 1")
        for point in self.__simplex:
            if len(point) != self.__function.dimension:
                raise AttributeError(f"{point} length less than dimension")

    def __create_simplex(self):
        self.__simplex = [np.array([0]*self.__function.dimension)]
        for point in range(self.__function.dimension):
            prev_point = self.__simplex[point]
            new_point = prev_point
            while np.equal(prev_point, new_point).all():
                new_point = np.random.randint(low=(min(prev_point) - 1),
                                              high=(max(prev_point) + 1),
                                              size=self.__function.dimension)
            self.__simplex.append(new_point)

    def __simplex_modify(self):
        for index, value in enumerate(self.__simplex):
            self.__simplex[index] = np.array(value)

    def __make_points(self):
        points_with_values = list()
        for point in self.__simplex:
            value = self.__function.calculate(list(point))
            points_with_values.append((point, value))
        return points_with_values

    def __reflection(self, centroid):
        worst = self.__simplex[-1]
        reflected = (1 + self.__alpha) * centroid - self.__alpha * worst[0]
        return reflected, self.__function.calculate(list(reflected))

    def __expansion(self, centroid, reflected):
        expanded = (1-self.__gamma) * centroid + self.__gamma * reflected[0]
        value = self.__function.calculate(list(expanded))
        if value <= reflected[1]:
            self.__simplex[-1] = (expanded, value)
        else:
            self.__simplex[-1] = reflected

    def __contraction(self, centroid):
        worst = self.__simplex[-1]
        condensed = (1-self.__betta) * centroid + self.__betta * worst[0]
        value = self.__function.calculate(list(condensed))
        if value < worst[1]:
            self.__simplex[-1] = (condensed, value)
        else:
            self.__global_contraction()

    def __global_contraction(self):
        best = self.__simplex[0][0]
        for index, (point, _) in enumerate(self.__simplex[1:]):
            new_point = best + (point - best)/2
            new_value = self.__function.calculate(list(new_point))
            self.__simplex[index] = (new_point, new_value)

    def __stop(self):
        points = [point for (point, value) in self.__simplex]
        dispersion = np.var(points)
        if dispersion < self.__eps0:
            return True
        else:
            return False
