import numpy as np
import functions


class Nelder_Mead:
    __function = None

    def __init__(self, *, alpha=1, betta=0.5, gamma=2, max_steps=1000,
                 eps0=0.001, max_blank=10, eps1=0.001):
        self.__alpha = alpha
        self.__betta = betta
        self.__gamma = gamma
        self.__max_steps = max_steps
        self.__eps0 = eps0
        self.__eps1 = eps1
        self.__max_blank = max_blank

    @property
    def params(self):
        return {"alpha": self.__alpha,
                "betta": self.__betta,
                "gamma": self.__gamma,
                "max_steps": self.__max_steps,
                "eps0": self.__eps0,
                "eps1": self.__eps1,
                "max_blank": self.__max_blank}
