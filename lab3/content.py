# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 3: Regresja logistyczna
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, P. Dąbrowski
#  2019
# --------------------------------------------------------------------------

import numpy as np


def sigmoid(x):
    """
    Wylicz wartość funkcji sigmoidalnej dla punktów *x*.

    :param x: wektor wartości *x* do zaaplikowania funkcji sigmoidalnej Nx1
    :return: wektor wartości funkcji sigmoidalnej dla wartości *x* Nx1
    """
    return np.array([1.0 / (1 + np.exp(-i)) for i in x])


def logistic_cost_function(w, x_train, y_train):
    """
    Wylicz wartość funkcji logistycznej oraz jej gradient po parametrach.

    :param w: wektor parametrów modelu Mx1
    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :return: krotka (log, grad), gdzie *log* to wartość funkcji logistycznej,
        a *grad* jej gradient po parametrach *w* Mx1
    """
    N = len(y_train)
    M = len(w)

    ln_p_d_w = 0.0
    for n in range(N):
        sigm = sigmoid(w.transpose() @ x_train[n])
        ln_p_d_w += y_train[n] * np.log(sigm) + (1 - y_train[n]) * np.log(1 - sigm)

    log = -ln_p_d_w / N

    grad = 0.0
    for n in range(N):
        grad += (sigmoid(w.transpose() @ x_train[n]) - y_train[n]) * x_train[n]

    grad /= N
    return log, grad.reshape(M, 1)


def gradient_descent(obj_fun, w0, epochs, eta):
    """
    Dokonaj *epochs* aktualizacji parametrów modelu metodą algorytmu gradientu
    prostego, korzystając z kroku uczenia *eta* i zaczynając od parametrów *w0*.
    Wylicz wartość funkcji celu *obj_fun* w każdej iteracji. Wyznacz wartość
    parametrów modelu w ostatniej epoce.

    :param obj_fun: optymalizowana funkcja celu, przyjmująca jako argument
        wektor parametrów *w* [wywołanie *val, grad = obj_fun(w)*]
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok algorytmu gradientu prostego
    :param eta: krok uczenia
    :return: krotka (w, log_values), gdzie *w* to znaleziony optymalny
        punkt *w*, a *log_values* to lista wartości funkcji celu w każdej
        epoce (lista o długości *epochs*)
    """
    w = w0
    log_values = []
    val, grad = obj_fun(w)

    for e in range(epochs):
        w -= eta * grad
        val, grad = obj_fun(w)
        log_values.append(float(val))

    return w, log_values


def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    """
    Dokonaj *epochs* aktualizacji parametrów modelu metodą stochastycznego
    algorytmu gradientu prostego, korzystając z kroku uczenia *eta*, paczek
    danych o rozmiarze *mini_batch* i zaczynając od parametrów *w0*. Wylicz
    wartość funkcji celu *obj_fun* w każdej iteracji. Wyznacz wartość parametrów
    modelu w ostatniej epoce.

    :param obj_fun: optymalizowana funkcja celu, przyjmująca jako argumenty
        wektor parametrów *w*, paczkę danych składających się z danych
        treningowych *x* i odpowiadających im etykiet *y*
        [wywołanie *val, grad = obj_fun(w, x, y)*]
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok stochastycznego algorytmu gradientu prostego
    :param eta: krok uczenia
    :param mini_batch: rozmiar paczki danych / mini-batcha
    :return: krotka (w, log_values), gdzie *w* to znaleziony optymalny
        punkt *w*, a *log_values* to lista wartości funkcji celu dla całego
        zbioru treningowego w każdej epoce (lista o długości *epochs*)
    """

    M = int(len(x_train) / mini_batch)
    x_train_nb = x_train.reshape(M, mini_batch, len(x_train[0]))
    y_train_nb = y_train.reshape(M, mini_batch, len(y_train[0]))

    w = w0.copy()
    log_values = []

    for e in range(epochs):
        for m in range(M):
            val, grad = obj_fun(w, x_train_nb[m], y_train_nb[m])
            w -= eta * grad

        val, _ = obj_fun(w, x_train, y_train)
        log_values.append(float(val.copy()))

    return w, log_values


def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    """
    Wylicz wartość funkcji logistycznej z regularyzacją l2 oraz jej gradient
    po parametrach.

    :param w: wektor parametrów modelu Mx1
    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :param regularization_lambda: parametr regularyzacji l2
    :return: krotka (log, grad), gdzie *log* to wartość funkcji logistycznej
        z regularyzacją l2, a *grad* jej6 gradient po parametrach *w* Mx1
    """
    val, grad = logistic_cost_function(w, x_train, y_train)

    delta_g = regularization_lambda * w
    delta_g[0] = 0

    delta_v = regularization_lambda / 2 * np.linalg.norm(w[1:]) ** 2
    return val + delta_v, grad + delta_g


def prediction(x, w, theta):
    """
    Wylicz wartości predykowanych etykiet dla obserwacji *x*, korzystając
    z modelu o parametrach *w* i progu klasyfikacji *theta*.

    :param x: macierz obserwacji NxM
    :param w: wektor parametrów modelu Mx1
    :param theta: próg klasyfikacji z przedziału [0,1]
    :return: wektor predykowanych etykiet ze zbioru {0, 1} Nx1
    """
    return np.array([float(sigmoid(w.transpose() @ row)) >= theta for row in x]).reshape(len(x), 1)


def f_measure(y_true, y_pred):
    """
    Wylicz wartość miary F (F-measure) dla zadanych rzeczywistych etykiet
    *y_true* i odpowiadających im predykowanych etykiet *y_pred*.

    :param y_true: wektor rzeczywistych etykiet Nx1
    :param y_pred: wektor etykiet predykowanych przed model Nx1
    :return: wartość miary F (F-measure)
    """
    N = len(y_true)

    TP = sum([y_true[i] == y_pred[i] == 1 for i in range(N)])
    FP = sum([y_true[i] == 0 and y_pred[i] == 1 for i in range(N)])
    FN = sum([y_true[i] == 1 and y_pred[i] == 0 for i in range(N)])

    return 2 * TP / (2 * TP + FP + FN)


def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas, thetas):
    """
    Policz wartość miary F dla wszystkich kombinacji wartości regularyzacji
    *lambda* i progu klasyfikacji *theta. Wyznacz parametry *w* dla modelu
    z regularyzacją l2, który najlepiej generalizuje dane, tj. daje najmniejszy
    błąd na ciągu walidacyjnym.

    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :param x_val: zbiór danych walidacyjnych NxM
    :param y_val: etykiety klas dla danych walidacyjnych Nx1
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok stochastycznego algorytmu gradientu prostego
    :param eta: krok uczenia
    :param mini_batch: rozmiar paczki danych / mini-batcha
    :param lambdas: lista wartości parametru regularyzacji l2 *lambda*,
        które mają być sprawdzone
    :param thetas: lista wartości progów klasyfikacji *theta*,
        które mają być sprawdzone
    :return: krotka (regularization_lambda, theta, w, F), gdzie
        *regularization_lambda* to wartość regularyzacji *lambda* dla
        najlepszego modelu, *theta* to najlepszy próg klasyfikacji,
        *w* to parametry najlepszego modelu, a *F* to macierz wartości miary F
        dla wszystkich par *(lambda, theta)* #lambda x #theta
    """
    L = len(lambdas)
    T = len(thetas)

    f_measures = np.zeros(shape=(L, T))
    ws = []

    for i in range(L):
        def obj_fun(wl, xl, yl):
            return regularized_logistic_cost_function(wl, xl, yl, lambdas[i])

        w, _ = stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch)
        ws.append(w.copy())

        for j in range(T):
            y_pred = prediction(x_val, w, thetas[j])
            f_measures[i][j] = f_measure(y_val, y_pred).copy()

    index = f_measures.argmax()

    x = index % T
    y = int(index / T)

    return lambdas[y], thetas[x], ws[y], f_measures
