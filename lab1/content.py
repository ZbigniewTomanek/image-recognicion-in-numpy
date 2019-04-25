# --------------------------------------------------------------------------
# -----------------------  Rozpoznawanie Obrazow  --------------------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, P. Dąbrowski
#  2018
# --------------------------------------------------------------------------

import numpy as np
from numpy import linalg as la
import pickle


def mean_squared_error(x, y, w):
    """
    :param x: ciag wejsciowy Nx1
    :param y: ciag wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: blad sredniokwadratowy pomiedzy wyjsciami y
    oraz wyjsciami uzyskanymi z wielowamiu o parametrach w dla wejsc x
    """
    d_matrix = design_matrix(x, len(w)-1)
    y_ = d_matrix @ w
    return (1.0/len(x)) * la.norm(y - y_)**2


def design_matrix(x_train, M):
    """
    :param x_train: ciag treningowy Nx1
    :param M: stopien wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzedu M
    """
    return np.array([[float(np.power(x, j)) for j in range(0, M+1)] for x in x_train])


def least_squares(x_train, y_train, M):
    """
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu, a err blad sredniokwadratowy
    dopasowania
    """
    d_matrix = design_matrix(x_train, M)
    inv_matrix = la.inv(d_matrix.transpose() @ d_matrix)

    w = inv_matrix @ d_matrix.transpose() @ y_train
    err = mean_squared_error(x_train, y_train, w)

    return w, err


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    """
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu zgodnie z kryterium z regularyzacja l2,
    a err blad sredniokwadratowy dopasowania
    """
    d_matrix = design_matrix(x_train, M)

    t_matrix = d_matrix.transpose() @ d_matrix
    i_matrix = regularization_lambda * np.identity(len(t_matrix))
    inv_matrix = la.inv(t_matrix + i_matrix)

    w = inv_matrix @ d_matrix.transpose() @ y_train
    err = mean_squared_error(x_train, y_train, w)

    return w, err


def model_selection(x_train, y_train, x_val, y_val, M_values):
    """
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, ktore maja byc sprawdzone
    :return: funkcja zwraca krotke (w,train_err,val_err), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym, train_err i val_err to bledy na sredniokwadratowe na ciagach treningowym
    i walidacyjnym
    """
    best_params = (None, np.inf, np.inf)  # 1 - train_err, 2 - val_err
    for M in M_values:
        w, t_err = least_squares(x_train, y_train, M)
        v_err = mean_squared_error(x_val, y_val, w)

        if v_err < best_params[2]:
            print('found better model for M =', M)
            best_params = (w, t_err, v_err)

    return best_params


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    """
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M: stopien wielomianu
    :param lambda_values: lista ze wartosciami roznych parametrow regularyzacji
    :return: funkcja zwraca krotke (w,train_err,val_err,regularization_lambda), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym. Wielomian dopasowany jest wg kryterium z regularyzacja. train_err i val_err to
    bledy na sredniokwadratowe na ciagach treningowym i walidacyjnym. regularization_lambda to najlepsza wartosc parametru regularyzacji
    """
    best_params = (None, np.inf, np.inf, np.inf)  # 1 - train_err, 2 - val_err, 3 - regularization_lambda
    for r_lambda in lambda_values:
        w, t_err = regularized_least_squares(x_train, y_train, M, r_lambda)
        v_err = mean_squared_error(x_val, y_val, w)

        if v_err < best_params[2]:
            print('found better model for lambda =', r_lambda)
            best_params = (w, t_err, v_err, r_lambda)

    return best_params


if __name__ == '__main__':
    with open('test_data.pkl', mode='rb') as file_:
        TEST_DATA = pickle.load(file_)

    x = TEST_DATA['mean_error']['x']
    y = TEST_DATA['mean_error']['y']
    w = TEST_DATA['mean_error']['w']
    err_expected = TEST_DATA['mean_error']['err']
    err = mean_squared_error(x, y, w)

    x_train = TEST_DATA['design_matrix']['x_train']
    M = TEST_DATA['design_matrix']['M']
    dm_expected = TEST_DATA['design_matrix']['dm']
    dm = design_matrix(x_train, M)

    x_train = TEST_DATA['ls']['x_train']
    y_train = TEST_DATA['ls']['y_train']
    M = TEST_DATA['ls']['M']
    w_expected = TEST_DATA['ls']['w']
    w, _ = least_squares(x_train, y_train, M)

    x_train = TEST_DATA['rls']['x_train']
    y_train = TEST_DATA['rls']['y_train']
    M = TEST_DATA['rls']['M']
    err_expected = TEST_DATA['rls']['err']
    regularization_lambda = TEST_DATA['rls']['lambda']

    _, err = regularized_least_squares(x_train, y_train, M, regularization_lambda)
    print(err_expected, err)
