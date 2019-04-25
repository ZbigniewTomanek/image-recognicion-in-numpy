# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, P. Dąbrowski
#  2019
# --------------------------------------------------------------------------

import numpy as np
from scipy.spatial.distance import cdist

M = 4  # liczba klas przyjęta w zadaniu


def hamming_distance(X, X_train):
    """
    Zwróć odległość Hamminga dla obiektów ze zbioru *X* od obiektów z *X_train*.

    :param X: zbiór porównywanych obiektów N1xD
    :param X_train: zbiór obiektów do których porównujemy N2xD
    :return: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    """

    X = X.toarray()
    X_train = X_train.toarray()
    X_train = X_train.transpose()

    ones = X.astype(np.uint8) @ X_train.astype(np.uint8)
    zeros = (~X).astype(np.uint8) @ (~X_train).astype(np.uint8)

    # cdist(X.todense(), X_train.todense(), 'hamming') * X.shape[1] oneliner xD

    return (np.ones(shape=ones.shape) * X.shape[1]) - ones - zeros



def sort_train_labels_knn(Dist, y):
    """
    Posortuj etykiety klas danych treningowych *y* względem prawdopodobieństw
    zawartych w macierzy *Dist*.

    :param Dist: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    :param y: wektor etykiet o długości N2
    :return: macierz etykiet klas posortowana względem wartości podobieństw
        odpowiadającego wiersza macierzy Dist N1xN2

    Do sortowania użyj algorytmu mergesort.
    """

    mtx = np.zeros(shape=(len(Dist), len(Dist[0])))
    for i in range(len(Dist)):
        indexes = np.argsort(Dist[i], kind='mergesort')
        for j in range(len(Dist[i])):
            mtx[i][j] = y[indexes[j]]

    return mtx.astype(int)


def p_y_x_knn(y, k):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) każdej z klas dla obiektów
    ze zbioru testowego wykorzystując klasyfikator KNN wyuczony na danych
    treningowych.

    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najbliższych sasiadow dla KNN
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" N1xM
    """
    N = len(y)

    dist = np.zeros(shape=(N, M))
    for n in range(N):
        for m in range(M):
            dist[n][m] = sum(num == m for num in y[n][:k])/k

    return dist


def classification_error(p_y_x, y_true):
    """
    Wyznacz błąd klasyfikacji.

    :param p_y_x: macierz przewidywanych prawdopodobieństw - każdy wiersz
        macierzy reprezentuje rozkład p(y|x) NxM
    :param y_true: zbiór rzeczywistych etykiet klas 1xN
    :return: błąd klasyfikacji
    """

    err = 0

    N = len(p_y_x)
    M = len(p_y_x[0])

    for n in range(N):
        index = M - p_y_x[n][::-1].argmax() - 1
        err += index != y_true[n]

    return err / N


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    Wylicz bład dla różnych wartości *k*. Dokonaj selekcji modelu KNN
    wyznaczając najlepszą wartość *k*, tj. taką, dla której wartość błędu jest
    najniższa.

    :param X_val: zbiór danych walidacyjnych N1xD
    :param X_train: zbiór danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartości parametru k, które mają zostać sprawdzone
    :return: krotka (best_error, best_k, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_k" to "k" dla którego błąd był
        najniższy, a "errors" - lista wartości błędów dla kolejnych
        "k" z "k_values"
    """
    dist = hamming_distance(X_val, X_train)
    labels = sort_train_labels_knn(dist, y_train)

    errors = [classification_error(p_y_x_knn(labels, k), y_val) for k in k_values]

    index = np.argmin(errors)
    best_err = errors[index]
    best_k = k_values[index]

    return best_err, best_k, errors


def estimate_a_priori_nb(y_train):
    """
    Wyznacz rozkład a priori p(y) każdej z klas dla obiektów ze zbioru
    treningowego.

    :param y_train: etykiety dla danych treningowych 1xN
    :return: wektor prawdopodobieństw a priori p(y) 1xM
    """

    _, counts = np.unique(y_train, return_counts=True)
    return [count / len(y_train) for count in counts]


def estimate_p_x_y_nb(X_train, y_train, a, b):
    """
    Wyznacz rozkład prawdopodobieństwa p(x|y) zakładając, że *x* przyjmuje
    wartości binarne i że elementy *x* są od siebie niezależne.

    :param X_train: dane treningowe NxD
    :param y_train: etykiety klas dla danych treningowych 1xN
    :param a: parametr "a" rozkładu Beta
    :param b: parametr "b" rozkładu Beta
    :return: macierz prawdopodobieństw p(x|y) dla obiektów z "X_train" MxD.
    """
    X_train = np.array(X_train.todense())
    D = len(X_train[1])
    N = len(X_train)
    res = np.zeros(shape=(M, D))

    for k in range(M):
        for d in range(D):
            ctr = sum(float(y_train[n] == k and X_train[n][d]) for n in range(N)) + a - 1
            denom = sum(float(y_train[n] == k) for n in range(N)) + a + b - 2
            res[k][d] = ctr / denom

    return res


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) dla każdej z klas z wykorzystaniem
    klasyfikatora Naiwnego Bayesa.

    :param p_y: wektor prawdopodobieństw a priori 1xM
    :param p_x_1_y: rozkład prawdopodobieństw p(x=1|y) MxD
    :param X: dane dla których beda wyznaczone prawdopodobieństwa, macierz NxD
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" NxM
    """
    X = np.array(X.todense())
    N = len(X)
    D = len(X[0])
    p_y = np.array(p_y)

    p_y_x = np.zeros(shape=(N, M))
    for n in range(N):
        p_x_y = np.array([1., 1., 1., 1.])
        for d in range(D):
            for m in range(M):
                p_x_y[m] *= (p_x_1_y[m][d]**X[n][d]) * ((1 - p_x_1_y[m][d])**(1 - X[n][d]))
        p_x_y *= p_y

        for m in range(M):
            p_y_x[n][m] = p_x_y[m] / sum(p_x_y)

    return p_y_x


def model_selection_nb(X_train, X_val, y_train, y_val, a_values, b_values):
    """
    Wylicz bład dla różnych wartości *a* i *b*. Dokonaj selekcji modelu Naiwnego
    Byesa, wyznaczając najlepszą parę wartości *a* i *b*, tj. taką, dla której
    wartość błędu jest najniższa.
    
    :param X_train: zbiór danych treningowych N2xD
    :param X_val: zbiór danych walidacyjnych N1xD
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrów "a" do sprawdzenia
    :param b_values: lista parametrów "b" do sprawdzenia
    :return: krotka (best_error, best_a, best_b, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_a" i "best_b" to para parametrów
        "a" i "b" dla której błąd był najniższy, a "errors" - lista wartości
        błędów dla wszystkich kombinacji wartości "a" i "b" (w kolejności
        iterowania najpierw po "a_values" [pętla zewnętrzna], a następnie
        "b_values" [pętla wewnętrzna]).
    """
    A = len(a_values)
    B = len(b_values)

    errors = np.zeros(shape=(A, B))

    for i in range(A):
        for j in range(B):
            a_priori_nb = estimate_a_priori_nb(y_train)
            p_x_y_nb = estimate_p_x_y_nb(X_train, y_train, a_values[i], b_values[j])
            p_y_x = p_y_x_nb(a_priori_nb, p_x_y_nb, X_val)
            errors[i][j] = classification_error(p_y_x, y_val)

    index = np.where(errors == np.amin(errors))
    crds = list(zip(index[0], index[1])) # lista zawierająca koordynanty najnmniejszego błędu w macierzy errors

    return np.amin(errors), a_values[crds[0][0]], b_values[crds[0][1]], errors
