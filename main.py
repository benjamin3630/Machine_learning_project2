import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier 


from sklearn.tree import DecisionTreeClassifier

def P(x, beta=5):
    """Returns the probability that y=1
    Input x is an array of last dimension of size 2
    Output is an array of the same shape but the last dimension size is 1
    """
    x = np.asarray(x) # allows to use a basic, 1D, python list like [0.3, 0.6] as an input
    x1 = x[...,0] # ... allows to use any array shape, and the output will follow the same array shape
    x2 = x[...,1]
    #print(x1)
    return 1 / (1 + np.exp(-beta*(x2-np.sin(np.pi* x1))))




def make_multiple_datasets(number_LS=10, size_LS=15, beta=5, rng=None):
    """Returns X_ls with shape (number_LS, size_LS, 2) and y_ls with shape (number_LS, size_LS) """

    if rng is None:
        rng = np.random.default_rng()
    X_ls = rng.uniform(-1, 1, (number_LS, size_LS, 2))
    probs = P(X_ls, beta) # Marche directement si P utilise des fonctions NumPy
    y_ls = rng.binomial(1, probs)
    return X_ls, y_ls


def make_dataset(size_LS=250, beta=5, rng=None): # return X_ls with shape (size_LS, 2) and y_ls with shape (size_LS)
    X, y =  make_multiple_datasets(number_LS=1, size_LS=size_LS, beta=beta, rng=rng)
    return X[0], y[0]



def nn(X_train, y_train, rng):
    clf = MLPClassifier(solver = "lbfgs", alpha = 10**-5, max_iter = 2000, random_state=get_random_seed(rng))

    return clf.fit(X_train, y_train)


def knn(X_train, y_train, rng):
    n_neighbors = 5
    
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    return clf.fit(X_train, y_train)


def dtree(X_train, y_train, rng):
    depth = 6
    
    clf = DecisionTreeClassifier(max_depth=depth, random_state=get_random_seed(rng))
    return clf.fit(X_train, y_train)


def get_random_seed(rng):
    return int(rng.integers(0, 2**31 - 1))


"""
def noise(x_test, x_train, y_train, rng, model=nn, beta=5): # x_train of shape (size_LS, 2) and y_train of shape (size_LS)

    # fonction à revoir car je ne sais pas s'il faut prendre juste x_test ou bien x_train en fait  ==> pas sur que le noise puisse réellement être calculé comme ça

    h_b = P(x_train, beta=beta) 
    number_sample = y_train.shape[0]
    return np.sum(np.square(y_train-h_b)) / number_sample 
"""


def bias(x_test, x_train, y_train, rng, model=nn, beta=5): # x_train of shape (nb_ls, size_LS, 2) and y_train of shape (nb_ls, size_LS)
    h_B = P(x_test, beta=beta) # todo changer cette assignation arbitraire de beta : soit on donne P dans les arguments soit beta
    E_LS = compute_E_LS_y(x_test, x_train, y_train, rng, model=model)
    
    bias_sqr = (h_B - E_LS)**2
    return bias_sqr


def compute_E_LS_y(x_test, x_train, y_train, rng, model=nn, beta=5):
    n_LS = np.shape(y_train)[0]
    sum_f = 0
    for i in range(np.shape(x_train)[0]):
        m = model(x_train[i],y_train[i], rng=rng)
        sum_f += m.predict_proba(x_test)[:,1] # predict proba gives [ [p1, p2]] where p1 is proba of having a class m.classes_[0] and p1 the proba of having a class m.classes[1]. Here m.classes_=[0,1] thus we use 1
    
    E_LS = sum_f / n_LS

    return E_LS


def variance(x_test, x_train, y_train, rng, model=nn, beta=5):
    n_LS = np.shape(y_train)[0]

    pred = np.zeros((y_train.shape[0], np.shape(x_test)[0])) # shape de x_test permet de fonctionner que x_test contiennent une data ou plusieurs

    for i in range(np.shape(x_train)[0]):
        m = model(x_train[i],y_train[i], rng=rng)
        pred[i, :] = m.predict_proba(x_test)[:,1]

    E_LS = np.mean(pred)

    return np.mean(np.square(pred - E_LS), axis = 0) # sommer sur l'axe 0 pour que ce soit bien sur pred qu'on somme (sinon on risque de sommer tout quand on envoie plusieurs x_test d'un coup)

    




def main():
    # Creation de plusieurs data set (avec des beta = 5 et N = 250)
    rng = np.random.default_rng(seed=42)
    nb_LS = 10
    size_LS = 15
    beta=5

    x1, y1 = make_multiple_datasets(number_LS=nb_LS, size_LS=size_LS, beta=beta, rng=rng)

    x_test = np.array([[0.6,0.5], [0.2, 0.3]])

    test = bias(x_test, x_train=x1, y_train=y1, rng=rng, model=nn)
    print(test)

    """x_a, y_a = make_dataset(size_LS=size_LS, beta=beta, rng=rng)

    a = noise(x_test, x_train=x_a, y_train=y_a, rng=rng, model=nn)"""

    test2 = variance(x_test, x_train=x1, y_train=y1, rng=rng, model=nn)

    print(test2)
    


if __name__ == "__main__":
    main()