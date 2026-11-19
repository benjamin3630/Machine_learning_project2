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
    """
    Returns X_ls with shape (number_LS, size_LS, 2) and y_ls with shape (number_LS, size_LS)
    """
    if rng is None:
        rng = np.random.default_rng()
    X_ls = rng.uniform(-1, 1, (number_LS, size_LS, 2))
    probs = P(X_ls, beta) # Marche directement si P utilise des fonctions NumPy
    y_ls = rng.binomial(1, probs)
    return X_ls, y_ls


def make_dataset(size_LS=250, beta=5, rng=None): 
    """
    Return X_ls with shape (size_LS, 2) and y_ls with shape (size_LS)
    """
    X, y =  make_multiple_datasets(number_LS=1, size_LS=size_LS, beta=beta, rng=rng)
    return X[0], y[0]



def nn(X_train, y_train, rng):
    """Returns a trained neural network classifier"""
    clf = MLPClassifier(solver = "lbfgs", alpha = 10**-5, max_iter = 2000, random_state=get_random_seed(rng))

    return clf.fit(X_train, y_train)


def knn(X_train, y_train, rng):
    """Returns a trained k-nearest neighbors classifier"""
    n_neighbors = 5
    
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    return clf.fit(X_train, y_train)


def dtree(X_train, y_train, rng):
    """Returns a trained decision tree classifier"""
    depth = 6
    
    clf = DecisionTreeClassifier(max_depth=depth, random_state=get_random_seed(rng))
    return clf.fit(X_train, y_train)


def get_random_seed(rng):
    """Converts rng generator from numpy to a random number for the seed of sickit learn"""
    #return int(rng.integers(0, 2**31 - 1))
    return 42

    
def compute_error_params(x_test, x_train, y_train, rng, model=nn, beta=5):
    """
    Computes the residual error, the squared bias and the variance over the LS, for a given model.
    Input : 
    - x_test : array of x where the calculation should be made
    - x_train : training samples, divided in several LS, shape (number_LS, size_LS, 2)
    - y_train : training outputs, divided in several LS, shape (number_LS, size_LS)

    Outputs the computed results, with the same shape as x_test
    """
    # variance and E_LS:
    n_LS = np.shape(y_train)[0]
    pred = np.zeros((y_train.shape[0], np.shape(x_test)[0])) # shape de x_test permet de fonctionner que x_test contiennent une data ou plusieurs

    for i in range(np.shape(x_train)[0]):
        m = model(x_train[i],y_train[i], rng=rng)
        pred[i, :] = m.predict_proba(x_test)[:,1]

    E_LS = np.mean(pred, axis=0) # axis 0 pour que si on donne plusieurs x_test, E_LS garde la forme de x_test et moyenne juste les pred "element wise" pour chaque x_test

    variance = np.mean(np.square(pred - E_LS), axis = 0) # sommer sur l'axe 0 pour que ce soit bien sur pred qu'on somme (sinon on risque de sommer tout quand on envoie plusieurs x_test d'un coup)

    # bias:
    h_B = P(x_test, beta=beta)
    bias_sqr = (h_B - E_LS)**2

    # residual error, aka var_y{y} = E_y{ (y-E_y{y})² }
    E_y = np.mean(y_train)
    residual_error = np.mean((h_B - E_y)**2)
    print(f'residual error is {residual_error}')

    return residual_error, bias_sqr, variance


"""def compute_error(x_test, x_train, y_train, rng, model=nn, beta=5):
    residual_error, bias_sqr, variance = compute_error_params(x_test, x_train, y_train, rng=rng, model=model, beta=beta)
    expected_error = residual_error + bias_sqr + variance
    return expected_error

def compute_residual_error(x_test, x_train, y_train, rng, model=nn, beta=5):
    residual_error, bias_sqr, variance = compute_error_params(x_test, x_train, y_train, rng=rng, model=model, beta=beta)
    return residual_error


def compute_bias_sqr(x_test, x_train, y_train, rng, model=nn, beta=5):
    residual_error, bias_sqr, variance = compute_error_params(x_test, x_train, y_train, rng=rng, model=model, beta=beta)
    return bias_sqr

def compute_var_LS(x_test, x_train, y_train, rng, model=nn, beta=5):
    residual_error, bias_sqr, variance = compute_error_params(x_test, x_train, y_train, rng=rng, model=model, beta=beta)
    return variance"""
    

def plot_question_2_3(model=nn, nb_LS=10, size_LS=250, beta=5):
    # Creation de plusieurs data set (avec des beta = 5 et N = 250)
    rng = np.random.default_rng(seed=42)

    x_train, y_train = make_multiple_datasets(number_LS=nb_LS, size_LS=size_LS, beta=beta, rng=rng)
    # this generates x_train with shape (number_LS, size_LS, 2) and y_train with shape (number_LS, size_LS)

    # Génération d'une grille de données linéairement espacées
    grid = np.mgrid[-1:1.01:0.01, -1:1.01:0.01]
    # grid est de shape (2, N, N) avec N le nombre de points par axe
    # gride.T donne (N, N, 2)
    # reshape : on demande une shape (?, 2) avec l'argument (-1,2) : le -1 dit : je ne sais pas combien il y aura d'élément mais calcule a partir de la taille du tableau et de l'autre taille (2)
    # donc on peut faire input_space[i] et ça donne [x1_i, x2_i]

    input_space = grid.T.reshape(-1, 2)
    # input space est de shape (N², 2) avec N le nombre de points par axe

    residual_error, bias_sqr, variance = compute_error_params(input_space, x_train, y_train, rng=rng, model=model, beta=beta)
    expected_error = residual_error + bias_sqr + variance

    # plot everything
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    plot_extent = (-1, 1, -1, 1)
    plot_data = (
        (bias_sqr, "Squared Bias"),
        (variance, "Variance"),
        (expected_error, "Expected Error"),
        (P(input_space, beta=beta), "Bayes Model"),
    )
    for ax, (values, title) in zip(axes, plot_data):
        image = ax.imshow(
            values.reshape(grid.shape[1:]), # value is a list of lenght N². grid.shape is (2, N, N) thus grid.shape[1:] give (N,N). This allow to put the values in a grid format for the raster image generation
            extent=plot_extent,
            origin="lower",
            cmap="plasma",
            aspect="auto",
        )
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        fig.colorbar(image, ax=ax)

    fig.tight_layout()
    plt.show()


def main():
    plot_question_2_3(model=nn, nb_LS=10, size_LS=250, beta=5)
    


if __name__ == "__main__":
    main()