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


def nn(X_train, y_train, rng, model_param=None):
    """Returns a trained neural network classifier"""
    if model_param is None:
        hidden_layer_sizes = (100,) # default value is 100 neurons in a single hidden layer
    else:
        hidden_layer_sizes = model_param

    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, solver = "lbfgs", alpha = 10**-5, max_iter = 2000, random_state=get_random_seed(rng))

    return clf.fit(X_train, y_train)


def knn(X_train, y_train, rng, model_param=None):
    """Returns a trained k-nearest neighbors classifier"""
    if model_param is None:
            n_neighbors = 5 # default value is 5 neighbors
    else:
        n_neighbors = model_param
    
    
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    return clf.fit(X_train, y_train)


def dtree(X_train, y_train, rng, model_param=None):
    """Returns a trained decision tree classifier"""
    if model_param is None:
        depth = 6   # default value is a depth of 6
    else:
        depth = model_param
    
    clf = DecisionTreeClassifier(max_depth=depth, random_state=get_random_seed(rng))
    return clf.fit(X_train, y_train)


def get_random_seed(rng):
    """Converts rng generator from numpy to a random number for the seed of sickit learn"""
    #return int(rng.integers(0, 2**31 - 1))
    return 42 # todo remove this


def compute_error_params(x_test, x_train, y_train, rng, model=nn, model_param=None, beta=5):
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
        m = model(x_train[i],y_train[i], rng=rng, model_param=model_param)
        pred[i, :] = m.predict_proba(x_test)[:,1]

    E_LS = np.mean(pred, axis=0) # axis 0 pour que si on donne plusieurs x_test, E_LS garde la forme de x_test et moyenne juste les pred "element wise" pour chaque x_test

    variance = np.mean(np.square(pred - E_LS), axis = 0) # sommer sur l'axe 0 pour que ce soit bien sur pred qu'on somme (sinon on risque de sommer tout quand on envoie plusieurs x_test d'un coup)

    # bias:
    h_B = P(x_test, beta=beta)
    bias_sqr = (h_B - E_LS)**2

    # residual error, aka var_y{y} = E_y{ (y-E_y{y})² }
    E_y = np.mean(y_train)
    residual_error = np.mean((h_B - E_y)**2)
    # print(f'residual error is {residual_error}')

    return residual_error, bias_sqr, variance



def compute_mean(rng, model, model_param, size_LS, beta, nb_LS=10):
    # Creation de plusieurs data set (avec des beta = 5 et N = 250)

    x_train, y_train = make_multiple_datasets(number_LS=nb_LS, size_LS=size_LS, beta=beta, rng=rng)
    # this generates x_train with shape (number_LS, size_LS, 2) and y_train with shape (number_LS, size_LS)

    # Génération d'une grille de données linéairement espacées
    grid = np.mgrid[-1:1.01:0.01, -1:1.01:0.01] 
    input_space = grid.T.reshape(-1, 2)

    residual_error, bias_sqr, variance = compute_error_params(input_space, x_train, y_train, rng=rng, model=model, model_param=model_param, beta=beta)
    expected_error = residual_error + bias_sqr + variance

    return np.array( [np.mean(residual_error), np.mean(bias_sqr), np.mean(variance), np.mean(expected_error)])




     
def plot_question_2_3(model, rng, nb_LS=10, size_LS=250, beta=5):
    # Creation de plusieurs data set (avec des beta = 5 et N = 250)

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

    residual_error, bias_sqr, variance = compute_error_params(input_space, x_train, y_train, rng=rng, model=model, model_param=None, beta=beta) # use model_param=None to keep default parameter value
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



    




def plot_mean_over_complexity(rng, models, model_parameters):
    # rng = np.random.default_rng(seed=42) # todo remove seed
    
    """models = [nn, knn, dtree]
    model_parameters = np.array([[1, 5, 20, 50, 100], # nn parameters (hidden layer size)
                                 [1, 3, 10, 30, 100], # knn parameters (number of neighbors)
                                 [1, 2, 4, 7, 12] # dtree parameters (depth)
                                 ])"""

    number_models = len(models)
    number_params = model_parameters.shape[1]
    number_outputs = 4 # compute_means returns 4 output : resid error, sqr bias, var and expected error

    print(np.max(model_parameters, axis=1))
    print(model_parameters)

    max_param = np.max(model_parameters, axis=1)
    normalized_model_parameters = np.zeros_like(model_parameters, dtype=float)
    for i in range(number_models):
        normalized_model_parameters[i, :] = model_parameters[i, :]/max_param[i]
    print(normalized_model_parameters)


    
    result = np.zeros((number_models, number_params, number_outputs))
    for i, (model, params) in enumerate(zip(models, model_parameters)):
        for j, param in enumerate(params):
            result[i, j,:] = compute_mean(rng=rng, model=model, model_param=param, size_LS=250, beta=5)

    #normalized_model_parameters = model_parameters / 
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # this plots each model on a graph and each line is a different result (var, bias, resid error)
    axes[0].plot(normalized_model_parameters[0], result[0, :, 0], label="residual error nn")
    axes[0].plot(normalized_model_parameters[0], result[0, :, 1], label="squared bias nn")
    axes[0].plot(normalized_model_parameters[0], result[0, :, 2], label="variance nn")
    axes[0].set_title("Neural Network")
    axes[0].set_xlabel("Model complexity")

    axes[1].plot(1-normalized_model_parameters[1], result[1, :, 0], label="residual error knn") # inverser le sens car knn est plus complexe quand le nbr de neighbors est faible
    axes[1].plot(1-normalized_model_parameters[1], result[1, :, 1], label="squared bias knn")
    axes[1].plot(1-normalized_model_parameters[1], result[1, :, 2], label="variance knn")
    axes[1].set_title("K-Nearest Neighbors")
    axes[1].set_xlabel("Model complexity")

    axes[2].plot(normalized_model_parameters[2], result[2, :, 0], label="residual error dtree")
    axes[2].plot(normalized_model_parameters[2], result[2, :, 1], label="squared bias dtree")
    axes[2].plot(normalized_model_parameters[2], result[2, :, 2], label="variance dtree")
    axes[2].set_title("Decision Tree")
    axes[2].set_xlabel("Model complexity")

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()

    plt.show()


def plot_mean_over_N(rng, models, sizes_LS):

    number_models = len(models)
    number_sizes = len(sizes_LS)
    number_outputs = 4 # compute_means returns 4 output : resid error, sqr bias, var and expected error


    result = np.zeros((number_models, number_sizes, number_outputs))
    for i, model in enumerate(models):
        for j, size in enumerate(sizes_LS):
            result[i, j,:] = compute_mean(rng=rng, model=model, model_param=None, size_LS=size, beta=5)

    #normalized_model_parameters = model_parameters / 
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # this plots each model on a graph and each line is a different result (var, bias, resid error)
    # axes[0].plot(sizes_LS, result[0, :, 0], label="residual error nn")
    axes[0].plot(sizes_LS, result[0, :, 1], label="squared bias nn")
    axes[0].plot(sizes_LS, result[0, :, 2], label="variance nn")
    axes[0].set_title("Neural Network")
    axes[0].set_xlabel("Number of learning samples")

    # axes[1].plot(sizes_LS, result[1, :, 0], label="residual error knn") # inverser le sens car knn est plus complexe quand le nbr de neighbors est faible
    axes[1].plot(sizes_LS, result[1, :, 1], label="squared bias knn")
    axes[1].plot(sizes_LS, result[1, :, 2], label="variance knn")
    axes[1].set_title("K-Nearest Neighbors")
    axes[1].set_xlabel("Number of learning samples")

    # axes[2].plot(sizes_LS, result[2, :, 0], label="residual error dtree")
    axes[2].plot(sizes_LS, result[2, :, 1], label="squared bias dtree")
    axes[2].plot(sizes_LS, result[2, :, 2], label="variance dtree")
    axes[2].set_title("Decision Tree")
    axes[2].set_xlabel("Number of learning samples")

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()

    plt.show()



def plot_mean_over_beta(rng, models, betas):

    number_models = len(models)
    number_betas = len(betas)
    number_outputs = 4 # compute_means returns 4 output : resid error, sqr bias, var and expected error


    result = np.zeros((number_models, number_betas, number_outputs))
    for i, model in enumerate(models):
        for j, beta in enumerate(betas):
            result[i, j,:] = compute_mean(rng=rng, model=model, model_param=None, size_LS=250, beta=beta)

    #normalized_model_parameters = model_parameters / 
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # this plots each model on a graph and each line is a different result (var, bias, resid error)
    axes[0].plot(betas, result[0, :, 0], label="residual error nn")
    axes[0].plot(betas, result[0, :, 1], label="squared bias nn")
    axes[0].plot(betas, result[0, :, 2], label="variance nn")
    axes[0].set_title("Neural Network")
    axes[0].set_xlabel("Beta")

    axes[1].plot(betas, result[1, :, 0], label="residual error knn") # inverser le sens car knn est plus complexe quand le nbr de neighbors est faible
    axes[1].plot(betas, result[1, :, 1], label="squared bias knn")
    axes[1].plot(betas, result[1, :, 2], label="variance knn")
    axes[1].set_title("K-Nearest Neighbors")
    axes[1].set_xlabel("Beta")

    axes[2].plot(betas, result[2, :, 0], label="residual error dtree")
    axes[2].plot(betas, result[2, :, 1], label="squared bias dtree")
    axes[2].plot(betas, result[2, :, 2], label="variance dtree")
    axes[2].set_title("Decision Tree")
    axes[2].set_xlabel("Beta")

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()

    plt.show()


#Q2.5
def plot_question_2_5_beta(rng, nb_LS = 10):
    
     #Evolution of the optimal hyper paramter for a fixed N
     
     N = 1000
     table_beta = [2,4,5,6,8,9,15,20] 
     table_number_n = [1, 9, 45, 225,901]
     
     mean_error_evolution = np.zeros( (len(table_beta), len(table_number_n)) )
     
     for i , beta in enumerate(table_beta):
         for j, param in enumerate(table_number_n):
             mean_expected_error = compute_mean(rng, model = knn, model_param = param, size_LS = N, beta= beta)[3]
             mean_error_evolution[i, j] = mean_expected_error     
     
     #Le meilleur hyper param est celui minimise l'erreur 
     best_param_index = np.argmin(mean_error_evolution, axis=1)#Donne un tableau du meilleur hyper paramettre pour un beta donné
     
     best_param = np.zeros_like(best_param_index)
     for i, bp in enumerate(best_param_index):
         best_param[i] = table_number_n[bp]
     
     plt.scatter(table_beta, best_param)
     plt.yscale('log')
     plt.xscale('linear')
     plt.xlabel(r'parameter $\beta$')
     plt.ylabel("number of neighbors")
     plt.show()
     
     best_param_for_beta = np.zeros((2, len(best_param)))
     best_param_for_beta[0,:] = table_beta
     best_param_for_beta[1,:] = best_param
     print(best_param_for_beta)
     
def plot_question_2_5_N(rng, nb_LS = 10):
    
     #Evolution of the optimal hyper paramter for a fixed beta
     
     beta = 6
     table_N = [50,200,500,1000,5000,10000]
     
     table_number_n = [1, 9, 45, 225,901]
     
     mean_error_evolution = np.zeros( (len(table_N), len(table_number_n)) )
     
     for i, N in enumerate(table_N):
         for j, param in enumerate(table_number_n):
             if(param <= N):
                 mean_expected_error = compute_mean(rng, model = knn, model_param = param, size_LS = N, beta= beta)[3]
                 mean_error_evolution[i, j] = mean_expected_error     
             else:
                 mean_error_evolution[i,j] = np.inf #Si le nombre de voisins > au nombre de data on met l'erreur à l'infini pour pas la compter
                 
     #Le meilleur hyper param est celui minimise l'erreur 
     best_param_index = np.argmin(mean_error_evolution, axis=1)#Donne un tableau du meilleur hyper paramettre pour un beta donné
     
     best_param = np.zeros_like(best_param_index)
     for i, bp in enumerate(best_param_index):
         best_param[i] = table_number_n[bp]
     
     plt.scatter(table_N, best_param)
     plt.loglog()
     plt.xlabel("Sample size (N)")
     plt.ylabel("number of neighbors")
     plt.show()
     
     best_param_for_N = np.zeros((2, len(table_N)))
     best_param_for_N[0,:] = table_N
     best_param_for_N[1,:] = best_param
     print(best_param_for_N)
     #Evolution of the optimal hyper paramter for a fixed beta


from sklearn.ensemble import BaggingClassifier

def question_2_6( X_train, y_train, X, rng, nb_LS = 10, estimator = nn):
   
    
    Bagging = BaggingClassifier(estimator = estimator(), max_samples = 1000, max_features= 2, n_estimators = 10).fit(X_train, y_train)
    Bagging.predict_proba(X)
    

if __name__ == "__main__":
    rng = np.random.default_rng()
    # plot_question_2_3(model=nn, rng=rng, nb_LS=10, size_LS=250, beta=5)
    # plot_question_2_3(model=knn, rng=rng, nb_LS=10, size_LS=250, beta=5)
    # plot_question_2_3(model=dtree, rng=rng, nb_LS=10, size_LS=250, beta=5)


    """models = [nn, knn, dtree]

    # plot resid error, bias and variance over model complexity for each model
    model_parameters = np.array([[1, 5, 20, 50, 100], # nn parameters (hidden layer size)
                                    [1, 3, 10, 30, 100], # knn parameters (number of neighbors)
                                    [1, 2, 4, 7, 12] # dtree parameters (depth)
                                    ])
    #plot_mean_over_complexity(rng=rng, models=models, model_parameters=model_parameters)



    # plot resid error, bias and variance over number of learning samples for each model
    sizes_LS = [100, 250, 500]
    # plot_mean_over_N(rng=rng, models=models, sizes_LS=sizes_LS)


    betas = [1, 2, 5, 10, 20]
    plot_mean_over_beta(rng=rng, models=models, betas=betas)"""
    
    plot_question_2_5_beta(rng=rng, nb_LS=10)
    plot_question_2_5_N(rng=rng, nb_LS =10)
    