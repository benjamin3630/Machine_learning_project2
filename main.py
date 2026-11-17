import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier 


from sklearn.tree import DecisionTreeClassifier

def P(x, beta=5):
    """Returns the probability that y=1
    Input x is an array of shape (N, 2)
    """
    x1 = x[0]
    x2 = x[1]
    #print(x1)
    return 1 / (1 + np.exp(-beta*(x2-np.sin(np.pi* x1))))


def make_dataset(N=250, beta=5):
    """Creates a dataset of N elements using P(x) function
    """
    x = np.random.uniform(-1,1, (N,2))
    print(x.shape)    
    
    return x, P(x, beta)

def nn(X_train, y_train, X):
    
    random_state = np.random.randint(0,1000)
    clf = MLPClassifier(solver = "lbfgs", alpha = 10**-5, max_iter = 2000, random_state= random_state).fit(X_train, y_train)
    f = clf.predict(X)
    return f

def knn(X_train, y_train, X):
    
    n_neighbors = 5
    
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
    f = clf.predict(X)
    return f

def dtree(X_train, y_train, X):
    
    random_seed = np.random.randint(0,1000)
    depth = 6
    
    clf = DecisionTreeClassifier(max_depth=depth, random_state=random_seed)
    f = clf.predict(X)
    return f

def bias(x, learning_samples, Y_training, f=nn):
    
    h_B = P(x, beta=5)
    n_LS = np.shape(learning_samples)[0]
    sum_f = 0
    for i in range(np.shape(learning_samples)[0]):
        
        sum_f += f(learning_samples[i],Y_training, x)
    
    E_LS = 1/n_LS*sum_f
    
    bias_sqr = (h_B - E_LS)**2
    return bias_sqr
    
def main():
    
    #Creation de plusieurs data set (avec des beta = 5 et N = 250)
    
    nb_LS = 10
    size_LS = 250
    
    learning_samples = np.zeros((nb_LS, size_LS, 2))
    Y = np.zeros((nb_LS,size_LS,1))
    
    for i in range(nb_LS):
        dataset = make_dataset(size_LS)
        learning_samples[i] = dataset[0]
        Y[i] = dataset[1]
        
    #Computation of the squarred bias
    test = bias((0.5,0.5), learning_samples= learning_samples,Y_training = Y, f = nn)
    print(test)
    
    """N = 2500
    beta = 5

    np.random.seed(42) # for debugging purposes
    x, Y = make_dataset(N, beta)

    #print(f"{x[0] = } : {Y[0]=}")

    # plot a color map of y as a function of x1 and x2
    plt.figure(figsize=(6,6))
    plt.scatter(x[:,0], x[:,1], c=Y, cmap='coolwarm', s=10)
    plt.colorbar(label='P(y=1)') 
    plt.show()


    x, Y = make_dataset(N=2500, beta=100)
    
    plt.figure(figsize=(6,6))
    plt.scatter(x[:,0], x[:,1], c=Y, cmap='coolwarm', s=10)
    plt.colorbar(label='P(y=1)') 
    plt.show()"""



    


    


if __name__ == "__main__":
    main()