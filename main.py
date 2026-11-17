import numpy as np
import matplotlib.pyplot as plt





def P(x, beta=5):
    """Returns the probability that y=1
    Input x is an array of shape (N, 2)
    """
    x1 = x[:,0]
    x2 = x[:,1]
    #print(x1)
    return 1 / (1 + np.exp(-beta*(x2-np.sin(np.pi* x1))))


def make_dataset(N=250, beta=5):
    """Creates a dataset of N elements using P(x) function
    """
    x = np.random.uniform(-1,1, (N,2))
    print(x.shape)    
    
    return x, P(x, beta)




def main():
    N = 250
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
    plt.show()



    


    


if __name__ == "__main__":
    main()