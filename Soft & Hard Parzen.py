import numpy as np


######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:
    
    def feature_means(self, iris):
        return np.mean(iris[:, :-1], axis = 0)

    def covariance_matrix(self, iris):
        return np.cov(np.transpose(iris[:, :-1]))

    def feature_means_class_1(self, iris):
        return np.mean(iris[iris[:,-1]==1, :-1], axis = 0)

    def covariance_matrix_class_1(self, iris):
        return np.cov(np.transpose(iris[iris[:,-1]==1, :-1]))


class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        # self.label_list = np.unique(train_labels)
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(train_labels)

    def compute_predictions(self, test_data):
        k = np.array(test_data).shape[0]
        pred = np.zeros(k)
        for i in range(k):
            dist = (np.sum((np.abs(self.train_inputs - test_data[i, :]))**2,
                              axis = 1))**0.5
            neighbour_labels = self.train_labels[dist < self.h]
            if neighbour_labels.shape[0] == 0:
                pred[i] = draw_rand_label(test_data[i,:], self.label_list)
            else:
                class_label, counts = np.unique(neighbour_labels,
                                            return_counts = True)
                pred[i] = class_label[np.argmax(counts)]
        return pred


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def train(self, train_inputs, train_labels):
        # self.label_list = np.unique(train_labels)
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(train_labels)

    def compute_predictions(self, test_data):
        k = np.array(test_data).shape[0]
        m = len(self.label_list)
        pred = np.zeros(k)
        total_label = np.zeros(m)
        for i in range(k):
            dist = (np.sum((np.abs(self.train_inputs - test_data[i, :]))**2,
                          axis = 1))**0.5
            kernel = np.exp(-dist**2/2/self.sigma**2)
            for j in range(m):
                total_label[j] = np.sum(kernel[self.train_labels ==
                                               self.label_list[j]])
            pred[i] = self.label_list[np.argmax(total_label)]
        return pred    
            

def split_dataset(iris):
    train = iris[[i for i in range(iris.shape[0]) if i % 5 in [0, 1, 2]], :]
    validation = iris[[i for i in range(iris.shape[0]) if i % 5 == 3], :]
    test = iris[[i for i in range(iris.shape[0]) if i % 5 == 4], :]
    return (train, validation, test)

class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        hardparzen_func = HardParzen(h)
        hardparzen_func.train(self.x_train, self.y_train)
        hardparzen_pred = hardparzen_func.compute_predictions(self.x_val)
        incorrect = hardparzen_pred[hardparzen_pred != self.y_val].shape[0]
        return incorrect/hardparzen_pred.shape[0]

    def soft_parzen(self, sigma):
        softparzen_func = SoftRBFParzen(sigma)
        softparzen_func.train(self.x_train, self.y_train)
        softparzen_pred = softparzen_func.compute_predictions(self.x_val)
        incorrect = softparzen_pred[softparzen_pred != self.y_val].shape[0]
        return incorrect/softparzen_pred.shape[0]

def get_test_errors(iris):
    train, validation, test = split_dataset(iris)
    h = [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 15.0, 20.0]
    sigma = [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 15.0, 20.0]
    validation_error = ErrorRate(train[:, :-1], train[:, -1],
                                 validation[:, :-1], validation[:, -1])
    h_list = []
    sigma_list = []
    for i in range(len(h)):
        h_list.append(validation_error.hard_parzen(h[i]))
        sigma_list.append(validation_error.soft_parzen(sigma[i]))
    test_error = ErrorRate(train[:, :-1], train[:, -1],
                           test[:, :-1], test[:, -1])
    hardparzen_error = test_error.hard_parzen(h[np.argmin(h_list)])
    softparzen_error = test_error.soft_parzen(sigma[np.argmin(sigma_list)])
    return (hardparzen_error, softparzen_error)

def random_projections(X, A):
    return (1/np.sqrt(2))*(np.dot(X, A))
