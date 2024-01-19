#Importing all the used libraries
from tqdm import tqdm
from matplotlib import pyplot as plt
import random, csv, math

#Import from the settings file
from settings import *

#Creation of the class 'Algorithm'
class Algorithm:

    #Function called on creation of this object
    def __init__(self, weights = None, target_day = 1, start_date = 19520101, data_length = 26315):
        self.predicted_day = target_day
        self.start_date = start_date
        self.data_length = data_length

        self.data = self.setup_data('data')
        if weights == None:
            self.weights = self.new_weights()
        else:
            self.weights = weights
        self.active_trend = None

    #Retrieve all the data from our data file
    def setup_data(self, path):
        file = open(f"{path}.csv", "r")
        data = list(csv.reader(file, delimiter=";"))
        file.close()
        for i in range(len(data)):
            if int(data[i][1]) == self.start_date:
                for j in range(i):
                    del data[0]
                break
        for i in range(len(data) - self.data_length):
            del data[len(data) - 1]
        return data
    
    #Set random weights
    def new_weights(self):
        weights = []
        for i in range(3):
            weights.append([])
            for j in range(perspective):
                weights[i].append(round(random.random(), decimals))
        return weights

    #Generates a new index for training data
    def generate_index(self):
        return random.randint(perspective - 1, math.floor(learning_data_percentage * len(self.data)) - self.predicted_day - 1)
    
    #Sets up the data to make a prediction
    def set_prediction_data(self, index):
        prediction_data = []
        for i in range(perspective):
                prediction_data.append(float(self.data[index - i][data_list_index]))
        self.analyse_trend(prediction_data)
        return prediction_data
    
    def analyse_trend(self, prediction_data):
        self.active_trend = None
        if prediction_data[0] - prediction_data[perspective - 1] < -margin:
            self.active_trend = 2
        elif prediction_data[0] - prediction_data[perspective - 1] > margin:
            self.active_trend = 0
        else:
            self.active_trend = 1

    #Applies the weights to the prediction data
    def predict(self):
        prediction_data = self.set_prediction_data(self.index)
        weighted_data = []
        for i in range(perspective):
             weighted_data.append(prediction_data[i] * self.weights[self.active_trend][i])
        return round(sum(weighted_data)/len(weighted_data), decimals)
    
    #Adjusts the weights
    def adjust_weights(self):
        for i in range(perspective):
            for j in range(i+1):
                self.weights[self.active_trend][j] += learning_rate
                error_increased_weight = abs(self.predict() - self.target)
                self.weights[self.active_trend][j] -= 2 * learning_rate
                error_decreased_weight = abs(self.predict() - self.target)
                if self.error > error_increased_weight or self.error > error_decreased_weight:
                    if error_increased_weight < error_decreased_weight:
                        self.weights[self.active_trend][j] += 2 * learning_rate
                        self.error = error_increased_weight
                    else:
                        self.error = error_decreased_weight
                else:
                    self.weights[self.active_trend][j] += learning_rate
                self.weights[self.active_trend][j] = round(self.weights[self.active_trend][j], decimals)

    #Train the algorithm to learn the correct weights
    def train(self):
        errors = []
        iterations = []
        for i in range(learning_iterations):
            self.index = self.generate_index()
            self.prediction = self.predict()
            self.target = float(self.data[self.index + self.predicted_day][data_list_index])
            self.error =  round(abs(self.target - self.prediction), decimals)
            if self.error < 5.0:
                self.adjust_weights()
            iterations.append(i)
            errors.append(self.error)

    #Test the current weights
    def test(self):
        errors = []
        for i in range(10000):
            self.index = random.randint(math.floor(learning_data_percentage * len(self.data)) + perspective, len(self.data) - 1 - self.predicted_day)
            self.prediction = self.predict()
            self.target = float(self.data[self.index + self.predicted_day][data_list_index])
            errors.append(round(abs(self.target - self.prediction), decimals))
        for i in range(perspective):
            self.weights[self.active_trend][i] = round(self.weights[self.active_trend][i], decimals)
        #print(f"\nAccuracy: {round(sum(errors)/len(errors), decimals)}")
        #print(f"\nWeights: {self.weights}")
        return round(sum(errors)/len(errors), decimals)



iterations = []
accuracies = []
weights = None
id = random.randint(100000, 999999)
for i in tqdm(range(400), desc = "Training Algorithm"):
    algorithm = Algorithm(weights=weights)#, start_date = 20220101, data_length = 90)
    algorithm.train()
    accuracy = algorithm.test()
    weights = algorithm.weights

    file = open(f"RESULTS/{id}.txt", "a")
    file.write(f"\nAccuracy: {accuracy}       Iteration: {i}          Weights: {weights}\n")
    file.close()

    iterations.append(i)
    accuracies.append(accuracy)

plt.plot(iterations, accuracies)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Progression')
plt.show()
