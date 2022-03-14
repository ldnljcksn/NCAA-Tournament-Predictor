# Lee D. Jackson
# Machine Learning
# Project 1: Linear Regression
# 2/1/21

import numpy as np


def read_file(file_name):
    file = open(file_name, 'r')

    # Get header information
    header = file.readline()
    header.rstrip('\n')
    header_list = header.split()
    header_list = list(map(int, header_list))  # Make everything an int
    num_lines = header_list[0]
    num_feats = header_list[1]

    # Read data into correct matrix
    lines = file.readlines()
    feature_matrix = []
    output_matrix = []
    for line in lines:
        line.rstrip('\n')
        split_line = line.split()
        this_line_matrix = [1.0]
        for counter, each_entry in enumerate(split_line):
            if counter < num_feats:
                this_line_matrix.append(float(each_entry))
            else:
                output_matrix.append(float(each_entry))
        feature_matrix.append(this_line_matrix)

    # Make it a numpy array
    feature_matrix = np.array(feature_matrix)
    output_matrix = np.array(output_matrix)

    return feature_matrix, output_matrix, num_lines


def find_weights(feature_matrix, output_matrix):
    a = np.linalg.pinv(np.dot(feature_matrix.T, feature_matrix))
    b = np.dot(feature_matrix.T, output_matrix)
    w = np.dot(a, b)

    return w


def find_j(weight_matrix, feature_matrix, output_matrix, num_examples):
    a = np.dot(feature_matrix, weight_matrix) - output_matrix
    j = (1 / num_examples) * np.dot(a.T, a)

    return j


def predict(weights, want_to_predict):
    prediction = 0
    for counter, each_weight in enumerate(weights):
        if counter == 0:
            prediction += each_weight
        else:
            prediction += each_weight * want_to_predict[counter - 1]

    return prediction


def read_teams(file_name):
    file = open(file_name, 'r')
    lines = file.readlines()
    team_names = []
    for each_line in lines:
        each_line.rstrip('\n')
        team_names.append(each_line)
    return team_names


def main():
    # Enter training file name and read file
    # training_file = input('Please enter the name of the training file: ')
    training_file = 'train/champ.txt'
    feature_matrix, output_matrix, num_examples = read_file(training_file)

    # Calculate the weights and print them
    weights = find_weights(feature_matrix, output_matrix)
    for counter, each_weight in enumerate(weights):
        print('w_' + str(counter) + ': ' + str(each_weight))

    # Calculate J value and print it
    error = find_j(weights, feature_matrix, output_matrix, num_examples)
    print('J: ' + str(error))
    print()

    choice_is_valid = False

    while not choice_is_valid:
        predict_or_test = input('Do you want to (1) predict or (2) test? ')
        if predict_or_test == '1':
            choice_is_valid = True

            # predict_file = input('Enter name of file to predict: ')
            predict_file = 'predict/v3/2021predictChamp.txt'
            feature_matrix, output_matrix, num_examples = read_file(predict_file)

            # names_file = input('Enter name of file with team names: ')
            names_file = 'predict/v3/champteams.txt'
            team_names = read_teams(names_file)

            predictions = []
            for counter, each_team in enumerate(feature_matrix):
                prediction = predict(weights, each_team[1:])
                predictions.append(prediction)

            # normalized = map(lambda x, r=float(predictions[-1] - predictions[0]): ((x - predictions[0]) / r),
            #                  predictions)

            # old_max = max([abs(val) for val in predictions])
            # new_range = 1
            # normalized = [float(val) / old_max * new_range for val in predictions]

            min_val = min(predictions)
            max_val = max(predictions)
            normalized = []

            for each_value in predictions:
                norm_val = (each_value - min_val) / (max_val - min_val)
                normalized.append(norm_val)

            for counter, each_prediction in enumerate(normalized):
                print(team_names[counter] + ': ', round(each_prediction, 2))

        elif predict_or_test == '2':
            choice_is_valid = True

            # Enter test file name and read file
            test_file = input('Please enter the name of the test file: ')
            feature_matrix, output_matrix, num_examples = read_file(test_file)

            # Calculate J value and print it
            error = find_j(weights, feature_matrix, output_matrix, num_examples)
            print('J: ' + str(error))
        else:
            print('Choice is not valid, please choose again.')
            print()


main()
