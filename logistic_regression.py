# Lee D. Jackson
# Machine Learning
# Project 4: Classification with Logistic Regression
# 3/8/21

import numpy as np
import matplotlib.pyplot as plt


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
	features = []
	outputs = []
	for line in lines:
		line.rstrip('\n')
		split_line = line.split()
		this_line = [1.0]
		for counter, each_entry in enumerate(split_line):
			if counter < num_feats:
				this_line.append(float(each_entry))
			else:
				outputs.append(float(each_entry))
		features.append(this_line)

	# Make it a numpy array
	features = np.array(features)
	outputs = np.array(outputs)

	return features, outputs, num_lines, num_feats


def hypothesis(x):
	return 1 / (1 + np.exp(-x))


def cost(h, y):
	return (-(y * np.log(h)) - (1 - y) * np.log(1 - h)).mean()


def train(iterations, learn_rate, w, features, outputs):
	j_values = []

	for _ in range(0, iterations):
		z = np.dot(features, w)
		h = hypothesis(z)
		# print(h)

		gradient = np.dot(features.T, (h - outputs)) / outputs.shape[0]
		# print(gradient)

		j = cost(h, outputs)
		# print(j)
		j_values.append(j)

		w -= learn_rate * gradient
	print(j)
	# print(w)

	x = range(0, len(j_values))
	plt.plot(x, j_values)
	plt.xlabel('Iterations')
	plt.ylabel('J values')
	plt.show()

	return w


def probability(x, w):
	return hypothesis(np.dot(x, w))


def predict(x, w):
	return probability(x, w) >= 0.5


def get_accuracy(tp, tn, fp, fn):
	return (tp + tn) / (tp + tn + fp + fn)


def get_precision(tp, fp):
	return tp / (tp + fp)


def get_recall(tp, fn):
	return tp / (tp + fn)


def get_f1(tp, fp, fn):
	return 2 * 1 / ((1 / get_precision(tp, fp)) + (1 / get_recall(tp, fn)))


def test(features, outputs, final_weights):
	z = np.dot(features, final_weights)
	h = hypothesis(z)
	j = cost(h, outputs)

	predictions = predict(features, final_weights)

	true_positives = 0
	true_negatives = 0
	false_positives = 0
	false_negatives = 0

	for each_output, each_prediction in zip(outputs, predictions):
		if each_output == each_prediction:
			if each_prediction == 1:
				true_positives += 1
			else:
				true_negatives += 1
		else:
			if each_prediction == 1:
				false_positives += 1
			else:
				false_negatives += 1

	print('J = ', j)
	print()
	print('TPs: ', true_positives)
	print('TNs: ', true_negatives)
	print('FPs: ', false_positives)
	print('FNs: ', false_negatives)

	acc = get_accuracy(true_positives, true_negatives, false_positives, false_negatives)
	pre = get_precision(true_positives, false_positives)
	rec = get_recall(true_positives, false_negatives)
	f1 = get_f1(true_positives, false_positives, false_negatives)

	print('acc: ', acc)
	print('pre: ', pre)
	print('rec: ', rec)
	print('f1 : ', f1)


def main():
	# Enter training file name and read file
	# TODO change this back to user input
	# training_file = input('Please enter the name of the training file: ')
	training_file = 'train/r1.txt'
	features, outputs, num_examples, num_features = read_file(training_file)

	iterations = 10000
	lr = 15
	w = np.zeros(num_features + 1)

	final_weights = train(iterations, lr, w, features, outputs)
	print('Weights: ', final_weights)
	print()

	choice_is_valid = False

	while not choice_is_valid:
		predict_or_test = input('Do you want to (1) predict or (2) test? ')
		if predict_or_test == '1':
			choice_is_valid = True

			predict_file = input('Enter name of file to predict: ')
			features, outputs, num_examples, num_features = read_file(predict_file)

			for counter, each_team in enumerate(features):
				pass

	# Enter test file name and read file
	# TODO change this back to user input
	# test_file = input('Please enter the name of the test file: ')
	test_file = 'r2.txt'
	features, outputs, num_examples, num_features = read_file(test_file)

	test(features, outputs, final_weights)


main()
