/**
 * The main class that puts the pieces of the neural network together from Weights.java and Node.java.
 * Represents a simple multilayer perceptron neural network, using back propagation to update weights.
 * 
 * @author Chao Zhang
 */
package com.network.main;

import java.io.FileNotFoundException;
import java.io.IOException;

import com.function.*;
import com.network.memory.Memory;
import com.network.node.Node;
import com.network.weights.Weights;
import com.throwables.BadLoadException;

public class BPNeuralNetwork {
	// private static boolean debug = false;

	private static Function idFun = new IdentityFunction();

	private final int inputLayerSize;
	private final int hiddenLayerSize;
	private final int outputLayerSize;
	private final int momentum;

	private final Node[] inputLayer;
	private final Node[] hiddenLayer;
	private final Node[] outputLayer;

	private final Weights input2hidden;
	private final Weights hidden2output;

	/**
	 * Constructor.
	 * 
	 * @param inputLayerSize
	 *            The number of nodes in the input layer.
	 * @param hiddenLayerSize
	 *            The number of nodes in the hidden layer.
	 * @param outputLayerSize
	 *            The number of nodes in the output layer.
	 * @param fun
	 *            The activation function for each node.
	 * @param momentum
	 *            The momentum for learning.
	 */
	public BPNeuralNetwork(int inputLayerSize, int hiddenLayerSize,
			int outputLayerSize, Function fun, int momentum) {
		this.inputLayerSize = inputLayerSize;
		this.hiddenLayerSize = hiddenLayerSize;
		this.outputLayerSize = outputLayerSize;
		this.momentum = momentum;

		inputLayer = new Node[inputLayerSize];
		hiddenLayer = new Node[hiddenLayerSize];
		outputLayer = new Node[outputLayerSize];

		input2hidden = new Weights(inputLayerSize, hiddenLayerSize);
		hidden2output = new Weights(hiddenLayerSize, outputLayerSize);

		for (int x = 0; x < inputLayerSize; x++) {
			inputLayer[x] = new Node(idFun);
		}
		for (int x = 0; x < hiddenLayerSize; x++) {
			hiddenLayer[x] = new Node(fun);
		}
		for (int x = 0; x < outputLayerSize; x++) {
			outputLayer[x] = new Node(fun);
		}
	}

	/**
	 * Propagate an array of doubles through the neural network.
	 * 
	 * @param input
	 *            The array of doubles to propagate through the neural network.
	 * @return An array of doubles representing the output of the neural
	 *         network.
	 * @throws IllegalArgumentException
	 *             If the input size does not equal the input layer of the
	 *             neural network.
	 */
	public double[] feedForward(double[] input) throws IllegalArgumentException {
		if (input.length != inputLayerSize) {
			throw new IllegalArgumentException(
					"Input size does not equal input layer size. Cannot feed forward.");
		}

		double[] outputs = null;
		try {
			outputs = new double[hiddenLayerSize];

			for (int x = 0; x < inputLayerSize; x++) {
				double output = inputLayer[x].calculate(input[x]);
				for (int y = 0; y < hiddenLayerSize; y++) {
					outputs[y] += input2hidden.calculateWeightedOutput(output,
							x, y);
				}
			}

			input = outputs;

			outputs = new double[outputLayerSize];

			for (int x = 0; x < hiddenLayerSize; x++) {
				double output = hiddenLayer[x].calculate(input[x]);
				for (int y = 0; y < outputLayerSize; y++) {
					outputs[y] += hidden2output.calculateWeightedOutput(output,
							x, y);
				}
			}

			input = outputs;

			outputs = new double[outputLayerSize];

			for (int x = 0; x < outputLayerSize; x++) {
				outputs[x] = outputLayer[x].calculate(input[x]);
			}
		} catch (Exception exc) {
			exc.printStackTrace();
			System.exit(1);
		}
		return outputs;
	}

	/**
	 * Back propagate and readjusts the weights of the neural network.
	 * 
	 * @param correctOutput
	 *            The array of double values that the neural network should
	 *            output.
	 * @return The total error of the neural network.
	 * @throws IllegalArgumentException
	 *             If the correct output array size is not the same as the
	 *             output layer of the neural network.
	 */
	public double backPropagate(double[] correctOutput)
			throws IllegalArgumentException {
		if (correctOutput.length != outputLayerSize) {
			throw new IllegalArgumentException(
					"Correct output size does not equal output layer size. Could not back propagate.");
		}
		double err = 0;

		try {

			double[] outputLayerErrors = new double[outputLayerSize];
			for (int x = 0; x < outputLayerSize; x++) {
				outputLayerErrors[x] = outputLayer[x]
						.calculateError(correctOutput[x]
								- outputLayer[x].getOutput());
				err += Math.abs(outputLayerErrors[x]);
			}

			double[] hiddenLayerErrors = new double[hiddenLayerSize];
			for (int x = 0; x < hiddenLayerSize; x++) {
				double error = 0;
				for (int y = 0; y < outputLayerSize; y++) {
					error += outputLayerErrors[y]
							* hidden2output.getWeight(x, y);
					hidden2output.updateWeights(x, y, outputLayerErrors[y],
							momentum, hiddenLayer[x].getOutput());
				}
				hiddenLayerErrors[x] = hiddenLayer[x].calculateError(error);
				err += Math.abs(hiddenLayerErrors[x]);
			}

			for (int x = 0; x < inputLayerSize; x++) {
				for (int y = 0; y < hiddenLayerSize; y++) {
					input2hidden.updateWeights(x, y, hiddenLayerErrors[y],
							momentum, inputLayer[x].getOutput());
				}
			}
		}

		catch (Exception exc) {
			exc.printStackTrace();
			System.exit(1);
		}

		return err;
	}

	/**
	 * Trains the neural network with an array of inputs against an array of
	 * outputs until the error value is below a certain threshold.
	 * 
	 * @param inputs
	 *            An array of double arrays representing the inputs into the
	 *            neural network.
	 * @param correctOutput
	 *            An array of double arrays representing the correct outputs the
	 *            inputs should produce. A single output array should be in the
	 *            same index position as its corresponding input array.
	 * @param threshold
	 *            The error threshold below which learning stops.
	 * @return The number of feed forward and back propagation pairs that occur
	 *         before the threshold was reached and learning stopped.
	 * @throws IllegalArgumentException
	 *             If the single input array length or the single output array
	 *             length does not correspond to the number of input layer or
	 *             output layers respectively.
	 */
	public int train(double[][] inputs, double[][] correctOutput,
			double threshold) throws IllegalArgumentException {
		if (inputs.length != correctOutput.length) {
			throw new IllegalArgumentException(
					"Number of inputs must equal number of outputs. Cannot train.");
		}

		double error;

		int count = 0;
		int numInputs = inputs.length;

		do {
			error = 0;
			for (int x = 0; x < numInputs; x++) {
				feedForward(inputs[x]);
				error += backPropagate(correctOutput[x]);
			}

			count++;
		} while (error > threshold);

		return count;
	}

	/**
	 * Default method for saving. Will create a save file called
	 * "BPNeuralNetworkProperties.txt". See <code>save(String fileName)</code>
	 * for more details.
	 * 
	 * @throws IOException
	 *             If there is a problem creating a save file.
	 */
	public final void save() throws IOException {
		save("BPNeuralNetworkProperties.txt");
	}

	/**
	 * Saves the neural network properties in a file called <fileName>.
	 * Specifically, uses the save method found in the Memory.java file. See
	 * Memory.java for more details on the save file configuration.
	 * 
	 * @param fileName
	 *            The name of the file that stores the properties of the neural
	 *            network.
	 * @throws IOException
	 *             If there is a problem creating a save file.
	 */
	public final void save(String fileName) throws IOException {
		int[] layerCounts = { inputLayerSize, hiddenLayerSize, outputLayerSize };
		double[] weights = new double[inputLayerSize * hiddenLayerSize
				+ hiddenLayerSize * outputLayerSize];

		int index = 0;

		for (int x = 0; x < inputLayerSize; x++) {
			for (int y = 0; y < hiddenLayerSize; y++) {
				weights[index] = input2hidden.getWeight(x, y);
				index++;
			}
		}

		for (int x = 0; x < hiddenLayerSize; x++) {
			for (int y = 0; y < outputLayerSize; y++) {
				weights[index] = hidden2output.getWeight(x, y);
				index++;
			}
		}
		Memory.save(fileName, layerCounts, weights);
	}

	/**
	 * Loads up the weights for a particular neural network configuration. If
	 * the weights do not match up for this neural network configuration, a
	 * BadLoadException will be thrown. Utilizes the load method from
	 * Memory.java. See the Memory.java file for more details on what the
	 * configuration for the loaded file should be.
	 * 
	 * @param fileName
	 *            The name of the file to be loaded.
	 * @throws FileNotFoundException
	 *             If the named file could not be found.
	 * @throws IOException
	 *             If there is a read problem.
	 * @throws BadLoadException
	 *             If the neural network configuration does not match that of
	 *             the one given in saved file.
	 */
	public final void load(String fileName) throws FileNotFoundException,
			IOException, BadLoadException {
		int index = 0;
		int[] layerCounts = { inputLayerSize, hiddenLayerSize, outputLayerSize };

		double[] weights = Memory.load(fileName, layerCounts);

		if (weights.length != inputLayerSize * hiddenLayerSize
				+ hiddenLayerSize * outputLayerSize) {
			throw new BadLoadException(
					"Number of weights in input file does not correspond to number of weights in neural network. Could not load weights from file.");
		}

		double[][] firstLayer = new double[inputLayerSize][hiddenLayerSize];
		double[][] secondLayer = new double[hiddenLayerSize][outputLayerSize];

		for (int x = 0; x < inputLayerSize; x++) {
			for (int y = 0; y < hiddenLayerSize; y++) {
				firstLayer[x][y] = weights[index];
				index++;
			}
		}

		for (int x = 0; x < hiddenLayerSize; x++) {
			for (int y = 0; y < outputLayerSize; y++) {
				secondLayer[x][y] = weights[index];
				index++;
			}
		}

		input2hidden.setWeights(firstLayer);
		hidden2output.setWeights(secondLayer);
	}

	public static void main(String[] args) {
		Function sig = new SigmoidFunction();
		// Function id = new IdentityFunction();
		BPNeuralNetwork n = new BPNeuralNetwork(2, 2, 2, sig, 2);

		double[] input = { 1, 1 };
		// double[] correctOutput = { .5, .5 };

		// try {
		// n.load("Out.txt");
		// }
		// catch (IOException e) {
		// // TODO Auto-generated catch block
		// e.printStackTrace();
		// System.exit(1);
		// }
		// catch (BadLoadException e) {
		// // TODO Auto-generated catch block
		// e.printStackTrace();
		// System.exit(1);
		// }

		double[][] layerOne = new double[2][2];
		double[][] layerTwo = new double[2][2];

		layerOne[0][0] = 0.2;
		layerOne[0][1] = 0.3;
		layerOne[1][0] = 0.4;
		layerOne[1][1] = 0.5;

		layerTwo[0][0] = 0.9;
		layerTwo[0][1] = 0.8;
		layerTwo[1][0] = 0.7;
		layerTwo[1][1] = 0.6;

		n.input2hidden.setWeights(layerOne);
		n.hidden2output.setWeights(layerTwo);

		try {
			n.save("Out.txt");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(1);
		}
		double[] output = n.feedForward(input);

		for (double d : output) {
			System.out.println(d);
		}

		// System.out.println(n.backPropagate(correctOutput));
	}

}
