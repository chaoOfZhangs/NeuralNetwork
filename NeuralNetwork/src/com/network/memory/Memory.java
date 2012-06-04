/**
 * Wrapper over the ReadWriteTextFile.java class to save and load the weights in the neural network. 
 * 
 * Saving produces a file with <fileName> of the form:
 * 
 * N
 * <Number of layers in the neural network>
 * L
 * <Number of nodes in the first layer>
 * <Number of nodes in the second layer>
 * <Number of nodes in the third layer>
 * N
 * <Number of weights totally throughout the neural network>
 * W
 * <First weight value>
 * <Second weight value>
 * <Third weight value>
 * ... etc.
 * 
 * Loading must receive a file with the same format, or exceptions will be thrown.
 * 
 * @author	Chao Zhang
 */

package com.network.memory;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;

import com.network.memory.ReadWriteTextFile;
import com.throwables.BadLoadException;

public class Memory {

	/**
	 * Private constructor to prevent instantiation.
	 */
	private Memory() {
	}

	/**
	 * Static method to save the properties of the neural network.
	 * 
	 * Saving produces a file with <fileName> of the form:
	 * 
	 * N
	 * <Number of layers in the neural network> 
	 * L 
	 * <Number of nodes in the first layer> 
	 * <Number of nodes in the second layer> 
	 * <Number of nodes in the third layer> 
	 * N 
	 * <Number of weights totally throughout the neural network> 
	 * W 
	 * <First weight value> 
	 * <Second weight value> 
	 * <Third weight value> 
	 * ... etc.
	 * 
	 * @param fileName
	 *            The name of the save file with the neural network properties.
	 * @param layerCounts
	 *            The number of nodes in each layer.
	 * @param weights
	 *            The weights for the neural network.
	 * @return True if saved successfully, false otherwise.
	 * @throws IOException
	 *             If there was an exception from saving the file.
	 */
	public final static boolean save(String fileName, int[] layerCounts,
			double[] weights) throws IOException {

		StringBuilder text = new StringBuilder();

		String NL = System.getProperty("line.separator");

		text.append("N" + NL);
		text.append(layerCounts.length + NL);

		text.append("L" + NL);
		for (int x = 0; x < layerCounts.length; x++) {
			text.append(layerCounts[x] + NL);
		}

		text.append("N" + NL);
		text.append(weights.length + NL);

		text.append("W" + NL);
		for (int x = 0; x < weights.length; x++) {
			text.append(weights[x] + NL);
		}

		return ReadWriteTextFile.write(fileName, text.toString());
	}

	/**
	 * Loads a file and creates a neural network with the weights specified by
	 * the file. The neural network should have the node configuration specified
	 * by the file beforehand or an exception will be thrown.
	 * 
	 * The loaded file should have the following format:
	 * 
	 * N
	 * <Number of layers in the neural network> 
	 * L 
	 * <Number of nodes in the first layer> 
	 * <Number of nodes in the second layer> 
	 * <Number of nodes in the third layer> 
	 * N 
	 * <Number of weights totally throughout the neural network> 
	 * W 
	 * <First weight value> 
	 * <Second weight value> 
	 * <Third weight value> 
	 * ... etc.
	 * 
	 * Any other formats will result in a BadLoadException.
	 * 
	 * @param fileName
	 *            The name of the file to be loaded.
	 * @param layerCounts
	 *            The number of layers.
	 * @return An array containing the weights.
	 * @throws IOException
	 *             If there is a problem loading the file.
	 * @throws BadLoadException
	 *             If the configuration of the neural network does not match
	 *             that given in the file, or if the file is not configured
	 *             correctly.
	 * @throws FileNotFoundException
	 *             If the file cannot be found.
	 */
	public final static double[] load(String fileName, int[] layerCounts)
			throws IOException, BadLoadException, FileNotFoundException {
		int numLayers = 0;
		int numWeights = 0;

		double[] weights = null;

		String storedText = ReadWriteTextFile.read(fileName);

		Scanner scan = new Scanner(storedText);

		consumeLetter(scan, "N");

		if (scan.hasNextInt()) {
			numLayers = scan.nextInt();
			if (numLayers != layerCounts.length) {
				throw new BadLoadException(
						"Cannot use input file to load weights. Number of layers given in input file does not match that of the neural network.");
			}
		} else {
			throw new BadLoadException(
					"Cannot use input file to load weights. Cannot read in number of layers.");
		}

		consumeLetter(scan, "L");

		int[] temp = new int[numLayers];

		for (int x = 0; x < numLayers; x++) {
			if (scan.hasNextInt()) {
				temp[x] = scan.nextInt();
			} else {
				throw new BadLoadException(
						"Cannot use input file to load weights. Number of layers given in input file does not match that of the neural network.");
			}
		}

		for (int x = 0; x < numLayers; x++) {
			if (temp[x] != layerCounts[x]) {
				throw new BadLoadException(
						"Cannot use input file to load weights. The number of nodes in layer "
								+ (x + 1)
								+ " given in the input file does not equal the number of nodes found in the neural network.");
			}
		}

		consumeLetter(scan, "N");

		if (scan.hasNextInt()) {
			numWeights = scan.nextInt();
			weights = new double[numWeights];
		} else {
			throw new BadLoadException(
					"Cannot use input file to load weights. Cannot read in number of weights.");
		}

		consumeLetter(scan, "W");

		for (int x = 0; x < numWeights; x++) {
			if (scan.hasNextDouble()) {
				weights[x] = scan.nextDouble();
			} else {
				throw new BadLoadException(
						"Cannot use input file to load weights. Not enough weights provided.");
			}
		}

		return weights;
	}

	/**
	 * Consumes a letter in a text file and checks that it is the specified
	 * letter.
	 * 
	 * @param scan
	 *            The scanner to consume the next letter.
	 * @param letter
	 *            The letter it should be.
	 * @throws BadLoadException
	 *             If the scanner picks up a letter that is not the one
	 *             specified.
	 */
	private static void consumeLetter(Scanner scan, String letter)
			throws BadLoadException {
		if (scan.hasNext(letter)) {
			scan.next();
		} else {
			throw new BadLoadException(
					"Cannot use input file to load weights. Bad input file format.");
		}
	}

	// public static void main(String[] args) {
	// String fileName = "Out.txt";
	// int[] layerCounts = { 2, 3, 4 };
	// double[] weights = { 5, 6, 7, 8, 9, 10 };
	//
	// double[] out = null;
	//
	// try {
	// // System.out.println(Memory.save(fileName, layerCounts, weights));
	// out = Memory.load(fileName, layerCounts);
	// for (double d : out)
	// System.out.println(d);
	// }
	// catch (IOException e) {
	// e.printStackTrace();
	// System.exit(1);
	// }
	// catch (BadLoadException b) {
	// b.printStackTrace();
	// System.exit(1);
	// }
	// }

}
