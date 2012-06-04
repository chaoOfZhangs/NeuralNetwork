package com.network.memory;

import java.io.*;
import java.util.Scanner;

/**
 * Read and write a file using an explicit encoding. Removing the encoding from
 * this code will simply cause the system's default encoding to be used instead.
 * 
 * Modeled from code found on:
 * http://www.javapractices.com/topic/TopicAction.do?Id=42
 * 
 * @author Chao Zhang
 */
public final class ReadWriteTextFile {

	// Prints to the command line if set to true.
	private static final boolean LOG = false;

	/**
	 * Private constructor to prevent instantiation.
	 */
	private ReadWriteTextFile() {
	}

	/**
	 * Write to a file.
	 * 
	 * @param fileName
	 *            The name of the file to write to.
	 * @param text
	 *            The text to write to the file.
	 * @return True if written successfully, false otherwise.
	 * @throws IOException
	 */
	static boolean write(String fileName, String text) throws IOException {
		boolean successful = false;
		log("Writing to file named " + fileName + ".");
		Writer out = new OutputStreamWriter(new FileOutputStream(fileName));
		try {
			out.write(text);
			successful = true;
		} finally {
			out.close();
		}
		return successful;
	}

	/**
	 * Reads from a text file.
	 * 
	 * @param fileName
	 *            The name of the file to read from.
	 * @return A string containing the text read from the file.
	 * @throws IOException
	 * @throws FileNotFoundException
	 */
	static String read(String fileName) throws IOException,
			FileNotFoundException {
		log("Reading from file " + fileName + ".");

		StringBuilder outputText = new StringBuilder();

		String NL = System.getProperty("line.separator");

		Scanner scanner = new Scanner(new FileInputStream(fileName));
		try {
			while (scanner.hasNextLine()) {
				outputText.append(scanner.nextLine() + NL);
			}
		} finally {
			scanner.close();
		}
		log("Text read in: " + outputText);

		return outputText.toString();
	}

	/**
	 * Prints to the command line if the <code>LOG</code> is set to
	 * <code>true</code>.
	 * 
	 * @param aMessage
	 *            The message to print to the command line.
	 */
	private static void log(String aMessage) {
		if (LOG) {
			System.out.println(aMessage);
		}
	}
}
