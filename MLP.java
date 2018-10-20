import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

/**
 * Multilayer perceptron.
 */
public class MLP {

	/** The index of the hidden layer blocked neuron in each training cycle. */
	private static int blocked = -1;

	/** Pseudo-random number generator instance. */
	private static final Random PRNG = new Random();

	/** Number of independent experiments. */
	private static final int NUMBER_OF_EXPERIMENTS = 30;

	/** Number of training cycles. */
	private static final int NUMBER_OF_TRAINING_CYCLES = 1000;

	/** Weigths permutation rate. */
	private static final double WEIGHTS_PERMUTATION_RATE = 0.01D;

	/** List of layers. */
	private ArrayList<Layer> layers;

	/** Delta for weights correction. */
	private ArrayList<float[][]> weightsDelta;

	/** Gradients. */
	private ArrayList<float[]> gradients;

	/** Index of the layer which will be used for weights swap. */
	private int swapLayerIndex = -1;

	/** Index of the first neuron which will be used for weights swap. */
	private int swapNuronIndex1 = -1;

	/** Index of the second neuron which will be used for weights swap. */
	private int swapNuronIndex2 = -1;

	/** Index of the weight which will be swapped. */
	private int swapWeightIndex = -1;

	/**
	 * Propagate the inputs through all neural network and return the outputs.
	 * 
	 * @param inputs
	 * @return
	 */
	public float[] evaluate(float[] inputs) {
		assert (false);

		float outputs[] = new float[inputs.length];

		for (int i = 0; i < layers.size(); ++i) {
			outputs = layers.get(i).evaluate(inputs);
			inputs = outputs;
		}

		return outputs;
	}

	/**
	 * Evaluate network error for particular desired output.
	 * 
	 * @param networkOutput
	 *            Network output.
	 * @param desiredOutput
	 *            Desired output.
	 * 
	 * @return Network error.
	 */
	private float evaluateError(float networkOutput[], float desiredOutput[]) {
		/* Add bias to input if necessary. */
		float desired[];
		if (desiredOutput.length != networkOutput.length) {
			desired = Layer.add_bias(desiredOutput);
		} else {
			desired = desiredOutput;
		}

		assert (networkOutput.length == desired.length);

		float error = 0;
		for (int i = 0; i < networkOutput.length; ++i) {
			error += (networkOutput[i] - desired[i]) * (networkOutput[i] - desired[i]);
		}

		return error;
	}

	/**
	 * This function calculate the quadratic error for the given examples/results
	 * sets.
	 * 
	 * @param examples
	 * @param results
	 * @return
	 */
	public float evaluateQuadraticError(ArrayList<float[]> examples, ArrayList<float[]> results) {
		assert (false);

		float error = 0;
		for (int i = 0; i < examples.size(); ++i) {
			error += evaluateError(evaluate(examples.get(i)), results.get(i));
		}

		return error;
	}

	private void evaluateGradients(float[] results) {
		/* For each neuron in each layer. */
		for (int c = layers.size() - 1; c >= 0; --c) {
			for (int i = 0; i < layers.get(c).size(); ++i) {
				if (c == layers.size() - 1) {
					/* If it's output layer neuron. */
					gradients.get(c)[i] = 2 * (layers.get(c).getOutput(i) - results[0])
							* layers.get(c).getActivationDerivative(i);
				} else {
					/* If it's neuron of the previous layers. */
					float sum = 0;
					for (int k = 1; k < layers.get(c + 1).size(); ++k) {
						sum += layers.get(c + 1).getWeight(k, i) * gradients.get(c + 1)[k];
					}

					gradients.get(c)[i] = layers.get(c).getActivationDerivative(i) * sum;
				}
			}
		}
	}

	/** Reset delta values for each weight. */
	private void resetWeightsDelta() {
		for (int c = 0; c < layers.size(); ++c) {
			for (int i = 0; i < layers.get(c).size(); ++i) {
				float weights[] = layers.get(c).getWeights(i);

				for (int j = 0; j < weights.length; ++j) {
					weightsDelta.get(c)[i][j] = 0;
				}
			}
		}
	}

	/** Evaluate delta values for each weight. */
	private void evaluateWeightsDelta() {
		for (int c = 1; c < layers.size(); ++c) {
			for (int i = 0; i < layers.get(c).size(); ++i) {
				float weights[] = layers.get(c).getWeights(i);

				for (int j = 0; j < weights.length; ++j) {
					weightsDelta.get(c)[i][j] += gradients.get(c)[i] * layers.get(c - 1).getOutput(j);
				}
			}
		}
	}

	/**
	 * Update weight between the layers.
	 * 
	 * @param learningRate
	 *            Learning rate constant.
	 */
	private void updateWeights(float learningRate) {
		for (int c = 0; c < layers.size(); ++c) {
			for (int i = 0; i < layers.get(c).size(); ++i) {
				if (c == 1 && blocked == i) {
					continue;
				}

				float weights[] = layers.get(c).getWeights(i);
				for (int j = 0; j < weights.length; ++j) {
					layers.get(c).setWeight(i, j,
							layers.get(c).getWeight(i, j) - (learningRate * weightsDelta.get(c)[i][j]));
				}
			}
		}
	}

	/**
	 * Do single cycle of back-propagation.
	 * 
	 * @param input
	 *            Examples set.
	 * @param output
	 *            Learning results.
	 * @param learningRate
	 *            Learning rate constant.
	 */
	private void batchBackPropagation(ArrayList<float[]> input, ArrayList<float[]> output, float learningRate) {
		resetWeightsDelta();

		/* Examples should be shuffled in order not ANN not to learn order pattern. */
		long seed = PRNG.nextInt(Integer.MAX_VALUE);
		Collections.shuffle(input, new Random(seed));
		Collections.shuffle(output, new Random(seed));

		/* Do a single training step. */
		for (int l = 0; l < input.size(); ++l) {
			evaluate(input.get(l));
			evaluateGradients(output.get(l));
			evaluateWeightsDelta();
		}

		updateWeights(learningRate);
	}

	/**
	 * This function implements a batched back propagation algorithm.
	 * 
	 * @param examples
	 *            Examples set.
	 * @param results
	 *            Training results.
	 * @param learningRate
	 *            Learning rate constant.
	 */
	private void learn(ArrayList<float[]> examples, ArrayList<float[]> results, float learningRate) {
		assert (false);

		float error = Float.POSITIVE_INFINITY;
		do {
			batchBackPropagation(examples, results, learningRate);
			error = evaluateQuadraticError(examples, results);
		} while (error > 0.0001f);
	}

	/** Permutate two randomly selected weights. */
	private void permutateRandomWeights() {
		/* Select random layer except the first one. */
		swapLayerIndex = 1 + PRNG.nextInt(layers.size() - 1);

		/* Select two random different neurons from the previous layer. */
		do {
			swapNuronIndex1 = PRNG.nextInt(layers.get(swapLayerIndex).size());
			swapNuronIndex2 = PRNG.nextInt(layers.get(swapLayerIndex).size());
		} while (swapNuronIndex1 == swapNuronIndex2);

		float weights1[] = layers.get(swapLayerIndex).getWeights(swapNuronIndex1);
		float weights2[] = layers.get(swapLayerIndex).getWeights(swapNuronIndex2);

		/* Swap randomly selected weights. */
		swapWeightIndex = PRNG.nextInt(Math.min(weights1.length, weights2.length));
		float buffer = weights1[swapWeightIndex];
		weights1[swapWeightIndex] = weights2[swapWeightIndex];
		weights2[swapWeightIndex] = buffer;
	}

	/**
	 * Swap weights back after random swap.
	 */
	private void swapBackWeights() {
		if (swapLayerIndex == -1) {
			return;
		}
		if (swapNuronIndex1 == -1) {
			return;
		}
		if (swapNuronIndex2 == -1) {
			return;
		}
		if (swapWeightIndex == -1) {
			return;
		}

		float weights1[] = layers.get(swapLayerIndex).getWeights(swapNuronIndex1);
		float weights2[] = layers.get(swapLayerIndex).getWeights(swapNuronIndex2);
		float buffer = weights1[swapWeightIndex];
		weights1[swapWeightIndex] = weights2[swapWeightIndex];
		weights2[swapWeightIndex] = buffer;

		swapLayerIndex = -1;
		swapNuronIndex1 = -1;
		swapNuronIndex2 = -1;
		swapWeightIndex = -1;
	}

	/**
	 * Main constructor.
	 * 
	 * @param topology
	 */
	public MLP(int topology[]) {
		/* Create the required layers. */
		layers = new ArrayList<Layer>();
		for (int i = 0; i < topology.length; ++i) {
			layers.add(new Layer(i == 0 ? topology[i] : topology[i - 1], topology[i], PRNG));
		}

		weightsDelta = new ArrayList<float[][]>();
		for (int i = 0; i < topology.length; ++i) {
			weightsDelta.add(new float[layers.get(i).size()][layers.get(i).getWeights(0).length]);
		}

		gradients = new ArrayList<float[]>();
		for (int i = 0; i < topology.length; ++i) {
			gradients.add(new float[layers.get(i).size()]);
		}
	}

	private static void regularTraining(int[] topology, ArrayList<float[]> input, ArrayList<float[]> output) {
		float values[][] = new float[NUMBER_OF_EXPERIMENTS][NUMBER_OF_TRAINING_CYCLES];

		for (int c = 0; c < NUMBER_OF_EXPERIMENTS; c++) {
			MLP mlp = new MLP(topology);

			for (int r = 0; r < NUMBER_OF_TRAINING_CYCLES; r++) {
				/* Single training step. */
				mlp.learn(input, output, 0.3f);

				/* Single evaluation step. */
				values[c][r] = mlp.evaluateQuadraticError(input, output);
			}
		}

		/* Console output. */
		for (int r = 0; r < NUMBER_OF_TRAINING_CYCLES; r++) {
			for (int c = 0; c < NUMBER_OF_EXPERIMENTS; c++) {
				System.out.print(values[c][r]);
				System.out.print("\t");
			}
			System.out.println();
		}
	}

	private static void blockedNeuronTraining(int[] topology, ArrayList<float[]> input, ArrayList<float[]> output) {
		float values[][] = new float[NUMBER_OF_EXPERIMENTS][NUMBER_OF_TRAINING_CYCLES];

		for (int c = 0; c < NUMBER_OF_EXPERIMENTS; c++) {
			MLP mlp = new MLP(topology);

			for (int r = 0; r < NUMBER_OF_TRAINING_CYCLES; r++) {
				/* Select index of a neuron to block in the hidden layer. */
				blocked = PRNG.nextInt(topology[1]);

				/* Single training step. */
				mlp.learn(input, output, 0.3f);

				/* Single evaluation step. */
				values[c][r] = mlp.evaluateQuadraticError(input, output);
			}
		}

		/* Console output. */
		for (int r = 0; r < NUMBER_OF_TRAINING_CYCLES; r++) {
			for (int c = 0; c < NUMBER_OF_EXPERIMENTS; c++) {
				System.out.print(values[c][r]);
				System.out.print("\t");
			}
			System.out.println();
		}
	}

	private static void weightsPermutationTraining(int[] topology, ArrayList<float[]> input,
			ArrayList<float[]> output) {
		float values[][] = new float[NUMBER_OF_EXPERIMENTS][NUMBER_OF_TRAINING_CYCLES];

		for (int c = 0; c < NUMBER_OF_EXPERIMENTS; c++) {
			MLP mlp = new MLP(topology);

			for (int r = 0; r < NUMBER_OF_TRAINING_CYCLES; r++) {
				/* Single training step. */
				mlp.learn(input, output, 0.3f);

				if (Math.random() < WEIGHTS_PERMUTATION_RATE) {
					mlp.permutateRandomWeights();
				} else {
					mlp.swapBackWeights();
				}

				/* Single evaluation step. */
				values[c][r] = mlp.evaluateQuadraticError(input, output);
			}
		}

		/* Console output. */
		for (int r = 0; r < NUMBER_OF_TRAINING_CYCLES; r++) {
			for (int c = 0; c < NUMBER_OF_EXPERIMENTS; c++) {
				System.out.print(values[c][r]);
				System.out.print("\t");
			}
			System.out.println();
		}
	}

	/**
	 * Application single entry point.
	 * 
	 * @param args
	 *            Command line arguments.
	 */
	public static void main(String[] args) throws IOException {
		/*
		 * layer 1: input layer - 2 neurons
		 * 
		 * layer 2: hidden layer - 10 neurons
		 * 
		 * layer 3: output layer - 1 neuron
		 */
		int topology[] = { 2, 10, 1, };

		/* Input-out put pairs. */
		ArrayList<float[]> input = new ArrayList<float[]>();
		ArrayList<float[]> output = new ArrayList<float[]>();
		for (int i = 0; i < 4; ++i) {
			input.add(new float[2]);
			output.add(new float[1]);
		}

		/* Fill the examples set. */
		input.get(0)[0] = -1;
		input.get(0)[1] = 1;
		output.get(0)[0] = 1;

		input.get(1)[0] = 1;
		input.get(1)[1] = 1;
		output.get(1)[0] = -1;

		input.get(2)[0] = 1;
		input.get(2)[1] = -1;
		output.get(2)[0] = 1;

		input.get(3)[0] = -1;
		input.get(3)[1] = -1;
		output.get(3)[0] = -1;

		regularTraining(topology, input, output);
		System.out.println();
		// blockedNeuronTraining(topology, input, output);
		// System.out.println();
		weightsPermutationTraining(topology, input, output);
		System.out.println();
	}
}
