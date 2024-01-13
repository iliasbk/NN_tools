package nn_interface;

/**
 * This is an interface to each part of the network. 
 * <br><i>Layer</i>, <i>Unit</i> and <i>Network</i> classes, all implement this interface.
 * <br>It brings uniformity to the architecture, in a way that allows each part to be seen
 * as a node that can be initialized and trained by itself, without requiring to be part of a network.
 * 
 * @author Ilias Bakhbukh
 */
public interface Node {
	
	/**
	 * Makes a deep copy of the node
	 * @return deep copy of the node
	 */
	public Node clone();
	
	/**
	 * Initializes the number of connections of the node, 
	 * and adapts the number of connections of its successor nodes.<br>
	 * In units initializes the weights array.
	 * <br><b>This method must be called to complete the node initialization and enable its functionalities.</b>
	 * @param nbInputs the number of connections
	 * @throws Exception if the number of inputs doesn't meet node's necessities or if node's parameters aren't fully set up
	 */
	public void initConnections(int nbInputs) throws Exception;
	
	/**
	 * Get the outputs values of the node
	 * @return the output values of the node
	 */
	public Double[] getOutputs();
	
	/**
	 * Returns the number of outputs of the node
	 * @return the amount of output values
	 */
	public int getOutputsNumber();
	
	/**
	 * Randomly updates some units of the node
	 * @param probability in [0, 1] with which the node gets updated or not<br>(i.e. if p >= 1 - random)
	 */
	public void updateWeightsRandom(double probability);
	
	/**
	 * Sets all weights of all the units of the node to the specified value<br>
	 * @param value the value assigned to the weights
	 */
	public void setAllWeights(double value);
	
	/**
	 * This method sets the learning rate of every unit to the specified value
	 * @param value
	 */
	public void setLearningRate(double learningRate);
	
	/**
	 * Computes outputs from the specified inputs
	 * @param inputs inputs to be processed
	 * @return the preceding outputs
	 */
	public Double[] compute(Double[] inputs);
	
	/**
	 * Updates its weights according to the received error<br>
	 * Returns the calculated gradients with respect to node's inputs
	 * @param receivedGradients cumulative gradients with respect to each output
	 * @return the preceding gradients
	 */
	public Double[] backpropagate(Double[] receivedGradients);
	
}





















