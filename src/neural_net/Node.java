package neural_net;

public interface Node {
	
	/**
	 * Makes a deep copy of the node
	 * @return deep copy of the node
	 */
	public Node clone();
	
	/**
	 * Initializes the number of connections of the node, 
	 * and adapts the number of connections of its successor nodes.<br>
	 * In layers initializes the inputs array. In units initializes the weights array.
	 * <br><b>This method must be called to complete the node initialization and enable its functionalities.</b>
	 * @param nbInputs the number of connections
	 * @throws Exception if the number of inputs doesn't meet node's necessities or if the node's parameters aren't completed
	 */
	public void initConnections(int nbInputs) throws Exception;
	
	/**
	 * Get the outputs values of the node
	 * @return
	 */
	public Double[] getOutputs();
	
	/**
	 * Get the number of outputs of the node
	 * @return the amount of output values
	 */
	public int getOutputsNumber();
	
	/**
	 * 
	 * @param probability in [0, 1] with which the node gets updated or not<br>(i.e. if p >= 1 - random)
	 */
	public void updateWeightsRandom(double probability);
	
	/**
	 * Sets all weights of a network to a certain value<br>
	 * Mainly useful for testing
	 * @param value the value of the tests
	 */
	public void setAllWeights(double value);
	
	/**
	 * Compute outputs
	 */
	public Double[] compute(Double[] inputs);
	
	/**
	 * Updates its weights accordingly to the received error,
	 * and sends the calculated error to its inputs
	 * @param error the received error
	 */
	public Double[] backpropagate(Double[] receivedGradients);
	
}





















