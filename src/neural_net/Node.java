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
	 * @param nb the number of connections
	 * @throws Exception if the number of inputs doesn't meet node's necessities
	 */
	public void initConnections(int nbInputs) throws Exception;
	
	/**
	 * Compute outputs
	 */
	public Double[] compute(Double[] inputs);
	
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
	
}







