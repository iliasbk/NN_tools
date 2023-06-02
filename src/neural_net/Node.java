package neural_net;

public interface Node {
	
	/**
	 * Makes a deep copy of the node
	 * @return deep copy of the node
	 */
	public Node clone();
	
	/**
	 * Initializes the number of connections<br>
	 * In layers initializes the inputs array. In units initializes the weights array.
	 * @param nb the number of connections
	 * @throws Exception 
	 */
	public void initConnections(int nbInputs) throws Exception;
	
	/**
	 * Compute outputs
	 */
	public void compute(Double[] inputs);
	
	/**
	 * Get the number of outputs of the node
	 * @return the amount of output values
	 */
	public int getOutputsNumber();
	
}
