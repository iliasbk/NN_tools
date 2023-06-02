package units;

import neural_net.Node;

/**
 * This class implements a network perceptron.<br>
 * It has 
 * @author ilias
 */
public abstract class Unit implements Node {
	
	Double[] weights;
	
	Double output;
	
	final int OUTPUT_AMOUNT = 1;

	@Override
	public abstract Node clone();
	
	@Override
	public void initConnections(int nbInputs) {
		weights = new Double[nbInputs+1];
	}

	@Override
	public abstract void compute(Double[] inputs);
	
	@Override
	public int getOutputsNumber() {
		return OUTPUT_AMOUNT;
	}
	
}
