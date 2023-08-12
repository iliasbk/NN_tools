package units;

import java.io.Serializable;

import neural_net.Node;

/**
 * This class implements a perceptron.<br>
 * It contains weights for each input and an additional weight for the y-intercept, which is always multiplied by 1.
 * @author ilias
 */
public abstract class Unit implements Node, Serializable {
	
	// UNIT'S ESSENTIALS
	protected Double[] weights;
	protected Double output;
	
	// BACKPTOPAGATE OPTIMIZATION
	protected Double[] inputs; //inputs to compute error gradient for each weight
	
	private final int OUTPUT_AMOUNT = 1;
	
	private final double LEARNING_RATE = 0.5;
	
	double learningRate = LEARNING_RATE;

	protected abstract double activate(double value);
	
	protected abstract double activationFunctionDerivative();
	
	protected void cloneProperties(Unit unit) {
		if(unit.weights != null)
			weights = unit.weights.clone();
		if(unit.output != null)
			output = unit.output;
	}
	
	public Double[] getWeights() {
		return weights;
	}
	
	public Double weightedSum(Double[] inputs) {
		Double sum = weights[0];
		for(int i=0;i<inputs.length;i++)
			sum += inputs[i]*weights[i+1];
		return sum;
	}
	
//	---NODE INTERFACE---
	
	@Override
	public abstract Node clone();
	
	@Override
	public void initConnections(int nbInputs) throws Exception {
		if(nbInputs <= 0)
			throw new Exception("Number of inputs should be greater than zero");
		
		weights = new Double[nbInputs+1];
		for(int i=0;i<weights.length;i++)
			weights[i] = Math.random() - 0.5; // initialize each weight to a value in the interval [-0.5, 0.5]
	}
	
	@Override
	public int getOutputsNumber() {
		return OUTPUT_AMOUNT;
	}

	@Override
	public Double[] getOutputs() {
		return new Double[] {output};
	}

	@Override
	public void updateWeightsRandom(double probability) {
		for(int i=0; i<weights.length; i++) {
			double rand = Math.random();
			if(probability >= rand) {
				// normalize rand (rand/probability = normalizedRand/1)
				rand = rand/probability - 0.5;
				// set the new weight, reusing rand
				weights[i] = (weights[i]+rand)%1;
			}
		}
	}
	
	@Override
	public void setAllWeights(double value) {
		for(int i=0; i<weights.length; i++)
			weights[i] = value;
	}

	@Override
	public Double[] compute(Double[] inputs) {
		this.inputs = inputs;
		this.output = activate(weightedSum(inputs));
		return new Double[]{output};
	}

	@Override
	public Double[] backpropagate(Double[] receivedGradients) {
		
		Double receivedGradient = receivedGradients[0];
		Double[] producedGradients = new Double[inputs.length];
		
		Double nodeGradient = receivedGradient * activationFunctionDerivative();
		
		// compute inputs' gradients
		for(int i=0; i<inputs.length; i++)
			producedGradients[i] = nodeGradient * weights[i+1];
		
//		learningRate *= 0.99995;
		
		// update weights
		weights[0] += nodeGradient * learningRate; // first weight's input is always = 1
		for(int i=1; i < weights.length; i++)
			weights[i] += nodeGradient * inputs[i-1] * learningRate;
		
		
		return producedGradients;
	}
}





















