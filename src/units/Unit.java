package units;

import java.io.Serializable;

import neural_net.Node;

/**
 * This class implements a perceptron.<br>
 * It has 
 * @author ilias
 */
public abstract class Unit implements Node, Serializable {
	
	protected Double[] weights;
	
	protected Double output;
	
	private final int OUTPUT_AMOUNT = 1;
	
	private final double LEARNING_RATE = 0.05;
	
	protected void cloneProperties(Unit unit) {
		if(unit.weights != null)
			weights = unit.weights.clone();
		if(unit.output != null)
			output = unit.output;
	}

	protected abstract double activate(double value);
	
	public Double[] getWeights() {
		return weights;
	}
	
//	---NODE INTERFACE---
	
	@Override
	public abstract Node clone();
	
	@Override
	public void initConnections(int nbInputs) {
		weights = new Double[nbInputs+1];
		for(int i=0;i<weights.length;i++)
			weights[i] = Math.random() - 0.5; // initialize each weight to a value in the interval [-0.5, 0.5]
	}

	@Override
	public Double[] compute(Double[] inputs) {
		double val = weights[0];
		for(int i=0;i<inputs.length;i++)
			val += inputs[i]*weights[i+1];
		return new Double[]{this.activate(val)};
	}
	
	@Override
	public int getOutputsNumber() {
		return OUTPUT_AMOUNT;
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
}













