package learning_module;

import neural_net.Node;

public class LearningModule {

	Node subject;
	
	int outputsNumber;
	
	public LearningModule(Node subject) {
		setSubject(subject);
	}
	
	public void setSubject(Node subject) {
		this.subject = subject;
		this.outputsNumber = subject.getOutputsNumber();
	}
	
	/**
	 * Computes and propagates the squared difference derivative with respect to each output
	 * <br>Squared difference derivative = -2(y - f)
	 * <br>We multiply by -1 the derivative, as we are looking to MINIMIZE the loss function.
	 * @param expectedValues all the expected values of the outputs, given a set of exemplary inputs
	 * @throws Exception if the number of outputs values and expected values doesn't match
	 */
	public void propagateSquaredDifference(Double[] expectedValues) throws Exception {
		
		checkValuesNumber(expectedValues);
		
		Double[] squaredDifferenceDerivatives = subject.getOutputs();
		
		// compute the squared difference derivatives with respect to each output
		for(int i=0; i<outputsNumber; i++) {
			squaredDifferenceDerivatives[i] -= expectedValues[i];
			squaredDifferenceDerivatives[i] *= -2;
		}
		
		subject.backpropagate(squaredDifferenceDerivatives);
	}
	
	private void checkValuesNumber(Double[] values) throws Exception {
		if(values.length != outputsNumber)
			throw new Exception("Quantities of values do not match, received "+values.length+" expected "+outputsNumber);
	}
}






















