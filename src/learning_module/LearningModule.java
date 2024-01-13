package learning_module;

import nn_interface.Node;

/**
 * This class implements supervised learning mechanisms<br>
 * Particularly it computes and propagates errors, computed by its methods
 * 
 * @author Ilias Bakhbukh
 */
public class LearningModule {

	/**
	 * The node that is trained
	 */
	Node subject;
	
	/**
	 * Node number of outputs
	 */
	int outputsNumber;
	
	/**
	 * Class constructor
	 * @param subject the node to be trained
	 */
	public LearningModule(Node subject) {
		setSubject(subject);
	}
	
	/**
	 * Sets the subject node
	 * @param subject the node to be trained
	 */
	public void setSubject(Node subject) {
		this.subject = subject;
		this.outputsNumber = subject.getOutputsNumber();
	}
	
	/**
	 * Computes and propagates the squared difference derivative with respect to each output<br>
	 * <b>The program should compute node outputs before calling this method</b><br>
	 * Squared difference derivative = -2(y - f(inputs))<br>
	 * We multiply by -1 the derivative, as we are looking to minimize the loss function.
	 * @param expectedValues all the expected values of the outputs, given a set of exemplary inputs
	 * @throws Exception if the number of outputs values and expected values doesn't match
	 */
	public void propagateSquaredDifference(Double[] expectedValues) {
		
		Double[] squaredDifferenceDerivatives = subject.getOutputs();
		
		// compute the squared difference derivatives with respect to each output
		for(int i=0; i<outputsNumber; i++) {
			squaredDifferenceDerivatives[i] -= expectedValues[i];
			squaredDifferenceDerivatives[i] *= -2;
		}
		
		subject.backpropagate(squaredDifferenceDerivatives);
	}
	
}






















