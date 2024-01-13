package units;

import nn_interface.Node;

/**
 * This class implements a perceptron with a sigmoid activation function<br>
 * @author Ilias Bakhbukh
 */
public class UnitSigmoid extends Unit {

	public UnitSigmoid() {}
	
	public UnitSigmoid(UnitSigmoid unit) {
		cloneProperties(unit);
	}

//	---UNIT INTERFACE---
	
	@Override
	protected double activate(double x) {
		return 1/(1 + Math.exp(-x));
	}
	
	@Override
	protected double activationFunctionDerivative() {
		// f'(x) = exp(-x) / (1 + exp(-x))^2 
		//       = f(x) * ( 1 - f(x) )
		// where f(x) = output
		return output * ( 1 - output );
	}

//	---NODE INTERFACE---
	
	@Override
	public Node clone() {
		return new UnitSigmoid(this);
	}

}
