package units;

import neural_net.Node;

public class UnitSigmoid extends Unit {

	public UnitSigmoid() {}
	
	public UnitSigmoid(UnitSigmoid unit) {
		cloneProperties(unit);
	}

//	---UNIT INTERFACE---
	
	@Override
	protected double activate(double value) {
		return 1/(1 + Math.exp(-value));
	}
	
	@Override
	protected double activationFunctionDerivative(double value) {
		// f'(x) = exp(-x) / (1 + exp(-x))^2
		double exp = Math.exp(-value);
		double denom = Math.pow(1+exp, 2);
		return exp / denom;
	}

//	---NODE INTERFACE---
	
	@Override
	public Node clone() {
		return new UnitSigmoid(this);
	}

}
