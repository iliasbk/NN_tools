package units;

import neural_net.Node;

public class UnitSigmoid extends Unit {

	public UnitSigmoid() {}
	
	public UnitSigmoid(Unit unit) {
		cloneProperties(unit);
	}

//	---UNIT INTERFACE---
	
	@Override
	protected double activate(double value) {
		return 1/(1 + Math.exp(-value));
	}

//	---NODE INTERFACE---
	
	@Override
	public Node clone() {
		return new UnitSigmoid(this);
	}
}
