package units;

import neural_net.Node;

public class UnitLinear extends Unit {
	
	public static void main(String[] args) {
		UnitLinear u1 = new UnitLinear();
		u1.initConnections(2);
		UnitLinear u2 = u1.clone();
		
		Double[] w1 = u1.getWeights().clone();
		Double[] w2 = u2.getWeights().clone();
		
		u1.setAllWeights(0.5);
		
		for(Double w : w1) 
			System.out.println("old1 : "+w);
		System.out.println();
		for(Double w : w2) 
			System.out.println("old2 : "+w);
		System.out.println();
		for(Double w : u1.getWeights()) 
			System.out.println("new1 : "+w);
		System.out.println();
		for(Double w : u2.getWeights()) 
			System.out.println("new2 : "+w);
	}
	
	public UnitLinear() {}
	
	public UnitLinear(Unit unit) {
		cloneProperties(unit);
	}

//	---UNIT INTERFACE---
	
	@Override
	protected double activate(double value) {
		return value;
	}

//	---NODE INTERFACE---
	
	@Override
	public UnitLinear clone() {
		return new UnitLinear(this);
	}
	
}
