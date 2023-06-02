package layers;

import neural_net.Node;
import units.UnitLinear;
import units.UnitSigmoid;

public class LayerSigmoid extends Layer {
	
	public LayerSigmoid() {
		nodes.add(new UnitSigmoid());
	}
	
	public LayerSigmoid(int nbNodes) throws Exception {
		if(nbNodes < 1) 
			throw new Exception("The number of nodes in the layer should be greater than zero.");
		System.out.println("ici");
		for(int i=0; i<nbNodes; i++)
			nodes.add(new UnitSigmoid());
	}
	
	public void addNode(Node n) {
		UnitSigmoid u = (UnitSigmoid) n;
		nodes.add(u);
	}
	
}
