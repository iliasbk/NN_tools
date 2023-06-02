package layers;

import neural_net.Node;
import units.Unit;
import units.UnitLinear;

public class LayerLinear extends Layer {

	public LayerLinear() {
		nodes.add(new UnitLinear());
	}
	
	public LayerLinear(int nbNodes) throws Exception {
		if(nbNodes < 1) 
			throw new Exception("The number of nodes in the layer should be greater than zero.");
		for(int i=0; i<nbNodes; i++)
			nodes.add(new UnitLinear());
	}
	
	public void addNode(Node n) {
		UnitLinear u = (UnitLinear) n;
		nodes.add(u);
	}
}
