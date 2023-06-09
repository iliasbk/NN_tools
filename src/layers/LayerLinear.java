package layers;

import java.io.Serializable;

import neural_net.Node;
import units.Unit;
import units.UnitLinear;

public class LayerLinear extends Layer implements Serializable {

	public LayerLinear() {
		nodes.add(new UnitLinear());
	}
	
	public LayerLinear(LayerLinear layer) {
		cloneProperties(layer);
	}
	
	public LayerLinear(int nbNodes) throws Exception {
		if(nbNodes < 1) 
			throw new Exception("The number of nodes in the layer should be greater than zero.");
		for(int i=0; i<nbNodes; i++)
			nodes.add(new UnitLinear());
	}
	
	public void addNode(Node n) throws Exception {
		UnitLinear u = (UnitLinear) n;
		super.addNode(u);
	}
	
	@Override
	public LayerLinear clone() {
		return new LayerLinear(this);
	}
}




