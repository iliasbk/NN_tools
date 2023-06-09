package layers;

import java.io.Serializable;

import neural_net.Node;
import units.UnitLinear;
import units.UnitSigmoid;

public class LayerSigmoid extends Layer implements Serializable {
	
	public LayerSigmoid() {
		nodes.add(new UnitSigmoid());
	}

	public LayerSigmoid(LayerSigmoid layer) {
		cloneProperties(layer);
	}
	
	public LayerSigmoid(int nbNodes) throws Exception {
		if(nbNodes < 1) 
			throw new Exception("The number of nodes in the layer should be greater than zero.");
		for(int i=0; i<nbNodes; i++)
			nodes.add(new UnitSigmoid());
	}
	
	@Override
	public void addNode(Node n) throws Exception {
		UnitSigmoid u = (UnitSigmoid) n;
		super.addNode(u);
	}
	
	@Override
	public LayerSigmoid clone() {
		return new LayerSigmoid(this);
	}
	
}
