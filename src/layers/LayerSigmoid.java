package layers;

import java.io.Serializable;

import nn_interface.Node;
import units.UnitSigmoid;

/**
 * This is an extension to the Layer class that exclusively contains nodes of type UnitSigmoid<br>
 * By default it is created with at least one sigmoid unit
 * 
 * @author Ilias Bakhbukh
 */
public class LayerSigmoid extends Layer implements Serializable {
	
	/**
	 * Class constructor that creates a layer with one unit
	 * @throws Exception
	 */
	public LayerSigmoid() throws Exception {
		this(1);
	}

	/**
	 * Class constructor that creates a layer as a copy of the specified layer
	 * @throws Exception
	 */
	public LayerSigmoid(LayerSigmoid layer) {
		cloneProperties(layer);
	}
	
	/**
	 * Class constructor that creates a layer with <i>n</> units
	 * @throws Exception
	 */
	public LayerSigmoid(int n) throws Exception {
		if(n < 1) 
			throw new Exception("The number of nodes in the layer should be greater than zero.");
		for(int i=0; i<n; i++)
			nodes.add(new UnitSigmoid());
	}
	
	@Override
	public void addNode(Node node) throws Exception {
		UnitSigmoid u = (UnitSigmoid) node;
		super.addNode(u);
	}
	
	@Override
	public LayerSigmoid clone() {
		return new LayerSigmoid(this);
	}
	
}
