package layers;

import java.io.Serializable;

import nn_interface.Node;
import units.UnitLinear;

/**
 * This is an extension to the Layer class that exclusively contains nodes of type UnitLinear<br>
 * By default it is created with at least one linear unit
 * 
 * @author Ilias Bakhbukh
 */
public class LayerLinear extends Layer implements Serializable {

	/**
	 * Class constructor that creates a layer with one unit
	 * @throws Exception
	 */
	public LayerLinear() throws Exception {
		this(1);
	}
	
	/**
	 * Class constructor that creates a layer as a copy of the specified layer
	 * @throws Exception
	 */
	public LayerLinear(LayerLinear layer) {
		cloneProperties(layer);
	}
	
	/**
	 * Class constructor that creates a layer with <i>n</> units
	 * @throws Exception
	 */
	public LayerLinear(int n) throws Exception {
		if(n < 1) 
			throw new Exception("The number of nodes in the layer should be greater than zero.");
		for(int i=0; i<n; i++)
			nodes.add(new UnitLinear());
	}
	
	@Override
	public void addNode(Node node) throws Exception {
		UnitLinear u = (UnitLinear) node;
		super.addNode(u);
	}
	
	@Override
	public LayerLinear clone() {
		return new LayerLinear(this);
	}
}




