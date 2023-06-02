package layers;

import java.util.ArrayList;

import neural_net.Node;
import units.Unit;

/**
 * This class implements a layer of a network.<br>
 * It contains some nodes, which can be units or other networks.<br>
 * The minimum possible layer is composed of one node.<br>
 * <b>There must be at least one node in the layer, for a network to be built.<b>
 * @author ilias
 */
public class Layer implements Node {

	protected ArrayList<Node> nodes = new ArrayList<Node>();

	protected Double[] inputs;
	
	/**
	 * Number of connections of each node in the layer with the inputs of the previous layer. <br>
	 * <b>ex</b> : connectionSize=3, then each node will connect to only 3 adjacent inputs starting from the first. <br>
	 * connectionSize = 0 means that it will connect to <b>all</b> the inputs.
	 */
	protected int connectionSize = 0;
	
	/**
	 * Spacing between the first input of each node <br>
	 * <b>ex</b> : connectionSize=2, stride=1, then the first node connects to inputs 0 and 1 
	 *  and the second to the inputs 1 and 2 <br>
	 * <b>ex</b> : connectionSize=3, stride=3, then the first node connects to inputs 0, 1, 2 
	 *  and the second to the inputs 3, 4, 5
	 */
	protected int stride = 0;
	
	public void addNode(Node n) {
		nodes.add(n);
	}
	
	public void setConnectionSize(int size) throws Exception {
		if(inputs == null)
			throw new Exception("Cannot set the connection size after the initialization of connections.");
		
		this.connectionSize = size;
	}
	
	public void setStride(int stride) {
		this.stride = stride;
	}
	
//	---NODE INTERFACE---

	@Override
	public Node clone() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void initConnections(int nbInputs) throws Exception {
		int nbConnections = nbInputs;
		
		if(connectionSize > 0) {
			int neccessaryNb = connectionSize+stride*(nodes.size()-1);
			if(nbInputs != neccessaryNb)
				throw new Exception("This number of inputs doen't meet layer' properties. "
						+ "The necessary amount of inputs is "+neccessaryNb);
			
			nbConnections = connectionSize;
		}
		
		inputs = new Double[nbInputs];
		
		for(Node n : nodes)
			n.initConnections(nbConnections);
	}

	@Override
	public int getOutputsNumber() {
		int total = 0;
		for(Node n : nodes)
			total += n.getOutputsNumber();
		return total;
	}

	@Override
	public void compute(Double[] inputs) {
		// TODO Auto-generated method stub
		
	}
}
















