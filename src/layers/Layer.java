package layers;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

import neural_net.Node;

/**
 * This class implements a layer of a network.<br>
 * It contains some nodes, which can be units or other networks.<br>
 * The minimum possible layer is composed of one node.<br>
 * <b>There must be at least one node in the layer, for a network to be built.<b>
 * @author ilias
 */
public class Layer implements Node, Serializable {

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
	
	public Layer() {}
	
	public Layer(Layer layer) {
		cloneProperties(layer);
	}
	
	protected void cloneProperties(Layer layer) {
		for(Node node : layer.nodes)
			this.nodes.add(node.clone());
		
		this.connectionSize = layer.connectionSize;
		this.stride = layer.stride;
		
		if(layer.inputs != null)
			this.inputs = layer.inputs.clone();
	}
	
	public void addNode(Node n) throws Exception {
		checkInputs();
		nodes.add(n);
	}
	
	public void setConnectionSize(int size) throws Exception {
		checkInputs();
		this.connectionSize = size;
	}
	
	public void setStride(int stride) throws Exception {
		checkInputs();
		this.stride = stride;
	}
	
	private void checkInputs() throws Exception {
		if(inputs != null)
			throw new Exception("Cannot modify the layer after the initialization of connections.");
	}
	
	public Double[] getInputsBlock(int nodeIndex) {
		if(nodeIndex < 0 || nodeIndex >= nodes.size())
			return null;
		if(connectionSize==0)
			return inputs;
		int begin = nodeIndex*stride;
		int end = begin+connectionSize;
		return Arrays.copyOfRange(inputs, begin, end);
	}
	
	public void setInputs(Double[] inputs) throws Exception {
		initConnections(inputs.length);
		this.inputs = inputs;
	}
	
	public Node[] getNodes() {
		return nodes.toArray(new Node[nodes.size()]);
	}
	
//	---NODE INTERFACE---

	@Override
	public Layer clone() {
		return new Layer(this);
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
	public Double[] compute(Double[] inputs) {
		this.inputs = inputs;
		Double[] outputs = new Double[this.getOutputsNumber()];
		int outputCount = 0;
		for(int i=0; i<nodes.size(); i++) {
			Double[] res = nodes.get(i).compute(getInputsBlock(i));
			for(Double r : res)
				outputs[outputCount++] = r; 
		}
		return outputs;
	}
	

	@Override
	public void updateWeightsRandom(double probability) {
		for(Node node : nodes)
			node.updateWeightsRandom(probability);
	}

	@Override
	public void setAllWeights(double value) {
		for(Node node : nodes)
			node.setAllWeights(value);
	}
}
















