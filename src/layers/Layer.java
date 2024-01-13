package layers;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

import nn_interface.Node;

/**
 * This class implements a layer of a network.<br>
 * It can contain arbitrarily many nodes, which can be units, layers or other networks.<br>
 * <b>It must contain at least one node in order to be initialized.</b><br>
 * @author Ilias Bakhbukh
 */
public class Layer implements Node, Serializable {

	/**
	 * This collection contains layer nodes
	 */
	protected ArrayList<Node> nodes = new ArrayList<Node>();
	
	/**
	 * This variable represents the number of inputs in this layer
	 */
	protected int nbInputs = 0;
	
	/**
	 * Number of connections of each node in the layer with the inputs. <br>
	 * <b>ex</b> : connectionSize=3, then each node will connect to only 3 adjacent inputs starting from the first. <br>
	 * connectionSize = 0 means that the layer is fully connected.
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
	
	/**
	 * The constructor of an empty layer
	 */
	public Layer() {}
	
	/**
	 * Layer copy constructor
	 * @param layer
	 */
	public Layer(Layer layer) {
		cloneProperties(layer);
	}
	
	/**
	 * This method transforms the current layer into a copy of the specified layer
	 * @param layer the layer to copy
	 */
	protected void cloneProperties(Layer layer) {
		for(Node node : layer.nodes)
			this.nodes.add(node.clone());
		
		this.connectionSize = layer.connectionSize;
		this.stride = layer.stride;
	}
	
	/**
	 * This method adds the copy of <i>node</i> to the layer 
	 * @param node - model node
	 * @throws Exception if the layer is already set up, 
	 * cannot modify the layer after the number of inputs is defined
	 */
	public void addNode(Node node) throws Exception {
		addNode(node, 1);
	}
	
	/**
	 * This method adds <i>n</i> copies of <i>node</i> to the layer
	 * @param node - model node
	 * @param n - number of new nodes to add
	 * @throws Exception if the layer is already set up, 
	 * cannot modify the layer after the number of inputs is defined
	 */
	public void addNode(Node node, int n) throws Exception {
		checkInputs();
		for(int i=0; i<n; i++)
			nodes.add(node.clone());
	}
	
	/**
	 * This method sets the number of inputs to each node in the layer
	 * @param n - number of inputs of each node, <i>0</i> means the layer is fully connected
	 * @throws Exception if the layer is already set up, 
	 * cannot modify the layer after the number of inputs is defined
	 */
	public void setConnectionSize(int n) throws Exception {
		checkInputs();
		if(n <= 0)
			connectionSize = stride = 0;
		else
			connectionSize = n;
	}
	
	/**
	 * This method sets the stride between the first input of each node
	 * @param stride
	 * @throws Exception if the layer is already set up, 
	 * cannot modify the layer after the number of inputs is defined. 
	 * Or, if the connection size = 0 (layer is fully connected), then the stride can't be defined
	 */
	public void setStride(int stride) throws Exception {
		checkInputs();
		if(connectionSize == 0)
			throw new Exception("The layer is fully connected. Set connection size first.");
		this.stride = stride;
	}
	
	/**
	 * This method checks whether the number of inputs is defined, that is if the layer is fully set up
	 * @throws Exception if the layer is already set up
	 */
	private void checkInputs() throws Exception {
		if(nbInputs > 0)
			throw new Exception("Cannot modify the layer after the initialization of connections.");
	}
	
	/**
	 * This method attributes inputs to the specified layer node 
	 * according to layer inputs array and the index of the node
	 * @param inputs - layer inputs array
	 * @param nodeIndex - node's index in the layer
	 * @return node inputs array
	 */
	public Double[] getInputsBlock(Double[] inputs, int nodeIndex) {
		if(nodeIndex < 0 || nodeIndex >= nodes.size())
			return null;
		if(connectionSize==0)
			return inputs;
		int begin = nodeIndex*stride;
		int end = begin+connectionSize;
		return Arrays.copyOfRange(inputs, begin, end);
	}
	
	/**
	 * Returns nodes as an array
	 * @return layer nodes as an array
	 */
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
		if(nodes.size() <= 0)
			throw new Exception("The layer should contain at least one node");
		if(nbInputs <= 0)
			throw new Exception("Number of inputs should be greater than zero");
		
		int nbConnections = nbInputs;
		
		if(connectionSize > 0) {
			int neccessaryNb = connectionSize+stride*(nodes.size()-1);
			if(nbInputs != neccessaryNb)
				throw new Exception("This number of inputs doen't meet layer' properties. "
						+ "The necessary amount of inputs is "+neccessaryNb);
			
			nbConnections = connectionSize;
		}
		
		this.nbInputs = nbInputs;
		
		for(Node n : nodes)
			n.initConnections(nbConnections);
	}

	@Override
	public Double[] getOutputs() {
		Double[] outputs = new Double[getOutputsNumber()];
		int outputCount = 0;
		for(int i=0; i<nodes.size(); i++) {
			Double[] out = nodes.get(i).getOutputs();
			for(Double o : out)
				outputs[outputCount++] = o; 
		}
		return outputs;
	}

	@Override
	public int getOutputsNumber() {
		int total = 0;
		for(Node n : nodes)
			total += n.getOutputsNumber();
		return total;
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
	
	@Override
	public void setLearningRate(double learningRate) {
		for(Node node : nodes)
			node.setLearningRate(learningRate);
	}
	
	@Override
	public Double[] compute(Double[] inputs) {
		Double[] outputs = new Double[getOutputsNumber()];
		int outputCount = 0;
		for(int i=0; i<nodes.size(); i++) {
			Double[] res = nodes.get(i).compute(getInputsBlock(inputs, i));
			for(Double r : res)
				outputs[outputCount++] = r; 
		}
		return outputs;
	}

	@Override
	public Double[] backpropagate(Double[] receivedGradients) {
		
		Double[] producedGradients = new Double[nbInputs];
		Arrays.fill(producedGradients, 0.0);
		
		int nbProducedGradientsPerNode = connectionSize > 0 ? connectionSize : nbInputs;
		
		int nodeGradientsStart = 0;
		
		for(int n=0; n<nodes.size();n++) {
			
			Node node = nodes.get(n);
			
			int nodeGradientsEnd = nodeGradientsStart + node.getOutputsNumber();
			Double[] recievedNodeGradients = Arrays.copyOfRange(receivedGradients, nodeGradientsStart, nodeGradientsEnd);
			
			Double[] producedNodeGradients = node.backpropagate(recievedNodeGradients);
			
			int offset = stride * n;
			
			for(int i=0; i<nbProducedGradientsPerNode; i++) 
				producedGradients[i+offset] += producedNodeGradients[i];
			
			nodeGradientsStart = nodeGradientsEnd;
			
		}
		
		return producedGradients;
	}
}
















