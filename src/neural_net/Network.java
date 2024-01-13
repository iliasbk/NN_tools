package neural_net;

import java.io.Serializable;
import java.util.ArrayList;

import layers.*;
import nn_interface.Node;
import tools.NodeFactory;
import units.Type;
import units.UnitSigmoid;

/**
 * This class implements a neural network.<br>
 * It is composed of layers which contain the nodes of the network.<br>
 * The minimum possible network is composed of one layer and one unit.<br>
 * <b>There's always an output layer, which cannot be removed</b>
 * <br>
 * <br>To build a network:
 * <br>1. Initialize a <i>Network</i> object
 * <br>2. Create <i>layers</i>
 * <br>3. Insert <i>nodes</i> in the layers ( Which can be units, layers or other networks)
 * <br>4. Define the <i>connection size</i> and the <i>stride</i> of each layer (by default it's fully connected with the previous layer)
 * <br>5. Insert <i>layers</i> in the <i>network</i> in the desired order
 * <br>6. Finally call <i>initConnections</i> method to denote the end of the building process and to initialize the connections
 * <br><b>(The network cannot be modified after the call to <i>initConnections</i>)</b>
 * 
 * @author Ilias Bakhbukh
 */
public class Network implements Node, Serializable {
	
	public static void main(String[] args) throws Exception {
		Network net = new Network(Type.LINEAR, 3);
		Layer hidden1 = new LayerSigmoid(5);
		Layer hidden2 = new LayerSigmoid(4);
		hidden2.setConnectionSize(2);
		hidden2.setStride(1);
		net.addLayer(hidden1);
		net.addLayer(hidden2);
		net.initConnections(4);
		Double[] out = net.compute(new Double[] {0.2, -0.5, 0.7, 0.33});
		for(Double o : out)
			System.out.println(o);
		
		Network net2 = new Network(Type.SIGMOID, 3);
		Layer hidden = new Layer();
		hidden.addNode(net);
		hidden.addNode(new UnitSigmoid());
		hidden.setConnectionSize(4);
		hidden.setStride(4);
		net2.addLayer(hidden);
		net2.initConnections(8);
		
		out = net2.compute(new Double[] {0.2, -0.5, 0.7, 0.33, -0.9, -0.2, 0.7, 0.5});
		for(Double o : out)
			System.out.println(o);

		System.out.println();
		System.out.println();
		
		Double[] graidents = net2.backpropagate(new Double[] {0.1, -0.5, 0.9});
		for(Double o : graidents)
			System.out.println(o);
	}
		
	
	/**
	 * Hidden layers of the network + an output layer
	 * <i>There's always an output layer, which cannot be removed</i>
	 */
	private ArrayList<Layer> layers = new ArrayList<Layer>();
	
	/**
	 * Creates a network with an output layer.
	 */
	public Network() {
		layers.add(new Layer());
	}
	
	/**
	 * Creates a network with an output layer of the right type with the right number of units
	 * @param outputType the type of the output layer
	 * @param nbOutputs the number of units in the output layer
	 * @throws Exception 
	 */
	public Network(Type outputType, int nbOutputs) throws Exception {
		Layer newUnit = NodeFactory.createLayer(outputType, nbOutputs);
		layers.add(newUnit);
	}
	
	/**
	 * Class constructor that creates a copy of the specified network
	 * @param network
	 */
	public Network(Network network) {
		for(Layer layer : network.layers)
			this.layers.add(layer.clone());
	}
	
//	---LAYERS MANAGEMENT---
	
	/**
	 * Adds the specified layer at the end of the hidden layers
	 * @param layer - layer to be inserted
	 */
	public void addLayer(Layer layer) {
		addLayer(layers.size()-1, layer);
	}
	
	/**
	 * Adds the specified layer at the position <i>index</i>
	 * @param index - index at which the specified layer is to be inserted
	 * @param layer - layer to be inserted
	 */
	public void addLayer(int index, Layer layer) {
		layers.add(index, layer);
	}
	
	/**
	 * Inserts layers of the specified type, starting from <i>startIndex</i> included up to <i>endIndex</i> excluded
	 * @param layerType - type of the layers
	 * @param startIndex - first index
	 * @param endIndex - last index
	 */
	public void addLayers(Type layerType, int startIndex, int endIndex) {
		for(; startIndex<endIndex; startIndex++)
			layers.add(startIndex, NodeFactory.createLayer(layerType));
	}
	
	/**
	 * Returns the layer at the specified position
	 * @param index - index of the layer to return
	 * @return the layer at the specified position
	 */
	public Layer getLayer(int index) {
		return layers.get(index);
	}
	
	/**
	 * Returns the output layer
	 * @return the output layer
	 */
	public Layer getOutputLayer() {
		return layers.get(layers.size()-1);
	}
	
	/**
	 * Returns the array of hidden layers, that is all the layers excluding the last, the output layer
	 * @return hidden layers
	 */
	public Layer[] getHiddenLayers() {
		Layer[] hidden = new Layer[layers.size()-1];
		return layers.subList(0, layers.size()-1).toArray(hidden);
	}
	
	/**
	 * Replaces the output layer
	 * @param layer the new output layer
	 */
	public void setOutputLayer(Layer layer) {
		setLayer(layers.size()-1, layer);
	}
	
	/**
	 * Replaces the layer at the specified position with the specified layer
	 * @param index position of the layer to replace
	 * @param layer the new layer to be set at the specified position
	 */
	public void setLayer(int index, Layer layer) {
		layers.set(index, layer);
	}

	
//	---NODE INTERFACE---
	
	@Override
	public Network clone() {
		return new Network(this);
	}
	
	@Override
	public void initConnections(int nbInputs) throws Exception {
		if(layers.size() <= 0)
			throw new Exception("The network should contain at least one layer");
		if(nbInputs <= 0)
			throw new Exception("Number of inputs should be greater than zero");
		
		for(Layer layer : layers) {
			layer.initConnections(nbInputs);
			nbInputs = layer.getOutputsNumber();
		}	
	}
	
	@Override
	public Double[] getOutputs() {
		return getOutputLayer().getOutputs();
	}

	@Override
	public int getOutputsNumber() {
		return getOutputLayer().getOutputsNumber();
	}

	@Override
	public void updateWeightsRandom(double d) {
		for(Layer layer : layers)
			layer.updateWeightsRandom(d);
	}

	@Override
	public void setAllWeights(double value) {
		for(Layer layer : layers)
			layer.setAllWeights(value);
	}

	@Override
	public void setLearningRate(double learningRate) {
		for(Layer layer : layers)
			layer.setLearningRate(learningRate);
	}

	@Override
	public Double[] compute(Double[] inputs) {
		for(Layer layer : layers)
			inputs = layer.compute(inputs);
		return inputs;
	}

	@Override
	public Double[] backpropagate(Double[] receivedGradients) {
		// travel layers starting from the end
		for(int l=layers.size(); --l>=0;)
			receivedGradients = layers.get(l).backpropagate(receivedGradients);
		return receivedGradients;
	}

}













