package neural_net;

import java.io.Serializable;
import java.util.ArrayList;

import layers.*;
import tools.Factory;
import units.Type;
import units.Unit;
import units.UnitLinear;
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
 * @author ilias
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
		Layer newUnit = Factory.createLayer(outputType, nbOutputs);
		layers.add(newUnit);
	}
	
	public Network(Network network) {
		for(Layer layer : network.layers)
			this.layers.add(layer.clone());
	}
	
//	---LAYERS MANAGEMENT---
	
	public void addLayer(Layer layer) {
		addLayer(layers.size()-1, layer);
	}
	
	public void addLayer(int index, Layer layer) {
		layers.add(index, layer);
	}
	
	public void addLayers(Type layerType, int startIndex, int endIndex) {
		for(; startIndex<endIndex; startIndex++)
			layers.add(startIndex, Factory.createLayer(layerType));
	}
	
	public Layer getLayer(int index) {
		return layers.get(index);
	}
	
	public Layer getOutputLayer() {
		return layers.get(layers.size()-1);
	}
	
	public Layer[] getHiddenLayers() {
		Layer[] hidden = new Layer[layers.size()-1];
		return layers.subList(0, layers.size()-1).toArray(hidden);
	}

	
//	---NODE INTERFACE---
	
	@Override
	public Network clone() {
		return new Network(this);
	}
	
	@Override
	public void initConnections(int nbInputs) throws Exception {
		for(Layer layer : layers) {
			layer.initConnections(nbInputs);
			nbInputs = layer.getOutputsNumber();
		}	
	}

	@Override
	public Double[] compute(Double[] inputs) {
		for(Layer layer : layers)
			inputs = layer.compute(inputs);
		return inputs;
	}

	@Override
	public int getOutputsNumber() {
		return layers.get(layers.size()-1).getOutputsNumber();
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

}













