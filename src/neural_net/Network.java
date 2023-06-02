package neural_net;

import java.util.ArrayList;

import layers.*;
import tools.Factory;
import units.Type;
import units.Unit;
import units.UnitLinear;

/**
 * This class implements a neural network.<br>
 * It is composed of layers which contain the nodes of the network.<br>
 * The minimum possible network is composed of one layer and one unit.<br>
 * <b>There's always an output layer, which cannot be removed</b>
 * <br>
 * <br>To build a network:
 * <br>1. Initialize a <i>Network</i> object
 * <br>2. Insert <i>layers</i>, of the desired type and in the desired order
 * <br>3. Define the <i>connection size</i> and the <i>stride</i> of each layer (by default it's fully connected)
 * <br>4. Insert <i>nodes</i> in the layers ( Which can be units or other networks)
 * <br>5. Finally call <i>initConnections</i> method to denote the end of the building process and to initialize the connections
 * <br><b>(The network cannot be modified after the call to <i>initConnections</i>)</b>
 * 
 * @author ilias
 */
public class Network implements Node {
	
	public static void main(String[] args) throws Exception {
		Layer layer = new LayerSigmoid();
		Unit unit = new UnitLinear();
		layer.addNode(unit);
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
	public Node clone() {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public void initConnections(int nbInputs) throws Exception {
		int nbConnections = nbInputs;
		for(Layer layer : layers) {
			layer.initConnections(nbConnections);
			nbConnections = layer.getOutputsNumber();
		}
			
	}

	@Override
	public void compute(Double[] inputs) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public int getOutputsNumber() {
		return layers.get(layers.size()-1).getOutputsNumber();
	}

}




