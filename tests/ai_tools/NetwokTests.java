package ai_tools;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import layers.Layer;
import layers.LayerLinear;
import layers.LayerSigmoid;
import neural_net.Network;
import nn_interface.Node;
import units.Type;

class NetwokTests {
	
	/**
	 * Considering that the numbers are rounded to 4 digits after the decimal point
	 */
	final double TEST_MARGIN = 0.0001;
	
	@Test
	@DisplayName("Should initialize a network with only an output layer of the type any")
	void networkWithAnOutputLayerOfTypeAny() {
		Network net = new Network();
		Layer layer0 = net.getLayer(0);
		Layer outputLayer = net.getOutputLayer();
		assertEquals(layer0, outputLayer);
		assertEquals(outputLayer.getClass(), new Layer().getClass());
	}

	@Test
	@DisplayName("Should initialize a network with only an output layer of the linear type")
	void networkWithAnOutputLayerOfTypeLinear() throws Exception {
		Network net = new Network(Type.LINEAR, 1);
		Layer layer0 = net.getLayer(0);
		Layer outputLayer = net.getOutputLayer();
		assertEquals(layer0, outputLayer);
		assertEquals(outputLayer.getClass(), new LayerLinear().getClass());
	}
	
	@Test
	@DisplayName("Should initialize a network with only an output layer of the sigmoid type")
	void networkWithAnOutputLayerOfTypeSigmoid() throws Exception {
		Network net = new Network(Type.SIGMOID, 1);
		Layer layer0 = net.getLayer(0);
		Layer outputLayer = net.getOutputLayer();
		assertEquals(layer0, outputLayer);
		assertEquals(outputLayer.getClass(), new LayerSigmoid().getClass());
	}

	@Test
	@DisplayName("Should add a new hidden layer at the specified index position")
	void addAHiddenLayerAtIndex() {
		try {
			Network net = new Network();
			Layer layer1 = new LayerSigmoid();
			Layer layer2 = new LayerLinear();
			Layer layer3 = new Layer();
			net.addLayer(0, layer1);
			net.addLayer(1, layer2);
			net.addLayer(1, layer3);
			
			Layer hidden1 = net.getLayer(0);
			Layer hidden2 = net.getLayer(2);
			Layer hidden3 = net.getLayer(1);
			Layer output = net.getLayer(3);
			
			assertEquals(layer1, hidden1);
			assertEquals(layer2, hidden2);
			assertEquals(layer3, hidden3);
			assertEquals(output, net.getOutputLayer());
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	@Test
	@DisplayName("Should add a new hidden layer at the end of the hidden layers")
	void addAHiddenLayerInTheEnd() {
		try {
			Network net = new Network();
			Layer layer1 = new LayerSigmoid();
			Layer layer2 = new LayerLinear();
			net.addLayer(layer1);
			net.addLayer(layer2);
			
			Layer hidden1 = net.getLayer(0);
			Layer hidden2 = net.getLayer(1);
			Layer output = net.getLayer(2);
			
			assertEquals(layer1, hidden1);
			assertEquals(layer2, hidden2);
			assertEquals(output, net.getOutputLayer());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	@Test
	@DisplayName("Should add hidden layers of the specified type in the specified interval")
	void addHiddenLayersInTheInterval() {
		final int START = 1;
		final int END = 3;
		
		try {
			Network net = new Network();
			Layer layer1 = new LayerSigmoid();
			Layer layer2 = new LayerSigmoid();
			net.addLayer(layer1);
			net.addLayer(layer2);
			
			// randomize the type of added layers
			Layer newLayer;
			if(Math.random()>0.5) {
				net.addLayers(Type.SIGMOID, START, END);
				newLayer = new LayerSigmoid();
			}else {
				net.addLayers(Type.LINEAR, START, END);
				newLayer = new LayerLinear();
			}
			
			Layer[] hidden = net.getHiddenLayers();
			
			assertEquals(layer1, hidden[0]);
			assertEquals(hidden[1].getClass(), newLayer.getClass());
			assertEquals(hidden[2].getClass(), newLayer.getClass());
			assertEquals(layer2, hidden[3]);
		
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	@Test
	@DisplayName("Should compute the right outputs of a network composed of units")
	void computeOutputsTest1() {
		try {
			Double[] testInputs = {0.1, -0.4, 0.7};
			Double weightsValue = 0.5;
			int nbOut = 2;
			
			// create network
			
			Network net = new Network(Type.SIGMOID, nbOut);
			
			Layer layer1 = new LayerSigmoid(4);
			Layer layer2 = new LayerLinear(3);
			net.addLayer(layer1);
			net.addLayer(layer2);
			
			net.initConnections(3);
			
			net.setAllWeights(weightsValue);
			
			Double[] out = net.compute(testInputs);
			
			// Compute testOutputs
			Double[] testOut = {0.96284, 0.96284};
			
			assertAll(()->{
				for(int i=0; i<out.length; i++)
					assertTrue(Math.abs(out[i]-testOut[i]) < TEST_MARGIN);
			});
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	@Test
	@DisplayName("Should compute the right outputs of a network composed of units and other networks")
	void computeOutputsTest2() {
		
		try {
			Double[] testInputs = {0.1, -0.4, 0.7, 0.1};
			Double weightsValue = 0.5;
			int nbOut = 3;
			
			// create the network
			
			Network net = new Network(Type.LINEAR, nbOut);
			
			// build layers
			
			Layer layer1 = new Layer();
			layer1.setConnectionSize(3);
			layer1.setStride(1);
			
			Layer layer2 = new LayerSigmoid(3);
			
			// create the subnetwork of the previous test
			
			Network subNet = new Network(Type.SIGMOID, 2);
			
			Layer subLayer1 = new LayerSigmoid(4);
			Layer subLayer2 = new LayerLinear(3);
			subNet.addLayer(subLayer1);
			subNet.addLayer(subLayer2);
			
			layer1.addNode(subNet);
			layer1.addNode(subNet);
			
			// add layers
			
			net.addLayer(layer1);
			net.addLayer(layer2);

			net.initConnections(4);
			
			net.setAllWeights(weightsValue);
			
			Double[] out = net.compute(testInputs);
			
			// Compute testOutputs
			Double[] testOut = {1.87814, 1.87814, 1.87814};
			
			assertAll(()->{
				for(int i=0; i<out.length; i++) 
					assertTrue(Math.abs(out[i]-testOut[i]) < TEST_MARGIN);
			});
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
	
	@Test
	@DisplayName("Should compute the right outputs of a network composed of units")
	void computeOutputsTest3() {
		try {
			Double[] inputs = {0.4, -0.3, 0.5};
			Double[] testOut = {0.9238, 0.9238};
			
			Network net = new Network(Type.LINEAR, 2);
			net.addLayer(new LayerSigmoid(3));
			Layer outputLayer = net.getOutputLayer();
			outputLayer.setConnectionSize(2);
			outputLayer.setStride(1);
			
			net.initConnections(3);
			net.setAllWeights(0.4);
			
			Double[] out = net.compute(inputs);
			
			assertAll(()->{
				for(int i=0; i<out.length; i++)
					assertTrue(Math.abs(out[i]-testOut[i]) < TEST_MARGIN);
			});
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	@Test
	@DisplayName("Should compute the right outputs of a network composed of units and subnets")
	void computeOutputsTest4() {	
		try {

			Double[] inputs = {0.4, -0.3, 0.5};
			Double[] testOut = {0.7575, 0.7575, 0.7575};
			
			Network net = new Network(Type.SIGMOID, 3);
			
			// add subnets of the previous test
			Network subnet = new Network(Type.LINEAR, 2);
			subnet.addLayer(new LayerSigmoid(3));
			Layer outputLayer = subnet.getOutputLayer();
			outputLayer.setConnectionSize(2);
			outputLayer.setStride(1);
			
			Layer subnetLayer = new Layer();
			subnetLayer.addNode(subnet);
			subnetLayer.addNode(subnet);
			
			net.addLayer(subnetLayer);
			
			// set connection size of the output layer
			outputLayer = net.getOutputLayer();
			outputLayer.setConnectionSize(2);
			outputLayer.setStride(1);
			
			// init connections
			net.initConnections(3);
			net.setAllWeights(0.4);
			
			Double[] out = net.compute(inputs);
			
			assertAll(()->{
				for(int i=0; i<out.length; i++) {
					assertTrue(Math.abs(out[i]-testOut[i]) <= TEST_MARGIN);
				}
			});
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
	
	
//	---NODE INTERFACE---
	
	@Test
	@DisplayName("Should return the right amount of outputs")
	void getOutputsNumberTest() {
		try {
			
			final int NB = 2;
			Network net = new Network(Type.SIGMOID, NB);
			int nbOut = net.getOutputsNumber();
			assertEquals(nbOut, NB);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
}



















