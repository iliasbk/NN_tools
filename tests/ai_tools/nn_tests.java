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
import units.Type;

class nn_tests {
	
	Network net;

	@BeforeEach
	void setUp() throws Exception {
	}

	@AfterEach
	void tearDown() throws Exception {
	}
	
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
	@DisplayName("Should add a new hidden layer at the end of the hidden layers")
	void addAHiddenLayerInTheEnd() {
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
	}
	
	@Test
	@DisplayName("Should add hidden layers of the specified type in the specified interval")
	void addHiddenLayersInTheInterval() {
		final int START = 1;
		final int END = 3;
		
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
	}
	
}



















