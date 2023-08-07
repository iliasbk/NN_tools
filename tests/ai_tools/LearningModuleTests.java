package ai_tools;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import layers.Layer;
import layers.LayerSigmoid;
import learning_module.LearningModule;
import neural_net.Network;
import units.Type;

class LearningModuleTests {

	
	@Test
	@DisplayName("Should initialize backpropagate the squared difference error")
	void squaredDifferenceBackpropagationTest() {
		Double[] expectedOutputs = {0.7, -0.8, 0.1};
		Double[] inputs = {0.4, -0.3, 0.5};
		Network net = new Network();
		
		// initialize the network from the test #4 of NetworkTests
		
		try {
			net = new Network(Type.SIGMOID, 3);
			
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
			
			// finalize
			net.initConnections(3);
			net.setAllWeights(0.4);
			
			net.compute(inputs);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		
		// Initialize the learning module
		
		try {
			
			LearningModule module = new LearningModule(net);
			module.propagateSquaredDifference(expectedOutputs);
			
			
			
			assertAll(()->{
				
			});
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}



















