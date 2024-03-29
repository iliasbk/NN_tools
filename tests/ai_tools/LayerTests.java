package ai_tools;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.reflect.Field;
import java.util.Arrays;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import layers.Layer;
import layers.LayerSigmoid;

class LayerTests {

	@Test
	@DisplayName("Should return the right inputs")
	void getInputsBlockTest() {
		try {
			int nbNode = 4;
			int connectionSize = 2;
			int stride = 1;
			Layer layer = new LayerSigmoid(nbNode);
			
			layer.setConnectionSize(connectionSize);
			layer.setStride(stride);
			
			layer.initConnections(5);
			
			Double[] inputs = {0.1, 0.2, 0.3, 0.4, 0.5};
			Double[][] inputBlocks = {{0.1, 0.2}, {0.2, 0.3}, {0.3, 0.4}, {0.4, 0.5}};
			
			assertAll(() -> {
				for(int i=0; i<nbNode; i++) {
					Double[] in = layer.getInputsBlock(inputs, i);
					Double[] inTest = inputBlocks[i];
					assertEquals(in.length, inTest.length);
					for(int v=0; v<in.length;v++)
						assertEquals(in[v], inTest[v]);
				}
			});
			
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
