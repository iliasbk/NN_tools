package ai_tools;

import static org.junit.jupiter.api.Assertions.*;

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
			int outNb = 4;
			int connectionSize = 2;
			int stride = 1;
			Layer layer = new LayerSigmoid(outNb);
			
			layer.setConnectionSize(connectionSize);
			layer.setStride(stride);
			
			layer.initConnections(5);
			
			Double[] inputs = {0.1, 0.2, 0.3, 0.4, 0.5};
			
			layer.setInputs(inputs);
			
			assertAll(() -> {
				for(int i=0; i<outNb; i++) {
					Double[] in = layer.getInputsBlock(i);
					Double[] inTest = Arrays.copyOfRange(inputs, i*stride, i+connectionSize);
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
