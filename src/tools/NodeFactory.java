package tools;

import layers.Layer;
import layers.LayerLinear;
import layers.LayerSigmoid;
import units.Type;
import units.Unit;
import units.UnitLinear;
import units.UnitSigmoid;

/**
 * This class implements a factory for typed layers and units
 * 
 * @author Ilias Bakhbukh
 *
 */
public class NodeFactory {
	
	/**
	 * Creates a unit of a given type
	 * @param unitType the type of the unit
	 * @return a unit of the specified type
	 */
	public static Unit createUnit(Type unitType) {
		switch(unitType) {
		case SIGMOID: return new UnitSigmoid();
		default: return new UnitLinear();
		}
	}
	
	/**
	 * Creates a layer of a given type
	 * @param layerType the type of the layer
	 * @return a layer of the specified type
	 */
	public static Layer createLayer(Type layerType) {
		try {
			switch(layerType) {
			case LINEAR: return new LayerLinear();
			case SIGMOID: return new LayerSigmoid();
			}
		} catch(Exception e) {
			e.printStackTrace();
		}
		return new Layer();
	}

	/**
	 * Creates a layer of a given type with the specified number of output units
	 * @param layerType the type of the layer
	 * @param nbOutputs the number of the output units
	 * @return a layer of the specified type
	 * @throws Exception 
	 */
	public static Layer createLayer(Type layerType, int nbOutputs) throws Exception {
		switch(layerType) {
		case SIGMOID: return new LayerSigmoid(nbOutputs);
		default: return new LayerLinear(nbOutputs);
		}
	}

}
