package org.nd4j.linalg.api.ops.impl.shape;

import lombok.val;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Split op
 */
public class Split extends DynamicCustomOp {

    private int numSplit;

    @Override
    public String opName() {
        return "split";
    }

    @Override
    public String tensorflowName() {
        return "Split";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
       val numSplits = (int) attributesForNode.get("num_split").getI();
       this.numSplit = numSplits;
       addIArgument(numSplits);

    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("numSplit",numSplit);
        return ret;
    }



    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String,Map<String,PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> map = new HashMap<>();

        val numSplit = PropertyMapping.builder()
                .tfAttrName("num_split")
                .propertyNames(new String[]{"numSplit"})
                .build();

        map.put("numSplit",numSplit);

        ret.put(tensorflowName(),map);
        ret.put(onnxName(),map);

        return ret;
    }

}
