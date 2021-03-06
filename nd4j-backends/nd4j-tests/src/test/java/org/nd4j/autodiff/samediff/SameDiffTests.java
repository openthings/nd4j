package org.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.controlflow.While;
import org.nd4j.linalg.api.ops.impl.layers.Linear;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMaxDerivative;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.weightinit.impl.UniformInitScheme;
import org.nd4j.weightinit.impl.ZeroInitScheme;

import java.lang.reflect.Field;
import java.util.*;

import static org.junit.Assert.*;
import static org.junit.Assume.assumeNotNull;

/**
 * Created by agibsonccc on 4/11/17.
 */
@Slf4j
public class SameDiffTests {
    static {
        Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    public Map<String, INDArray> variablesForInput() {
        INDArray inputs = Nd4j.create(new double[][]{
                {0.52, 1.12, 0.77},
                {0.88, -1.08, 0.15},
                {0.52, 0.06, -1.30},
                {0.74, -2.49, 1.39}
        });

        INDArray labels = Nd4j.create(new double[]{1, 1, 0, 1}).reshape(4, 1);

        INDArray weights = Nd4j.zeros(3, 1);

        Map<String, INDArray> inputMap = new HashMap<>();
        inputMap.put("x", inputs);
        inputMap.put("w", weights);
        inputMap.put("y", labels);
        return inputMap;
    }




    @Test
    public void testAddArgsAndOutput() {
        SameDiff sameDiff = SameDiff.create();
        val varOne = sameDiff.var("one",Nd4j.ones(2));

    }

    @Test
    public void testEvalVariable() {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        INDArray twos = ones.add(ones);
        SDVariable inputOne = sameDiff.var("inputone", ones);
        SDVariable inputResult = inputOne.add("extravarname",inputOne);
        assertEquals(twos, inputResult.eval());
    }



    @Test
    public void testSum() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4));
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result = sameDiff.sum(x, 1);
        assertArrayEquals(arr.shape(), result.getShape());
    }



    @Test
    public void testReshape() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4)).reshape(2, 2);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result = sameDiff.reshape(x, 2, 2);
        assertArrayEquals(new int[]{2, 2}, result.getShape());

    }

    @Test
    public void testTranspose() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4));
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result = sameDiff.transpose(x);
        sameDiff.exec();
        assertArrayEquals(new int[]{4, 1}, result.getArr().shape());

    }


    @Test
    public void testDynamicOp() {
        SameDiff sameDiff = SameDiff.create();
        DynamicCustomOp dynamicCustomOp = DynamicCustomOp.
                sameDiffBuilder("testop",sameDiff)
                .addInputs(sameDiff.var("i1",new int[]{2,2}),
                        sameDiff.var("i2",new int[]{2,2}))
                .addOutputShape(new int[]{2,2})
                .addOutputShape(new int[]{2,3})
                .build();
        assertEquals(2,dynamicCustomOp.outputVariables().length);


    }




    @Test
    public void testDistance() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4)).reshape(2, 2);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", arr);
        SDVariable result = sameDiff.cosineSimilarity(x, y, 1);
        SDVariable addResult = result.add(result);

        assertArrayEquals(new int[]{1, 2}, result.getShape());
    }

    @Test
    public void testTensorGradMmul() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4)).reshape(2, 2);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", arr);
        SDVariable result = sameDiff.mmul(x, y);
        SDVariable otherResult = result.add(result);
        assertArrayEquals(new int[]{2, 2}, result.getShape());
    }



    @Test
    public void testEval() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 4, 4);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable sigmoid = sameDiff.sigmoid(x);
        INDArray assertion = Transforms.sigmoid(arr);
        INDArray[] eval = sameDiff.eval(Collections.singletonMap("x", arr));
        assertEquals(assertion, eval[0]);

    }




    @Test
    public void testUpdateVariableName() throws Exception {
        INDArray inArr = Nd4j.create(1,4);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable s = sd.tanh("s", in);

        List<SDVariable> l = sd.variables();
        assertEquals(2, l.size());      //Fails here: returns 3 (inc "tanh" variable that should have been replaced)

        for(SDVariable sdv : l ){
            String n = sdv.getVarName();
            assertTrue(n.equals("in") || n.equals("s"));
        }

        Field f = SameDiff.class.getDeclaredField("incomingArgsReverse");
        f.setAccessible(true);
        Map<String,String[]> incomingArgsReverse = (Map<String, String[]>) f.get(sd);

        for(Map.Entry<String,String[]> e : incomingArgsReverse.entrySet()){
            for(String str : e.getValue()){
                assertTrue( str, str.equals("in") || str.equals("s"));
            }
        }

        f = SameDiff.class.getDeclaredField("ougoingArgsReverse");      //Also: typo in the SameDiff class field name
        f.setAccessible(true);
        Map<String,String[]> outgoingArgsReverse = (Map<String, String[]>) f.get(sd);
        for(Map.Entry<String,String[]> e : outgoingArgsReverse.entrySet()){
            for(String str : e.getValue()){
                assertTrue( str, str.equals("in") || str.equals("s"));  //Also fails here due to "tanh" variable
            }
        }
    }

    @Test
    public void testFunctionInputsAndArgs() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable var = sameDiff.var("one",Nd4j.scalar(1.0));
        SDVariable variable2 = sameDiff.var("two",Nd4j.scalar(1.0));
        val sum = var.add(variable2);
        assertArrayEquals(new int[]{1,1},sum.getShape());


    }



    @Test
    public void testCrossSameDiffVariableInitWithAlloc() {
        SameDiff first = SameDiff.create();
        SameDiff second = SameDiff.create();


        SDVariable firstVar = first.var("one",new int[]{2,2});
        SDVariable secondVar = second.var(firstVar);
        assertTrue(firstVar.getArr() == secondVar.getArr());
        assertEquals(firstVar.getVarName(),secondVar.getVarName());

    }



    @Test
    public void testCrossSameDiffVariableInitWithPlaceHolder() {
        SameDiff first = SameDiff.create();
        SameDiff second = SameDiff.create();


        SDVariable firstVar = first.var("one",new int[]{2,2});
        SDVariable secondVar = second.var(firstVar);
        assumeNotNull(firstVar.getArr());

        assertTrue(firstVar.getArr() == secondVar.getArr());
        assertEquals(firstVar.getVarName(),secondVar.getVarName());

    }


    @Test
    public void testVariableArrayReference() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable arr = sameDiff.var("one",new int[]{2,2});
        assertArrayEquals(new int[]{2,2},arr.getShape());
        assumeNotNull(arr.getArr());
        assertArrayEquals(new int[]{2,2},arr.getArr().shape());

    }

    @Test
    public void testEvalAddSelf() {
        /**
         * Note this test fails yet due to needing
         * to validate simple cases like x * x
         * matching number of inputs.
         */
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 4, 4);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable sigmoid = x.mul(x);
        INDArray assertion = arr.mul(arr);
        INDArray[] eval = sameDiff.eval(Collections.singletonMap("x", arr));
        assertEquals(assertion, eval[0]);

    }

    @Test
    public void testEvalAdd() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 4, 4);
        INDArray yArr = arr.dup();
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", yArr);

        SDVariable sigmoid = x.mul(y);
        INDArray assertion = arr.mul(arr);
        Map<String, INDArray> vars = new HashMap<>();
        vars.put("x", arr);
        vars.put("y", yArr);
        INDArray[] eval = sameDiff.eval(vars);
        assertEquals(assertion, eval[0]);

    }


    @Test
    public void testTensorGradTensorMmul() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 8, 8)).reshape(2, 2, 2);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", arr);
        SDVariable result = sameDiff.tensorMmul(x, y, new int[][]{{0}, {1}});
        assertArrayEquals(ArrayUtil.getTensorMmulShape(new int[]{2, 2, 2}, new int[]{2, 2, 2}, new int[][]{{0}, {1}}), result.getShape());
        assertEquals(32, sameDiff.numElements());
    }

    @Test
    public void testDup() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 8, 8)).reshape(2, 2, 2);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", arr);
        SameDiff tg2 = sameDiff.dup();
    }


    @Test
    public void testLogGrad() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable input = sameDiff.var("x", Nd4j.linspace(1, 4, 4));
        SDVariable log = sameDiff.log(input);
        SDVariable sum = sameDiff.sum(log,Integer.MAX_VALUE);
        INDArray result = null;
        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> execBackwards = sameDiff.execBackwards();
        System.out.println(execBackwards);
        //INDArray assertion = Nd4j.create(new double[]{1, 0.5, 0.33, 0.25});
        // assertTrue(assertion.equalsWithEps(result, 1e-2));
    }


    @Test
    public void testElementWiseDivAndRDiv() {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        INDArray toDivBy = Nd4j.valueArrayOf(4, 0.25);
        Map<String, INDArray> xAndY = new HashMap<>();
        xAndY.put("x", ones);
        xAndY.put("y", toDivBy);
        sameDiff.defineFunction("div", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable x = sameDiff.var("x", inputs.get("x"));
                SDVariable y = sameDiff.var("y", inputs.get("y"));
                return new SDVariable[] {x.div(y)};
            }
        }, xAndY);

        sameDiff.defineFunction("rdiv", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable x = sameDiff.var("x", inputs.get("x"));
                SDVariable y = sameDiff.var("y", inputs.get("y"));
                return new SDVariable[] {x.rdiv(y)};
            }
        }, xAndY);


        INDArray assertionForDiv = Nd4j.valueArrayOf(4, 4.0);
        INDArray assertionForRDiv = Nd4j.valueArrayOf(4, 0.25);
        assertEquals(assertionForDiv, sameDiff.getFunction("div").execAndEndResult());
        assertEquals(assertionForRDiv, sameDiff.getFunction("rdiv").execAndEndResult());

    }


    @Test
    public void testNegativeGradient() {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        Map<String, INDArray> xAndY = new HashMap<>();
        xAndY.put("x", ones);
        sameDiff.defineFunction("neg", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable x = sameDiff.var("x", inputs.get("x"));
                return new SDVariable[] {sameDiff.neg(x)};
            }
        }, xAndY);

        INDArray assertionForDiv = Nd4j.valueArrayOf(4, -1);
        assertEquals(assertionForDiv, sameDiff.getFunction("neg").execAndEndResult());

    }



    @Test
    public void testSumOp() {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("x", sumInput);
        sameDiff.defineFunction("sum", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable sum = sameDiff.sum(input, 1);
                return new SDVariable[] {sum};
            }
        }, inputs);

        INDArray assertion = sumInput.sum(1);
        INDArray executions = sameDiff.execAndEndResult("sum");
        assertEquals(assertion, executions);
    }


    @Test
    public void testVariableReferenceNoFunction() {
        /**
         * Creating a variable should not create a differential function.
         */
        SameDiff sameDiff = SameDiff.create();
        SDVariable sdVariable = sameDiff.var("one",Nd4j.scalar(1.0));
        assumeNotNull(sameDiff.getVariable(sdVariable.getVarName()));
    }


    @Test
    public void testVariableWithFunction() {
        /**
         * A variable's function should be null
         * when just a variable but
         * have a function result
         * when the variable itself is the result of a function.
         *
         */
        SameDiff sameDiff = SameDiff.create();
        SDVariable sdVariable = sameDiff.var("one",Nd4j.scalar(1.0));
        SDVariable add = sdVariable.add(1.0);
        assertEquals(sameDiff.getVariable(add.getVarName()),add);
    }



    @Test
    public void testUpdateVariable() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable one = sameDiff.one("one",new int[]{1,1});
        sameDiff.updateVariableName(one.getVarName(),"one-diff");
        assertEquals(one.getArr(),sameDiff.getVariable("one-diff").getArr());
    }


    @Test
    public void testMulGradient() {
        INDArray arr1 = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray arr2 = Nd4j.linspace(1,4,4).reshape(2,2);

        INDArray gradAssertion =  Nd4j.ones(arr1.shape());
        INDArray scalar = Nd4j.scalar(1.0);
        INDArray aGradAssertion = Nd4j.create(new double[][]{
                {1,4},
                {9,16}
        });

        INDArray cGradAssertion = Nd4j.create(new double[][]{
                {1,2},
                {3,4}
        });

        INDArray wGradAssertion = Nd4j.create(new double[][]{
                {2,8},
                {18,32}
        });

        INDArray dGradAssertion = Nd4j.ones(2,2);

        SameDiff sameDiff = SameDiff.create();

        SDVariable sdVariable = sameDiff.var("a",arr1);
        SDVariable sdVariable1 = sameDiff.var("w",arr2);
        SDVariable varMulPre = sdVariable.mul("c",sdVariable1);
        SDVariable varMul = varMulPre.mul("d",sdVariable1);
        SDVariable sum = sameDiff.sum("ret",varMul,Integer.MAX_VALUE);

        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> mapListPair = sameDiff.execBackwards();

        SDVariable finalResult = sameDiff.grad(sum.getVarName());

        SDVariable cGrad = sameDiff.grad(varMulPre.getVarName());

        SDVariable mulGradResult = sameDiff.grad(varMul.getVarName());
        SDVariable aGrad = sameDiff.grad(sdVariable.getVarName());
        SDVariable wGrad = sameDiff.grad(sdVariable1.getVarName());
        SDVariable dGrad = sameDiff.grad(varMul.getVarName());

        INDArray scalarGradTest = finalResult.getArr();
        assertEquals(scalar,scalarGradTest);


        INDArray gradTest = mulGradResult.getArr();
        assertEquals(gradAssertion,gradTest);

        INDArray aGradTest = aGrad.getArr();
        assertEquals(aGradAssertion,aGradTest);

        INDArray cGradTest = cGrad.getArr();
        assertEquals(cGradAssertion,cGradTest);

        INDArray wGradTest = wGrad.getArr();
        assertEquals(wGradAssertion,wGradTest);

        INDArray dGradTest = dGrad.getArr();
        assertEquals(dGradAssertion,dGradTest);


    }


    @Test
    public void testLinearModule() {
        int nIn = 5;
        Linear linear = Linear.execBuilder()
                .nIn(nIn)
                .nOut(4)
                .weightInitScheme(new UniformInitScheme('f',nIn))
                .biasWeightInitScheme(new ZeroInitScheme('f'))
                .build();
        linear.exec(Nd4j.linspace(1,20,20).reshape(4,5));
        assertEquals(1,linear.numOutputArguments());

    }



    @Test
    public void testInPlaceAdd() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable toAdd = sameDiff.var("arr1",Nd4j.ones(2,2));
        SDVariable add = sameDiff.var("arr2",Nd4j.valueArrayOf(2,2,2.0));
        SDVariable result = toAdd.addi(add);
        INDArray result2 = sameDiff.execAndEndResult();
        INDArray arr = result.getArr();
        INDArray assertion = Nd4j.ones(2,2).addi(Nd4j.valueArrayOf(2,2,2.0));
        assertEquals(assertion,result2);
    }




    @Test
    public void testDefineFunctionArrayExistence() {
        SameDiff sameDiff = SameDiff.create();
        String testFunctionName = "testfunction";
        SDVariable[] inputVars = new SDVariable[] {
                sameDiff.var("one",new int[]{1,1}),
                sameDiff.var("two",new int[]{1,1}),

        };

        SameDiff functionDef = sameDiff.defineFunction(testFunctionName, new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                return new SDVariable[] {variableInputs[0].add(variableInputs[1])};
            }
        },inputVars);


        //1 input plus 2 outputs
        assertEquals(3,functionDef.variables().size());



    }


    @Test
    public void testWhileLoop() {
        SameDiff sameDiff = SameDiff.create();
        sameDiff.whileStatement(new SameDiff.DefaultSameDiffConditional(), new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable eqResult = sameDiff.neq(variableInputs[0],variableInputs[1]);
                return new SDVariable[]{eqResult};
            }
        }, new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable ret = variableInputs[1].addi(1.0);
                return new SDVariable[]{variableInputs[0],ret};
            }
        },new SDVariable[] {
                sameDiff.one("one",new int[]{1,1}),
                sameDiff.var("two",new int[]{1,1}),

        });

        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> exec = sameDiff.exec();
        assertFalse(exec.getRight().isEmpty());
        While function = (While) exec.getRight().get(exec.getRight().size() - 1);
        assumeNotNull(function.getOutputVars());
        assertEquals(1,function.getNumLooped());
        sameDiff.toString();
    }



    @Test
    public void testIfStatementTrueBodyBackwards() {
        SameDiff sameDiff = SameDiff.create();
        SameDiff.SameDiffFunctionDefinition conditionBody = new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable sum = sameDiff.sum(variableInputs[0],Integer.MAX_VALUE);
                SDVariable result = sameDiff.gt(sum,1.0);
                return new SDVariable[] {result};
            }
        };


        SameDiff.SameDiffFunctionDefinition trueBody = new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable add = variableInputs[0].add(1.0);
                return new SDVariable[] {add};
            }
        };

        SameDiff.SameDiffFunctionDefinition falseBody =  new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable sub = variableInputs[0].sub(1.0);
                return new SDVariable[] {sub};
            }
        };

        //true body trigger
        SDVariable[] firstInputs = new SDVariable[] {
                sameDiff.var("one",new int[]{1,1})

        };



        sameDiff.ifStatement(new SameDiff.DefaultSameDiffConditional(), conditionBody, trueBody, falseBody,firstInputs);
        sameDiff.execBackwards();
        SameDiff grad = sameDiff.getFunction("grad");
       /* If ifBlock = (If) grad.getFunction(new int[]{1},new int[]{2});
        SameDiff assertComparision = SameDiff.create();
        SDVariable initialInput = assertComparision.zero("zero",new int[]{1,1});
        initialInput.addi(1.0);
        assumeNotNull(ifBlock.getTrueBodyExecuted());
        assertTrue(ifBlock.getTrueBodyExecuted());
        assertEquals(Nd4j.scalar(1.00),initialInput.getArr());
        assertEquals(Nd4j.scalar(1.0),ifBlock.getLoopBodyExecution().getVariableForVertexId(2).getArr());
*/
    }



    @Test
    public void testWhileBackwards() {
        SameDiff sameDiff = SameDiff.create();
        sameDiff.whileStatement(new SameDiff.DefaultSameDiffConditional(), new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable eqResult = sameDiff.neq(variableInputs[0],variableInputs[1]);
                return new SDVariable[]{eqResult};
            }
        }, new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable ret = variableInputs[1].addi(1.0);
                return new SDVariable[]{variableInputs[0],ret};
            }
        },new SDVariable[] {
                sameDiff.one("one",new int[]{1,1}),
                sameDiff.var("two",new int[]{1,1}),

        });

        sameDiff.execBackwards();
        SameDiff exec = sameDiff.getFunction("grad");
        System.out.println(exec);
    }


    @Test
    public void testIfStatementTrueBody() {
        SameDiff sameDiff = SameDiff.create();

        SameDiff.SameDiffFunctionDefinition conditionBody = new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable sum = sameDiff.sum(variableInputs[0],Integer.MAX_VALUE);
                SDVariable result = sameDiff.gt(sum,1.0);
                return new SDVariable[] {result};
            }
        };


        SameDiff.SameDiffFunctionDefinition trueBody = new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable add = variableInputs[0].add(1.0);
                return new SDVariable[] {add};
            }
        };

        SameDiff.SameDiffFunctionDefinition falseBody =  new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable sub = variableInputs[0].sub(1.0);
                return new SDVariable[] {sub};
            }
        };

        //true body trigger
        SDVariable[] firstInputs = new SDVariable[] {
                sameDiff.var("one",new int[]{1,1})

        };



        sameDiff.ifStatement(new SameDiff.DefaultSameDiffConditional(), conditionBody, trueBody, falseBody,firstInputs);
        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> exec = sameDiff.exec();

    }


    @Test
    public void testIfStatementFalseBody() {
        SameDiff sameDiff = SameDiff.create();

        SameDiff.SameDiffFunctionDefinition conditionBody = new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable sum = sameDiff.sum(variableInputs[0],Integer.MAX_VALUE);
                SDVariable result = sameDiff.gt(sum,1.0);
                return new SDVariable[] {result};
            }
        };


        SameDiff.SameDiffFunctionDefinition trueBody = new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable add = variableInputs[0].add(1.0);
                return new SDVariable[] {add};
            }
        };

        SameDiff.SameDiffFunctionDefinition falseBody =  new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable sub = variableInputs[0].sub(1.0);
                return new SDVariable[] {sub};
            }
        };


        //false body trigger
        SDVariable[] secondInputs = new SDVariable[] {
                sameDiff.setupFunction(sameDiff.var("two",new int[]{1,1}))

        };

        sameDiff.ifStatement(new SameDiff.DefaultSameDiffConditional(), conditionBody, trueBody, falseBody,secondInputs);

        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> exec = sameDiff.exec();




    }


    @Test
    public void testAutoBroadcastAddMatrixector() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray row = Nd4j.ones(2);
        INDArray assertion = arr.add(1.0);
        SDVariable left = sameDiff.var("arr",arr);
        SDVariable right = sameDiff.var("row",row);
        SDVariable test = left.add(right);
        sameDiff.exec();
        assertEquals(assertion,test.getArr());
    }


    @Test
    public void testNegativeOneShape() {
        val sd = SameDiff.create();
        val var = sd.var("test",new int[] {-1,3});
        assertNull(var.getShape());
        assertTrue(var.isPlaceHolder());
    }

    @Test
    public void testShapeResolutionMinus1(){
        int nIn = 3;
        int nOut = 4;

        int minibatch = 3;

        for(boolean useMinus1 : new boolean[]{false, true}) {
            log.info("Starting: {}", (useMinus1 ? "minibatch -1" : "minibatch 3"));

            int[] inShape;
            if(useMinus1){
                inShape = new int[]{-1, nIn};
            } else {
                inShape = new int[]{minibatch, nIn};
            }
            int[] wShape = new int[]{nIn, nOut};
            int[] bShape = new int[]{1, nOut};

            SameDiff sd = SameDiff.create();
            SDVariable layerInput = sd.var("in", inShape);
            SDVariable weights = sd.var("W", wShape);
            SDVariable bias = sd.var("b", bShape);

            SDVariable mmul = sd.mmul("mmul", layerInput, weights);
            SDVariable z = mmul.add("z", bias);
            SDVariable out = sd.sigmoid("out", z);

            Map<String, INDArray> m = new HashMap<>();
            INDArray in = Nd4j.rand(new int[]{minibatch, nIn});
            INDArray w = Nd4j.rand(wShape);
            INDArray b = Nd4j.rand(bShape);

            sd.associateArrayWithVariable(in, sd.getVariable("in"));
            sd.associateArrayWithVariable(w, sd.getVariable("W"));
            sd.associateArrayWithVariable(b, sd.getVariable("b"));

            INDArray outArr = sd.execAndEndResult();

            assertArrayEquals(new int[]{minibatch, nOut}, outArr.shape());
        }
    }


    @Test
    public void testRunLogisticRegression() {
        Map<String,INDArray> vars = this.variablesForInput();
        SameDiff outside = SameDiff.create();
        outside.defineFunction("activate", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                sameDiff.enableDebugMode();
                SDVariable x = sameDiff.var("x",inputs.get("x"));
                SDVariable w = sameDiff.var("w",inputs.get("w"));
                SDVariable y = sameDiff.var("y",inputs.get("y"));
                SDVariable activation = sameDiff.sigmoid("activation",sameDiff.mmul("mmul",x,w));
                SDVariable oneMinusY = y.rsub("oneminusy",1.0);
                SDVariable oneMinusPredictions = activation.rsub("oneminusactivations",1.0);
                SDVariable outputTimesY = y.mul("output * y",activation);
                SDVariable yHat = oneMinusPredictions.mul("yhat",oneMinusY);
                SDVariable probs = outputTimesY.add("probs",yHat);
                SDVariable logProbs = sameDiff.log("logprob",probs);
                SDVariable ret = sameDiff.sum("totalsum",logProbs,Integer.MAX_VALUE);
                SDVariable ret2 = sameDiff.neg("negtotalsum",ret);
                return new SDVariable[] {ret2};
            }
        },vars);

        SameDiff activation = outside.getFunction("activate");
        int epochsToRun = 5;
        double lr = 0.1;
     /*   for(int i = 0; i < epochsToRun; i++) {
            activation.execBackwards();
            INDArray wGrad = activation.grad("w").getArr().reshape(vars.get("w").shape());
            vars.get("w").subi(wGrad.mul(lr));
            System.out.println("Score: " + activation.getVariable("negtotalsum").getArr());
        }*/

    }


    @Test
    public void testSoftmaxRegression() {
        Map<String,INDArray> vars = this.variablesForInput();
        SameDiff outside = SameDiff.create();
        outside.defineFunction("activate", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                sameDiff.enableDebugMode();
                SDVariable x = sameDiff.var("x",inputs.get("x"));
                SDVariable w = sameDiff.var("w",inputs.get("w"));
                SDVariable y = sameDiff.var("y",inputs.get("y"));
                SDVariable activation = sameDiff.softmax("activation",sameDiff.mmul("mmul",x,w));
                SDVariable ret = sameDiff.sum("totalsum",activation,Integer.MAX_VALUE);
                SDVariable ret2 = sameDiff.neg("negtotalsum",ret);
                return new SDVariable[] {ret2};
            }
        },vars);


        /**
         * Backwards should be:
         * neg score
         * sum sum of log
         * log (log probs)
         * add
         * mul
         * mul
         * rsub (predictions)
         * sigmoid
         * rsub
         * matrix multiply
         *
         */


        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> opsBackward = outside.getFunction("activate").execBackwards();
        SameDiff gradSameDiff = outside.getFunction("activate").getFunction("grad");

        SDVariable gradWrtX = outside.getFunction("activate").grad("x");
        SDVariable gradWrtW = outside.getFunction("activate").grad("w");
        assertNotNull(gradWrtX);
        assertNotNull(gradWrtW);

        INDArray wGradAssertion = Nd4j.create(new double[]{0,0,0}).reshape(3,1);
        assertEquals(wGradAssertion,outside.getFunction("activate").grad("w").getArr());
        //note here that the gradients here end up being some weird really low eps where it
        //isn't exactly zero
        //        assertEquals(inputAssertion,outside.getFunction("activate").grad("x").getArr());



        System.out.println(gradWrtX);
        System.out.println(gradWrtW);


    }



    @Test
    public void testLogisticRegression() {
        Map<String,INDArray> vars = this.variablesForInput();
        SameDiff outside = SameDiff.create();
        outside.defineFunction("activate", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                sameDiff.enableDebugMode();
                SDVariable x = sameDiff.var("x",inputs.get("x"));
                SDVariable w = sameDiff.var("w",inputs.get("w"));
                SDVariable y = sameDiff.var("y",inputs.get("y"));
                SDVariable activation = sameDiff.sigmoid("activation",sameDiff.mmul("mmul",x,w));
                SDVariable oneMinusY = y.rsub("oneminusy",1.0);
                SDVariable oneMinusPredictions = activation.rsub("oneminusactivations",1.0);
                SDVariable outputTimesY = y.mul("output * y",activation);
                SDVariable yHat = oneMinusPredictions.mul("yhat",oneMinusY);
                SDVariable probs = outputTimesY.add("probs",yHat);
                SDVariable logProbs = sameDiff.log("logprob",probs);
                SDVariable ret = sameDiff.sum("totalsum",logProbs,Integer.MAX_VALUE);
                SDVariable ret2 = sameDiff.neg("negtotalsum",ret);
                return new SDVariable[] {ret2};
            }
        },vars);


        /**
         * Backwards should be:
         * neg score
         * sum sum of log
         * log (log probs)
         * add
         * mul
         * mul
         * rsub (predictions)
         * sigmoid
         * rsub
         * matrix multiply
         *
         */


        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> opsBackward = outside.getFunction("activate").execBackwards();
        SameDiff gradSameDiff = outside.getFunction("activate").getFunction("grad");

        SDVariable gradWrtX = outside.getFunction("activate").grad("x");
        SDVariable gradWrtW = outside.getFunction("activate").grad("w");
        assertNotNull(gradWrtX);
        assertNotNull(gradWrtW);

        INDArray wGradAssertion = Nd4j.create(new double[]{-0.81,1.255,-1.80499983}).reshape(3,1);
        INDArray inputAssertion = Nd4j.valueArrayOf(vars.get("x").shape(),1e-1);
        INDArray yGradAssertion = Nd4j.zeros(vars.get("y").shape());
        INDArray mmulGrad = Nd4j.create(new double[]{-0.5,-0.5,0.5,-0.5}).reshape(4,1);
        INDArray predsGradAssertion = Nd4j.create(new double[]{-2,-2,2,-2}).reshape(4,1);
        INDArray oneMinusPredsGradAssertion = Nd4j.create(new double[]{0,0,-2,0}).reshape(4,1);
        INDArray oneMinusLabelsAssertion = Nd4j.valueArrayOf(4,-1).reshape(4,1);
        INDArray outputTimesYGradAssertion = Nd4j.valueArrayOf(4,-2).reshape(4,1);
        INDArray yHatAssertion = outputTimesYGradAssertion.dup();
        INDArray labelProbsGradAssertion = yHatAssertion.dup();
        INDArray logProbsGradAssertion = Nd4j.valueArrayOf(4,-1).reshape(4,1);

        assertEquals(logProbsGradAssertion,outside.getFunction("activate").grad("logprob").getArr());
        assertEquals(labelProbsGradAssertion,outside.getFunction("activate").grad("probs").getArr());
        assertEquals(yHatAssertion,outside.getFunction("activate").grad("yhat").getArr());
        assertEquals(outputTimesYGradAssertion,outside.getFunction("activate").grad("output * y").getArr());
        assertEquals(oneMinusLabelsAssertion,outside.getFunction("activate").grad("oneminusy").getArr());
        assertEquals(oneMinusPredsGradAssertion,outside.getFunction("activate").grad("oneminusactivations").getArr());
        assertEquals(predsGradAssertion,outside.getFunction("activate").grad("activation").getArr());
        assertEquals(mmulGrad,outside.getFunction("activate").grad("mmul").getArr());
        assertEquals(yGradAssertion,outside.getFunction("activate").grad("y").getArr());
        assertEquals(wGradAssertion,outside.getFunction("activate").grad("w").getArr());
        //note here that the gradients here end up being some weird really low eps where it
        //isn't exactly zero
        //        assertEquals(inputAssertion,outside.getFunction("activate").grad("x").getArr());



        System.out.println(gradWrtX);
        System.out.println(gradWrtW);


    }



    @Test
    public void testNestedExecution() {
        final SameDiff outer = SameDiff.create();
        Map<String, INDArray> input = new HashMap<>();
        input.put("x", Nd4j.ones(2));
        outer.defineFunction("firstadd", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable ret = input.add(input);
                return new SDVariable[] {ret};
            }
        }, input);

        outer.defineFunction("secondadd", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable result = outer.invokeFunctionOn("firstadd", sameDiff);
                return new SDVariable[] {result.add(1.0)};
            }
        });

        SameDiff secondAdd = outer.getFunction("secondadd");
        INDArray[] outputs = secondAdd.eval(input);
        INDArray outputsAssertion = Nd4j.valueArrayOf(2, 2.0);
        assertEquals(outputsAssertion, outputs[0]);
    }


    @Test
    public void testResultPropagation() {
        SameDiff sameDiff = SameDiff.create();
        INDArray inputs = Nd4j.create(new double[][]{
                {0.52, 1.12, 0.77},
                {0.88, -1.08, 0.15},
                {0.52, 0.06, -1.30},
                {0.74, -2.49, 1.39}
        });


        INDArray weights = Nd4j.randn(3, 1);

        SDVariable x = sameDiff.var("x", inputs);
        SDVariable w = sameDiff.var("w", weights);
        SDVariable preOutput = sameDiff.mmul(x, w);

        SDVariable outputs = sameDiff.sigmoid(preOutput);
        List<DifferentialFunction> ops = sameDiff.exec().getRight();
        DifferentialFunction firstOp = ops.get(0);
        val firstResult = sameDiff.getVariable(firstOp.outputVariables()[0].getVarName()).getArr();

    }

    @Test
    public void testSimpleDefineFunction() {
        SameDiff sameDiffOuter = SameDiff.create();
        Map<String, INDArray> inputs = variablesForInput();
        inputs.remove("y");
        String logisticForward = "logisticPredictions";
        sameDiffOuter.defineFunction(logisticForward, new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {

                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable w = sameDiff.var("w", inputs.get("w"));
                SDVariable preOutput = sameDiff.mmul(input, w);
                SDVariable sigmoid = sameDiff.sigmoid(preOutput);
                return new SDVariable[] {sigmoid};
            }

        }, inputs);

        assertEquals(1, sameDiffOuter.definedFunctionNames().size());

        //note here that we don't add the duplicate ops with define function anymore
    }


    @Test
    public void testSoftmax() {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("x", sumInput);
        sameDiff.defineFunction("softmax", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x").dup());
                SDVariable softmax = sameDiff.softmax(input);
                //original shape ends up being 2,2
                return new SDVariable[] {softmax};
            }
        }, inputs);

        INDArray executions = sameDiff.execAndEndResult("softmax");
        INDArray assertions = Transforms.softmax(sumInput.dup());
        assertArrayEquals(sumInput.shape(), executions.shape());
        System.out.println(executions);
        assertEquals(assertions, executions);


        SoftMaxDerivative softMaxDerivative = new SoftMaxDerivative(sumInput);
        Nd4j.getExecutioner().exec(softMaxDerivative);
        System.out.println(softMaxDerivative.z());
    }




    @Test
    public void testSigmoidBackwards() {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("x",sumInput);
        SDVariable input = sameDiff.var("x",inputs.get("x"));
        SDVariable sigmoid = sameDiff.sigmoid(input);
        SDVariable sum = sameDiff.sum(sigmoid,Integer.MAX_VALUE);
        List<DifferentialFunction> backwardsOps = sameDiff.execBackwards().getRight();
        Op finalOp = (Op)  backwardsOps.get(backwardsOps.size() - 1);
        assertTrue(Nd4j.create(new double[][]{
                {0.1966 , 0.1050},
                {0.0452 , 0.0177}
        }).equalsWithEps(
                finalOp.z(),1e-2));
        System.out.println(backwardsOps);
    }


    @Test
    public void testSumGradient() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable twoByTwo = sameDiff.var("initial",Nd4j.linspace(1,4,4).reshape(2,2));
        SDVariable sum = sameDiff.sum(twoByTwo,Integer.MAX_VALUE);
        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> execBackwards = sameDiff.execBackwards();
        SameDiff grad = sameDiff.getFunction("grad");
        SDVariable gradArr = sameDiff.grad(twoByTwo.getVarName());
        assertEquals(Nd4j.ones(2,2),gradArr.getArr());
    }






    @Test
    public void testMmulGradient() {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1,4,4).reshape(2,2);
        Map<String,INDArray> inputs = new HashMap<>();
        inputs.put("x",sumInput);
        inputs.put("y",sumInput.dup());

        sameDiff.defineFunction("mmulGradient", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x",inputs.get("x"));
                SDVariable input2 = sameDiff.var("y",inputs.get("y"));
                SDVariable exp = sameDiff.mmul(input,input2);
                SDVariable sum = sameDiff.sum(exp,Integer.MAX_VALUE);
                return new SDVariable[] {sum};
            }
        },inputs);

        List<DifferentialFunction> ops = sameDiff.getFunction("mmulGradient").execBackwards().getRight();
        String print = sameDiff.asFlatPrint();


        assumeNotNull(sameDiff.getFunction("mmulGradient").getFunction("grad"));
        assumeNotNull(sameDiff.getFunction("mmulGradient").grad("x"));
        assumeNotNull(sameDiff.getFunction("mmulGradient").grad("y"));

        SDVariable gradWrtX = sameDiff.getFunction("mmulGradient").grad("x");
        SDVariable gradWrtY = sameDiff.getFunction("mmulGradient").grad("y");
        assumeNotNull(gradWrtX.getArr());
        assumeNotNull(gradWrtY.getArr());


        INDArray xGradAssertion = Nd4j.create(new double[][]{
                {3,7},
                {3,7}
        });

        INDArray yGradAssertion = Nd4j.create(new double[][]{
                {4,4},
                {6,6}
        });

        assertEquals(xGradAssertion,gradWrtX.getArr());
        assertEquals(yGradAssertion,gradWrtY.getArr());

    }

    @Test
    public void testExpGradient() {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1,4,4).reshape(2,2);
        Map<String,INDArray> inputs = new HashMap<>();
        inputs.put("x",sumInput);
        sameDiff.defineFunction("expGradient", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x",inputs.get("x"));
                SDVariable exp = sameDiff.exp(input);
                SDVariable sum = sameDiff.sum(exp,Integer.MAX_VALUE);
                return new SDVariable[] {sum};
            }
        },inputs);


        List<DifferentialFunction> ops = sameDiff.getFunction("expGradient").execBackwards().getRight();

        INDArray executions = ops.get(ops.size() - 1).outputVariables()[0].getArr();
        INDArray assertion = Nd4j.create(new double[][]{
                {2.7183  , 7.3891},
                {20.0855  ,54.5981}
        });
        assertArrayEquals(sumInput.shape(),executions.shape());
        assertEquals(assertion,executions);
        System.out.println(executions);
        //assertEquals(Nd4j.ones(2,2),executions);
    }


/*    @Test
    public void testDepth() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable x = sameDiff.one("one",new int[]{2,2});
        assertEquals(0,x.depth());
        SDVariable sigmoid = sameDiff.sigmoid("sigmoid",x);
        assertEquals(1,sigmoid.depth());
    }*/


    @Test
    public void testTanhGradient() {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1,4,4).reshape(2,2);
        Map<String,INDArray> inputs = new HashMap<>();
        inputs.put("x",sumInput);
        sameDiff.defineFunction("tanhGradient", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x",inputs.get("x"));
                SDVariable tanh = sameDiff.tanh(input);
                SDVariable sum = sameDiff.sum(tanh,Integer.MAX_VALUE);
                return new SDVariable[] {tanh};
            }
        },inputs);

        INDArray executions = sameDiff.getFunction("tanhGradient").execBackwardAndEndResult();
        //[0.41997434161402614,0.07065082485316443,0.009866037165440211,0.0013409506830258655]
        INDArray assertion = Nd4j.create(new double[][]{
                {0.41997434161402614 , 0.07065082485316443},
                {0.009866037165440211 , 0.0013409506830258655}
        });

        assertTrue(assertion.equalsWithEps(
                executions,1e-3));

        assertArrayEquals(sumInput.shape(),executions.shape());
        assertEquals(assertion,executions);
        System.out.println(executions);
        //assertEquals(Nd4j.ones(2,2),executions);
    }



    @Test
    public void testRsubScalar() {
        SameDiff sameDiff = SameDiff.create();
        Map<String,INDArray> params = new HashMap<>();
        INDArray var = Nd4j.valueArrayOf(4,2);
        params.put("x",var);
        sameDiff.defineFunction("rsubop", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x",inputs.get("x"));
                SDVariable ret = input.rsub(1.0);
                return new SDVariable[] {ret};
            }
        },params);

        SameDiff logisticGraph = sameDiff.getFunction("rsubop");
        INDArray[] outputs = logisticGraph.eval(params);
        assertEquals(Nd4j.ones(4).muli(-1),outputs[0]);
        System.out.println(Arrays.toString(outputs));



    }




    @Test
    public void testFunctionScalarResultPropagation() {
        SameDiff sameDiffOuter = SameDiff.create();
        Map<String,INDArray> inputs = variablesForInput();

        sameDiffOuter.defineFunction("logisticPredictions", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x",inputs.get("x"));
                SDVariable w = sameDiff.var("w",inputs.get("w"));
                SDVariable preOutput = sameDiff.mmul(input,w);
                SDVariable sigmoid = sameDiff.sigmoid(preOutput);
                return new SDVariable[] {sigmoid};
            }
        },inputs);

        sameDiffOuter.defineFunction("oneminuspredictions", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable y = sameDiff.var("y",inputs.get("y"));
                SDVariable oneMinusPredictions = y.rsub(1.0);
                return new SDVariable[] {oneMinusPredictions};
            }
        },inputs);


        SameDiff logisticGraph = sameDiffOuter.getFunction("oneminuspredictions");
        INDArray[] outputs = logisticGraph.eval(inputs);
        INDArray assertion = Nd4j.create(new double[]{0,0,1,0});
        assertEquals(assertion,outputs[outputs.length - 1]);

    }

    @Test
    public void testInplaceSubi() {
        SameDiff sameDiffOuter = SameDiff.create();
        Map<String,INDArray> params = new HashMap<>();
        params.put("x",Nd4j.ones(4));
        sameDiffOuter.defineFunction("inplacesubi", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable inplace = sameDiff.var("x",inputs.get("x"));
                return new SDVariable[] {inplace.subi(1.0)};
            }
        },params);

        sameDiffOuter.getFunction("inplacesubi").eval(params);
        assertEquals(Nd4j.zeros(4),params.get("x"));
    }


    @Test
    public void testMmul() {
        SameDiff sameDiffOuter = SameDiff.create();
        Map<String,INDArray> inputs = variablesForInput();
        SDVariable x = sameDiffOuter.var("x",inputs.get("x"));
        SDVariable w = sameDiffOuter.var("w",inputs.get("w"));
        SDVariable output = sameDiffOuter.mmul(x,w);
    }





    @Test
    public void testGraphBuilding() {
        final SameDiff sameDiffOuter = SameDiff.create();
        Map<String, INDArray> inputs = variablesForInput();

        sameDiffOuter.defineFunction("logisticPredictions", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable w = sameDiff.var("w", inputs.get("w"));
                SDVariable y = sameDiff.var("y", inputs.get("y"));
                SDVariable preOutput = sameDiff.mmul(input, w);
                SDVariable sigmoid = sameDiff.sigmoid(preOutput);

                return new SDVariable[]{sigmoid};
            }
        }, inputs);

        sameDiffOuter.defineFunction("loss", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable outputs = sameDiffOuter.invokeFunctionOn("logisticPredictions", sameDiff);
                SDVariable y = sameDiff.getVariable("y");
                SDVariable outputTimesY = outputs.mul(y);
                return new SDVariable[]{outputTimesY};

            }
        }, inputs);



        SameDiff logisticPrediction = sameDiffOuter.getFunction("logisticPredictions");
        List<String> logisticOpNameAssertions = Arrays.asList("mmul", "sigmoid");


    }


    @Test
    public void testScalarAdd() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable twoByTwo = sameDiff.var("first",Nd4j.linspace(1,4,4).reshape(2,2));
        SDVariable add = twoByTwo.add(1.0);
        INDArray test = sameDiff.execAndEndResult();
        INDArray assertion = Nd4j.linspace(1,4,4).reshape('f',2,2).add(1.0);
        assertEquals(assertion,test);
    }


    @Test
    public void testSums() {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        SDVariable sdVariable = sameDiff.var("ones",ones);
        SDVariable result = sdVariable.addi(1.0);
        SDVariable total = sameDiff.sum(result,Integer.MAX_VALUE);
        List<DifferentialFunction> ops = sameDiff.exec().getRight();
        INDArray output = null;
        for(int i = 0; i < 5; i++) {
            output = sameDiff.execAndEndResult(ops);
            System.out.println("Ones " + ones);
            System.out.println(output);
        }

        assertEquals(Nd4j.valueArrayOf(4,7),ones);
        assertEquals(28,output.getDouble(0),1e-1);
    }


    @Test
    public void testDenseLayerForwardPass() {
        Nd4j.getRandom().setSeed(12345);

        SameDiff sd = SameDiff.create();

        INDArray iInput = Nd4j.rand(3,4);
        INDArray iWeights = Nd4j.rand(4,5);
        INDArray iBias = Nd4j.rand(1,5);

        SDVariable input = sd.var("input", iInput);
        SDVariable weights = sd.var("weights", iWeights);
        SDVariable bias = sd.var("bias", iBias);

        SDVariable mmul = sd.mmul("mmul", input, weights);
        SDVariable z = mmul.add("z", bias);
        SDVariable out = sd.sigmoid("out", z);

        INDArray expMmul = iInput.mmul(iWeights);
        INDArray expZ = expMmul.addRowVector(iBias);
        INDArray expOut = Transforms.sigmoid(expZ, true);

        sd.exec();

        assertEquals(expMmul, mmul.getArr());
        assertEquals(expZ, z.getArr());
        assertEquals(expOut, out.getArr());
    }

    @Test
    public void testActivationBackprop() {

        Activation[] afns = new Activation[]{
                Activation.TANH,
                Activation.SIGMOID,
                Activation.ELU,
                Activation.SOFTPLUS,
                Activation.SOFTSIGN,
                Activation.HARDTANH,
                Activation.CUBE,            //WRONG output - see issue https://github.com/deeplearning4j/nd4j/issues/2426
                Activation.RELU,            //JVM crash
                Activation.LEAKYRELU        //JVM crash
        };

        for (Activation a : afns) {

            if(a == Activation.RELU || a == Activation.LEAKYRELU){
                //TODO REMOVE THIS ONCE FIXED
                fail("Failing now to avoid JVM crash");
            }

            SameDiff sd = SameDiff.create();
            INDArray inArr = Nd4j.linspace(-3, 3, 7);
            INDArray labelArr = Nd4j.linspace(-3, 3, 7).muli(0.5);
            SDVariable in = sd.var("in", inArr.dup());

//            System.out.println("inArr: " + inArr);

            INDArray outExp;
            SDVariable out;
            switch (a) {
                case ELU:
                    out = sd.elu("out", in);
                    outExp = Transforms.elu(inArr, true);
                    break;
                case HARDTANH:
                    out = sd.hardTanh("out", in);
                    outExp = Transforms.hardTanh(inArr, true);
                    break;
                case LEAKYRELU:
                    out = sd.leakyRelu("out", in, 0.0);
                    outExp = Transforms.leakyRelu(inArr, true);
                    break;
                case RELU:
                    out = sd.relu("out", in, 0.0);
                    outExp = Transforms.relu(inArr, true);
                    break;
                case SIGMOID:
                    out = sd.sigmoid("out", in);
                    outExp = Transforms.sigmoid(inArr, true);
                    break;
                case SOFTPLUS:
                    out = sd.softplus("out", in);
                    outExp = Transforms.softPlus(inArr, true);
                    break;
                case SOFTSIGN:
                    out = sd.softsign("out", in);
                    outExp = Transforms.softsign(inArr, true);
                    break;
                case TANH:
                    out = sd.tanh("out", in);
                    outExp = Transforms.tanh(inArr, true);
                    break;
                case CUBE:
                    out = sd.pow("out", in, 3);
                    outExp = Transforms.pow(inArr, 3, true);
                    break;
                default:
                    throw new RuntimeException(a.toString());
            }

            //Sum squared error loss:
            SDVariable label = sd.var("label", labelArr.dup());
            SDVariable diff = label.sub("diff", out);
            SDVariable sqDiff = diff.mul("sqDiff", diff);
            SDVariable totSum = sd.sum("totSum", sqDiff, Integer.MAX_VALUE);    //Loss function...

            sd.exec();
            INDArray outAct = sd.getVariable("out").getArr();
            assertEquals(outExp, outAct);

            // L = sum_i (label - out)^2
            //dL/dOut = 2(out - label)
            INDArray dLdOutExp = outExp.sub(labelArr).mul(2);
            INDArray dLdInExp = a.getActivationFunction().backprop(inArr.dup(), dLdOutExp.dup()).getFirst();

            sd.execBackwards();
            SameDiff gradFn = sd.getFunction("grad");
            INDArray dLdOutAct = gradFn.getVariable("out-grad").getArr();
            INDArray dLdInAct = gradFn.getVariable("in-grad").getArr();

            assertEquals(a.toString(), dLdOutExp, dLdOutAct);
            assertEquals(a.toString(), dLdInExp, dLdInAct);
        }
    }

    @Test
    public void testMmulWithTranspose(){
        //Here: [x,3]^T * [x,4] = [3,4]

        for( int i : new int[]{2,1}) {
            System.out.println("i = " + i);
            INDArray first = Nd4j.linspace(1, 3*i, 3*i).reshape('c', i,3);      //To [1,3] or [2,3]
            INDArray second = Nd4j.linspace(4, 4 + 4*i, 4*i).reshape('c',i,4);  //To [1,4] or [2,4]

            System.out.println("Shapes: " + Arrays.toString(first.shape()) + "\t" + Arrays.toString(second.shape()));

            SameDiff sd = SameDiff.create();
            SDVariable f = sd.var("in1", first);
            SDVariable s = sd.var("in2", second);

            MMulTranspose mt = MMulTranspose.builder()
                    .transposeA(true)
                    .transposeB(false)
                    .transposeResult(false)
                    .a(first)
                    .b(second)
                    .build();
            SDVariable mmul = sd.f().mmul(f, s, mt);
            sd.updateVariableNameAndReference(mmul, "mmul");

            INDArray out = sd.execAndEndResult();

            INDArray exp = first.transpose().mmul(second);
            assertEquals(exp, out);
            System.out.println("----- Finished: i = " + i + " ------");
        }
    }

    @Test
    public void testConv2dBasic(){
        int nIn = 3;
        int nOut = 4;
        int kH = 2;
        int kW = 2;

        int mb = 3;
        int imgH = 28;
        int imgW = 28;

        SameDiff sd = SameDiff.create();
        INDArray wArr = Nd4j.create(nOut, nIn, kH, kW); //As per DL4J
        INDArray bArr = Nd4j.create(1, nOut);
        INDArray inArr = Nd4j.create(mb, nIn, imgH, imgW);

        SDVariable in = sd.var("in", inArr);
        SDVariable w = sd.var("W", wArr);
        SDVariable b = sd.var("b", bArr);

        //Order: https://github.com/deeplearning4j/libnd4j/blob/6c41ea5528bb1f454e92a9da971de87b93ff521f/include/ops/declarable/generic/convo/conv2d.cpp#L20-L22
        //in, w, b - bias is optional
        SDVariable[] vars = new SDVariable[]{in, w, b};

        Conv2DConfig c = Conv2DConfig.builder()
                .kh(kH).kw(kW)
                .ph(0).pw(0)
                .sy(1).sx(1)
                .dh(1).dw(1)
                .isSameMode(false)
                .build();

        SDVariable out = sd.conv2d(vars, c);

        INDArray outArr = sd.execAndEndResult();
        //Expected output size: out = (in - k + 2*p)/s + 1 = (28-2+0)/1+1 = 27
        int[] outShape = outArr.shape();
        assertArrayEquals(new int[]{mb, nOut, 27, 27}, outShape);
    }
}

