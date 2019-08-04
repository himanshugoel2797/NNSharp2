using NNSharp2.Kernels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp2
{
    public class ContextOutputEntry
    {
        public string Name { get; set; }
        public Tensor Expression { get; set; }
        internal Node Variable { get; set; }
    }

    public class ContextInputEntry
    {
        public string Name { get; set; }
        public Tensor Value { get; set; }
    }

    public class ContextResultEntry
    {
        public string Name { get; set; }
        public Tensor Result { get; set; }
        public bool Evaluated { get; set; }
    }

    public class Context
    {
        private static int ID = 0;
        private Tensor builtExpression;
        private ContextOutputEntry[] outputs;
        private ContextResultEntry[] results;

        private void OptimizeInternal(Tensor a, Tensor c0)
        {

            if (a.node.Operation == NodeOperation.AddFloat && c0.node.Operation == NodeOperation.AddFloat)
            {
                c0.node.Children[0].Parents.Remove(c0.node);
                a.node.Children[0] = c0.node.Children[0];
                a.node.Children[1].Value = (float)a.node.Children[1].Value + (float)c0.node.Children[1].Value;
                c0.node.Children[0].Parents.Add(a.node);
            }
            else if (a.node.Operation == NodeOperation.AddFloat && c0.node.Operation == NodeOperation.SubtractFloat)
            {
                a.node.Operation = NodeOperation.SubtractFloat;
                float f = (float)a.node.Children[1].Value + (float)c0.node.Children[0].Value;
                a.node.Children[0] = c0.node.Children[0];
                a.node.Children[0].Value = f;

                c0.node.Children[1].Parents.Remove(a.node.Children[0]);
                a.node.Children[1] = c0.node.Children[1];
                a.node.Children[1].Parents.Add(a.node);
            }
            else if (a.node.Operation == NodeOperation.MultiplyFloat && c0.node.Operation == NodeOperation.MultiplyFloat)
            {
                c0.node.Children[0].Parents.Remove(c0.node);
                a.node.Children[0] = c0.node.Children[0];
                a.node.Children[1].Value = (float)a.node.Children[1].Value * (float)c0.node.Children[1].Value;
                c0.node.Children[0].Parents.Add(a.node);
            }
        }
        private void Optimize(Tensor a)
        {
            if (a.node.Value is Tensor && string.IsNullOrEmpty(((Tensor)a.node.Value).name))
                ((Tensor)a.node.Value).name = $"tmp{ID++}";

            if (a.node.Operation == NodeOperation.Operand)
                return;

            //Solve constant operations
            switch (a.node.Operation)
            {
                case NodeOperation.MultiplyFloat:
                case NodeOperation.AddFloat:
                    {
                        var c0 = (Tensor)(a.node.Children[0].Value);
                        OptimizeInternal(a, c0);
                    }
                    break;
                case NodeOperation.SubtractFloat:
                    {
                        var c0 = (Tensor)(a.node.Children[1].Value);
                        OptimizeInternal(a, c0);
                    }
                    break;
                case NodeOperation.Exp:
                    {
                        if (a.node.Children[0].Operation == NodeOperation.MultiplyFloat && (float)a.node.Children[0].Children[1].Value == -1.0f)
                        {
                            a.node.Children[0].Children[0].Parents.Remove(a.node.Children[0]);
                            a.node.Children[0] = a.node.Children[0].Children[0];
                            a.node.Children[0].Parents.Add(a.node);
                        }
                    }
                    break;
            }

            for (int i = 0; i < a.node.Children.Length; i++)
                if (a.node.Children[i].Value is Tensor)
                {
                    Optimize(a.node.Children[i].Value as Tensor);
                }
        }
        private void ProcessOutputs(Tensor a, ContextOutputEntry[] desiredOutputs)
        {
            if (a.node.Operation == NodeOperation.Operand)
                return;

            for (int i = 0; i < a.node.Children.Length; i++)
                if (a.node.Children[i].Value is Tensor) ProcessOutputs(a.node.Children[i].Value as Tensor, desiredOutputs);

            for (int j = 0; j < a.node.Children.Length; j++)
                if (a.node.Children[j].Value is Tensor)
                    for (int i = 0; i < desiredOutputs.Length; i++)
                        if (desiredOutputs[i].Expression == (Tensor)a.node.Children[j].Value)
                            a.node.Children[j] = desiredOutputs[i].Variable;

        }

        public void Build(Tensor a, params ContextOutputEntry[] desiredOutputs)
        {
            outputs = desiredOutputs;
            results = new ContextResultEntry[desiredOutputs.Length];
            for (int i = 0; i < desiredOutputs.Length; i++)
            {
                desiredOutputs[i].Variable = new Node()
                {
                    Operation = NodeOperation.Operand,
                    ResultType = NodeResultType.ParameterMatrix,
                    Children = null,
                    Parents = desiredOutputs[i].Expression.node.Parents,
                    Value = new Tensor(desiredOutputs[i].Name, desiredOutputs[i].Expression.Axes)
                };
                results[i] = new ContextResultEntry()
                {
                    Name = desiredOutputs[i].Name,
                    Result = new Tensor(desiredOutputs[i].Expression.Axes)
                };
            }

            Optimize(a);
            ProcessOutputs(a, desiredOutputs);
        }

        #region CPU Implementation
        private Tensor ComputeCPUInternal(ContextInputEntry[] inputs, int i, Tensor m)
        {
            switch (m.node.Operation)
            {
                case NodeOperation.Operand:
                    switch (m.node.ResultType)
                    {
                        case NodeResultType.ParameterMatrix:
                            //find the value if computed, else compute it
                            for (int j = 0; j < results.Length; j++)
                            {
                                if (results[j].Name == ((Tensor)m.node.Value).name)
                                {
                                    if (!results[j].Evaluated)
                                    {
                                        results[j].Result = ComputeCPUInternal(inputs, j, outputs[j].Expression);
                                        results[j].Evaluated = true;
                                    }
                                    return results[j].Result;
                                }
                            }
                            for (int j = 0; j < inputs.Length; j++)
                                if (inputs[j].Name == ((Tensor)m.node.Value).name)
                                    return inputs[j].Value;

                            throw new Exception();
                        case NodeResultType.CommonConstantMatrix:
                        case NodeResultType.InitializedMatrix:
                            return (Tensor)m.node.Value;
                    }
                    break;
                case NodeOperation.AddFloat:
                    {
                        var op0 = ComputeCPUInternal(inputs, i, (Tensor)m.node.Children[0].Value);
                        var op1 = (float)m.node.Children[1].Value;
                        var fmem = new float[op0.Axes.Aggregate((a, b) => a * b)];
                        for (int j = 0; j < fmem.Length; j++)
                            switch (op0.node.ResultType)
                            {
                                case NodeResultType.CommonConstantMatrix:
                                    return new Tensor(op0.mem_const + op1, op0.Axes);
                                case NodeResultType.InitializedMatrix:
                                    fmem[j] = op0.mem[j] + op1;
                                    break;
                            }
                        return new Tensor(fmem, op0.Axes);
                    }
                case NodeOperation.Addition:
                    {
                        var op0 = ComputeCPUInternal(inputs, i, (Tensor)m.node.Children[0].Value);
                        var op1 = ComputeCPUInternal(inputs, i, (Tensor)m.node.Children[1].Value);
                        var fmem = new float[op0.Axes.Aggregate((a, b) => a * b)];
                        for (int j = 0; j < fmem.Length; j++)
                            switch (op0.node.ResultType)
                            {
                                case NodeResultType.CommonConstantMatrix:
                                    return new Tensor(op0.mem_const + op1.mem_const, op0.Axes);
                                case NodeResultType.InitializedMatrix:
                                    fmem[j] = op0.mem[j] + op1.mem[j];
                                    break;
                            }
                        return new Tensor(fmem, op0.Axes);
                    }
                case NodeOperation.MultiplyFloat:
                    {
                        var op0 = ComputeCPUInternal(inputs, i, (Tensor)m.node.Children[0].Value);
                        var op1 = (float)m.node.Children[1].Value;
                        if (op1 == 0) return new Tensor(op0.Axes);
                        if (op1 == 1) return op0;
                        var fmem = new float[op0.Axes.Aggregate((a, b) => a * b)];

                        for (int j = 0; j < fmem.Length; j++)
                            switch (op0.node.ResultType)
                            {
                                case NodeResultType.CommonConstantMatrix:
                                    return new Tensor(op0.mem_const * op1, op0.Axes);
                                case NodeResultType.InitializedMatrix:
                                    fmem[j] = op0.mem[j] * op1;
                                    break;
                            }
                        return new Tensor(fmem, op0.Axes);
                    }
                case NodeOperation.Hadamard:
                    {
                        var op0 = ComputeCPUInternal(inputs, i, (Tensor)m.node.Children[0].Value);
                        var op1 = ComputeCPUInternal(inputs, i, (Tensor)m.node.Children[1].Value);
                        var fmem = new float[op0.Axes.Aggregate((a, b) => a * b)];
                        for (int j = 0; j < fmem.Length; j++)
                            switch (op0.node.ResultType)
                            {
                                case NodeResultType.CommonConstantMatrix:
                                    return new Tensor(op0.mem_const * op1.mem_const, op0.Axes);
                                case NodeResultType.InitializedMatrix:
                                    fmem[j] = op0.mem[j] * op1.mem[j];
                                    break;
                            }
                        return new Tensor(fmem, op0.Axes);
                    }
                case NodeOperation.SubtractFloat:
                    {
                        var op0 = ComputeCPUInternal(inputs, i, (Tensor)m.node.Children[1].Value);
                        var op1 = (float)m.node.Children[0].Value;
                        var fmem = new float[op0.Axes.Aggregate((a, b) => a * b)];
                        for (int j = 0; j < fmem.Length; j++)
                            switch (op0.node.ResultType)
                            {
                                case NodeResultType.CommonConstantMatrix:
                                    return new Tensor(op1 - op0.mem_const, op0.Axes);
                                case NodeResultType.InitializedMatrix:
                                    fmem[j] = op1 - op0.mem[j];
                                    break;
                            }
                        return new Tensor(fmem, op0.Axes);
                    }
                case NodeOperation.Reciprocal:
                    {
                        var op0 = ComputeCPUInternal(inputs, i, (Tensor)m.node.Children[0].Value);
                        var fmem = new float[op0.Axes.Aggregate((a, b) => a * b)];
                        for (int j = 0; j < fmem.Length; j++)
                            switch (op0.node.ResultType)
                            {
                                case NodeResultType.CommonConstantMatrix:
                                    return new Tensor(1.0f / op0.mem_const, op0.Axes);
                                case NodeResultType.InitializedMatrix:
                                    fmem[j] = 1.0f / op0.mem[j];
                                    break;
                            }
                        return new Tensor(fmem, op0.Axes);
                    }
                case NodeOperation.Exp:
                    {
                        var op0 = ComputeCPUInternal(inputs, i, (Tensor)m.node.Children[0].Value);
                        var fmem = new float[op0.Axes.Aggregate((a, b) => a * b)];
                        for (int j = 0; j < fmem.Length; j++)
                            switch (op0.node.ResultType)
                            {
                                case NodeResultType.CommonConstantMatrix:
                                    return new Tensor((float)Math.Exp(op0.mem_const), op0.Axes);
                                case NodeResultType.InitializedMatrix:
                                    fmem[j] = (float)Math.Exp(op0.mem[j]);
                                    break;
                            }
                        return new Tensor(fmem, op0.Axes);
                    }
                case NodeOperation.Pow:
                    {
                        var op0 = ComputeCPUInternal(inputs, i, (Tensor)m.node.Children[0].Value);
                        var op1 = (float)m.node.Children[1].Value;
                        if (op1 == 0) return new Tensor(1, op0.Axes);
                        if (op1 == 1) return op0;
                        var fmem = new float[op0.Axes.Aggregate((a, b) => a * b)];

                        for (int j = 0; j < fmem.Length; j++)
                            switch (op0.node.ResultType)
                            {
                                case NodeResultType.CommonConstantMatrix:
                                    return new Tensor((float)Math.Pow(op0.mem_const, op1), op0.Axes);
                                case NodeResultType.InitializedMatrix:
                                    fmem[j] = (float)Math.Pow(op0.mem[j], op1);
                                    break;
                            }
                        return new Tensor(fmem, op0.Axes);
                    }
                case NodeOperation.Dot:
                    {
                        var op0 = ComputeCPUInternal(inputs, i, (Tensor)m.node.Children[0].Value);
                        var op1 = ComputeCPUInternal(inputs, i, (Tensor)m.node.Children[1].Value);

                        if (op0.Axes.Length != 2) throw new Exception();
                        if (op1.Axes.Length != 2) throw new Exception();

                        if (op0.Axes[1] != op1.Axes[0]) throw new Exception();

                        var fmem = new float[op0.Axes[0] * op1.Axes[1]];
                        //Compute dot product
                        Parallel.For(0, op0.Axes[0], (i0) =>
                        {
                            for (int j = 0; j < op1.Axes[1]; j++)
                            {
                                float acc = 0;
                                for (int k = 0; k < op0.Axes[1]; k++)
                                {
                                    //TODO: Check if either one is continuously indexed, if so, use a version that uses and increments pointers directly
                                    acc += op0.mem[op0.Index(i0, k)] * op1.mem[op1.Index(k, j)];
                                }

                                fmem[i0 * op1.Axes[1] + j] = acc;
                            }
                        });
                        return new Tensor(fmem, op0.Axes[0], op1.Axes[1]);
                    }
                case NodeOperation.VectorProduct:
                    {
                        var op0 = ComputeCPUInternal(inputs, i, (Tensor)m.node.Children[0].Value);
                        var op1 = ComputeCPUInternal(inputs, i, (Tensor)m.node.Children[1].Value);

                        if (op0.Axes.Length != 2 | op1.Axes.Length != 2) throw new Exception();

                        //Verify that broadcasting will result in equivalent tensors
                        var a_oaxes = new int[op0.Axes.Length];
                        var b_oaxes = new int[op1.Axes.Length];
                        for (int a_i = 0; a_i < a_oaxes.Length; a_i++)
                            if (op0.Axes[a_i] == 1)
                                a_oaxes[a_i] = op1.Axes[a_i];
                            else
                                a_oaxes[a_i] = op0.Axes[a_i];

                        for (int b_i = 0; b_i < b_oaxes.Length; b_i++)
                            if (op1.Axes[b_i] == 1)
                                b_oaxes[b_i] = op0.Axes[b_i];
                            else
                                b_oaxes[b_i] = op1.Axes[b_i];

                        if (a_oaxes.SequenceEqual(b_oaxes))
                        {
                            var fmem = new float[a_oaxes.Aggregate((a, b) => a * b)];

                            for (int i0 = 0; i0 < a_oaxes[0]; i0++)
                                for (int j0 = 0; j0 < a_oaxes[1]; j0++)
                                    if (op0.node.ResultType == NodeResultType.InitializedMatrix && op1.node.ResultType == NodeResultType.CommonConstantMatrix)
                                        fmem[i0 * a_oaxes[1] + j0] = op0.mem[op0.Index(op0.Axes[0] == 1 ? 0 : i0, op0.Axes[1] == 1 ? 0 : j0)] * op1.mem_const;
                                    else if (op0.node.ResultType == NodeResultType.CommonConstantMatrix && op1.node.ResultType == NodeResultType.InitializedMatrix)
                                        fmem[i0 * a_oaxes[1] + j0] = op0.mem_const * op1.mem[op1.Index(op1.Axes[0] == 1 ? 0 : i0, op1.Axes[1] == 1 ? 0 : j0)];
                                    else if (op0.node.ResultType == NodeResultType.InitializedMatrix && op1.node.ResultType == NodeResultType.InitializedMatrix)
                                        fmem[i0 * a_oaxes[1] + j0] = op0.mem[op0.Index(op0.Axes[0] == 1 ? 0 : i0, op0.Axes[1] == 1 ? 0 : j0)] * op1.mem[op1.Index(op1.Axes[0] == 1 ? 0 : i0, op1.Axes[1] == 1 ? 0 : j0)];

                            Tensor c = new Tensor(fmem, a_oaxes);
                            return c;
                        }
                        throw new Exception();
                    }
                case NodeOperation.Transpose:
                    {
                        return Tensor.Transpose(ComputeCPUInternal(inputs, i, (Tensor)m.node.Children[0].Value));
                    }
            }
            throw new Exception();
        }
        public ContextResultEntry[] ComputeCPU(params ContextInputEntry[] inputs)
        {
            //Interpret each expression, start by evaluating the root, then substituting any provided inputs, upon encountering unevaluated result variables, evaluate them
            for (int i = 0; i < results.Length; i++)
                results[i].Evaluated = false;

            for (int i = 0; i < results.Length; i++)
                results[i].Result = ComputeCPUInternal(inputs, i, outputs[i].Expression);

            return results;
        }
        #endregion

        #region GPU Implementation
        List<CommandBuffer> cmdBuffers;
        bool half = true;

        private Tensor ComputeGPUInternal(ContextInputEntry[] inputs, int i, Tensor m, CommandBuffer cmdBuffer)
        {
            switch (m.node.Operation)
            {
                case NodeOperation.Operand:
                    switch (m.node.ResultType)
                    {
                        case NodeResultType.ParameterMatrix:
                            //find the value if computed, else compute it
                            for (int j = 0; j < results.Length; j++)
                            {
                                if (results[j].Name == ((Tensor)m.node.Value).name)
                                {
                                    if (!results[j].Evaluated)
                                    {
                                        var cmdBuffer2 = new CommandBuffer(half);
                                        results[j].Result = ComputeGPUInternal(inputs, j, outputs[j].Expression, cmdBuffer2);
                                        results[j].Evaluated = true;
                                        cmdBuffers.Add(cmdBuffer2);
                                    }
                                    return results[j].Result;
                                }
                            }
                            for (int j = 0; j < inputs.Length; j++)
                                if (inputs[j].Name == ((Tensor)m.node.Value).name)
                                {
                                    inputs[j].Value.name = inputs[j].Name;
                                    return inputs[j].Value;
                                }

                            throw new Exception();
                        case NodeResultType.CommonConstantMatrix:
                        case NodeResultType.InitializedMatrix:
                            return (Tensor)m.node.Value;
                    }
                    break;
                case NodeOperation.AddFloat:
                    {
                        var op0 = ComputeGPUInternal(inputs, i, (Tensor)m.node.Children[0].Value, cmdBuffer);
                        var op1 = (float)m.node.Children[1].Value;
                        var res = new Tensor($"tmp{ID++}", op0.Axes);

                        cmdBuffer.Add(Commands.AddFloat,
                            new int[] { op0.Axes.Aggregate((a, b) => a * b), 1 },
                            new CommandParams[] {
                                new CommandParams() {
                                    Name = op0.name,
                                    Axes = op0.Axes,
                                    Transpose = op0.transposed,
                                    Strides = op0.Strides,
                                }
                            },
                            new CommandParams[] {
                                new CommandParams()
                                {
                                    Name = res.name,
                                    Axes = res.Axes,
                                    Transpose = false,
                                }
                            },
                            new float[] { op1 });

                        return res;
                    }
                case NodeOperation.Addition:
                    {
                        var op0 = ComputeGPUInternal(inputs, i, (Tensor)m.node.Children[0].Value, cmdBuffer);
                        var op1 = ComputeGPUInternal(inputs, i, (Tensor)m.node.Children[1].Value, cmdBuffer);
                        var res = new Tensor($"tmp{ID++}", op0.Axes);

                        cmdBuffer.Add(Commands.Addition,
                            new int[] { op0.Axes.Aggregate((a, b) => a * b), 1 },
                            new CommandParams[] {
                                new CommandParams() {
                                    Name = op0.name,
                                    Axes = op0.Axes,
                                    Transpose = op0.transposed,
                                    Strides = op0.Strides,
                                },
                                new CommandParams() {
                                    Name = op1.name,
                                    Axes = op1.Axes,
                                    Transpose = op1.transposed,
                                    Strides = op1.Strides,
                                }
                            },
                            new CommandParams[] {
                                new CommandParams()
                                {
                                    Name = res.name,
                                    Axes = res.Axes,
                                    Transpose = false,
                                }
                            },
                            null);

                        return res;
                    }
                case NodeOperation.MultiplyFloat:
                    {
                        var op0 = ComputeGPUInternal(inputs, i, (Tensor)m.node.Children[0].Value, cmdBuffer);
                        var op1 = (float)m.node.Children[1].Value;
                        if (op1 == 0) return new Tensor(op0.Axes) { name = $"tmp{ID++}" };
                        if (op1 == 1) return op0;

                        var res = new Tensor($"tmp{ID++}", op0.Axes);

                        cmdBuffer.Add(Commands.MultiplyFloat,
                            new int[] { op0.Axes.Aggregate((a, b) => a * b), 1 },
                            new CommandParams[] {
                                new CommandParams() {
                                    Name = op0.name,
                                    Axes = op0.Axes,
                                    Transpose = op0.transposed,
                                    Strides = op0.Strides,
                                }
                            },
                            new CommandParams[] {
                                new CommandParams()
                                {
                                    Name = res.name,
                                    Axes = res.Axes,
                                    Transpose = false,
                                }
                            },
                            new float[] { op1 });

                        return res;
                    }
                case NodeOperation.Hadamard:
                    {
                        var op0 = ComputeGPUInternal(inputs, i, (Tensor)m.node.Children[0].Value, cmdBuffer);
                        var op1 = ComputeGPUInternal(inputs, i, (Tensor)m.node.Children[1].Value, cmdBuffer);
                        var res = new Tensor($"tmp{ID++}", op0.Axes);

                        cmdBuffer.Add(Commands.Hadamard,
                            new int[] { op0.Axes.Aggregate((a, b) => a * b), 1 },
                            new CommandParams[] {
                                new CommandParams() {
                                    Name = op0.name,
                                    Axes = op0.Axes,
                                    Transpose = op0.transposed,
                                    Strides = op0.Strides,
                                },
                                new CommandParams() {
                                    Name = op1.name,
                                    Axes = op1.Axes,
                                    Transpose = op1.transposed,
                                    Strides = op1.Strides,
                                }
                            },
                            new CommandParams[] {
                                new CommandParams()
                                {
                                    Name = res.name,
                                    Axes = res.Axes,
                                    Transpose = false,
                                }
                            },
                            null);

                        return res;
                    }
                case NodeOperation.SubtractFloat:
                    {
                        var op0 = ComputeGPUInternal(inputs, i, (Tensor)m.node.Children[1].Value, cmdBuffer);
                        var op1 = (float)m.node.Children[0].Value;
                        var res = new Tensor($"tmp{ID++}", op0.Axes);

                        cmdBuffer.Add(Commands.SubtractFloat,
                            new int[] { op0.Axes.Aggregate((a, b) => a * b), 1 },
                            new CommandParams[] {
                                new CommandParams() {
                                    Name = op0.name,
                                    Axes = op0.Axes,
                                    Transpose = op0.transposed,
                                    Strides = op0.Strides,
                                }
                            },
                            new CommandParams[] {
                                new CommandParams()
                                {
                                    Name = res.name,
                                    Axes = res.Axes,
                                    Transpose = false,
                                }
                            },
                            new float[] { op1 });

                        return res;
                    }
                case NodeOperation.Reciprocal:
                    {
                        var op0 = ComputeGPUInternal(inputs, i, (Tensor)m.node.Children[0].Value, cmdBuffer);
                        var res = new Tensor($"tmp{ID++}", op0.Axes);

                        cmdBuffer.Add(Commands.Reciprocal,
                            new int[] { op0.Axes.Aggregate((a, b) => a * b), 1 },
                            new CommandParams[] {
                                new CommandParams() {
                                    Name = op0.name,
                                    Axes = op0.Axes,
                                    Transpose = op0.transposed,
                                    Strides = op0.Strides,
                                }
                            },
                            new CommandParams[] {
                                new CommandParams()
                                {
                                    Name = res.name,
                                    Axes = res.Axes,
                                    Transpose = false,
                                }
                            },
                            null);

                        return res;
                    }
                case NodeOperation.Exp:
                    {
                        var op0 = ComputeGPUInternal(inputs, i, (Tensor)m.node.Children[0].Value, cmdBuffer);
                        var res = new Tensor($"tmp{ID++}", op0.Axes);

                        cmdBuffer.Add(Commands.Exp,
                            new int[] { op0.Axes.Aggregate((a, b) => a * b), 1 },
                            new CommandParams[] {
                                new CommandParams() {
                                    Name = op0.name,
                                    Axes = op0.Axes,
                                    Transpose = op0.transposed,
                                    Strides = op0.Strides,
                                }
                            },
                            new CommandParams[] {
                                new CommandParams()
                                {
                                    Name = res.name,
                                    Axes = res.Axes,
                                    Transpose = false,
                                }
                            },
                            null);

                        return res;
                    }
                case NodeOperation.Pow:
                    {
                        var op0 = ComputeGPUInternal(inputs, i, (Tensor)m.node.Children[0].Value, cmdBuffer);
                        var op1 = (float)m.node.Children[1].Value;
                        if (op1 == 0) return new Tensor(1, op0.Axes) { name = $"tmp{ID++}" };
                        if (op1 == 1) return op0;
                        var res = new Tensor($"tmp{ID++}", op0.Axes);

                        cmdBuffer.Add(Commands.Pow,
                            new int[] { op0.Axes.Aggregate((a, b) => a * b), 1 },
                            new CommandParams[] {
                                new CommandParams() {
                                    Name = op0.name,
                                    Axes = op0.Axes,
                                    Transpose = op0.transposed,
                                    Strides = op0.Strides,
                                }
                            },
                            new CommandParams[] {
                                new CommandParams()
                                {
                                    Name = res.name,
                                    Axes = res.Axes,
                                    Transpose = false,
                                }
                            },
                            new float[] { op1 });

                        return res;
                    }
                case NodeOperation.Dot:
                    {
                        var op0 = ComputeGPUInternal(inputs, i, (Tensor)m.node.Children[0].Value, cmdBuffer);
                        var op1 = ComputeGPUInternal(inputs, i, (Tensor)m.node.Children[1].Value, cmdBuffer);

                        if (op0.Axes.Length != 2) throw new Exception();
                        if (op1.Axes.Length != 2) throw new Exception();
                        if (op0.Axes[1] != op1.Axes[0]) throw new Exception();

                        var res = new Tensor($"tmp{ID++}", op0.Axes[0], op1.Axes[1]);

                        cmdBuffer.Add(Commands.Dot,
                            new int[] { op0.Axes[0], op1.Axes[1] },
                            new CommandParams[] {
                                new CommandParams() {
                                    Name = op0.name,
                                    Axes = op0.Axes,
                                    Transpose = op0.transposed,
                                    Strides = op0.Strides,
                                },
                                new CommandParams() {
                                    Name = op1.name,
                                    Axes = op1.Axes,
                                    Transpose = op1.transposed,
                                    Strides = op1.Strides,
                                }
                            },
                            new CommandParams[] {
                                new CommandParams()
                                {
                                    Name = res.name,
                                    Axes = res.Axes,
                                    Transpose = false,
                                }
                            },
                            null);

                        return res;
                    }
                case NodeOperation.VectorProduct:
                    {
                        var op0 = ComputeGPUInternal(inputs, i, (Tensor)m.node.Children[0].Value, cmdBuffer);
                        var op1 = ComputeGPUInternal(inputs, i, (Tensor)m.node.Children[1].Value, cmdBuffer);

                        if (op0.Axes.Length != 2 | op1.Axes.Length != 2) throw new Exception();

                        //Verify that broadcasting will result in equivalent tensors
                        var a_oaxes = new int[op0.Axes.Length];
                        var b_oaxes = new int[op1.Axes.Length];
                        for (int a_i = 0; a_i < a_oaxes.Length; a_i++)
                            if (op0.Axes[a_i] == 1)
                                a_oaxes[a_i] = op1.Axes[a_i];
                            else
                                a_oaxes[a_i] = op0.Axes[a_i];

                        for (int b_i = 0; b_i < b_oaxes.Length; b_i++)
                            if (op1.Axes[b_i] == 1)
                                b_oaxes[b_i] = op0.Axes[b_i];
                            else
                                b_oaxes[b_i] = op1.Axes[b_i];

                        if (a_oaxes.SequenceEqual(b_oaxes))
                        {
                            Tensor res = new Tensor($"tmp{ID++}", a_oaxes);
                            cmdBuffer.Add(Commands.Dot,
                                a_oaxes,
                                new CommandParams[] {
                                new CommandParams() {
                                    Name = op0.name,
                                    Axes = op0.Axes,
                                    Transpose = op0.transposed,
                                    Strides = op0.Strides,
                                },
                                new CommandParams() {
                                    Name = op1.name,
                                    Axes = op1.Axes,
                                    Transpose = op1.transposed,
                                    Strides = op1.Strides,
                                }
                                },
                                new CommandParams[] {
                                new CommandParams()
                                {
                                    Name = res.name,
                                    Axes = res.Axes,
                                    Transpose = false,
                                }
                                },
                                null);
                            return res;
                        }
                        throw new Exception();
                    }
                case NodeOperation.Transpose:
                    {
                        return Tensor.Transpose(ComputeGPUInternal(inputs, i, (Tensor)m.node.Children[0].Value, cmdBuffer));
                    }
            }
            throw new Exception();
        }
        public ContextResultEntry[] ComputeGPU(params ContextInputEntry[] inputs)
        {
            if (cmdBuffers == null)
            {
                cmdBuffers = new List<CommandBuffer>();

                //Interpret each expression, start by evaluating the root, then substituting any provided inputs, upon encountering unevaluated result variables, evaluate them
                for (int i = 0; i < results.Length; i++)
                    results[i].Evaluated = false;

                for (int i = 0; i < results.Length; i++)
                {
                    if (results[i].Evaluated) continue;

                    var cmdBuffer = new CommandBuffer(half);
                    results[i].Result = ComputeGPUInternal(inputs, i, outputs[i].Expression, cmdBuffer);
                    results[i].Evaluated = true;
                    cmdBuffers.Add(cmdBuffer);
                }
            }
            return results;
        }
        #endregion
    }
}
