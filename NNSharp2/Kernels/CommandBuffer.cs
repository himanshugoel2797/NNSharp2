using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp2.Kernels
{
    public enum Commands
    {
        Addition,
        Pow,
        Exp,

        Reciprocal,
        MultiplyFloat,
        AddFloat,
        SubtractFloat,

        Dot,
        Hadamard,
        VectorProduct,
    }

    public class CommandParams
    {
        public string Name { get; set; }
        public bool Transpose { get; set; }
        public int[] Axes { get; set; }
        public int[] Strides { get; set; }
        public Tensor Value { get; set; }
        public override string ToString()
        {
            return $"{Name}, ({Axes[0]}, {Axes[1]}), {(Transpose ? "T" : "")}";
        }
    }

    public class CommandEntry
    {
        public Commands Command { get; set; }
        public int[] WorkDimensions { get; set; }
        public CommandParams[] Inputs { get; set; }
        public CommandParams[] Outputs { get; set; }
        public float[] Constants { get; set; }

        public override string ToString()
        {
            return $"{Command}, ({WorkDimensions[0]}, {WorkDimensions[1]})";
        }
    }
    public class CommandBuffer
    {
        private bool half;
        private List<CommandEntry> Commands;
        public CommandBuffer(bool half) { this.half = half; Commands = new List<CommandEntry>(); }

        public void Add(Commands cmd, int[] work_dims, CommandParams[] inputs, CommandParams[] outputs, float[] constants)
        {
            Commands.Add(new CommandEntry()
            {
                Command = cmd,
                WorkDimensions = work_dims,
                Inputs = inputs,
                Outputs = outputs,
                Constants = constants
            });
        }

        public void Simplify()
        {
            //Perform SSA optimizations
            //identify chained -1 multiplys
            //identify multiplication by 1
            //identify multiplication by 0
            for (int i = 0; i < Commands.Count; i++)
            {
                var cmd = Commands[i];

                //If both inputs are transposed, swap their multiplication order and transpose the output result
                if (cmd.Command == Kernels.Commands.Dot && cmd.Inputs[0].Transpose && cmd.Inputs[1].Transpose)
                {
                    var tmp = cmd.Inputs[0];
                    cmd.Inputs[0] = cmd.Inputs[1];
                    cmd.Inputs[1] = tmp;

                    for (int j = i + 1; j < Commands.Count; j++)
                        for (int k = 0; k < Commands[j].Inputs.Length; k++)
                            if (Commands[j].Inputs[k].Name == cmd.Outputs[0].Name)
                                Commands[j].Inputs[k].Transpose = true;
                }
            }
        }

        #region OpenCL
        public void EmitCL()
        {
            //Emit OpenCL code
        }

        public RealTensor RunCL()
        {
            return null;
        }
        #endregion

        #region C
        public void EmitC()
        {
            //Emit vectorized C code
        }

        public RealTensor RunC()
        {
            return null;
        }
        #endregion

        #region CSharp
        private Matrix<float> ToDenseMatrix(Tensor a)
        {
            if (a.node.Operation == NodeOperation.Operand || a.node.Operation == NodeOperation.Transpose)
                switch (a.node.ResultType)
                {
                    case NodeResultType.InitializedMatrix:
                        return DenseMatrix.Create(a.Axes[0], a.Axes[1], (i, j) => a.mem[a.Index(i, j)]);
                    case NodeResultType.CommonConstantMatrix:
                        return DenseMatrix.Create(a.Axes[0], a.Axes[1], a.mem_const);
                    default:
                        throw new Exception();
                }
            throw new Exception();
        }
        private void ProcessCS(Dictionary<string, Matrix<float>> vars, int i)
        {
            var cmd = Commands[i];
            var inputParams = new Matrix<float>[cmd.Inputs.Length];
            for (int j = 0; j < cmd.Inputs.Length; j++)
            {
                if (vars.ContainsKey(cmd.Inputs[j].Name)) inputParams[j] = vars[cmd.Inputs[j].Name];
                else inputParams[j] = ToDenseMatrix(cmd.Inputs[j].Value);
            }

            switch (cmd.Command)
            {
                case Kernels.Commands.AddFloat:
                    vars[cmd.Outputs[0].Name] = inputParams[0].Add(cmd.Constants[0]);
                    break;
                case Kernels.Commands.Addition:
                    vars[cmd.Outputs[0].Name] = inputParams[0].Add(inputParams[1]);
                    break;
                case Kernels.Commands.Dot:
                    if (!cmd.Inputs[0].Transpose && !cmd.Inputs[1].Transpose)
                        vars[cmd.Outputs[0].Name] = inputParams[0].Multiply(inputParams[1]);
                    else if (!cmd.Inputs[1].Transpose)
                        vars[cmd.Outputs[0].Name] = inputParams[0].TransposeThisAndMultiply(inputParams[1]);
                    else if (!cmd.Inputs[0].Transpose)
                        vars[cmd.Outputs[0].Name] = inputParams[0].TransposeAndMultiply(inputParams[1]);
                    else
                        throw new Exception("This situation should not occur, both inputs are transposed");
                    break;
                case Kernels.Commands.Exp:
                    vars[cmd.Outputs[0].Name] = inputParams[0].PointwiseExp();
                    break;
                case Kernels.Commands.Hadamard:
                    vars[cmd.Outputs[0].Name] = inputParams[0].PointwiseMultiply(inputParams[1]);
                    break;
                case Kernels.Commands.MultiplyFloat:
                    vars[cmd.Outputs[0].Name] = inputParams[0].Multiply(cmd.Constants[0]);
                    break;
                case Kernels.Commands.Pow:
                    vars[cmd.Outputs[0].Name] = inputParams[0].PointwisePower(cmd.Constants[0]);
                    break;
                case Kernels.Commands.Reciprocal:
                    vars[cmd.Outputs[0].Name] = inputParams[0].DivideByThis(1);
                    break;
                case Kernels.Commands.SubtractFloat:
                    vars[cmd.Outputs[0].Name] = inputParams[0].SubtractFrom(cmd.Constants[0]);
                    break;
                case Kernels.Commands.VectorProduct:
                    vars[cmd.Outputs[0].Name] = inputParams[0].Multiply(inputParams[1]);
                    break;
            }
        }

        //Add 'computable matrix' type which has float[] backend and OpenCL backend
        public RealTensor RunCS(ContextInputEntry[] inputs)
        {
            var vars = new Dictionary<string, Matrix<float>>();
            for (int i = 0; i < inputs.Length; i++)
                vars[inputs[i].Name] = ToDenseMatrix(inputs[i].Value);

            for (int i = 0; i < Commands.Count; i++)
            {
                ProcessCS(vars, i);
            }

            return null;
        }
        #endregion
    }
}
