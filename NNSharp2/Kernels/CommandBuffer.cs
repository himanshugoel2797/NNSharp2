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
        }

        #region OpenCL
        public void EmitCL()
        {
            //Emit OpenCL code
        }

        public void RunCL()
        {

        }
        #endregion

        #region C
        public void EmitC()
        {
            //Emit vectorized C code
        }

        public void RunC()
        {

        }
        #endregion

        #region CSharp
        public void RunCS()
        {
            //Add 'computable matrix' type which has float[] backend and OpenCL backend
            for(int i = 0; i < Commands.Count; i++)
            {
                switch (Commands[i].Command)
                {

                }
            }
        }
        #endregion
    }
}
