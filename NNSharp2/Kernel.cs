using OpenCL.Net;
using OpenCL.Net.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp2
{
    public class Kernel
    {
        public string Name { get; set; }

#if DELAY_COMPILE
        public string SourceCode { get; set; }
        public bool Initialized { get; set; }
#endif
        internal Event PendingExecution { get; set; }
        internal OpenCL.Net.Kernel kern;
        private CLExtensions.KernelArgChain chain;
        private bool reset = true;

        public void Reset()
        {
            reset = true;
        }

        public Kernel SetArgument<T>(T val) where T : struct, IComparable
        {
#if DELAY_COMPILE 
            if (!Initialized)
            {
                Initialized = true;
                kern = Device.GetDevice().env.Context.CompileKernelFromSource(SourceCode, Name, "-cl-unsafe-math-optimizations");
            }
#endif
            if (PendingExecution.IsValid())
            {
                Device.GetDevice().HandleEvent();
            }

            if (reset)
            {
                chain = kern.SetKernelArg(val);
                reset = false;
            }
            else
                chain = chain.SetKernelArg(val);
            return this;
        }

        public Kernel SetArgumentMemory(Memory val)
        {
#if DELAY_COMPILE
            if (!Initialized)
            {
                Initialized = true;
                kern = Device.GetDevice().env.Context.CompileKernelFromSource(SourceCode, Name, "-cl-unsafe-math-optimizations");
            }
#endif
            if (PendingExecution.IsValid())
            {
                Device.GetDevice().HandleEvent();
            }
            if (val == null)
            {
                chain = chain.SetKernelArg(null);
                return this;
            }

            if (reset)
            {
                chain = kern.SetKernelArg((IMem)val.buf);
                reset = false;
            }
            else
                chain = chain.SetKernelArg((IMem)val.buf);
            return this;
        }
    }
}
