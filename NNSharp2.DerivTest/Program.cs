using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp2.DerivTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Tensor x = new Tensor(nameof(x), 2, 1);
            Tensor eo = new Tensor(nameof(eo), 2, 1);

            Tensor b = new Tensor(nameof(b), 2, 1);
            Tensor w = new Tensor(nameof(w), 2, 2);

            var f = Tensor.Dot(w, x) + b;
            var z = Tensor.Recip(1 + Tensor.Exp(-1 * f));
            var dz = Tensor.Deriv(z, w);
            var E = 0.5f * Tensor.Pow(z - eo, 2);
            var dE = Tensor.Hadamard(z - eo, dz);

            Context ctxt = new Context();
            ctxt.Build(dE, new ContextOutputEntry[] {
                new ContextOutputEntry(){
                    Expression = f,
                    Name = "f"
                },
                new ContextOutputEntry(){
                    Expression = z,
                    Name = "z"
                },
                new ContextOutputEntry(){
                    Expression = dz,
                    Name = "dz"
                },
                new ContextOutputEntry(){
                    Expression = E,
                    Name = "E"
                },
                new ContextOutputEntry()
                {
                    Expression = dE,
                    Name = "dE"
                }
            });

            Console.WriteLine($"{Tensor.PrintAxes(f)}, f = {(f)}");
            Console.WriteLine($"{Tensor.PrintAxes(z)}, z = {(z)}");
            Console.WriteLine($"{Tensor.PrintAxes(E)}, E = {(E)}");
            Console.WriteLine();
            Console.WriteLine($"{Tensor.PrintAxes(dz)}, dz = {(dz)}");
            Console.WriteLine($"{Tensor.PrintAxes(dE)}, dE = {(dE)}");

            var results = ctxt.ComputeGPU(new ContextInputEntry[]
            {
                new ContextInputEntry()
                {
                    Name = "x",
                    Value = new Tensor(new float[]{ 1, 2 }, 2, 1)
                },
                new ContextInputEntry()
                {
                    Name = "eo",
                    Value = new Tensor(new float[] { 3, 4 }, 2, 1)
                },
                new ContextInputEntry()
                {
                    Name = "b",
                    Value = new Tensor(new float[] { 5, 6 }, 2, 1)
                },
                new ContextInputEntry()
                {
                    Name = "w",
                    Value = new Tensor(new float[] { 7, 8, 9, 10 }, 2, 2)
                },
            });

            Console.WriteLine();
            for (int i = 0; i < results.Length; i++)
                Console.WriteLine($"{results[i].Name} = \n{Tensor.PrintValue(results[i].Result)}");

            Console.ReadLine();
        }
    }
}
