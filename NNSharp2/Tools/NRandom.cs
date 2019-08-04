using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp2.Tools
{
    [Serializable]
    public class NRandom
    {
        private Random rng;
        private ulong seed;
        private readonly object lock_obj;

        private const ulong mult = 6364136223846793005;

        public NRandom(NRandom r)
        {
            this.seed = r.seed;
            lock_obj = new object();
        }

        public NRandom() : this((int)(DateTime.Now.Ticks & 0x7fffffff))
        {

        }

        public NRandom(int seed)
        {
            this.seed = unchecked((ulong)seed);
            //for (int i = 0; i < 4; i++) Next();
            lock_obj = new object();
            rng = new Random(seed);
        }

        public int Next()
        {
            lock (lock_obj)
            {
                return rng.Next();
            }
            //seed = unchecked(seed * mult + 1);
            //return unchecked((int)(seed >> 33));
        }

        public double NextDouble()
        {
            lock (lock_obj)
            {
                return rng.NextDouble();
            }
            //return unchecked((Next() % 2000000) + 1 - double.Epsilon) / 2000000 + double.Epsilon;
        }

        public double NextStdNormal()
        {
            double r0 = 1 - NextDouble();
            double r1 = 1 - NextDouble();

            var rand_std_normal = Math.Sqrt(-2.0 * Math.Log(r0)) * Math.Sin(2.0 * Math.PI * r1);
            return rand_std_normal;
        }

        public double NextGaussian(double mu = 0, double sigma = 1)
        {
            var rand_normal = mu + sigma * NextStdNormal();
            return rand_normal;
        }
    }
}
