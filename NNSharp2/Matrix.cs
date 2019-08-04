using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp2
{
    public class Tensor
    {
        public int[] Axes { get; private set; }
        public int[] Strides { get; private set; }

        internal Node node;
        internal float[] mem;
        internal string name;
        internal float mem_const;
        internal bool transposed;

        public Tensor(params int[] axes) : this(0, axes) { }

        public Tensor(float c, params int[] axes)
        {
            Axes = axes;
            Strides = new int[Axes.Length];
            for (int j = 0; j < Strides.Length; j++)
                Strides[j] = 1;
            for (int j = 1; j < Strides.Length; j++)
                for (int i = 0; i < Strides.Length; i++)
                    if (i + j < Axes.Length)
                        Strides[i] *= Axes[i + j];

            mem_const = c;
            node = new Node()
            {
                Operation = NodeOperation.Operand,
                ResultType = NodeResultType.CommonConstantMatrix,
                Value = this,
                Children = null,
            };
        }

        public Tensor(float[] mem, params int[] axes)
        {
            Axes = axes;
            Strides = new int[Axes.Length];
            for (int j = 0; j < Strides.Length; j++)
                Strides[j] = 1;
            for (int j = 1; j < Strides.Length; j++)
                for (int i = 0; i < Strides.Length; i++)
                    if (i + j < Axes.Length)
                        Strides[i] *= Axes[i + j];

            this.mem = mem;
            node = new Node()
            {
                Operation = NodeOperation.Operand,
                ResultType = NodeResultType.InitializedMatrix,
                Value = this,
                Children = null,
            };
        }

        public Tensor(string name, params int[] axes)
        {
            Axes = axes;
            Strides = new int[Axes.Length];
            for (int j = 0; j < Strides.Length; j++)
                Strides[j] = 1;
            for (int j = 1; j < Strides.Length; j++)
                for (int i = 0; i < Strides.Length; i++)
                    if (i + j < Axes.Length)
                        Strides[i] *= Axes[i + j];

            this.name = name;
            node = new Node()
            {
                Operation = NodeOperation.Operand,
                ResultType = NodeResultType.ParameterMatrix,
                Value = this,
                Children = null,
            };
        }

        internal Tensor(string name, float[] mem, float c, int[] axes, int[] strides)
        {
            this.name = name;
            this.mem = mem;
            this.mem_const = c;
            this.Axes = axes;
            this.Strides = strides;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Index(params int[] pos)
        {
#if INDEXING_CHECK
            if (row >= Rows | row < 0)
                throw new Exception();

            if (col >= Columns | col < 0)
                throw new Exception();
#endif
            int idx = 0;
            for (int i = 0; i < Axes.Length; i++)
                idx += pos[i] * Strides[i];
            return idx;
        }

        public Tensor T()
        {
            return Transpose(this);
        }

        public static Tensor Transpose(Tensor a)
        {
            var m = new Tensor(a.name, a.mem, a.mem_const, a.Axes.Reverse().ToArray(), a.Strides.Reverse().ToArray());
            m.transposed = true;
            m.node = new Node()
            {
                Operation = NodeOperation.Transpose,
                Children = new Node[] { a.node },
                ResultType = a.node.ResultType,
                Value = m,
            };
            a.node.Parents.Add(m.node);
            return m;
        }

        public static Tensor Deriv(Tensor a, Tensor b)
        {
            if (a == b)
            {
                return new Tensor(1.0f, 1, 1);
            }

            switch (a.node.Operation)
            {
                case NodeOperation.Operand:
                    {
                        return new Tensor(0.0f, 1, 1);
                    }
                case NodeOperation.Transpose:
                    {
                        return Transpose(Deriv((Tensor)a.node.Children[0].Value, b));
                    }
                case NodeOperation.Pow:
                    {
                        var m = (float)a.node.Children[1].Value * Pow((Tensor)a.node.Children[0].Value, (float)a.node.Children[1].Value - 1);
                        m = Hadamard(Deriv((Tensor)a.node.Children[0].Value, b), m);
                        return m;
                    }
                case NodeOperation.Exp:
                    {
                        var m = Hadamard(Deriv((Tensor)a.node.Children[0].Value, b), a);
                        return m;
                    }
                case NodeOperation.Dot:
                    {
                        var t_l = (Tensor)a.node.Children[0].Value;
                        var t_r = (Tensor)a.node.Children[1].Value;

                        if (t_l == b && t_r.Axes[1] == 1)
                            return t_r.T();

                        var left = Deriv((Tensor)a.node.Children[0].Value, b);
                        var right = Deriv((Tensor)a.node.Children[1].Value, b);

                        bool left_isZero = left.node.Operation == NodeOperation.Operand && (left.node.ResultType == NodeResultType.Constant && left.node.Value is float && (float)left.node.Value == 0.0f) || (left.node.ResultType == NodeResultType.CommonConstantMatrix && left.node.Value is Tensor && ((Tensor)left.node.Value).mem_const == 0.0f);
                        bool left_isOne = left.node.Operation == NodeOperation.Operand && (left.node.ResultType == NodeResultType.Constant && left.node.Value is float && (float)left.node.Value == 1.0f) || (left.node.ResultType == NodeResultType.CommonConstantMatrix && left.node.Value is Tensor && ((Tensor)left.node.Value).mem_const == 1.0f);

                        bool right_isZero = right.node.Operation == NodeOperation.Operand && (right.node.ResultType == NodeResultType.Constant && right.node.Value is float && (float)right.node.Value == 0.0f) || (right.node.ResultType == NodeResultType.CommonConstantMatrix && right.node.Value is Tensor && ((Tensor)right.node.Value).mem_const == 0.0f);
                        bool right_isOne = right.node.Operation == NodeOperation.Operand && (right.node.ResultType == NodeResultType.Constant && right.node.Value is float && (float)right.node.Value == 1.0f) || (right.node.ResultType == NodeResultType.CommonConstantMatrix && right.node.Value is Tensor && ((Tensor)right.node.Value).mem_const == 1.0f);

                        if (left_isZero && !right_isOne && !right_isZero)
                            return Tensor.Dot((Tensor)a.node.Children[0].Value, right);
                        if (left_isOne && !right_isOne && !right_isZero)
                            return (Tensor)a.node.Children[1].Value + Tensor.Dot((Tensor)a.node.Children[0].Value, right);
                        if (right_isZero && !left_isOne && !left_isZero)
                            return Tensor.Dot(left, (Tensor)a.node.Children[1].Value);
                        if (right_isOne && !left_isOne && !left_isZero)
                            return Tensor.Dot(left, (Tensor)a.node.Children[1].Value) + (Tensor)a.node.Children[0].Value;
                        if (left_isZero && right_isZero)
                            return new Tensor(1, 1, 0);
                        if (left_isZero && right_isOne)
                            return (Tensor)a.node.Children[0].Value;
                        if (right_isZero && left_isOne)
                            return (Tensor)a.node.Children[1].Value;

                        return Tensor.Dot(left, (Tensor)a.node.Children[1].Value) + Tensor.Dot((Tensor)a.node.Children[0].Value, right);
                    }
                case NodeOperation.Addition:
                    {
                        var left = Deriv((Tensor)a.node.Children[0].Value, b);
                        var right = Deriv((Tensor)a.node.Children[1].Value, b);

                        //if left result is zero, return right result only
                        if (left.node.Operation == NodeOperation.Operand && (left.node.ResultType == NodeResultType.Constant && left.node.Value is float && (float)left.node.Value == 0.0f) || (left.node.ResultType == NodeResultType.CommonConstantMatrix && left.node.Value is Tensor && ((Tensor)left.node.Value).mem_const == 0.0f))
                            return right;

                        //if right result is zero, return left result only
                        if (right.node.Operation == NodeOperation.Operand && (right.node.ResultType == NodeResultType.Constant && right.node.Value is float && (float)right.node.Value == 0.0f) || (right.node.ResultType == NodeResultType.CommonConstantMatrix && right.node.Value is Tensor && ((Tensor)right.node.Value).mem_const == 0.0f))
                            return left;

                        return left + right;
                    }
                    break;
                case NodeOperation.MultiplyFloat:
                    {
                        var chain = Deriv((Tensor)a.node.Children[0].Value, b);
                        return chain * (float)a.node.Children[1].Value;
                    }
                    break;
                case NodeOperation.Reciprocal:
                    {
                        var deriv_F = Deriv((Tensor)a.node.Children[0].Value, b);
                        return Hadamard(-1 * deriv_F, Recip(Pow((Tensor)a.node.Children[0].Value, 2)));
                    }
                    break;
                case NodeOperation.AddFloat:
                    {
                        var deriv_F = Deriv((Tensor)a.node.Children[0].Value, b);
                        return deriv_F;
                    }
                    break;
            }

            throw new Exception();
        }

        public static bool operator ==(Tensor a, Tensor b)
        {
            if (( (object)a == null && (object)b != null) || ((object)a != null && (object)b == null))
                return false;

            if ((object)a == null && (object)b == null)
                return true;

            var res = a.Axes.SequenceEqual(b.Axes) && a.name == b.name && a.mem_const == b.mem_const && a.node == b.node;
            if (!res)
                return false;

            if (a.mem != null && b.mem != null)
            {
                if (!a.mem.SequenceEqual(b.mem))
                    return false;
            }
            else if (!(a.mem == null && b.mem == null))
                return false;

            return true;
        }

        public static bool operator !=(Tensor a, Tensor b)
        {
            return !(a == b);
        }

        public static Tensor operator +(Tensor a, Tensor b)
        {
            if (!a.Axes.SequenceEqual(b.Axes)) throw new Exception();

            Tensor c = new Tensor(a.Axes);
            c.node = new Node()
            {
                Operation = NodeOperation.Addition,
                ResultType = NodeResultType.ResultMatrix,
                Children = new Node[] { a.node, b.node },
                Value = c,
            };
            a.node.Parents.Add(c.node);
            b.node.Parents.Add(c.node);

            return c;
        }
        public static Tensor operator -(Tensor a, Tensor b)
        {
            return a + (b * -1);
        }

        public static Tensor operator +(Tensor a, float b)
        {
            Node b_node = new Node()
            {
                Operation = NodeOperation.Operand,
                ResultType = NodeResultType.Constant,
                Children = null,
                Value = b,
            };

            Tensor c = new Tensor(a.Axes);
            c.node = new Node()
            {
                Operation = NodeOperation.AddFloat,
                ResultType = NodeResultType.ResultMatrix,
                Children = new Node[] { a.node, b_node },
                Value = c,
            };
            a.node.Parents.Add(c.node);
            b_node.Parents.Add(c.node);

            return c;
        }
        public static Tensor operator -(Tensor a, float b)
        {
            return a + (-b);
        }
        public static Tensor operator *(Tensor a, float b)
        {
            Node b_node = new Node()
            {
                Operation = NodeOperation.Operand,
                ResultType = NodeResultType.Constant,
                Children = null,
                Value = b,
            };

            Tensor c = new Tensor(a.Axes);
            c.node = new Node()
            {
                Operation = NodeOperation.MultiplyFloat,
                ResultType = NodeResultType.ResultMatrix,
                Children = new Node[] { a.node, b_node },
                Value = c,
            };
            a.node.Parents.Add(c.node);
            b_node.Parents.Add(c.node);

            return c;
        }
        public static Tensor operator /(Tensor a, float b)
        {
            return a * (1.0f / b);
        }
        public static Tensor operator +(float b, Tensor a)
        {
            Node b_node = new Node()
            {
                Operation = NodeOperation.Operand,
                ResultType = NodeResultType.Constant,
                Children = null,
                Value = b,
            };

            Tensor c = new Tensor(a.Axes);
            c.node = new Node()
            {
                Operation = NodeOperation.AddFloat,
                ResultType = NodeResultType.ResultMatrix,
                Children = new Node[] { a.node, b_node },
                Value = c,
            };
            a.node.Parents.Add(c.node);
            b_node.Parents.Add(c.node);

            return c;
        }
        public static Tensor operator -(float b, Tensor a)
        {
            Node b_node = new Node()
            {
                Operation = NodeOperation.Operand,
                ResultType = NodeResultType.Constant,
                Children = null,
                Value = -b,
            };

            Tensor c = new Tensor(a.Axes);
            c.node = new Node()
            {
                Operation = NodeOperation.SubtractFloat,
                ResultType = NodeResultType.ResultMatrix,
                Children = new Node[] { b_node, a.node },
                Value = c,
            };
            a.node.Parents.Add(c.node);
            b_node.Parents.Add(c.node);

            return c;
        }
        public static Tensor operator *(float b, Tensor a)
        {
            Node b_node = new Node()
            {
                Operation = NodeOperation.Operand,
                ResultType = NodeResultType.Constant,
                Children = null,
                Value = b,
            };

            Tensor c = new Tensor(a.Axes);
            c.node = new Node()
            {
                Operation = NodeOperation.MultiplyFloat,
                ResultType = NodeResultType.ResultMatrix,
                Children = new Node[] { a.node, b_node },
                Value = c,
            };
            a.node.Parents.Add(c.node);
            b_node.Parents.Add(c.node);

            return c;
        }
        public static Tensor operator /(float b, Tensor a)
        {
            return b * Recip(a);
        }

        public static Tensor Dot(Tensor a, Tensor b)
        {
            //#error err
            //Tensor Dot Product
            //f = Wx
            //  = IWx
            //  = (x.T dot I)vec(W)
            //  = (x.T dot I)w
            //  = (x.T dot I)

            if (a.Axes.Length != 2) throw new Exception();
            if (b.Axes.Length != 2) throw new Exception();

            if (a.Axes[1] != b.Axes[0]) throw new Exception();

            Tensor c = new Tensor(a.Axes[0], b.Axes[1]);
            c.node = new Node()
            {
                Operation = NodeOperation.Dot,
                ResultType = NodeResultType.ResultMatrix,
                Children = new Node[] { a.node, b.node },
                Value = c,
            };
            a.node.Parents.Add(c.node);
            b.node.Parents.Add(c.node);

            return c;
        }

        public static Tensor Hadamard(Tensor a, Tensor b)
        {
            bool a_isZero = a.node.Operation == NodeOperation.Operand && a.Axes.All(z => z == 1) && (a.node.ResultType == NodeResultType.Constant && a.node.Value is float && (float)a.node.Value == 0.0f) || (a.node.ResultType == NodeResultType.CommonConstantMatrix && a.node.Value is Tensor && ((Tensor)a.node.Value).mem_const == 0.0f);
            bool a_isOne = a.node.Operation == NodeOperation.Operand && a.Axes.All(z => z == 1) && (a.node.ResultType == NodeResultType.Constant && a.node.Value is float && (float)a.node.Value == 1.0f) || (a.node.ResultType == NodeResultType.CommonConstantMatrix && a.node.Value is Tensor && ((Tensor)a.node.Value).mem_const == 1.0f);

            bool b_isZero = b.node.Operation == NodeOperation.Operand && b.Axes.All(z => z == 1) && (b.node.ResultType == NodeResultType.Constant && b.node.Value is float && (float)b.node.Value == 0.0f) || (b.node.ResultType == NodeResultType.CommonConstantMatrix && b.node.Value is Tensor && ((Tensor)b.node.Value).mem_const == 0.0f);
            bool b_isOne = b.node.Operation == NodeOperation.Operand && b.Axes.All(z => z == 1) && (b.node.ResultType == NodeResultType.Constant && b.node.Value is float && (float)b.node.Value == 1.0f) || (b.node.ResultType == NodeResultType.CommonConstantMatrix && b.node.Value is Tensor && ((Tensor)b.node.Value).mem_const == 1.0f);

            if (a_isOne)
                return b;

            if (b_isOne)
                return a;

            if (a_isZero | b_isZero) throw new Exception();

            //Shapes are not exactly the same
            if (!a.Axes.SequenceEqual(b.Axes))
            {
                //Verify that broadcasting will result in equivalent tensors
                var a_oaxes = new int[a.Axes.Length];
                var b_oaxes = new int[b.Axes.Length];
                for (int a_i = 0; a_i < a_oaxes.Length; a_i++)
                    if (a.Axes[a_i] == 1)
                        a_oaxes[a_i] = b.Axes[a_i];
                    else
                        a_oaxes[a_i] = a.Axes[a_i];

                for (int b_i = 0; b_i < b_oaxes.Length; b_i++)
                    if (b.Axes[b_i] == 1)
                        b_oaxes[b_i] = a.Axes[b_i];
                    else
                        b_oaxes[b_i] = b.Axes[b_i];

                if (a_oaxes.SequenceEqual(b_oaxes))
                {
                    Tensor c = new Tensor(a_oaxes);
                    c.node = new Node()
                    {
                        Operation = NodeOperation.VectorProduct,
                        ResultType = NodeResultType.ResultMatrix,
                        Children = new Node[] { a.node, b.node },
                        Value = c,
                    };
                    a.node.Parents.Add(c.node);
                    b.node.Parents.Add(c.node);

                    return c;
                }

                throw new Exception();
            }
            else
            {
                Tensor c = new Tensor(a.Axes);
                c.node = new Node()
                {
                    Operation = NodeOperation.Hadamard,
                    ResultType = NodeResultType.ResultMatrix,
                    Children = new Node[] { a.node, b.node },
                    Value = c,
                };
                a.node.Parents.Add(c.node);
                b.node.Parents.Add(c.node);

                return c;
            }
        }

        public static Tensor Pow(Tensor a, float b)
        {
            if (b == 1) return a;
            Node b_node = new Node()
            {
                Operation = NodeOperation.Operand,
                ResultType = NodeResultType.Constant,
                Children = null,
                Value = b,
            };

            Tensor c = new Tensor(a.Axes);
            c.node = new Node()
            {
                Operation = NodeOperation.Pow,
                ResultType = NodeResultType.ResultMatrix,
                Children = new Node[] { a.node, b_node },
                Value = c,
            };
            a.node.Parents.Add(c.node);
            b_node.Parents.Add(c.node);

            return c;
        }

        public static Tensor Exp(Tensor a)
        {
            Tensor c = new Tensor(a.Axes);
            c.node = new Node()
            {
                Operation = NodeOperation.Exp,
                ResultType = NodeResultType.ResultMatrix,
                Children = new Node[] { a.node },
                Value = c,
            };
            a.node.Parents.Add(c.node);

            return c;
        }

        public static Tensor Recip(Tensor a)
        {
            Tensor c = new Tensor(a.Axes);
            c.node = new Node()
            {
                Operation = NodeOperation.Reciprocal,
                ResultType = NodeResultType.ResultMatrix,
                Children = new Node[] { a.node },
                Value = c,
            };
            a.node.Parents.Add(c.node);

            return c;
        }

        public static string PrintValue(Tensor a)
        {
            if (a == null) return "null";
            if (a.node.Operation == NodeOperation.Operand && a.node.ResultType == NodeResultType.InitializedMatrix)
            {
                string v = "";
                if(a.Axes.Length == 2)
                for (int i = 0; i < a.Axes[0]; i++)
                {
                    v += "[";
                    for (int j = 0; j < a.Axes[1]; j++)
                        if (j < a.Axes[1] - 1)
                            v += $"{a.mem[a.Index(i, j)]}, ";
                        else
                            v += $"{a.mem[a.Index(i, j)]}";
                    v += "]\n";
                }
                return v;
            }else if(a.node.Operation == NodeOperation.Transpose && a.node.ResultType == NodeResultType.InitializedMatrix)
            {
                string v = "";
                if (a.Axes.Length == 2)
                    for (int i = 0; i < a.Axes[0]; i++)
                    {
                        v += "[";
                        for (int j = 0; j < a.Axes[1]; j++)
                            if (j < a.Axes[1] - 1)
                                v += $"{a.mem[a.Index(i, j)]}, ";
                            else
                                v += $"{a.mem[a.Index(i, j)]}";
                        v += "]\n";
                    }
                return v;
            }else if(a.node.Operation == NodeOperation.Operand && a.node.ResultType == NodeResultType.CommonConstantMatrix)
            {
                string v = $"[{a.mem_const}]\n";
                return v;
            }
            return "";
        }

        public static string PrintAxes(Tensor t)
        {
            string dims = "(";
            for (int i = 0; i < t.Axes.Length; i++)
                if (i < t.Axes.Length - 1)
                    dims += t.Axes[i] + ", ";
                else
                    dims += t.Axes[i];
            dims += ")";
            return dims;
        }
        public override string ToString()
        {
            return node.ToString();
        }
    }
}
