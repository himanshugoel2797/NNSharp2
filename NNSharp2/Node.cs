using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp2
{
    public enum NodeOperation
    {
        Operand,
        Addition,
        Dot,
        Pow,
        Exp,
        Reciprocal,
        MultiplyFloat,
        AddFloat,
        SubtractFloat,
        Transpose,
        Hadamard,
        VectorProduct,
    }

    public enum NodeResultType
    {
        ResultMatrix,
        InitializedMatrix,
        ParameterMatrix,
        CommonConstantMatrix,
        Constant,
    }

    public class Node
    {
        public NodeOperation Operation;
        public Node[] Children;
        public List<Node> Parents;

        public NodeResultType ResultType;
        public object Value;

        public Node()
        {
            Parents = new List<Node>();
        }

        public static bool operator ==(Node a, Node b)
        {
            var cmn = (a.Operation == b.Operation && a.ResultType == b.ResultType);
            if (!cmn)
                return false;

            if (a.Children != null && b.Children != null)
            {
                if (a.Children.Length != b.Children.Length)
                    return false;

                for (int i = 0; i < a.Children.Length; i++)
                    if (a.Children[i] != b.Children[i])
                        return false;
            }
            else if (!(a.Children == null && b.Children == null))
                return false;

            if (a.Parents.Count != b.Parents.Count)
                return false;

            return true;
        }
        public static bool operator !=(Node a, Node b)
        {
            return !(a == b);
        }

        public override string ToString()
        {
            switch (Operation)
            {
                case NodeOperation.Operand:
                    {
                        string dims = "";
                        switch (ResultType)
                        {
                            case NodeResultType.CommonConstantMatrix:
                            case NodeResultType.InitializedMatrix:
                            case NodeResultType.ParameterMatrix:
                                dims = Tensor.PrintAxes((Tensor)Value);
                                break;
                            case NodeResultType.Constant:
                                return ((float)Value).ToString();
                            default:
                                throw new Exception();
                        }

                        switch (ResultType)
                        {
                            case NodeResultType.CommonConstantMatrix:
                                return $"[{((Tensor)Value).mem_const}, {dims}]";
                            case NodeResultType.InitializedMatrix:
                                return $"[Preinit, {dims}]";
                            case NodeResultType.ParameterMatrix:
                                return $"[{((Tensor)Value).name}, {dims}]";
                            default:
                                throw new Exception();
                        }
                    }
                    break;
                case NodeOperation.AddFloat:
                case NodeOperation.Addition:
                    return $"({Children[0].ToString()} + {Children[1].ToString()})";
                case NodeOperation.Dot:
                    return $"({Children[0].ToString()} . {Children[1].ToString()})";
                case NodeOperation.Exp:
                    return $"Exp({Children[0].ToString()})";
                case NodeOperation.Reciprocal:
                    return $"1 / {Children[0].ToString()}";
                case NodeOperation.Hadamard:
                case NodeOperation.VectorProduct:
                    return $"({Children[0].ToString()} * {Children[1].ToString()})";
                case NodeOperation.MultiplyFloat:
                    return $"{Children[0].ToString()} * {Children[1].ToString()}";
                case NodeOperation.Pow:
                    return $"{Children[0].ToString()} ^ {Children[1].ToString()}";
                case NodeOperation.SubtractFloat:
                    return $"({Children[0].ToString()} - {Children[1].ToString()})";
                case NodeOperation.Transpose:
                    return $"{Children[0].ToString()}.T";
                default:
                    throw new Exception();
            }
        }
    }
}
