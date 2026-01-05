using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace Dense
{
    public static class Calculator
    {
        private static readonly Dictionary<string, int> Precedence = new()
        {
            { "+", 1 }, { "-", 1 }, { "*", 2 }, { "/", 2 }, { "^", 3 }
        };

        private static readonly HashSet<string> RightAssociative = new() { "^" };

        private static readonly HashSet<string> Functions = new()
        {
            "sin","cos","tan","log","exp","sqrt"
        };

        public static double Evaluate(string expr)
        {
            var rpn = ToRpn(expr);
            return EvalRpn(rpn);
        }

        // Tokenize then shunting-yard
        private static List<string> ToRpn(string expr)
        {
            var tokens = Tokenize(expr);
            var output = new List<string>();
            var ops = new Stack<string>();

            for (int i = 0; i < tokens.Count; i++)
            {
                string t = tokens[i];
                if (double.TryParse(t, NumberStyles.Number, CultureInfo.InvariantCulture, out _))
                {
                    output.Add(t);
                }
                else if (IsConstant(t))
                {
                    output.Add(ConstantValue(t).ToString(CultureInfo.InvariantCulture));
                }
                else if (Functions.Contains(t))
                {
                    ops.Push(t);
                }
                else if (t == ",")
                {
                    while (ops.Count > 0 && ops.Peek() != "(")
                        output.Add(ops.Pop());
                }
                else if (IsOperator(t))
                {
                    while (ops.Count > 0 && IsOperator(ops.Peek()))
                    {
                        var o2 = ops.Peek();
                        if ((RightAssociative.Contains(t) && Precedence[t] < Precedence[o2]) ||
                            (!RightAssociative.Contains(t) && Precedence[t] <= Precedence[o2]))
                        {
                            output.Add(ops.Pop());
                        }
                        else break;
                    }
                    ops.Push(t);
                }
                else if (t == "(")
                {
                    // handle unary minus: if previous token is operator or left paren or start, allow unary
                    // keep "(" 
                    ops.Push(t);
                }
                else if (t == ")")
                {
                    while (ops.Count > 0 && ops.Peek() != "(")
                        output.Add(ops.Pop());
                    if (ops.Count == 0) throw new Exception("括号不匹配");
                    ops.Pop(); // pop "("
                    if (ops.Count > 0 && Functions.Contains(ops.Peek()))
                        output.Add(ops.Pop());
                }
                else
                {
                    throw new Exception($"无法识别的 token: {t}");
                }
            }

            while (ops.Count > 0)
            {
                var op = ops.Pop();
                if (op == "(" || op == ")") throw new Exception("括号不匹配");
                output.Add(op);
            }

            return output;
        }

        private static double EvalRpn(List<string> tokens)
        {
            var st = new Stack<double>();
            foreach (var t in tokens)
            {
                if (double.TryParse(t, NumberStyles.Number, CultureInfo.InvariantCulture, out double n))
                {
                    st.Push(n);
                }
                else if (IsOperator(t))
                {
                    if (st.Count < 2) throw new Exception("运算符参数不足");
                    var b = st.Pop();
                    var a = st.Pop();
                    switch (t)
                    {
                        case "+": st.Push(a + b); break;
                        case "-": st.Push(a - b); break;
                        case "*": st.Push(a * b); break;
                        case "/":
                            if (b == 0) throw new DivideByZeroException();
                            st.Push(a / b); break;
                        case "^": st.Push(Math.Pow(a, b)); break;
                    }
                }
                else if (Functions.Contains(t))
                {
                    if (st.Count < 1) throw new Exception("函数参数不足");
                    var a = st.Pop();
                    switch (t)
                    {
                        case "sin": st.Push(Math.Sin(a)); break;
                        case "cos": st.Push(Math.Cos(a)); break;
                        case "tan": st.Push(Math.Tan(a)); break;
                        case "log":
                            if (a <= 0) throw new Exception("log 的参数必须大于 0");
                            st.Push(Math.Log(a)); break;
                        case "exp": st.Push(Math.Exp(a)); break;
                        case "sqrt":
                            if (a < 0) throw new Exception("sqrt 的参数不能为负数");
                            st.Push(Math.Sqrt(a)); break;
                    }
                }
                else
                {
                    throw new Exception($"无法在求值阶段识别 token: {t}");
                }
            }
            if (st.Count != 1) throw new Exception("表达式求值后栈不为 1");
            return st.Pop();
        }

        private static bool IsOperator(string s) => Precedence.ContainsKey(s);

        private static bool IsConstant(string s) => s == "pi" || s == "e";

        private static double ConstantValue(string s) => s switch
        {
            "pi" => Math.PI,
            "e" => Math.E,
            _ => throw new Exception("未知常数")
        };

        // 简单 tokenizer，支持函数名、数字、小数点、常数、运算符、括号
        private static List<string> Tokenize(string expr)
        {
            var tokens = new List<string>();
            var i = 0;
            expr = expr.Replace(" ", "");
            while (i < expr.Length)
            {
                var c = expr[i];
                if (char.IsDigit(c) || c == '.')
                {
                    var sb = new StringBuilder();
                    while (i < expr.Length && (char.IsDigit(expr[i]) || expr[i] == '.'))
                    {
                        sb.Append(expr[i]);
                        i++;
                    }
                    tokens.Add(sb.ToString());
                }
                else if (char.IsLetter(c))
                {
                    var sb = new StringBuilder();
                    while (i < expr.Length && char.IsLetter(expr[i]))
                    {
                        sb.Append(expr[i]); i++;
                    }
                    tokens.Add(sb.ToString());
                }
                else if ("+-*/^(),".IndexOf(c) >= 0)
                {
                    // handle unary minus: if '-' and (start or previous is operator or '('), treat as unary by adding 0 then '-'
                    if (c == '-')
                    {
                        bool unary = tokens.Count == 0 || tokens[^1] == "(" || IsOperator(tokens[^1]) || tokens[^1] == ",";
                        if (unary)
                        {
                            tokens.Add("0");
                        }
                    }
                    tokens.Add(c.ToString());
                    i++;
                }
                else
                {
                    throw new Exception($"无效字符: {c}");
                }
            }
            return tokens;
        }
    }
}
