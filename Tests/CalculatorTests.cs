using System;
using Xunit;
using Dense;

namespace Dense.Tests
{
    public class CalculatorTests
    {
        [Theory]
        [InlineData("1+2*3", 7)]
        [InlineData("(1+2)*3", 9)]
        [InlineData("2^3^2", 512)] // 2^(3^2) = 512
        [InlineData("sin(0)", 0)]
        [InlineData("cos(0)", 1)]
        [InlineData("log(exp(1))", 1)]
        [InlineData("sqrt(4)+pi", 2 + Math.PI)]
        [InlineData("-3+5", 2)]
        public void Evaluate_ReturnsExpected(string expr, double expected)
        {
            var res = Calculator.Evaluate(expr);
            Assert.Equal(expected, res, 10);
        }
    }
}
