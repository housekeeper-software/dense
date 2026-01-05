using System;
using System.Drawing;
using System.Windows.Forms;

namespace Dense
{
    public class MainForm : Form
    {
        private TextBox display;
        private ListBox history;

        public MainForm()
        {
            Text = "Dense Calculator";
            ClientSize = new Size(420, 520);
            FormBorderStyle = FormBorderStyle.FixedSingle;
            MaximizeBox = false;

            display = new TextBox
            {
                ReadOnly = true,
                Font = new Font("Consolas", 18),
                TextAlign = HorizontalAlignment.Right,
                Dock = DockStyle.Top,
                Height = 50
            };
            Controls.Add(display);

            history = new ListBox
            {
                Dock = DockStyle.Right,
                Width = 140
            };
            Controls.Add(history);

            var panel = new TableLayoutPanel
            {
                RowCount = 6,
                ColumnCount = 5,
                Dock = DockStyle.Fill,
                Padding = new Padding(6),
            };
            for (int i = 0; i < panel.RowCount; i++) panel.RowStyles.Add(new RowStyle(SizeType.Percent, 100f / panel.RowCount));
            for (int j = 0; j < panel.ColumnCount; j++) panel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f / panel.ColumnCount));
            Controls.Add(panel);

            string[,] buttons =
            {
                { "7","8","9","/","sqrt" },
                { "4","5","6","*","^" },
                { "1","2","3","-","(" },
                { "0",".","=","+"," )" },
                { "sin","cos","tan","log","exp" },
                { "pi","e","C","⌫","Ans" }
            };

            for (int r = 0; r < buttons.GetLength(0); r++)
            {
                for (int c = 0; c < buttons.GetLength(1); c++)
                {
                    var text = buttons[r, c];
                    var btn = new Button
                    {
                        Text = text,
                        Dock = DockStyle.Fill,
                        Font = new Font("Segoe UI", 10),
                        Tag = text
                    };
                    btn.Click += Button_Click;
                    panel.Controls.Add(btn, c, r);
                }
            }

            KeyPreview = true;
            KeyDown += MainForm_KeyDown;
        }

        private string lastAnswer = "";

        private void Button_Click(object? sender, EventArgs e)
        {
            if (sender is Button b && b.Tag is string tag)
            {
                switch (tag)
                {
                    case "=":
                        Evaluate();
                        break;
                    case "C":
                        display.Text = "";
                        break;
                    case "⌫":
                        if (display.Text.Length > 0) display.Text = display.Text[..^1];
                        break;
                    case "Ans":
                        display.Text += lastAnswer;
                        break;
                    case "pi":
                        display.Text += "pi";
                        break;
                    case "e":
                        display.Text += "e";
                        break;
                    default:
                        // ensure function names are appended with a trailing '(' for convenience where appropriate
                        if (IsFunctionName(tag))
                            display.Text += tag + "(";
                        else
                            display.Text += tag;
                        break;
                }
            }
        }

        private bool IsFunctionName(string s)
        {
            return s is "sin" or "cos" or "tan" or "log" or "exp" or "sqrt";
        }

        private void Evaluate()
        {
            try
            {
                var expr = display.Text;
                if (string.IsNullOrWhiteSpace(expr)) return;
                var result = Calculator.Evaluate(expr);
                lastAnswer = result.ToString();
                history.Items.Insert(0, $"{expr} = {result}");
                display.Text = result.ToString();
            }
            catch (Exception ex)
            {
                MessageBox.Show("错误: " + ex.Message, "计算失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void MainForm_KeyDown(object? sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Enter) { Evaluate(); e.SuppressKeyPress = true; }
            else if (e.KeyCode == Keys.Back)
            {
                if (display.Text.Length > 0) display.Text = display.Text[..^1];
            }
            else
            {
                // simple mapping for digits and operators
                char ch = (char)0;
                if (e.KeyCode >= Keys.D0 && e.KeyCode <= Keys.D9)
                    ch = (char)('0' + (e.KeyCode - Keys.D0));
                else if (e.KeyCode >= Keys.NumPad0 && e.KeyCode <= Keys.NumPad9)
                    ch = (char)('0' + (e.KeyCode - Keys.NumPad0));
                else
                {
                    if (e.KeyCode == Keys.Add) ch = '+';
                    if (e.KeyCode == Keys.Subtract) ch = '-';
                    if (e.KeyCode == Keys.Multiply) ch = '*';
                    if (e.KeyCode == Keys.Divide) ch = '/';
                    if (e.KeyCode == Keys.OemPeriod) ch = '.';
                    if (e.KeyCode == Keys.OemMinus) ch = '-';
                }

                if (ch != 0) display.Text += ch;
            }
        }
    }
}
