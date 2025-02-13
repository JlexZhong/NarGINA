import gradio as gr
from bokeh.plotting import figure
from bokeh.embed import components
import numpy as np

# 定义绘图函数
def plot_function_bokeh(x):
    x_values = np.linspace(-10, 10, 400)
    y_values = np.sin(x_values) if x == "sin" else np.cos(x_values)

    # 创建Bokeh图
    p = figure(title=f'{x} Function Plot', x_axis_label='x-axis', y_axis_label='y-axis')
    p.line(x_values, y_values, legend_label=f'{x} function', line_width=2)

    # 使用Bokeh组件嵌入
    script, div = components(p)
    return f"{script}\n{div}"

# 创建Gradio Interface
interface = gr.Interface(
    fn=plot_function_bokeh,
    inputs=gr.inputs.Radio(["sin", "cos"], label="Select Function"),
    outputs="html",  # 使用HTML输出来嵌入Bokeh的图
    title="Interactive Plotting with Bokeh",
    description="Select either a sine or cosine function to visualize the plot interactively."
)

# 启动应用
interface.launch()
