import gradio as gr
from paddlespeech.cli.asr.infer import ASRExecutor

# 初始化ASR识别器
asr = ASRExecutor()

# 语音识别函数
def recognize_audio(audio):
    result = asr(audio_file=audio,force_yes=True)
    return result

# 创建Gradio应用
with gr.Blocks() as app:
    # 文件上传或麦克风录制音频
    audio_input = gr.Audio(source="microphone", type="filepath", label="录制或上传语音")
    result_output = gr.Textbox(label="识别结果")

    # 按钮点击触发语音识别
    record_button = gr.Button("开始语音识别")
    record_button.click(fn=recognize_audio, inputs=audio_input, outputs=result_output)

# 启动应用
app.launch()
