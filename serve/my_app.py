
import sys
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "7"
sys.path.append("./")
sys.path.append("./utils")
sys.path.append("./train")
import gradio as gr
import spacy
import torch
from torch_geometric.data import Data
from PIL import Image
import json
import os
from utils.utils import tokenizer_graph_token
from ChildText.prompt import LABEL_TEMPLATE_SCORE, LABEL_TEMPLATE_TRAIT, LABEL_TEMPLATE_TRAIT_COMMENT, PROMPT_TEMPLATE,PROMPT_TEMPLATE_TRAIT_COMMENT
from train.train_childtext import  build_heterogeneous_graph_string 
from utils.metric import analyze_composition
import torch
torch.autograd.set_detect_anomaly(True)
import os
from serve.server_utils import COMMENT_GENERATE_PROMPT, CRITERIA, ST_emb_text, cl_model_encode, get_graph_encoder, get_pydot_graph_img, get_text_encoder, gold_image_to_plotly, image_to_plotly, load_model, recognize_audio
import json
from utils.constants import GRAPH_TOKEN_INDEX
from utils.conversation import conv_templates, SeparatorStyle
from utils.utils import tokenizer_graph_token

from serve.explian import explain_visualization


dtype = torch.bfloat16

conv_mode = 'conv_childtext'

model_paths = {
    "vicuna-7b-v1.5-score-nolora": {
        "model_path": '/disk/NarGINA/checkpoints/ChildText/score/llaga-vicuna7b-GRACE_512_STv2-mlpv2-no_edgestr-pre_metric-stage2',    
        #"model_path": "/disk/NarGINA/checkpoints/ChildText/llaga-vicuna7b-GRACE_512_ST-mlpv2-no_edgestr",  # 替换为你的实际模型路径
        "model_base": "/disk/NarGINA/weights/vicuna-7b-v1.5",  # 替换为基座模型路径
        "encoder_weight_path": "/disk/NarGINA/contrastive_learning/weights/ChildText/v2/encoder_model_ST_hidden=512_numlayers=4_headnum=8_loss=3.92_weights.pth",
    },    
    "vicuna-7b-v1.5-multi-trait-nolora": {
        "model_path": '/disk/NarGINA/checkpoints/ChildText/score_trait/llaga_vicuna7b_GRACE_512_STv2_mlpv2_no_edgestr_trait/checkpoint-204',    
        #"model_path": "/disk/NarGINA/checkpoints/ChildText/llaga-vicuna7b-GRACE_512_ST-mlpv2-no_edgestr",  # 替换为你的实际模型路径
        "model_base": "/disk/NarGINA/weights/vicuna-7b-v1.5",  # 替换为基座模型路径
        "encoder_weight_path": '/disk/NarGINA/contrastive_learning/weights/ChildText/v2/encoder_model_ST_hidden=512_numlayers=4_headnum=8_loss=3.92_weights.pth',
    },
    # "vicuna-7b-v1.5-multi-trait-nolora": {
    #     "model_path": '/disk/NarGINA/checkpoints/ChildText/llaga_vicuna7b_GRACE_512_7bv2_mlpv2_no_edgestr_trait/checkpoint-192',    
    #     #"model_path": "/disk/NarGINA/checkpoints/ChildText/llaga-vicuna7b-GRACE_512_ST-mlpv2-no_edgestr",  # 替换为你的实际模型路径
    #     "model_base": "/disk/NarGINA/weights/vicuna-7b-v1.5",  # 替换为基座模型路径
    #     "encoder_weight_path": '/disk/NarGINA/contrastive_learning/weights/ChildText/v2/encoder_model_vicuna-7b-v1.5_hidden=512_numlayers=4_headnum=8_loss=4.32_weights.pth',
    # },#TODO还未实现预处理函数，加文本前缀

    "vicuna-7b-v1.5-score": {
        "model_path": "/disk/NarGINA/checkpoints/ChildText/llaga-vicuna7b-GRACE_512_ST-mlpv2-no_edgestr-lora",  # 替换为你的实际模型路径
        "model_base": "/disk/NarGINA/weights/vicuna-7b-v1.5",  # 替换为基座模型路径
        #"encoder_weight_path":
    },    
    "vicuna-7b-v1.5-multi-trait": {
        "model_path": "/disk/NarGINA/checkpoints/ChildText/llaga_vicuna7b_GRACE_512_ST_mlpv2_no_edgestr_trait_premicro_lora",  # 替换为你的实际模型路径
        "model_base": "/disk/NarGINA/weights/vicuna-7b-v1.5",  # 替换为基座模型路径
        #"encoder_weight_path":
    },
}

def get_all_model(select_model_item):
    tokenizer, model = load_model(select_model_item["model_path"], select_model_item["model_base"])
    graph_encoder_model = get_graph_encoder(select_model_item["encoder_weight_path"],select_model_item["model_path"])
    text_encoder = get_text_encoder(select_model_item["model_path"])
    return tokenizer,model,graph_encoder_model,text_encoder

# 加载优先模型模型
first_model_item = list(model_paths.items())[0]
current_model_name = first_model_item[0]

tokenizer,model,graph_encoder_model,text_encoder = get_all_model(first_model_item[1])


# 显示故事图图片
image_folder = "/disk/NarGINA/serve/asset/story"  # 替换为你的图片文件夹路径
# 获取并排序图片路径，确保顺序为 0.png, 1.png, 2.png...
image_paths = sorted(
    [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png") and 'graph' not in f],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
)
images = [Image.open(path) for path in image_paths]
total_images = len(images)
current_index = 0# 全局变量来跟踪当前图片索引


custom_css = """
.icon-btn {
    max-width: 30px;
    border: none;
    padding: 5px; /* 减少内边距 */
    margin: 2px; /* 减少外边距，使按钮更紧凑 */
}
.textbox-custom {
    font-weight: bold; /* 使文本加粗 */
    font-size: 50px; /* 可选：设置字体大小 */
}

"""

def model_inference(input_text, model_name,generate_comment):
    response = chat_with_model(input_text, model_name,generate_comment)
    return response

# 定义模型切换函数
def change_model(model_name):
    global tokenizer, model, graph_encoder_model,text_encoder,current_model_name
    if model_name != current_model_name:
        #tokenizer, model = load_model(model_paths[model_name]['model_path'], model_paths[model_name]["model_base"])
        tokenizer,model,graph_encoder_model,text_encoder = get_all_model(model_paths[model_name])
        current_model_name = model_name
    return f"已加载模型：{model_name}"

    
def process(input_text,g,tokenizer):
    edge_code_str = build_heterogeneous_graph_string(g.edge_index,g.edge_type)
    if 'trait' in current_model_name :    # 叙事图+文本，多维度评分加评论
        qs = PROMPT_TEMPLATE_TRAIT_COMMENT
    elif 'score' in current_model_name :                          # 叙事图+文本，只做评分       
        qs = PROMPT_TEMPLATE

    qs = qs.replace("<essay_text>",input_text)
    ## 计算微观指标
    if 'metric' in model_paths[current_model_name]['model_path']:
        nlp_model = spacy.load("zh_core_web_sm")
        micro_metric = analyze_composition(text=input_text,nlp=nlp_model)
        qs = qs.replace("<micro_metrics_str>",micro_metric)
    else:
        qs = qs.replace("3. 微观结构维度评分时，请你使用以下量化数据作为参考：<micro_metrics_str>\n","")
    
    if 'trait' in current_model_name and 'comment' in current_model_name:
        label_template = LABEL_TEMPLATE_TRAIT_COMMENT
    elif 'trait' in current_model_name:
        label_template = LABEL_TEMPLATE_TRAIT
    else:
        label_template = LABEL_TEMPLATE_SCORE
    qs = qs.replace("<label_template>",label_template)
    if "no_edgestr" in current_model_name:
        qs = qs.replace("- 边结构：<edge_code_str>\n",'')
    else:
        qs = qs.replace("<edge_code_str>",edge_code_str)

    g.id = id
    g.conversations = [{"from":"human","value":qs},
                    {"from":"gpt","value": None}]

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    #print(prompt)
    input_ids = tokenizer_graph_token(prompt, tokenizer, GRAPH_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    g.x = g.x.to(dtype)
    g.edge_attr = g.edge_attr.to(dtype)
    # g_batch = Batch.from_data_list([g])
    graph = torch.LongTensor(range(g.x.size(0))).unsqueeze(0)
    graph_emb = g.x.unsqueeze(0)
    edge_index = g.edge_index
    edge_attr = g.edge_attr.to(dtype)
    edge_type = g.edge_type
    return input_ids,graph_emb,graph,edge_index,edge_attr,edge_type,conv,qs


def graph_json_2_pyg(g_json,encoder):
    # 边类型文本描述
    relation_descriptions = [
        "使能-因果",
        "动机-因果",
        "物理-因果",
        "心理-因果",
        "并列",
    ]
    if 'STv2' in model_paths[current_model_name]['model_path']:
        edge_embeddings = ST_emb_text(relation_descriptions,encoder).cpu()
    else:
        edge_embeddings = encoder.encode(relation_descriptions).cpu()
    # Step 1: 获取所有唯一节点并分配索引
    node_dict = {}  # 存储每个节点和它的索引
    edges = []  # 存储边
    edge_types = []  # 存储边的类型索引
    edge_attr_list = []  # 存储每个边的特征向量
    node_texts = []  # 存储每个节点的文本

    # 分配索引并创建边，收集节点文本
    for graph in g_json:
        for edge in graph:
            first_event = edge["first_event"]
            second_event = edge["second_event"]
            relation = edge["relation"]

            # 如果节点未见过，分配索引并存储文本
            if first_event not in node_dict:
                node_dict[first_event] = len(node_dict)
                node_texts.append(first_event)  # 添加到文本列表
            if second_event not in node_dict:
                node_dict[second_event] = len(node_dict)
                node_texts.append(second_event)  # 添加到文本列表

            # 处理边，跳过"无"类型的边
            if relation != "无":
                first_idx = node_dict[first_event]
                second_idx = node_dict[second_event]
                edges.append([first_idx, second_idx])

                # 存储边的类型索引
                edge_type = relation_descriptions.index(relation)
                edge_types.append(edge_type)
                
                # 添加边的特征向量
                edge_attr_list.append(edge_embeddings[edge_type])

    # Step 3: 构建边索引张量
    if edges:  # 如果有边
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # 转置以符合PyG格式
        edge_attr = torch.stack(edge_attr_list)
    else:  # 如果没有边
        edge_index = torch.empty((2, 0), dtype=torch.long)  # 空边张量
        edge_attr = None

    # Step 4: 创建数据对象
    edge_type_tensor = torch.tensor(edge_types, dtype=torch.long)  # 将边类型索引转换为张量
    if 'STv2' in model_paths[current_model_name]['model_path']:
        node_embeddings = ST_emb_text(node_texts,encoder).cpu()
    else:
        node_embeddings = encoder.encode(node_texts).cpu()
    data = Data(x=node_embeddings, edge_index=edge_index, edge_type=edge_type_tensor, edge_attr=edge_attr)##TODO 7bv2的数据集需要多一步预处理，给每个节点和边加上前缀文本
    return data


# 定义对话函数
def chat_with_model(input_text, model_name,generate_comment):
    
    #tokenizer, model = model_dict.get(model_name)
    #g_json = narrative_graph_api(input_text=input_text)
    with open('/disk/NarGINA/ChildText/gold_data/content.json', 'r', encoding='utf-8') as file:
        g_json = json.load(file)
    with open('/disk/NarGINA/ChildText/gold_data/gold_graph_content.json', 'r', encoding='utf-8') as file:
        gold_g_json = json.load(file)
    g = graph_json_2_pyg(g_json,encoder=text_encoder)
    # data_path = '/disk/NarGINA/dataset/ChildText/embedings/GRACE_7b/pretrained_GAT_hidden=512_test.pkl'
    # with open(data_path, 'rb') as f:
    #     data = pickle.load(f)
    # g = data[0]
    #0:小狗小青蛙小朋友就要准备睡觉了.于是小青蛙没睡觉.他们两个睡觉了.然后他们醒了.咦?小青蛙不见了.他们在找小青蛙.到处找啊找,找啊找.于是他在找啊.他掉下去了.小狗接住了他.然后他们在找小青蛙.地洞里是不是有小青蛙呢?不是小青蛙.他们在树洞里找.又不是小青蛙.他们在山上找.和小驯鹿就一起找了.掉下去了.他们在河里了.他掉进河底也要开始找小青蛙.嘘.然后他找啊找.忽然找到了两只小青蛙.还有几只小青蛙.然后他找到了小青蛙.他就高兴的笑了.
    # 使用 to_graph 函数生成图数据
    g = cl_model_encode(g,graph_encoder_model)

    input_ids,graph_emb,graph,edge_index,edge_attr,edge_type,conv,prompt= process(input_text,g,tokenizer)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    outputs = model_inference_fn(model,tokenizer, input_ids,graph_emb,graph,edge_index,edge_attr,edge_type,stop_str)
    comment_outputs = ''
    if generate_comment:
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], outputs)
        conv.append_message(conv.roles[0], COMMENT_GENERATE_PROMPT)
        conv.append_message(conv.roles[1], None)
        new_input_text = conv.get_prompt()
        input_ids = tokenizer_graph_token(new_input_text, tokenizer, GRAPH_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        comment_outputs = model_inference_fn(model,tokenizer, input_ids,graph_emb,graph,edge_index,edge_attr,edge_type,stop_str)
    # iimage generate
    
    # main_filename,isolated_filename = get_pydot_graph_img(g_json,'child')
    gold_graph_filename,gold_isolated_filename = get_pydot_graph_img(gold_g_json,'gold')
    #gold_graph_filename = "/disk/NarGINA/serve/asset/gold.png"
    gold_main_fig = image_to_plotly(gold_graph_filename)


    main_filename = "/disk/NarGINA/serve/asset/explain.png"
    explain_visualization(g_json,gold_g_json,main_filename)
    main_fig = image_to_plotly(main_filename)

    # img = post_graph_image()
    final_output = outputs + '\n' +comment_outputs
    return final_output,main_fig,gold_main_fig

def model_inference_fn(model,tokenizer, input_ids,graph_emb,graph,edge_index,edge_attr,edge_type,stop_str):
    # 生成输出
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            graph_emb=graph_emb.to(dtype),
            graph=graph.to('cuda').to(dtype),  # 简单的 graph 示例
            edge_index=edge_index.to('cuda').to(dtype),
            edge_attr=edge_attr.to('cuda').to(dtype),
            edge_type=edge_type.to('cuda').to(dtype),
            max_new_tokens=1024,
            do_sample=True,
            temperature=1.2,
            use_cache=True
        )
    # 解码输出
    input_token_len = input_ids.shape[1]#2414
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    #output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()    
    return outputs

# 用于显示当前图片的函数
def show_image(index):
    return images[index]

# 按钮函数：切换到下一张图片
def next_image():
    global current_index
    if current_index < total_images - 1:
        current_index += 1
    return show_image(current_index)

# 按钮函数：切换到上一张图片
def prev_image():
    global current_index
    if current_index > 0:
        current_index -= 1
    return show_image(current_index)

# 按钮函数：切换到下一张图片
def reset_image():
    global current_index
    current_index =0
    return show_image(current_index)


with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("## 儿童叙事能力评价")
    with gr.Row():
        with gr.Column(scale=3):
            image_display = gr.Image(show_image(current_index),height=750, show_label=True,label='《青蛙，你在哪里？》故事书')#width=800, height=400
            with gr.Row():
                prev_btn = gr.Button(value="⬅️", elem_classes=["icon-btn"])
                next_btn = gr.Button(value="➡️", elem_classes=["icon-btn"])
                reset_btn = gr.Button(value="🔄", elem_classes=["icon-btn"])   
                # example_display  = gr.Examples(
                #         examples=truncated_text_examples,
                #         inputs=[input_text, generate_comment_checkbox, model_dropdown],  # 对应输入组件
                #         label="Example Inputs"  # 示例标签
                #     )     
        with gr.Column(scale=2):
            gr.Markdown("### 讲讲《青蛙，你在哪里？》的故事吧！")
            with gr.Row():
               
                    # 文件上传或麦克风录制音频
                audio_input = gr.Audio(source="microphone", type="filepath", label="录制或上传语音",scale=3)
                record_button = gr.Button("语音识别",scale=1)

            with gr.Row():
                input_text = gr.Textbox(
                    placeholder="或用文字讲述这个故事",
                    lines=8,
                    show_label=False
                )
            with gr.Row():
            # 模型选择下拉菜单
                model_dropdown = gr.Dropdown(
                    label="选择模型",
                    choices=list(model_paths.keys()),
                    value=first_model_item[0],
                    allow_custom_value=False,
                )
                # 输出加载状态
                load_status = gr.Textbox(value=f"加载模型成功：{current_model_name}",
                                          interactive=False,
                                         show_label=False, 
                                        )
            # 新增“生成评语”复选框
            with gr.Row():
                generate_comment_checkbox = gr.Checkbox(label="生成评语", value=False,visible=True)
                submit_btn = gr.Button("提交")
    with gr.Row():
        gr.Markdown(CRITERIA,scale=2)
        # 显示模型输出的文本
        output_text = gr.Textbox(
            label="模型评价",
            lines=12,
            interactive=False,  # 设置为不可编辑
            show_copy_button=True,
            elem_classes=["textbox-custom"],
            scale=3
        )
    # 评价部分
    with gr.Row():
        with gr.Tabs():
            with gr.TabItem("叙事图"): 
                graph_display_plotly = gr.Plot(label='叙事图')
            with gr.TabItem("金标图"):
                gold_graph_plotly =  gr.Plot(label='金标叙事图')
        #graph_display = gr.Image(show_label=True,label='叙事图')#width=800, height=500


    submit_btn.click(model_inference, inputs=[input_text,model_dropdown,generate_comment_checkbox], outputs=[output_text,graph_display_plotly,gold_graph_plotly])
    # 图片切换按钮交互逻辑
    # 按钮交互逻辑
    next_btn.click(next_image, outputs=image_display)
    prev_btn.click(prev_image, outputs=image_display)
    reset_btn.click(reset_image,outputs=image_display)
    
    # 更改模型选择时加载新模型并显示状态
    model_dropdown.change(fn=change_model, inputs=model_dropdown, outputs=load_status)
    # 按钮点击触发语音识别
    
    record_button.click(fn=recognize_audio, inputs=audio_input, outputs=input_text)

# 启动 Gradio 应用
demo.launch(server_name="0.0.0.0", server_port=7860)


