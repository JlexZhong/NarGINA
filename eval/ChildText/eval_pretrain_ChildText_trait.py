
import os

from termcolor import colored

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from sklearn.metrics import cohen_kappa_score
import spacy
import pickle
import re
import sys
sys.path.append("./")
sys.path.append("./utils")
sys.path.append("./train")
from ChildText.prompt import LABEL_TEMPLATE_TRAIT, LABEL_TEMPLATE_TRAIT_COMMENT, LABEL_TEMPLATE_V2, PROMPT_TEMPLATE, PROMPT_TEMPLATE_TRAIT_ONLY_GRAPH,PROMPT_TEMPLATE_only_graph,PROMPT_TEMPLATE_TRAIT_COMMENT,PROMPT_TEMPLATE_TRAIT_COMMENT_FEW_SHOT,PROMPT_TEMPLATE_TRAIT_COMMENT_V2
from train.train_childtext import  build_heterogeneous_graph_string, extract_prediction_score, extract_prediction_trait_score 
from utils.metric import analyze_composition
from torch_geometric.nn.models import GAT
import argparse
import torch
torch.autograd.set_detect_anomaly(True)
import os
import json
from tqdm import tqdm
import shortuuid

from eval.ChildText.eval_pretrain_ChildText import get_checkpoint_file_paths
from utils.constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN
from utils.conversation import conv_templates, SeparatorStyle
from model.builder import load_pretrained_model
from utils.utils import disable_torch_init, tokenizer_graph_token, get_model_name_from_path
from torch_geometric.utils import k_hop_subgraph, degree, remove_self_loops, add_self_loops
from torch_geometric.data import Batch

def eval_model(args):
    # Model
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"Loaded from {model_path}. Model Base: {args.model_base}")
    tokenizer, model, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                          cache_dir=args.cache_dir)
    dtype = torch.float16
    model = model.to(dtype)
    model = model.cuda()
    # print(model)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)


    ans_file = open(answers_file, "w",encoding='utf-8')
    data_path = args.data_path
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print("Dataset size:",len(data))
    preds=[]
    labels=[]
    if args.is_precompute_micro_metric:
        nlp_model = spacy.load("zh_core_web_sm")
    for id,g in enumerate(tqdm(data)):
        edge_code_str = build_heterogeneous_graph_string(g.edge_index,g.edge_type)
        # 不要加标签
        if args.is_only_graph:         # 只输入叙事图，只做评分
            qs = PROMPT_TEMPLATE_TRAIT_ONLY_GRAPH
        elif args.is_trait_comment:    # 叙事图+文本，多维度评分加评论
            qs = PROMPT_TEMPLATE_TRAIT_COMMENT
            qs = qs.replace("<essay_text>",g.essay_text)
            if args.is_precompute_micro_metric:
                micro_metric = analyze_composition(text=g.essay_text,nlp=nlp_model)
                qs = qs.replace("<micro_metrics_str>",micro_metric)
            else:
                qs = qs.replace("3. 微观结构维度评分时，请你使用以下量化数据作为参考：<micro_metrics_str>\n","")
        elif args.is_trait_comment or args.is_trait :    # 叙事图+文本，多维度评分加评论
            qs = PROMPT_TEMPLATE_TRAIT_COMMENT
            qs = qs.replace("<essay_text>",g.essay_text)
            if args.is_precompute_micro_metric:
                micro_metric = analyze_composition(text=g.essay_text,nlp=nlp_model)
                qs = qs.replace("<micro_metrics_str>",micro_metric)
            else:
                qs = qs.replace("3. 微观结构维度评分时，请你使用以下量化数据作为参考：<micro_metrics_str>\n","")
            qs = qs.replace("<label_template>",LABEL_TEMPLATE_TRAIT_COMMENT if args.is_trait_comment else LABEL_TEMPLATE_TRAIT)                      

        else:                          # 叙事图+文本，只做评分       
            qs = PROMPT_TEMPLATE
            qs = qs.replace("<essay_text>",g.essay_text[0])

        if args.is_edge_str:
            qs = qs.replace("<edge_code_str>",edge_code_str)
        else:
            qs = qs.replace("- 边结构：<edge_code_str>\n",'')
        g.id = id
        label_str = make_label(data_args=args,g=g)
        g.conversations = [{"from":"human","value":qs},
                        {"from":"gpt","value": label_str}]

        cur_prompt = qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        qs = conv.get_prompt()
        #print(prompt)
        input_ids = tokenizer_graph_token(qs, tokenizer, GRAPH_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        g.x = g.x.to(dtype)
        g.edge_attr = g.edge_attr.to(dtype)
        # g_batch = Batch.from_data_list([g])
        graph = torch.LongTensor(range(g.x.size(0))).unsqueeze(0)
        graph_emb = g.x.unsqueeze(0)
        edge_index = g.edge_index
        edge_attr = g.edge_attr.to(dtype)
        edge_type = g.edge_type
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        # try:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,#[1,1832]
                graph_emb=graph_emb.cuda(),#torch.half() 函数用于将张量的数据类型转换为 16 位浮点数,加速计算
                graph=graph.cuda(),
                edge_index = edge_index.cuda(),
                edge_attr = edge_attr.cuda(),
                edge_type = edge_type.cuda(),
                # g = g_batch.cuda(),
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)
            
        # input_clone = input_ids.clone()
        # input_clone[input_clone == -200] = tokenizer.pad_token_id
        # input_tmp = tokenizer.batch_decode(input_clone, skip_special_tokens=True)[0]
        # #print(input_tmp)

        # output_clone = output_ids.clone()
        # output_clone[output_clone == -200] = tokenizer.pad_token_id
        # tmp = tokenizer.batch_decode(output_clone, skip_special_tokens=True)[0]
        #print(tmp)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # except Exception as e:
        #     print(f"!!!!!!Error!!!!! {e}")
        #     outputs=""
        #print(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": id,
                                   "prediction": outputs,
                                   "gt":g.conversations[1]['value'],
                                   "essay_text":g.essay_text,
                                   "prompt": cur_prompt,
                                   "answer_id": ans_id},ensure_ascii=False) + "\n")
        ans_file.flush()
        # 计算
        preds.append(outputs)
        labels.append(label_str)


    ans_file.close()
    assert len(preds) == len(labels), "预测和标签的长度不匹配"
    qwk = compute_trait_qwk( preds,labels)

        # 打印 QWK 分数
    QWK_macro,QWK_micro,QWK_psych,QWK_total = qwk['宏观结构得分'],qwk['微观结构得分'],qwk['叙事心理描写得分'],qwk['总分']
    QWK_avg = (QWK_macro+QWK_micro+QWK_psych+QWK_total) / 4
    print(colored(f"Model:{model_path}=========QWK-macro:{qwk['宏观结构得分']:.3f};QWK-micro:{qwk['微观结构得分']:.3f};QWK-psych:{qwk['叙事心理描写得分']:.3f};QWK-total:{qwk['总分']:.3f};QWK-avg:{QWK_avg:.3f}",'light_green'))
    new_name = os.path.splitext(args.answers_file)[0]+ f'-QWK_macro={QWK_macro:.3f}-QWK_micro={QWK_micro:.3f}-QWK_psych={QWK_psych:.3f}-QWK_total={QWK_total:.3f}-QWK_avg={QWK_avg:.3f}' + os.path.splitext(args.answers_file)[1]
    os.rename(args.answers_file,new_name)
    # print("保存到：",new_name)
    del model
    torch.cuda.empty_cache()  # 清理GPU缓存

def make_label(data_args,g):
    """
    构建标签string"""
    if data_args.is_trait_comment:
        labels = LABEL_TEMPLATE_TRAIT_COMMENT.replace("<comment>",g.comment).replace("<macro_score>",str(g.macro_score.item())).replace("<micro_score>",str(g.micro_score.item())).replace("<psych_score>",str(g.psych_score.item())).replace("<total_score>",str(g.total_score.item()))
    elif data_args.is_trait:
        labels = LABEL_TEMPLATE_TRAIT.replace("<macro_score>",str(g.macro_score.item())).replace("<micro_score>",str(g.micro_score.item())).replace("<psych_score>",str(g.psych_score.item())).replace("<total_score>",str(g.total_score.item()))    
    else:
        labels = "预测得分："+str(g.y.item())
    return labels

def compute_trait_qwk(pred_str,label_str):

    # 初始化保存每个维度的 QWK 分数
    qwk_scores = {}

    # 针对每个维度分别提取并计算 QWK
    for score_type in ["宏观结构得分", "微观结构得分", "叙事心理描写得分", "总分"]:
        # 提取相应维度的得分
        filtered_results = [(extract_prediction_trait_score(p).get(score_type), extract_prediction_trait_score(l).get(score_type)) 
                            for p, l in zip(pred_str, label_str) 
                            if extract_prediction_trait_score(p).get(score_type) is not None and extract_prediction_trait_score(l).get(score_type) is not None]

        if filtered_results:
            preds_score, y = zip(*filtered_results)
            qwk_score = cohen_kappa_score(preds_score, y, weights='quadratic')
            qwk_scores[score_type] = qwk_score
        else:
            qwk_scores[score_type] = 0.0  # 如果没有有效数据

    # 打印并返回每个维度的 QWK 分数
    for score_type, qwk in qwk_scores.items():
        print(f"QWK for {score_type}: {qwk}")
    
    return qwk_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/disk/NarGINA/checkpoints/ChildText/score_trait/llaga_vicuna7b_GRACE_512_STv1_mlpv2_no_edgestr_trait_lora_v2")
    parser.add_argument("--model_base", type=str, default="/disk/NarGINA/weights/vicuna-7b-v1.5")
    parser.add_argument("--data_path", type=str, default="/disk/NarGINA/dataset/ChildText_test/teacher_2/embedings/GRACE_ST_t5uie-vicuna/pretrained_GAT_hidden=512_test.pkl")
    #parser.add_argument("--pretrained_embedding_type", type=str, default="GRACE_512")#ST_encoder,GRACE_512
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=10)
    parser.add_argument("--answers_file", type=str, default="/disk/NarGINA/output/ChildText/score_trait_v2/answer_childtext_vicuna_7b_mlpv2_GRACE_512_t5uie-vicuna_trait_lora")
    parser.add_argument("--conv_mode", type=str, default="conv_childtext") #conv_childtext_llama2
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--mm_use_graph_start_end",default=False, action="store_true")
    parser.add_argument("--task", type=str, default="graph_score")
    parser.add_argument("--dataset", type=str, default="ChildText")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--template", type=str, default=None)
    parser.add_argument("--is_only_graph", type=bool, default=False)#TODO
    parser.add_argument("--is_trait_comment", type=bool, default=False)#TODO
    parser.add_argument("--is_trait", type=bool, default=True)#TODO
    parser.add_argument("--is_precompute_micro_metric", type=bool, default=False)#TODO
    parser.add_argument("--is_edge_str", type=bool, default=False)#TODO
    args = parser.parse_args()

    is_eval_all_checkpoint = True
    if is_eval_all_checkpoint:
        checkpoint_files,answers_file_paths = get_checkpoint_file_paths(args.model_path,args.answers_file)
        for i in range(len(checkpoint_files)):
            args.model_path,args.answers_file = checkpoint_files[i],answers_file_paths[i]
            eval_model(args)

    else:
        eval_model(args)

