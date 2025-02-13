import pickle
import re
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append("./")
sys.path.append("./utils")
sys.path.append("./train")
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from sklearn.metrics import cohen_kappa_score
from termcolor import colored

from eval.ChildText.eval_pretrain_ChildText import get_checkpoint_file_paths
from train.train_childtext import  extract_prediction_score
from train.train_childtext_onlytext import SupervisedDataset, extract_prediction_trait_score
from ChildText.prompt import PROMPT_TEMPLATE
from torch_geometric.nn.models import GAT
import argparse
import torch
torch.autograd.set_detect_anomaly(True)
import os
import json
from tqdm import tqdm
import shortuuid
from utils import conversation as conversation_lib
from utils.constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN
from utils.conversation import conv_templates, SeparatorStyle
from model.builder import load_pretrained_model
from utils.utils import disable_torch_init, tokenizer_graph_token, get_model_name_from_path
from torch_geometric.utils import k_hop_subgraph, degree, remove_self_loops, add_self_loops


def eval_model(args):
    # Model
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"Loaded from {model_path}. Model Base: {args.model_base}")
    tokenizer, model, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                          cache_dir=args.cache_dir)
    tokenizer.pad_token_id = 0
    #tokenizer.padding_side = "left"
    dtype = torch.bfloat16
    model = model.to(dtype)
    model = model.cuda()
    # print(model)
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode]

    test_data = json.load(open(args.data_path, "r"))
    dataset = SupervisedDataset(test_data,tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w",encoding='utf-8')

    preds=[]
    labels=[]
    conv = conv_templates[args.conv_mode].copy()
    for id, data in enumerate(tqdm(dataloader)):
        input_ids,label,attention_mask=data['input_ids'].cuda(),data['labels'].cuda(),data['attention_mask'].cuda()
        # 推理时不给标签
        input_str = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
        if "llama" in model_name:
            input_str = input_str[:input_str.find("[/INST]") + len("[/INST]")]
        elif  "vicuna" in model_name:
            input_str = input_str[:input_str.find("ASSISTANT:") + len("ASSISTANT:")]

        input_ids = tokenizer(
            input_str,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids.cuda()
        
        # labels.append(int(label[-3]))
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,#[1,1832]
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)
        
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

        except Exception as e:
            print(f"!!!!!!Error!!!!! {e}")
            outputs=""
        #print(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": id,
                                   "prediction": outputs,
                                   "gt":test_data[id]['conversations'][1]['value'],
                                   "prompt": test_data[id]['conversations'][0]['value'],
                                   "answer_id": ans_id},ensure_ascii=False) + "\n")
        ans_file.flush()



        preds.append(outputs)
        labels.append(test_data[id]['conversations'][1]['value'])

    ans_file.close()
    assert len(preds) == len(labels), "预测和标签的长度不匹配"
    if args.is_trait_comment or args.is_trait:

        qwk = compute_trait_qwk( preds,labels)
        # 打印 QWK 分数
        print(colored(f"Model:{model_path}=========QWK-macro:{qwk['宏观结构得分']:.3f};QWK-micro:{qwk['微观结构得分']:.3f};QWK-psych:{qwk['叙事心理描写得分']:.3f};QWK-total:{qwk['总分']:.3f};",'light_green'))
        QWK_macro,QWK_micro,QWK_psych,QWK_total = qwk['宏观结构得分'],qwk['微观结构得分'],qwk['叙事心理描写得分'],qwk['总分']
        QWK_avg = (QWK_macro+QWK_micro+QWK_psych+QWK_total) / 4
        new_name = os.path.splitext(args.answers_file)[0]+ f'-QWK_macro={QWK_macro:.3f}-QWK_micro={QWK_micro:.3f}-QWK_psych={QWK_psych:.3f}-QWK_total={QWK_total:.3f}-QWK_avg={QWK_avg:.3f}' + os.path.splitext(args.answers_file)[1]
        os.rename(args.answers_file,new_name)
    else:
        qwk = compute_qwk( preds,labels)
    del model
    torch.cuda.empty_cache()  # 清理GPU缓存


def compute_qwk(pred_str,label_str):
    # eval_pred:input_id=none;label_ids=array(val-dataset-num,4096) = label;predictions=(val-dataset-num,3496,32000)=logits
    # 从 pred_str 提取 "<>" 之间的整数，并丢弃 preds_score 中为 None 的项及对应的 label
    filtered_results = [(extract_prediction_score(p), extract_prediction_score(l)) for p, l in zip(pred_str, label_str) if extract_prediction_score(p) is not None]
    preds_score, y = zip(*filtered_results) if filtered_results else ([], [])

    # 如果 preds_score 和 y 都不为空，计算 QWK
    if preds_score and y:
        qwk_score = cohen_kappa_score(preds_score, y,weights='quadratic')
        print("Quadratic Weighted Kappa (QWK):", qwk_score)
        return {"QWK": qwk_score}
    else:
        print("No valid data to compute QWK.")
        return {"QWK": 0.0}


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
    parser.add_argument("--model_path", type=str, default="/disk/NarGINA/checkpoints/ChildText/score_trait/llama2_7b-nograph-trait-lora")
    parser.add_argument("--model_base", type=str, default="/disk/NarGINA/weights/Llama-2-7b-chat-hf")
    parser.add_argument("--data_path", type=str, default="/disk/NarGINA/dataset/ChildText_onlytext/score_trait/trait_test_data.json")
    # parser.add_argument("--data_dir", type=str, default=None)
    #parser.add_argument("--pretrained_embedding_type", type=str, default="attn_mlp")
    #parser.add_argument("--use_hop", type=int, default=2)
    #parser.add_argument("--sample_neighbor_size", type=int, default=10)
    parser.add_argument("--answers_file", type=str, default="/disk/NarGINA/output/ChildText/score_trait/answer_childtext_llama2_7b_trait_lora")
    parser.add_argument("--conv_mode", type=str, default="conv_childtext_llama2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    # parser.add_argument("--prompt", type=str, default=None)
    # parser.add_argument("--start", type=int, default=-1)
    # parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--test_path", type=str, default=None)
    # parser.add_argument("--mm_use_graph_start_end",default=False, action="store_true")
    #parser.add_argument("--task", type=str, default="graph_score")
    #parser.add_argument("--dataset", type=str, default="ChildText")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--template", type=str, default=None)
    parser.add_argument("--is_only_graph", type=bool, default=False)#TODO
    parser.add_argument("--is_trait_comment", type=bool, default=False)#TODO
    parser.add_argument("--is_trait", type=bool, default=True)#TODO
    parser.add_argument("--is_precompute_micro_metric", type=bool, default=False)#TODO
    args = parser.parse_args()

    is_eval_all_checkpoint = True
    if is_eval_all_checkpoint:
        checkpoint_files,answers_file_paths = get_checkpoint_file_paths(args.model_path,args.answers_file)
        for i in range(len(checkpoint_files)):
            args.model_path,args.answers_file = checkpoint_files[i],answers_file_paths[i]
            eval_model(args)

    else:
        eval_model(args)
