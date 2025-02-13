import pickle
import re
import sys
import os
from sklearn.metrics import cohen_kappa_score
from termcolor import colored
from sklearn.metrics import precision_score, recall_score, f1_score
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,6,7"
sys.path.append("./")
sys.path.append("./utils")
sys.path.append("./train")
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from eval.ChildText.eval_pretrain_ChildText import get_checkpoint_file_paths
from train.train_asap import extract_prediction_trait_score
from train.train_childtext_onlytext import SupervisedDataset
from ChildText.prompt import PROMPT_TEMPLATE
from train.train_asap_onlytext import   extract_prediction_score
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
from accelerate import Accelerator


def eval_model(args):
    # Model
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"Loaded from {model_path}. Model Base: {args.model_base}")
    tokenizer, model, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                          cache_dir=args.cache_dir)
    print("加载成功")
    tokenizer.pad_token_id = 0
    #tokenizer.padding_side = "left"
    accelerator = Accelerator()

    #model = model.to(torch.float16).cuda()
    model = accelerator.prepare(model)
    #print(model)
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

        outputs = ""
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,#[1,1832]
                #do_sample=True,
                #temperature=args.temperature,
                #top_p=args.top_p,
                #num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=512,
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
    # 过滤和计算 Precision, Recall, F1
    metrics = compute_precision_recall_f1(preds, labels)

    # 保存结果文件名附带指标值
    precision_w = metrics["precision_weighted"]
    recall_w = metrics["recall_weighted"]
    f1_w = metrics["f1_weighted"]
    result_file = (os.path.splitext(args.answers_file)[0] +
                   f"-P={precision_w:.3f}-R={recall_w:.3f}-F1={f1_w:.3f}.json")


    print(f"结果保存到 {result_file}")
    
    # print("保存到：",new_name)
    os.rename(args.answers_file,result_file)
    del model
    torch.cuda.empty_cache()  # 清理GPU缓存
    

def compute_precision_recall_f1(pred_str, label_str):
    """
    计算 Precision、Recall 和 F1 分数。

    Args:
        pred_str (list): 模型输出的文本列表。
        label_str (list): 真实标签的文本列表。

    Returns:
        dict: 包含每个类别的 P, R, F1 以及整体加权平均值。
    """
    # 预定义的类别
    predefined_classes = ["并列", "动机-因果", "心理-因果", "物理-因果", "使能-因果", "无"]
    class_to_id = {cls: idx for idx, cls in enumerate(predefined_classes)}

    # 过滤并映射
    y_true, y_pred = [], []
    for pred, label in zip(pred_str, label_str):
        if pred in class_to_id and label in class_to_id:
            y_true.append(class_to_id[label])
            y_pred.append(class_to_id[pred])

    if not y_true or not y_pred:
        print("没有有效数据用于计算指标。")
        return {}

    # 计算 Precision、Recall 和 F1
    results = {
        "precision_per_class": precision_score(y_true, y_pred, average=None, labels=list(range(len(predefined_classes)))),
        "recall_per_class": recall_score(y_true, y_pred, average=None, labels=list(range(len(predefined_classes)))),
        "f1_per_class": f1_score(y_true, y_pred, average=None, labels=list(range(len(predefined_classes)))),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }

    # 打印结果
    for i, cls in enumerate(predefined_classes):
        print(f"{cls}: Precision={results['precision_per_class'][i]:.3f}, "
              f"Recall={results['recall_per_class'][i]:.3f}, "
              f"F1={results['f1_per_class'][i]:.3f}")

    print(f"加权平均: Precision={results['precision_weighted']:.3f}, "
          f"Recall={results['recall_weighted']:.3f}, "
          f"F1={results['f1_weighted']:.3f}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/disk/NarGINA/checkpoints/relation_preds/vicuna_7b")
    parser.add_argument("--model_base", type=str, default="/disk/NarGINA/weights/vicuna-7b-v1.5")
    # parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="/disk/NarGINA/graph_construction_model/ft-vicuna-relation_extraction/dataset/xlw_test_data_2.json")    
    parser.add_argument("--pretrained_embedding_type", type=str, default="attn_mlp")
    #parser.add_argument("--use_hop", type=int, default=2)
    #parser.add_argument("--sample_neighbor_size", type=int, default=10)
    parser.add_argument("--answers_file", type=str, default="/disk/NarGINA/output/relation_preds/vicuna_test_xlw_2.json")
    parser.add_argument("--conv_mode", type=str, default="conv_edge_pred")
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
    parser.add_argument("--is_trait", type=bool, default=False)#TODO
    parser.add_argument("--is_precompute_micro_metric", type=bool, default=False)#TODO
    args = parser.parse_args()

    is_eval_all_checkpoint = False
    if is_eval_all_checkpoint:
        checkpoint_files,answers_file_paths = get_checkpoint_file_paths(args.model_path,args.answers_file)
        for i in range(len(checkpoint_files)):
            args.model_path,args.answers_file = checkpoint_files[i],answers_file_paths[i]
            # try:
            #     eval_model(args)
            # except Exception as e :
            #     print(e)
            eval_model(args)

    else:
        eval_model(args)