import os
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar

from OneForAll.gp.lightning.metric import EvalKit
from OneForAll.gp.utils.utils import dict_res_summary, load_pretrained_state
import torch

from sklearn.metrics import classification_report


def lightning_fit(
    logger,
    model,
    data_module,
    metrics: EvalKit,
    num_epochs,
    profiler=None,
    cktp_prefix="",
    load_best=False,
    prog_freq=20,
    test_rep=1,
    save_model=True,
    prog_bar=True,
    accelerator="auto",
    detect_anomaly=False,
    reload_freq=0,
    val_interval=1,
    strategy=None,
):
    callbacks = []
    if prog_bar:
        callbacks.append(TQDMProgressBar(refresh_rate=20))
    if save_model:
        callbacks.append(
            ModelCheckpoint(
                monitor=metrics.val_metric,
                mode=metrics.eval_mode,
                save_last=True,
                filename=cktp_prefix + "{epoch}-{step}",
            )
        )
    
    trainer = Trainer(
        accelerator=accelerator,
        strategy=strategy,
        # devices=1 if torch.cuda.is_available() else 10,
        max_epochs=num_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=prog_freq,
        profiler=profiler,
        enable_checkpointing=save_model,
        enable_progress_bar=prog_bar,
        detect_anomaly=detect_anomaly,
        reload_dataloaders_every_n_epochs=reload_freq,
        check_val_every_n_epoch=val_interval,
    )
    print('训练！')
    print(model)
    trainer.fit(model, datamodule=data_module)
    print('结束！')

    if load_best:
        model_dir = trainer.checkpoint_callback.best_model_path
        deep_speed = False
        if strategy[:9] == "deepspeed":
            deep_speed = True
        state_dict = load_pretrained_state(model_dir, deep_speed)
        model.load_state_dict(state_dict)


    val_col = []
    for i in range(test_rep):
        val_col.append(
            trainer.validate(model, datamodule=data_module, verbose=False)[0]
        )

    val_res = dict_res_summary(val_col)
    for met in val_res:
        val_mean = np.mean(val_res[met])
        val_std = np.std(val_res[met])
        print("{}:{:f}±{:f}".format(met, val_mean, val_std))

    target_val_mean = np.mean(val_res[metrics.val_metric])
    target_val_std = np.std(val_res[metrics.val_metric])

    test_col = []
    for i in range(test_rep):
        test_col.append(
            trainer.test(model, datamodule=data_module, verbose=False)[0]
        )

    test_res = dict_res_summary(test_col)
    for met in test_res:
        test_mean = np.mean(test_res[met])
        test_std = np.std(test_res[met])
        print("{}:{:f}±{:f}".format(met, test_mean, test_std))

    target_test_mean = np.mean(test_res[metrics.test_metric])
    target_test_std = np.std(test_res[metrics.test_metric])
    return [target_val_mean, target_val_std], [
        target_test_mean,
        target_test_std,
    ]


def lightning_test(
    logger,
    model,
    data_module,
    metrics: EvalKit,
    model_dir: str,
    strategy="auto",
    profiler=None,
    prog_freq=20,
    test_rep=1,
    prog_bar=True,
    accelerator="auto",
    detect_anomaly=False,
    deep_speed=False,
):
    callbacks = []
    if prog_bar:
        callbacks.append(TQDMProgressBar(refresh_rate=20))
    trainer = Trainer(
        accelerator=accelerator,
        strategy=strategy,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=prog_freq,
        profiler=profiler,
        enable_progress_bar=prog_bar,
        detect_anomaly=detect_anomaly,
    )
    
    state_dict = load_pretrained_state(model_dir, deep_speed)
    model.load_state_dict(state_dict)

    val_col = []
    for i in range(test_rep):
        val_col.append(
            trainer.validate(model, datamodule=data_module, verbose=False)[0]
        )

    val_res = dict_res_summary(val_col)
    for met in val_res:
        val_mean = np.mean(val_res[met])
        val_std = np.std(val_res[met])
        print("{}:{:f}±{:f}".format(met, val_mean, val_std))

    target_val_mean = np.mean(val_res[metrics.val_metric])
    target_val_std = np.std(val_res[metrics.val_metric])

    test_col = []
    for i in range(test_rep):
        test_col.append(
            trainer.test(model, datamodule=data_module, verbose=True)[0]
        )

    test_res = dict_res_summary(test_col)
    for met in test_res:
        test_mean = np.mean(test_res[met])
        test_std = np.std(test_res[met])
        print("{}:{:f}±{:f}".format(met, test_mean, test_std))

    target_test_mean = np.mean(test_res[metrics.test_metric])
    target_test_std = np.std(test_res[metrics.test_metric])

    return [target_val_mean, target_val_std], [
        target_test_mean,
        target_test_std,
    ]

def lightning_predict(
    logger,
    model,
    model_dir: str,
    strategy="auto",
    profiler=None,
    prog_freq=1,
    prog_bar=True,
    accelerator="auto",
    detect_anomaly=False,
    deep_speed=False,
    output_file="model_predictions.txt",
    num_classes = -1,
    datamodule = None,
    stroy_index = []
):
    callbacks = []
    if prog_bar:
        callbacks.append(TQDMProgressBar(refresh_rate=20))
    trainer = Trainer(
        devices=1,
        accelerator=accelerator,
        strategy=strategy,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=prog_freq,
        profiler=profiler,
        enable_progress_bar=prog_bar,
        detect_anomaly=detect_anomaly,
    )

    # predict_dataloader = datamodule.predict_dataloader()[0]
    
    state_dict = load_pretrained_state(model_dir, deep_speed)
    model.load_state_dict(state_dict)

    label_dict ={0:"动机-因果",1:"心理-因果",2:"物理-因果",3:"使能-因果",4:"并列",5:"无"}

    with open(output_file, 'w',encoding='utf-8') as f:

        # 进行模型的预测，使用 trainer.predict 获取输出

        # result = trainer.predict(model,datamodule=datamodule)
        
        result = trainer.predict(model,dataloaders=datamodule.predict_dataloader())
    
        # correct = 0
        # correct_no_negative = 0
        # correct_negative=0

        # all = 0
        # all_no_negative = 0
        # all_negative = 0


        count = 0
        count_index = 0

        # # 初始化每个类别的统计量
        # tp = [0] * num_classes  # True Positives
        # fp = [0] * num_classes  # False Positives
        # fn = [0] * num_classes  # False Negatives
        # tn = [0] * num_classes  # True Negatives

        p=[]
        l=[]
        # 对每次的预测输出进行处理并保存到文件中
        for i, batch_result in enumerate(result):

            batch_preds = batch_result['predictions']
            batch_label = batch_result['batch']

            batch_preds = batch_preds.view(-1, num_classes)
            batch_preds = torch.nn.functional.softmax(batch_preds,dim=-1)

            for j, pred in enumerate(batch_preds):

                # if count == i * batch_preds.shape[0]+j and batch_preds.shape[0] == datamodule.datasets['test'][0].batch_size:
                #     count += stroy_index[count_index]
                #     count_index +=1
                #     f.write(f"story {count_index} relations: \n")

                predicted_class = torch.argmax(pred).item()
                actual_label = batch_label[j].item()

                p.append(predicted_class)
                l.append(actual_label)

                # all +=1
                # if predicted_class == actual_label:
                #     correct+=1
                
                # if actual_label != 5:
                #     all_no_negative+=1
                #     if predicted_class == actual_label:
                #         correct_no_negative+=1
                # else:
                #     all_negative+=1
                #     if predicted_class == actual_label:
                #         correct_negative+=1

                # # 分类统计
                # for c in range(num_classes):
                #     if predicted_class == c and actual_label == c:
                #         tp[c] += 1  # True Positive
                #     elif predicted_class == c and actual_label != c:
                #         fp[c] += 1  # False Positive
                #     elif predicted_class != c and actual_label == c:
                #         fn[c] += 1  # False Negative
                #     elif predicted_class != c and actual_label != c:
                #         tn[c] += 1  # True Negative

                f.write(f"Sample {j} predicted class: {label_dict[predicted_class]}, actual label: {label_dict[actual_label]}\n")
            
        # # 计算每个类别的 Precision, Recall 和 F1-score
        # precision = [0] * num_classes
        # recall = [0] * num_classes
        # f1_score = [0] * num_classes

        # for c in range(num_classes):
        #     if tp[c] + fp[c] > 0:
        #         precision[c] = tp[c] / (tp[c] + fp[c])
        #     if tp[c] + fn[c] > 0:
        #         recall[c] = tp[c] / (tp[c] + fn[c])
        #     if precision[c] + recall[c] > 0:
        #         f1_score[c] = 2 * (precision[c] * recall[c]) / (precision[c] + recall[c])

        # # 计算整体的 F1-score
        # macro_f1 = sum(f1_score) / num_classes
        # print(f"整体 F1-score: {macro_f1:.4f}")

        # # 计算无负样本的 F1-score
        # no_negative_f1 = sum(f1_score[c] for c in range(num_classes) if c != 5) / (num_classes - 1)
        # print(f"无负样本 F1-score: {no_negative_f1:.4f}")
        
        # for c in range(num_classes):
        #     print(f"Class {c} - Precision: {precision[c]:.4f}, Recall: {recall[c]:.4f}, F1-score: {f1_score[c]:.4f}")
        report = classification_report(l,p)
        print(report)
        # acc = correct/all
        # acc_no_negative = correct_no_negative/all_no_negative
        # acc_negative = correct_negative/all_negative
        # print(f"正确率: {acc}")
        # print(f"无负样本正确率: {acc_no_negative}")
        # print(f"负样本正确率: {acc_negative}")

    

