import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from data.data_loader import load_ft
from model.HGCN import HGCN
from model.TMO import TMO
from utils.hypergraph_utils import LearnableHypergraph
from visualization import TrainingVisualizer


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(data_list, learnable_hg_list, label, model, optimizer, scheduler_dict,
                epoch, idx_tr, visualizer, hg_update_interval, cached_g_list):
    model.train()
    loss_meter = AverageMeter()

    if epoch % hg_update_interval == 0:
        g_list = []
        for i, hg_module in enumerate(learnable_hg_list):
            G = hg_module(data_list[i])
            g_list.append(G)
        cached_g_list[:] = [g.detach() for g in g_list]
    else:
        g_list = cached_g_list

    optimizer.zero_grad()
    if len(data_list) >= 2:
        evidence_a, u_a, loss = model(data_list, g_list, label, epoch, idx_tr)
    else:
        ci = model(data_list[0], g_list[0])
        loss = torch.mean(F.cross_entropy(ci[idx_tr], label[idx_tr]))
        u_a = None

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(learnable_hg_list.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler_dict.step()
    loss_meter.update(loss.item())

    if visualizer is not None:
        visualizer.update(epoch, train_loss=loss.item())

    return loss.item()


def test_epoch(data_list, learnable_hg_list, label, te_idx, model, epoch, cached_g_list):
    model.eval()
    with torch.no_grad():
        g_list = cached_g_list
        if len(data_list) >= 2:
            evidence_a, u_a, loss = model(data_list, g_list, label, epoch, list(range(len(label))))
        else:
            evidence_a = model(data_list[0], g_list[0])
            u_a = None
            loss = 0.0

    prob = F.softmax(evidence_a[te_idx], dim=1).data.cpu().numpy()
    u = u_a[te_idx].cpu().numpy() if u_a is not None else None
    return prob, u, loss


def train_fold(data_tensor_list, labels_tensor, model, learnable_hg_list, optimizer, scheduler_dict,
               num_epochs, idx_dict, num_class, test_interval, visualizer, fold_idx,
               hg_update_interval, patience=20):
    fold_history = {
        'epochs': [],
        'val_acc': [],
        'val_loss': [],
        'train_loss': [],
        'f1_scores': [],
        'auc_scores': [],
        'best_acc_sofar': [],
        'uncertainties': [],
        'correctness': [],
        'best_pred_labels': None,
        'best_true_labels': None
    }

    best_acc = 0.0
    best_f1 = 0.0
    best_auc = 0.0
    best_model_state = None
    best_hg_state = None
    wait = 0
    best_pred_labels = None
    best_true_labels = None

    cached_g_list = [None] * len(learnable_hg_list)

    for epoch in range(num_epochs + 1):
        train_loss = train_epoch(
            data_tensor_list, learnable_hg_list, labels_tensor,
            model, optimizer, scheduler_dict, epoch,
            idx_tr=idx_dict["tr"], visualizer=visualizer,
            hg_update_interval=hg_update_interval,
            cached_g_list=cached_g_list
        )

        if epoch % test_interval == 0:
            te_prob, te_uncertainty, val_loss = test_epoch(
                data_tensor_list, learnable_hg_list, labels_tensor,
                idx_dict["te"], model, epoch, cached_g_list
            )

            val_acc = accuracy_score(labels_tensor[idx_dict["te"]].cpu(), te_prob.argmax(1))
            val_f1 = f1_score(labels_tensor[idx_dict["te"]].cpu(), te_prob.argmax(1), average='weighted')
            if num_class == 2:
                val_auc = roc_auc_score(labels_tensor[idx_dict["te"]].cpu(), te_prob[:, 1])
            else:
                val_auc = f1_score(labels_tensor[idx_dict["te"]].cpu(), te_prob.argmax(1), average='macro')

            # 记录不确定性（数组）和正确性
            if te_uncertainty is not None:
                fold_history['uncertainties'].append(te_uncertainty)
                correct = (labels_tensor[idx_dict["te"]].cpu().numpy() == te_prob.argmax(1)).astype(int)
                fold_history['correctness'].append(correct)
            else:
                fold_history['uncertainties'].append(np.array([]))
                fold_history['correctness'].append(np.array([]))

            if val_acc > best_acc:
                best_acc = val_acc
                best_f1 = val_f1
                best_auc = val_auc
                wait = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_hg_state = {k: v.cpu().clone() for k, v in learnable_hg_list.state_dict().items()}
                best_pred_labels = te_prob.argmax(1).copy()
                best_true_labels = labels_tensor[idx_dict["te"]].cpu().numpy().copy()
                print(f"-！！！- New best accuracy: {best_acc:.4f} at epoch {epoch} (F1: {best_f1:.4f}, AUC: {best_auc:.4f})")
            else:
                wait += 1
                if wait >= patience:
                    print(f"\n⏹️ Early stopping triggered at epoch {epoch} (best acc: {best_acc:.4f})")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                        learnable_hg_list.load_state_dict(best_hg_state)
                        device = next(model.parameters()).device
                        model.to(device)
                        learnable_hg_list.to(device)
                    break

            fold_history['epochs'].append(epoch)
            fold_history['val_acc'].append(val_acc)
            fold_history['val_loss'].append(val_loss)
            fold_history['train_loss'].append(train_loss)
            fold_history['f1_scores'].append(val_f1)
            fold_history['auc_scores'].append(val_auc)
            fold_history['best_acc_sofar'].append(best_acc)

            if epoch % (test_interval * 500) == 0 or epoch == 0:
                mean_unc = np.mean(te_uncertainty) if te_uncertainty is not None else 0.0
                print(f"\nFold {fold_idx + 1} - Epoch {epoch}/{num_epochs}")
                print(f"  Val ACC: {val_acc:.4f}, Best ACC so far: {best_acc:.4f}, F1: {val_f1:.4f}, {'AUC' if num_class == 2 else 'F1_macro'}: {val_auc:.4f}")
                print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Mean Uncertainty: {mean_unc:.4f}")

    fold_history['best_pred_labels'] = best_pred_labels
    fold_history['best_true_labels'] = best_true_labels

    if visualizer is not None:
        visualizer.record_fold(fold_idx, fold_history)
    return best_acc, best_f1, best_auc


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', '-fd', type=str, required=True, help='数据集文件夹路径')
    parser.add_argument('--seed', '-s', type=int, default=20, help='随机种子')
    parser.add_argument('--num_epoch', '-ne', type=int, default=40000, help='训练轮数')
    parser.add_argument('--lr_e', '-lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--dim_he_list', '-dh', nargs='+', type=int, default=[400,200,200], help='HGCN隐藏层维度')
    parser.add_argument('--num_class', '-nc', type=int, required=True, help='类别数')
    parser.add_argument('--k_sparse', '-ks', type=int, default=20, help='稀疏注意力中每个节点的邻居数')
    parser.add_argument('--hg_update_interval', '-hui', type=int, default=10, help='超图更新间隔（epoch）')
    parser.add_argument('--test_interval', '-ti', type=int, default=50, help='测试间隔')
    parser.add_argument('--plot_freq', '-pf', type=int, default=10, help='绘图频率')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值（连续未提升准确率的验证周期数）')
    parser.add_argument('--attn_proj_dim', type=int, default=None,
                        help='Attention projection dimension (default: same as input dim)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    visualizer = TrainingVisualizer(
        save_dir=f'./visualization_{args.file_dir}',
        lr=args.lr_e,
        k_sparse=args.k_sparse,
        hg_update_interval=args.hg_update_interval
    )

    data_folder = 'data'
    omics_list = ['miRNA', 'meth', 'mRNA']
    num_omics = len(omics_list)

    data_tensor_list, labels_tensor = load_ft(data_folder, omics_list, args.file_dir)
    data_tensor_list = [x.to(device) for x in data_tensor_list]
    labels_tensor = labels_tensor.to(device)

    dim_list = [x.shape[1] for x in data_tensor_list]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    acc_res, F1_res, AUC_res = [], [], []

    print("=" * 60)
    print("Starting 5-fold Cross Validation with Sparse Learnable Hypergraph")
    print("=" * 60)

    fold_idx = 0
    for idx_train, idx_test in skf.split(pd.DataFrame(data=data_tensor_list[0].cpu()),
                                         pd.DataFrame(labels_tensor.cpu())):
        fold_idx += 1
        print(f"\n{'=' * 40}")
        print(f"Training Fold {fold_idx}/5")
        print(f"{'=' * 40}")

        idx_dict = {"tr": idx_train, "te": idx_test}

        if num_omics >= 2:
            model = TMO(dim_list, args.num_class, num_omics, args.dim_he_list)
        else:
            model = HGCN(dim_list[0], args.num_class, args.dim_he_list)
        model = model.to(device)

        learnable_hg_list = nn.ModuleList()
        for i in range(num_omics):
            learnable_hg_list.append(
                LearnableHypergraph(in_features=dim_list[i],
                                    proj_dim=args.attn_proj_dim,
                                    k=args.k_sparse,
                                    num_heads=4,
                                    dropout=0.2)
            )
        learnable_hg_list = learnable_hg_list.to(device)

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(learnable_hg_list.parameters()),
            lr=args.lr_e, weight_decay=0.0005
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.9)

        best_acc, best_f1, best_auc = train_fold(
            data_tensor_list, labels_tensor, model, learnable_hg_list, optimizer, scheduler,
            args.num_epoch, idx_dict, args.num_class, args.test_interval,
            visualizer, fold_idx - 1, args.hg_update_interval, patience=args.patience
        )

        acc_res.append(best_acc)
        F1_res.append(best_f1)
        AUC_res.append(best_auc)

        print(f"\nFold {fold_idx} Results:")
        print(f"  Best ACC: {best_acc:.4f}")
        print(f"  Best F1: {best_f1:.4f}")
        print(f"  Best {'AUC' if args.num_class == 2 else 'F1_macro'}: {best_auc:.4f}")

        if fold_idx % args.plot_freq == 0 or fold_idx == 5:
            visualizer.plot_accuracy_curve(fold_idx=fold_idx - 1, show=False)

    print("\n" + "=" * 60)
    print("5-fold Cross Validation Results:")
    print("=" * 60)
    print('Acc(%.4f ± %.4f)  F1(%.4f ± %.4f)  AUC/F1_mac(%.4f ± %.4f)'
          % (float(np.mean(acc_res)), float(np.std(acc_res)),
             float(np.mean(F1_res)), float(np.std(F1_res)),
             float(np.mean(AUC_res)), float(np.std(AUC_res))))

    print("\nGenerating visualizations...")
    visualizer.plot_accuracy_curve(save=True, show=True)
    visualizer.plot_f1_curve(save=True, show=True)
    if args.num_class == 2:
        visualizer.plot_auc_curve(save=True, show=True)
    else:
        visualizer.plot_macro_f1_curve(save=True, show=True)
    visualizer.plot_uncertainty_curve(save=True, show=True)
    visualizer.plot_reliability_diagram(save=True, show=True)
    visualizer.plot_accuracy_vs_uncertainty(save=True, show=True)
    # visualizer.plot_loss_curve(save=True, show=True)   # 已取消
    visualizer.plot_confusion_matrix(save=True, show=True)
    visualizer.plot_final_metrics(acc_res, F1_res, AUC_res, save=True, show=True)
    visualizer.save_history_to_csv()

    print("\nTraining completed!")
    print(f"Visualizations saved to: {visualizer.save_dir}")