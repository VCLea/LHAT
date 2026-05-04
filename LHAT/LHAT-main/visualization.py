import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix


class TrainingVisualizer:
    def __init__(self, save_dir='./visualization', lr=None, k_sparse=None, hg_update_interval=None):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.lr = lr
        self.k_sparse = k_sparse
        self.hg_update_interval = hg_update_interval
        self.history = {
            'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [],
            'f1_scores': [], 'auc_scores': [], 'epochs': []
        }
        self.fold_histories = []

    def _get_title_suffix(self):
        parts = []
        if self.lr is not None:
            parts.append(f"lr={self.lr}")
        if self.k_sparse is not None:
            parts.append(f"k={self.k_sparse}")
        if self.hg_update_interval is not None:
            parts.append(f"update_interval={self.hg_update_interval}")
        return " (" + ", ".join(parts) + ")" if parts else ""

    def record_fold(self, fold_idx, history):
        self.fold_histories.append({
            'fold': fold_idx,
            'history': history.copy(),
            'timestamps': [datetime.now()] * len(history['epochs'])
        })

    def update(self, epoch, train_acc=None, val_acc=None, train_loss=None, val_loss=None,
               f1=None, auc=None):
        self.history['epochs'].append(epoch)
        if train_acc is not None:
            self.history['train_acc'].append(train_acc)
        if val_acc is not None:
            self.history['val_acc'].append(val_acc)
        if train_loss is not None:
            self.history['train_loss'].append(train_loss)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if f1 is not None:
            self.history['f1_scores'].append(f1)
        if auc is not None:
            self.history['auc_scores'].append(auc)

    def _sample_curve(self, epochs, values, bin_size=100):
        if not epochs:
            return [], []
        max_epoch = max(epochs)
        sampled_epochs, sampled_values = [], []
        for start in range(0, max_epoch + bin_size, bin_size):
            end = start + bin_size
            indices = [i for i, e in enumerate(epochs) if start <= e < end]
            if indices:
                max_val = max(values[i] for i in indices)
                sampled_epochs.append(start + bin_size/2)
                sampled_values.append(max_val)
        return sampled_epochs, sampled_values

    # ------------------- 性能曲线 -------------------
    def plot_accuracy_curve(self, fold_idx=None, save=True, show=True):
        plt.figure(figsize=(12, 8))
        suffix = self._get_title_suffix()
        bin_size = 100
        color = 'blue'
        if fold_idx is not None and self.fold_histories:
            fd = self.fold_histories[fold_idx]['history']
            e = fd['epochs']
            best_acc = fd['best_acc_sofar']
            se, sv = self._sample_curve(e, best_acc, bin_size)
            plt.plot(se, sv, color=color, linewidth=2, label=f'Fold {fold_idx+1} Best Accuracy')
        else:
            if not self.fold_histories:
                return
            all_epochs = sorted(set().union(*[set(f['history']['epochs']) for f in self.fold_histories]))
            max_epoch = max(all_epochs)
            bin_starts = list(range(0, max_epoch+bin_size, bin_size))
            common_epochs = [s + bin_size/2 for s in bin_starts]
            aligned = []
            for fold in self.fold_histories:
                ep = fold['history']['epochs']
                acc = fold['history']['best_acc_sofar']
                d = dict(zip(ep, acc))
                bin_vals = []
                for s in bin_starts:
                    e_end = s + bin_size
                    vals = [d[e] for e in ep if s <= e < e_end]
                    bin_vals.append(max(vals) if vals else (bin_vals[-1] if bin_vals else 0.0))
                aligned.append(bin_vals)
            mean_acc = np.mean(aligned, axis=0)
            std_acc = np.std(aligned, axis=0)
            plt.plot(common_epochs, mean_acc, color=color, linewidth=2, label='Mean Best Accuracy')
            plt.fill_between(common_epochs, mean_acc - std_acc, mean_acc + std_acc, color=color, alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel('Best Accuracy (Cumulative Max)')
        plt.title(f'Best Accuracy Curve{suffix}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0.65, 0.9)
        if save:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            param = f"_lr{self.lr}_k{self.k_sparse}_int{self.hg_update_interval}" if self.lr else ""
            fname = f"accuracy_curve_fold{fold_idx if fold_idx is not None else 'all'}{param}_{ts}.png"
            plt.savefig(os.path.join(self.save_dir, fname), dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def plot_f1_curve(self, fold_idx=None, save=True, show=True):
        self._plot_metric_curve('f1_scores', 'Weighted F1 Score', fold_idx, save, show, color='cyan')

    def plot_auc_curve(self, fold_idx=None, save=True, show=True):
        self._plot_metric_curve('auc_scores', 'AUC', fold_idx, save, show, color='green')

    def plot_macro_f1_curve(self, fold_idx=None, save=True, show=True):
        self._plot_metric_curve('auc_scores', 'Macro F1', fold_idx, save, show, color='purple')

    def _plot_metric_curve(self, metric_key, ylabel, fold_idx, save, show, color):
        plt.figure(figsize=(12, 8))
        suffix = self._get_title_suffix()
        bin_size = 100
        if fold_idx is not None and self.fold_histories:
            fd = self.fold_histories[fold_idx]['history']
            e = fd['epochs']
            raw_vals = fd[metric_key]
            best_vals = np.maximum.accumulate(raw_vals)
            se, sv = self._sample_curve(e, best_vals, bin_size)
            plt.plot(se, sv, color=color, linewidth=2, label=f'Fold {fold_idx+1} Best {ylabel}')
        else:
            if not self.fold_histories:
                return
            all_epochs = sorted(set().union(*[set(f['history']['epochs']) for f in self.fold_histories]))
            max_epoch = max(all_epochs)
            bin_starts = list(range(0, max_epoch+bin_size, bin_size))
            common_epochs = [s + bin_size/2 for s in bin_starts]
            aligned = []
            for fold in self.fold_histories:
                ep = fold['history']['epochs']
                raw_vals = fold['history'][metric_key]
                best_vals = np.maximum.accumulate(raw_vals)
                d = dict(zip(ep, best_vals))
                bin_vals = []
                for s in bin_starts:
                    e_end = s + bin_size
                    vals = [d[e] for e in ep if s <= e < e_end]
                    bin_vals.append(max(vals) if vals else (bin_vals[-1] if bin_vals else 0.0))
                aligned.append(bin_vals)
            mean_vals = np.mean(aligned, axis=0)
            std_vals = np.std(aligned, axis=0)
            plt.plot(common_epochs, mean_vals, color=color, linewidth=2, label=f'Mean Best {ylabel}')
            plt.fill_between(common_epochs, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel(f'Best {ylabel} (Cumulative Max)')
        plt.title(f'{ylabel} Curve{suffix}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0.65, 0.9)
        if save:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            param = f"_lr{self.lr}_k{self.k_sparse}_int{self.hg_update_interval}" if self.lr else ""
            fname = f"{metric_key}_curve_fold{fold_idx if fold_idx is not None else 'all'}{param}_{ts}.png"
            plt.savefig(os.path.join(self.save_dir, fname), dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def plot_loss_curve(self, fold_idx=None, save=True, show=True):
        plt.figure(figsize=(12, 8))
        suffix = self._get_title_suffix()
        bin_size = 100
        if fold_idx is not None and self.fold_histories:
            fd = self.fold_histories[fold_idx]['history']
            e = fd['epochs']
            v = fd['val_loss']
            se, sv = self._sample_curve(e, v, bin_size)
            plt.plot(se, sv, 'b-', linewidth=2, label=f'Fold {fold_idx+1} Validation Loss')
        else:
            if not self.fold_histories:
                return
            all_epochs = sorted(set().union(*[set(f['history']['epochs']) for f in self.fold_histories]))
            max_epoch = max(all_epochs)
            bin_starts = list(range(0, max_epoch+bin_size, bin_size))
            common_epochs = [s + bin_size/2 for s in bin_starts]
            aligned = []
            for fold in self.fold_histories:
                ep = fold['history']['epochs']
                loss = fold['history']['val_loss']
                d = dict(zip(ep, loss))
                bin_vals = []
                for s in bin_starts:
                    e_end = s + bin_size
                    cur = [d[e] for e in ep if s <= e < e_end]
                    bin_vals.append(min(cur) if cur else (bin_vals[-1] if bin_vals else 0.0))
                aligned.append(bin_vals)
            mean_loss = np.mean(aligned, axis=0)
            std_loss = np.std(aligned, axis=0)
            plt.plot(common_epochs, mean_loss, 'b-', linewidth=2, label='Mean Validation Loss')
            plt.fill_between(common_epochs, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Validation Loss Curve{suffix}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.yscale('log')
        if save:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            param = f"_lr{self.lr}_k{self.k_sparse}_int{self.hg_update_interval}" if self.lr else ""
            fname = f"loss_curve_fold{fold_idx if fold_idx is not None else 'all'}{param}_{ts}.png"
            plt.savefig(os.path.join(self.save_dir, fname), dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    # ------------------- 不确定性相关 -------------------
    def plot_uncertainty_curve(self, fold_idx=None, save=True, show=True):
        plt.figure(figsize=(12, 8))
        suffix = self._get_title_suffix()
        bin_size = 100
        color = 'gray'
        if fold_idx is not None and self.fold_histories:
            fd = self.fold_histories[fold_idx]['history']
            e = fd['epochs']
            u_list = fd['uncertainties']
            mean_u = [np.mean(arr) for arr in u_list]
            se, sv = self._sample_curve(e, mean_u, bin_size)
            plt.plot(se, sv, color=color, linewidth=2, label=f'Fold {fold_idx+1} Mean Uncertainty')
        else:
            if not self.fold_histories:
                return
            all_epochs = sorted(set().union(*[set(f['history']['epochs']) for f in self.fold_histories]))
            max_epoch = max(all_epochs)
            bin_starts = list(range(0, max_epoch+bin_size, bin_size))
            common_epochs = [s + bin_size/2 for s in bin_starts]
            aligned = []
            for fold in self.fold_histories:
                ep = fold['history']['epochs']
                u_list = fold['history']['uncertainties']
                mu = [np.mean(arr) for arr in u_list]
                d = dict(zip(ep, mu))
                bin_vals = []
                for s in bin_starts:
                    e_end = s + bin_size
                    cur = [d[e] for e in ep if s <= e < e_end]
                    bin_vals.append(np.mean(cur) if cur else (bin_vals[-1] if bin_vals else 0.0))
                aligned.append(bin_vals)
            mean_unc = np.mean(aligned, axis=0)
            std_unc = np.std(aligned, axis=0)
            plt.plot(common_epochs, mean_unc, color=color, linewidth=2, label='Mean Uncertainty')
            plt.fill_between(common_epochs, mean_unc - std_unc, mean_unc + std_unc, color=color, alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Uncertainty')
        plt.title(f'Uncertainty Curve{suffix}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        if save:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            param = f"_lr{self.lr}_k{self.k_sparse}_int{self.hg_update_interval}" if self.lr else ""
            fname = f"uncertainty_curve_fold{fold_idx if fold_idx is not None else 'all'}{param}_{ts}.png"
            plt.savefig(os.path.join(self.save_dir, fname), dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def plot_reliability_diagram(self, fold_idx=None, save=True, show=True, n_bins=10):
        plt.figure(figsize=(8, 8))
        suffix = self._get_title_suffix()
        if fold_idx is not None and self.fold_histories:
            histories = [self.fold_histories[fold_idx]]
        else:
            histories = self.fold_histories
        all_unc = []
        all_corr = []
        for hist in histories:
            fd = hist['history']
            if 'uncertainties' not in fd or 'correctness' not in fd:
                continue
            for unc_arr, corr_arr in zip(fd['uncertainties'], fd['correctness']):
                if len(unc_arr) > 0:
                    unc_arr = np.asarray(unc_arr).flatten()
                    corr_arr = np.asarray(corr_arr).flatten()
                    all_unc.extend(unc_arr.tolist())
                    all_corr.extend(corr_arr.tolist())
        if not all_unc:
            print("No reliability data.")
            return
        all_unc = np.array(all_unc)
        all_corr = np.array(all_corr)
        bins = np.linspace(0, 1, n_bins+1)
        bin_indices = np.digitize(all_unc, bins) - 1
        bin_acc, bin_unc = [], []
        for b in range(n_bins):
            mask = (bin_indices == b)
            if np.sum(mask) == 0:
                continue
            acc = np.mean(all_corr[mask])
            unc = np.mean(all_unc[mask])
            bin_acc.append(acc)
            bin_unc.append(unc)
        plt.plot(bin_unc, bin_acc, 'bo-', label='Model')
        plt.xlabel('Predicted Uncertainty')
        plt.ylabel('Accuracy')
        plt.title(f'Reliability Diagram{suffix}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlim(0,1)
        plt.ylim(0,1)
        if save:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            param = f"_lr{self.lr}_k{self.k_sparse}_int{self.hg_update_interval}" if self.lr else ""
            fname = f"reliability_diagram_fold{fold_idx if fold_idx is not None else 'all'}{param}_{ts}.png"
            plt.savefig(os.path.join(self.save_dir, fname), dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def plot_accuracy_vs_uncertainty(self, fold_idx=None, save=True, show=True):
        plt.figure(figsize=(10, 8))
        suffix = self._get_title_suffix()
        if fold_idx is not None and self.fold_histories:
            histories = [self.fold_histories[fold_idx]]
        else:
            histories = self.fold_histories
        all_unc = []
        all_corr = []
        for hist in histories:
            fd = hist['history']
            if 'uncertainties' not in fd or 'correctness' not in fd:
                continue
            for unc_arr, corr_arr in zip(fd['uncertainties'], fd['correctness']):
                if len(unc_arr) > 0:
                    unc_arr = np.asarray(unc_arr).flatten()
                    corr_arr = np.asarray(corr_arr).flatten()
                    all_unc.extend(unc_arr.tolist())
                    all_corr.extend(corr_arr.tolist())
        if not all_unc:
            print("No uncertainty data.")
            return
        all_unc = np.array(all_unc)
        all_corr = np.array(all_corr)
        hb = plt.hexbin(all_unc, all_corr, gridsize=40, cmap='YlOrRd', mincnt=1)
        plt.colorbar(hb, label='Number of samples')
        plt.xlabel('Uncertainty')
        plt.ylabel('Correct (1) / Wrong (0)')
        plt.yticks([0,1], ['Wrong','Correct'])
        plt.title(f'Density of Samples (Uncertainty vs Correctness){suffix}')
        plt.grid(alpha=0.3)
        if save:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            param = f"_lr{self.lr}_k{self.k_sparse}_int{self.hg_update_interval}" if self.lr else ""
            fname = f"accuracy_vs_uncertainty_heatmap_fold{fold_idx if fold_idx is not None else 'all'}{param}_{ts}.png"
            plt.savefig(os.path.join(self.save_dir, fname), dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    # ------------------- 混淆矩阵 -------------------
    def plot_confusion_matrix(self, fold_idx=None, save=True, show=True):
        suffix = self._get_title_suffix()
        if fold_idx is not None and self.fold_histories:
            folds = [self.fold_histories[fold_idx]]
        else:
            folds = self.fold_histories

        all_true = []
        all_pred = []
        for fold in folds:
            true = fold['history'].get('best_true_labels')
            pred = fold['history'].get('best_pred_labels')
            if true is not None and pred is not None:
                all_true.extend(true)
                all_pred.extend(pred)

        if len(all_true) == 0:
            print("No confusion matrix data.")
            return

        cm = confusion_matrix(all_true, all_pred)
        # 使用 matplotlib 自带方法绘制热力图（无需 seaborn）
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix (All Folds Combined){suffix}')
        if save:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            param = f"_lr{self.lr}_k{self.k_sparse}_int{self.hg_update_interval}" if self.lr else ""
            fname = f"confusion_matrix_fold{fold_idx if fold_idx is not None else 'all'}{param}_{ts}.png"
            plt.savefig(os.path.join(self.save_dir, fname), dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    # ------------------- 最终指标箱线图 -------------------
    def plot_final_metrics(self, acc_res, f1_res, auc_res, save=True, show=True):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        suffix = self._get_title_suffix()
        metrics = [('Accuracy', acc_res, 'blue'), ('F1 Score', f1_res, 'cyan'), ('AUC/F1_macro', auc_res, 'green')]
        for idx, (title, data, color) in enumerate(metrics):
            ax = axes[idx]
            ax.boxplot(data, patch_artist=True, boxprops=dict(facecolor=color, alpha=0.5))
            x = np.random.normal(1, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.6, color=color, s=100)
            ax.set_title(f'{title} Distribution (5-fold)')
            ax.set_ylabel(title)
            ax.set_xticklabels([''])
            ax.grid(True, alpha=0.3)
            mean_val, std_val = np.mean(data), np.std(data)
            ax.text(0.95, 0.95, f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}',
                    transform=ax.transAxes, va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.suptitle(f'Model Performance Metrics (5-fold Cross Validation){suffix}')
        plt.tight_layout()
        if save:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            param = f"_lr{self.lr}_k{self.k_sparse}_int{self.hg_update_interval}" if self.lr else ""
            fname = f"final_metrics{param}_{ts}.png"
            plt.savefig(os.path.join(self.save_dir, fname), dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def save_history_to_csv(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_history_{timestamp}.csv"
        save_path = os.path.join(self.save_dir, filename)
        df_main = pd.DataFrame(self.history)
        df_main.to_csv(save_path.replace('.csv', '_main.csv'), index=False)
        for i, fold in enumerate(self.fold_histories):
            fold_df = pd.DataFrame(fold['history'])
            fold_df['fold'] = i + 1
            fold_df['timestamp'] = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in fold['timestamps']]
            if i == 0:
                all_folds = fold_df
            else:
                all_folds = pd.concat([all_folds, fold_df], ignore_index=True)
        all_folds.to_csv(save_path.replace('.csv', '_folds.csv'), index=False)
        print(f"Training history saved to: {save_path}")