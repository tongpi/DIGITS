# -*- coding: utf_8 -*-
"""
@auth: XH
@time: 2020-01-18 14:01
@project: DIGITS
@file: visualization.py
@desc: 新增yellowbrick绘图功能，此模块提供三种图形：混淆矩阵，ROC，预测误差
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.classification import _check_targets
from sklearn.metrics import roc_curve, auc
from sklearn.utils.multiclass import unique_labels
from yellowbrick.draw import bar_stack
from yellowbrick.style.palettes import color_sequence, LINE_COLOR
from yellowbrick.style import find_text_color, resolve_colors


class DrawVisualization(object):

    def __init__(self, model, y, y_train, labels, save):
        self.y = y
        self.y_pred = model.predict_classes(y_train)
        self.cmy_pred = model.predict(y_train)
        self.labels = labels
        self.classes = len(labels)
        self.save = save
        self.cmap = color_sequence("YlOrRd")
        plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']

    def draw_confusion_matrix(self):
        yp = np.asarray(self.y)
        if yp.dtype.kind in {"i", "u"}:
            idx = yp
        else:
            idx = LabelEncoder().fit_transform(yp)
        y_true = np.asarray(self.labels)[idx]

        yp = np.asarray(self.y_pred)
        if yp.dtype.kind in {"i", "u"}:
            idx = yp
        else:
            idx = LabelEncoder().fit_transform(yp)
        y_pred = np.asarray(self.labels)[idx]

        c_m = confusion_matrix(y_true, y_pred, labels=self.labels)

        cm_display = c_m[::-1, ::]
        n_classes = len(self.labels)
        X, Y = np.arange(n_classes + 1), np.arange(n_classes + 1)

        plt.cla()
        ax = plt.gca()

        ax.set_ylim(bottom=0, top=cm_display.shape[0])
        ax.set_xlim(left=0, right=cm_display.shape[1])

        xticklabels = self.labels
        yticklabels = self.labels[::-1]
        ticks = np.arange(n_classes) + 0.5

        ax.set(xticks=ticks, yticks=ticks)
        ax.set_xticklabels(xticklabels, rotation="vertical")
        ax.set_yticklabels(yticklabels)

        edgecolors = []

        for x in X[:-1]:
            for y in Y[:-1]:
                value = cm_display[x, y]
                svalue = "{:0.0f}".format(value)

                base_color = self.cmap(value / cm_display.max())
                text_color = find_text_color(base_color)

                if cm_display[x, y] == 0:
                    text_color = "0.75"

                cx, cy = x + 0.5, y + 0.5
                ax.text(
                    cy,
                    cx,
                    svalue,
                    va="center",
                    ha="center",
                    color=text_color
                )
                lc = "k" if xticklabels[x] == yticklabels[y] else "w"
                edgecolors.append(lc)

        vmin = 0.00001
        vmax = cm_display.max()

        ax.pcolormesh(
            X,
            Y,
            cm_display,
            vmin=vmin,
            vmax=vmax,
            edgecolor=edgecolors,
            cmap=self.cmap,
            linewidth="0.01",
        )

        ax.set_title("混淆矩阵")
        ax.set_ylabel("真值类")
        ax.set_xlabel("预测类")

        cm_image_path = os.path.join(self.save, 'confusion_matrix.png')
        plt.savefig(cm_image_path)

    def draw_roc_auc(self):
        fpr = dict()
        tpr = dict()
        thresholds = dict()

        for i, c in enumerate(self.labels):
            fpr[i], tpr[i], _ = roc_curve(self.y, self.cmy_pred[:, i], pos_label=c)
            thresholds[i] = auc(fpr[i], tpr[i])

        colors = resolve_colors(self.classes)

        plt.cla()
        ax = plt.gca()
        for i, color in zip(range(self.classes), colors):
            ax.plot(
                fpr[i],
                tpr[i],
                linestyle="--",
                color=color,
                label="ROC of class {}, AUC = {:0.2f}".format(
                    self.labels[i], thresholds[i]
                )
            )
        ax.plot([0, 1], [0, 1], linestyle=":", c=LINE_COLOR)
        ax.set_title("ROC曲线")
        ax.legend(loc="lower right", frameon=True)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])

        ax.set_ylabel("真阳性率")
        ax.set_xlabel("假阳性率")

        roc_image_path = os.path.join(self.save, 'roc_auc.png')
        plt.savefig(roc_image_path)

    def draw_class_prediction_error(self):
        y_type, y_true, y_pred = _check_targets(self.y, self.y_pred)

        indices = unique_labels(y_true, y_pred)

        predictions_ = np.array(
            [
                [(y_pred[self.y == label_t] == label_p).sum() for label_p in indices]
                for label_t in indices
            ]
        )

        self.classes = len(self.labels)
        colors = resolve_colors(self.classes)

        plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']

        legend_kws = {"bbox_to_anchor": (1, 0.5), "loc": "center left"}
        plt.cla()
        ax = plt.gca()

        bar_stack(
            predictions_,
            ax,
            labels=list(self.labels),
            ticks=self.labels,
            colors=colors,
            legend_kws=legend_kws
        )

        ax.set_title("分类预测误差图")
        ax.set_xlabel("实际分类")
        ax.set_ylabel("预测类数量")

        cmax = max([sum(predictions) for predictions in predictions_])
        ax.set_ylim(0, cmax + cmax * 0.1)
        plt.tight_layout(rect=[0, 0, 0.90, 1])

        cpe_image_path = os.path.join(self.save, 'class_prediction_error.png')
        plt.savefig(cpe_image_path)



