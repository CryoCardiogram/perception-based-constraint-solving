# from https://github.com/Jonathan-Pearce/calibration_library
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


import numpy as np
from scipy.special import softmax


class CELoss:
    def compute_bin_boundaries(self, probabilities=np.array([])):
        # uniform bin spacing
        if probabilities.size == 0:
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]
        else:
            # size of bins
            bin_n = int(self.n_data / self.n_bins)

            bin_boundaries = np.array([])

            probabilities_sort = np.sort(probabilities)

            for i in range(0, self.n_bins):
                bin_boundaries = np.append(
                    bin_boundaries, probabilities_sort[i * bin_n]
                )
            bin_boundaries = np.append(bin_boundaries, 1.0)

            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]

    def get_probabilities(self, output, labels, logits):
        # If not probabilities apply softmax!
        if logits:
            self.probabilities = softmax(output, axis=1)
        else:
            self.probabilities = output

        self.labels = labels
        self.confidences = np.max(self.probabilities, axis=1)
        self.predictions = np.argmax(self.probabilities, axis=1)
        self.accuracies = np.equal(self.predictions, labels)

    def binary_matrices(self):
        idx = np.arange(self.n_data)
        # make matrices of zeros
        pred_matrix = np.zeros([self.n_data, self.n_class])
        label_matrix = np.zeros([self.n_data, self.n_class])
        # self.acc_matrix = np.zeros([self.n_data,self.n_class])
        pred_matrix[idx, self.predictions] = 1
        label_matrix[idx, self.labels] = 1

        self.acc_matrix = np.equal(pred_matrix, label_matrix)

    def compute_bins(self, index=None):
        self.bin_prop = np.zeros(self.n_bins)
        self.bin_acc = np.zeros(self.n_bins)
        self.bin_conf = np.zeros(self.n_bins)
        self.bin_score = np.zeros(self.n_bins)

        if index == None:
            confidences = self.confidences
            accuracies = self.accuracies
        else:
            confidences = self.probabilities[:, index]
            accuracies = self.acc_matrix[:, index]

        for i, (bin_lower, bin_upper) in enumerate(
            zip(self.bin_lowers, self.bin_uppers)
        ):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(confidences, bin_lower.item()) * np.less_equal(
                confidences, bin_upper.item()
            )
            self.bin_prop[i] = np.mean(in_bin)

            if self.bin_prop[i].item() > 0:
                self.bin_acc[i] = np.mean(accuracies[in_bin])
                self.bin_conf[i] = np.mean(confidences[in_bin])
                self.bin_score[i] = np.abs(self.bin_conf[i] - self.bin_acc[i])


class MaxProbCELoss(CELoss):
    def loss(self, output, labels, n_bins=15, logits=False):
        self.n_bins = n_bins
        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().compute_bins()


class ConfidenceHistogram(MaxProbCELoss):
    def plot(self, output, labels, n_bins=15, logits=False, title=None):
        super().loss(output, labels, n_bins, logits)
        # scale each datapoint
        n = len(labels)
        w = np.ones(n) / n

        plt.rcParams["font.family"] = "serif"
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # size and axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
        )
        ax.set_yticks(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
        )
        # plot grid
        ax.grid(color="tab:grey", linestyle=(0, (1, 5)), linewidth=1, zorder=0)
        # plot histogram
        ax.hist(
            self.confidences,
            n_bins,
            weights=w,
            color="b",
            range=(0.0, 1.0),
            edgecolor="k",
        )

        # plot vertical dashed lines
        acc = np.mean(self.accuracies)
        conf = np.mean(self.confidences)
        ax.axvline(x=acc, color="tab:grey", linestyle="--", linewidth=3)
        ax.axvline(x=conf, color="tab:grey", linestyle="--", linewidth=3)
        if acc > conf:
            ax.text(acc + 0.03, 0.9, "Accuracy", rotation=90, fontsize=11)
            ax.text(conf - 0.07, 0.9, "Avg. Confidence", rotation=90, fontsize=11)
        else:
            ax.text(acc - 0.07, 0.9, "Accuracy", rotation=90, fontsize=11)
            ax.text(conf + 0.03, 0.9, "Avg. Confidence", rotation=90, fontsize=11)

        ax.set_ylabel("% of Samples", fontsize=13)
        ax.set_xlabel("Confidence", fontsize=13)
        fig.tight_layout()
        if title is not None:
            ax.title(title, fontsize=16)
        return ax, fig


class ReliabilityDiagram(MaxProbCELoss):
    def plot(self, output, labels, n_bins=15, logits=False, title=None, gap_message=""):
        super().loss(output, labels, n_bins, logits)

        # computations
        delta = 1.0 / n_bins
        x = np.arange(0, 1, delta)
        mid = np.linspace(delta / 2, 1 - delta / 2, n_bins)
        error = np.abs(np.subtract(mid, self.bin_acc))

        plt.rcParams["font.family"] = "serif"
        # size and axis limits
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        # plt.figure(figsize=(5,5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # plt.xlim(0,1)
        # plt.ylim(0,1)
        # plot grid
        # plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)
        ax.grid(color="tab:grey", linestyle=(0, (1, 5)), linewidth=1, zorder=0)
        # plot bars and identity line
        # plt.bar(x, self.bin_acc, color = 'b', width=delta,align='edge',edgecolor = 'k',label='Outputs',zorder=5)
        ax.bar(
            x,
            self.bin_acc,
            color="b",
            width=delta,
            align="edge",
            edgecolor="k",
            label="Observed accuracy",
            zorder=5,
        )
        # plt.bar(x, error, bottom=np.minimum(self.bin_acc,mid), color = 'mistyrose', alpha=0.5, width=delta,align='edge',edgecolor = 'r',hatch='/',label='Gap',zorder=10)
        ax.bar(
            x,
            error,
            bottom=np.minimum(self.bin_acc, mid),
            color="mistyrose",
            alpha=0.5,
            width=delta,
            align="edge",
            edgecolor="r",
            hatch="/",
            label=f"Gap{gap_message}",
            zorder=10,
        )
        ident = [0.0, 1.0]
        # plt.plot(ident,ident,linestyle='--',color='tab:grey',zorder=15)
        ax.plot(ident, ident, linestyle="--", color="tab:grey", zorder=15)
        # labels and legend
        ax.set_ylabel("Accuracy", fontsize=13)
        # plt.ylabel('Accuracy',fontsize=13)
        # plt.xlabel('Confidence',fontsize=13)
        ax.set_xlabel("Confidence", fontsize=13)
        # plt.legend(loc='upper left',framealpha=1.0,fontsize='medium')
        ax.legend(loc="upper left", framealpha=1.0, fontsize="medium")
        if title is not None:
            # plt.title(title,fontsize=16)
            ax.set_title(title, fontsize=16)
        # plt.tight_layout()
        fig.tight_layout()
        return ax, fig
