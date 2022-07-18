from os import PathLike
import numpy as np
from scipy.special import digamma, polygamma, loggamma
from pathlib import Path
import time


class Corpus:
    def __init__(self, data_path):
        # corpus = docIdx: document
        # document: list of word index
        self.corpus = {}
        self.words = set()

        # Read data and create corpus
        with open(data_path / "train.data", "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break

                # each line = docIdx wordIdx count
                docIdx, wordIdx, count = line.split()
                docIdx = int(docIdx)
                wordIdx = int(wordIdx)
                count = int(count)
                if docIdx not in self.corpus:
                    self.corpus[docIdx] = count * [wordIdx]
                else:
                    self.corpus[docIdx] += count * [wordIdx]

                if wordIdx not in self.words:
                    self.words.add(wordIdx)

        # V
        self.num_words = len(self.words)

        # M
        self.num_doc = len(self.corpus)

        # K
        self.num_topic = 0
        with open(data_path / "train.map", "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.num_topic += 1

    def __getitem__(self, index):
        # docIdx starts with 1
        return self.corpus[index + 1]


class LDA:
    def __init__(
        self, corpus: Corpus, verbose: bool = False, random_init: bool = False
    ):
        self.data = corpus
        self.random_init = random_init
        self.verbose = verbose
        self.K = corpus.num_topic
        self.V = corpus.num_words
        self.M = corpus.num_doc
        if random_init:
            self.alpha = np.random.random((self.K,))
            self.beta = np.random.random((self.K, self.V))
            self.beta /= self.beta.sum(axis=1).reshape((-1, 1))
        else:
            self.alpha = (50 / self.K) * np.ones(self.K)
            self.beta = np.ones((self.K, self.V)) / self.V
        self.gamma = np.zeros((self.M, self.K))

        self.phi = []
        for d in range(self.M):
            Nd = len(self.data[d])
            if self.random_init:
                phi = np.random.random((Nd, self.K))
                phi /= phi.sum(axis=1).reshape((-1, 1))
                self.phi += [phi]
                self.gamma[d] = self.alpha + phi.sum(axis=0)
            else:
                self.gamma[d] = self.alpha + (Nd / self.K) * np.ones((self.K,))
                self.phi += [np.ones((Nd, self.K)) / self.K]

    # Lower bound of log-likelihood for given document
    def L(self, doc: int):
        # To use numpy's vectorization, we used np.sum and np.einsum
        # TODO: check this equation
        alpha_term = (
            loggamma(self.alpha.sum())
            - np.sum(loggamma(self.alpha))
            + np.sum(
                (self.alpha - 1)
                * (digamma(self.gamma[doc]) - digamma(self.gamma[doc].sum()))
            )
        )
        ztheta_term = np.sum(
            self.phi[doc] * (digamma(self.gamma[doc]) - digamma(self.gamma[doc].sum()))
        )

        word_list = np.array(self.data[doc], dtype=int) - 1

        beta_term = np.einsum(
            "ni, in -> ", self.phi[doc], np.log(self.beta)[:, word_list]
        )

        theta_term = (
            -loggamma(self.gamma[doc].sum())
            + np.sum(loggamma(self.gamma[doc]))
            - np.sum(
                (self.gamma[doc] - 1)
                * (digamma(self.gamma[doc]) - digamma(self.gamma[doc].sum()))
            )
        )
        z_term = -np.sum(self.phi[doc] * np.log(self.phi[doc]))

        return alpha_term + ztheta_term + beta_term + theta_term + z_term

    def newton_rhapson(self, max_iter=100):
        # Phi = digamma = (log Gamma)'
        # Phi'(x) = polygamma(1, x)

        # Hessian is the form of
        # d^2L / da[i]a[j] = delta[i, j] M Phi'(a[i]) - Phi'(sum j=1 to k a[j])
        # = diag(Phi'(a)) + 1 (-Phi'(sum j=1 to k a[j])) 1.T
        # Therefore, we can use matrix inversion lemma
        # (H.inv g)[i] = (g[i] - c) / h[i]
        # where g is gradient and c = (sum j=1 to k g[j] / h[j]) / (1/z + sum j=1 to k 1/h[j])

        # Gradient is
        # dL / da[i] = M(phi(sum j=1 to k a[j]) - Phi(a[i])) + sum d=1 to M (Phi(gamma[d, i]) - Phi(sum j=1 to k gamma[d, j]))
        num_iter = 0
        converged = False
        new_alpha = self.alpha
        BATCH_SIZE = 64

        # MiniBatch Stochastic Gradient Descent

        for e in range(max_iter):
            indices = np.arange(self.M)
            np.random.shuffle(indices)

            for i in range(self.M // BATCH_SIZE):
                idx = indices[i : i + BATCH_SIZE]

                g = self.M * (
                    digamma(new_alpha.sum()) * np.ones(new_alpha.shape)
                    - digamma(new_alpha)
                ) + np.sum(
                    digamma(self.gamma[idx, :])
                    - digamma(self.gamma[idx, :].sum(axis=1).reshape((-1, 1))),
                    axis=0,
                )
                new_alpha = new_alpha - 1e-4 * g

        # newton_rhapson method

        # while (not converged) and (num_iter < max_iter):
        #     # TODO: check this equation
        #     g = self.M * (
        #         digamma(new_alpha.sum()) * np.ones(new_alpha.shape) - digamma(new_alpha)
        #     ) + np.sum(
        #         digamma(self.gamma) - digamma(self.gamma.sum(axis=1).reshape((-1, 1))),
        #         axis=0,
        #     )

        #     h = self.M * polygamma(1, new_alpha)
        #     z = -polygamma(1, self.alpha.sum())

        #     c = (np.sum(g / h)) / (1 / z + (1 / h).sum())
        #     delta = (g - c) / h
        #     new_alpha = new_alpha - delta

        #     converged = np.linalg.norm(g) < 1e-4
        #     num_iter += 1

        self.alpha = new_alpha

    def EStep(self, max_iter=100):
        if self.verbose:
            print("E step start")
        start = time.time()
        for d in range(self.M):
            converged = False
            num_iter = 0
            word_list = np.array(self.data[d], dtype=int) - 1
            beta_w = self.beta[:, word_list]
            while (not converged) and (num_iter < max_iter):
                # TODO: check this equation
                phi_new = (
                    beta_w
                    * np.exp(
                        digamma(self.gamma[d]) - digamma(self.gamma[d].sum())
                    ).T.reshape((-1, 1))
                ).T
                phi_new /= phi_new.sum(axis=1).reshape((-1, 1))
                gamma_new = self.alpha + phi_new.sum(axis=0)

                converged = (
                    np.linalg.norm(phi_new - self.phi[d]) < 1e-4
                    and np.linalg.norm(gamma_new - self.gamma[d]) < 1e-4
                )
                self.phi[d] = phi_new
                self.gamma[d] = gamma_new
                num_iter += 1
        end = time.time()
        if self.verbose:
            print(f"E step Ended: time: {end-start:.05}")

    def MStep(self, max_iter=100):
        start = time.time()
        if self.verbose:
            print(f"M step start")
        beta_new = np.zeros((self.K, self.V))
        for d in range(self.M):
            # TODO: check this equation
            word_list = np.array(self.data[d], dtype=int) - 1
            beta_new[:, word_list] += self.phi[d].T

        beta_new /= beta_new.sum(axis=1).reshape((-1, 1))
        self.beta = beta_new

        self.newton_rhapson()
        end = time.time()
        if self.verbose:
            print(f"M step Ended: time: {end-start:.05} secs")

    def train_variational_em(self, max_iter=100):

        converged = False
        num_iter = 0
        lbound = 0
        L_vec = np.vectorize(self.L)
        doc_idx = np.arange(self.M)

        while (not converged) and (num_iter < max_iter):
            # E step
            print(f"Step {num_iter + 1}/{max_iter} start")
            self.EStep()

            # M step
            self.MStep()

            start = time.time()
            lbound_new = L_vec(doc_idx).mean()
            converged = np.abs(lbound - lbound_new) < 1e-4
            end = time.time()
            if self.verbose:
                print(
                    f"Log Likelihood Lower Bound Mean After step {num_iter + 1}: {lbound_new}, converged?: {converged}, time for computing L: {end-start:.05} secs"
                )
            num_iter += 1


if __name__ == "__main__":

    # corpus = dict of doc
    # doc = list of words

    base_dir = Path().resolve().parent
    corpus = Corpus(base_dir / "data" / "20newsgroup")
    lda_model = LDA(corpus, verbose=True, random_init=False)
    lda_model.train_variational_em()
