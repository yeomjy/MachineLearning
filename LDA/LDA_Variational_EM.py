from os import PathLike
import numpy as np
from scipy.special import digamma, polygamma, loggamma
from pathlib import Path
import time


class Corpus:
    def __init__(self, data_path=PathLike | str):
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
    def __init__(self, corpus: Corpus, verbose: bool = False):
        self.data = corpus
        self.verbose = verbose
        self.K = corpus.num_topic
        self.V = corpus.num_words
        self.M = corpus.num_doc
        self.alpha = (50 / self.K) * np.ones(self.K)
        self.beta = np.ones((self.K, self.V)) / self.K
        self.gamma = np.zeros((self.M, self.K))

        self.phi = []
        for d in range(self.M):
            Nd = len(self.data[d])
            # self.gamma += [self.alpha + (Nd / self.K) * np.ones((self.K,))]
            self.gamma[d] = self.alpha + (Nd / self.K) * np.ones((self.K,))
            self.phi += [np.ones((Nd, self.K)) / self.K]

    # Lower bound of log-likelihood for given document
    def L(self, doc: int):
        # To use numpy's vectorization, we used np.sum and np.einsum
        Nd = len(self.data[doc])
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

    def newton_rhapson(self):
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

        g = self.M * (
            digamma(self.alpha.sum()) * np.ones(self.alpha.shape) - digamma(self.alpha)
        ) + np.sum(
            digamma(self.gamma) - digamma(self.gamma.sum(axis=1).reshape((-1, 1))),
            axis=0,
        )

        h = self.M * polygamma(1, self.alpha)
        z = -polygamma(1, self.alpha.sum())

        c = (np.sum(g / h)) / (1 / z + (1 / h).sum())
        delta = (g - c) / h
        self.alpha -= delta

    def EStep(self, max_iter=100):
        if self.verbose:
            print("E step start")
        start = time.time()
        for d in range(self.M):
            converged = False
            num_iter = 0
            word_list = np.array(self.data[d], dtype=int) - 1
            beta_w = self.beta[:, word_list]
            Nd = len(self.data[d])
            while (not converged) and (num_iter < max_iter):
                phi_new = (beta_w * np.exp(digamma(self.gamma[d])).T.reshape((-1, 1))).T
                phi_new /= phi_new.sum(axis=1).reshape((-1, 1))
                gamma_new = self.alpha + phi_new.sum(axis=0)

                converged = (
                    np.linalg.norm(phi_new - self.phi[d]) < 1e-4
                    and np.linalg.norm(gamma_new - self.gamma[d]) < 1e-4
                )
                self.phi[d] = phi_new
                self.gamma[d] = gamma_new
                num_iter += 1
        #     if self.verbose:
        #         print(
        #             f"E step for document {d}/{self.M}: {num_iter + 1}/{max_iter}, converged?: {converged}, time: {end - start}"
        #         )
        end = time.time()
        if self.verbose:
            print(f"E step Ended: time: {end-start}")

    def MStep(self, max_iter=100):
        start = time.time()
        if self.verbose:
            print(f"M step start")
        beta_new = np.zeros((self.K, self.V))
        for d in range(self.M):
            Nd = len(self.data[d])
            word_list = np.array(self.data[d], dtype=int) - 1
            beta_new[:, word_list] += self.phi[d].T

        beta_new /= beta_new.sum(axis=0)
        self.beta = beta_new

        self.newton_rhapson()
        end = time.time()
        if self.verbose:
            print(f"M step end: {end-start} secs")

    def train(self, max_iter=100):

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
            lbound_new = L_vec(doc_idx).sum()
            converged = np.abs(lbound - lbound_new) < 1e-4
            end = time.time()
            if self.verbose:
                print(
                    f"Log Likelihood Lower Bound After step {num_iter + 1}: {lbound_new}, converged?: {converged}, time for computing L: {start -end}"
                )
            num_iter += 1


# def newton_rhapson(alpha, gamma, M, k):
#     log_alpha = np.log(alpha)
#     dLda = M * (k * polygamma(1, k * alpha) - k * polygamma(1, alpha)) + np.sum(digamma(gamma) - digamma(np.sum(gamma)), axis=0)
#     d2Lda2 = M * (k ** 2 * polygamma(2, k * alpha) - k * polygamma(2, alpha))
#     log_new_alpha = log_alpha - dLda / (d2Lda2 * alpha + dLda)
#     return np.exp(log_new_alpha)


# def L(gamma, phi, alpha, beta, doc, V, K):

#     term1 = loggamma(np.sum(alpha)) - np.sum(loggamma(alpha))
#     gamma_sum = digamma(np.sum(gamma))

#     for i in range(K):
#         term1 += (alpha[i] - 1) * (digamma(gamma[i]) - gamma_sum)

#     term2 = 0
#     for n in range(len(doc)):
#         for i in range(K):
#             term2 += phi[n, i] * (digamma(gamma[i]) - gamma_sum)

#     term3 = 0
#     word_list = list(doc.keys())
#     for n in range(len(doc)):
#         for i in range(K):
#             term3 += phi[n, i] * np.log(beta[i, word_list[n]])

#     term4 = -loggamma(np.sum(gamma)) + np.sum(loggamma(gamma))
#     for i in range(K):
#         term4 -= (gamma[i] - 1) * (digamma(gamma[i]) - gamma_sum)

#     term5 = 0
#     for n in range(len(doc)):
#         for i in range(K):
#             term5 -= phi[n, i] * np.log(phi[n, i])

#     # print(term1, term2, term3, term4, term5)

#     return term1 + term2 + term3 + term4 + term5


# def lda_em(K, V, corpus):
#     # Initialize parameters
#     alpha = (50 / K) * np.ones(K)
#     phi_list = []
#     M = len(corpus)

#     gamma_list = []
#     for d in range(M):
#         Nd = len(corpus[d])
#         phi = np.ones((Nd, K))
#         phi = phi / K
#         phi_list.append(phi)
#         gamma = alpha + (Nd / K) * np.ones((K,))
#         gamma_list.append(gamma)

#     beta = np.ones((K, V)) / K
#     likelihood_converged = False
#     loglikelihood = 0
#     converged = False

#     loop_count = 0
#     likelihood_old = 0
#     while not likelihood_converged and loop_count < 100:
#         # E-step
#         loglikelihood = 0
#         print(f"E step start: {loop_count}")
#         for d in range(M):
#             phi = phi_list[d]
#             phi_old = phi_list[d]
#             gamma = gamma_list[d]
#             gamma_old = gamma_list[d]
#             inner_loop_count = 0
#             word_list = list(corpus[d].keys())
#             while not converged and inner_loop_count < 100:
#                 Nd = len(corpus[d])
#                 for n in range(Nd):
#                     for i in range(K):
#                         wn = word_list[n]
#                         phi[n][i] = beta[i][wn] * np.exp(digamma(gamma[i]))
#                     norm_factor = np.sum(phi[n])
#                     phi[n] = phi[n] / norm_factor
#                 gamma = alpha + np.sum(phi, axis=0)

#                 gamma_diff = np.linalg.norm(gamma - gamma_old)
#                 phi_diff = np.linalg.norm(phi - phi_old)
#                 converged = (gamma_diff < 1e-3 and phi_diff < 1e-3)
#                 gamma_list[d] = gamma
#                 phi_list[d] = phi
#                 inner_loop_count += 1

#             loglikelihood = loglikelihood + L(gamma, phi, alpha, beta, corpus[d], V, K)
#         print(f"E step end: {loop_count}")

#         # M-step
#         print(f"M step start: {loop_count}")
#         beta_new = np.zeros(beta.shape)
#         for d in range(M):
#             phi = phi_list[d]
#             word_list = list(corpus[d].keys())
#             for i in range(beta.shape[0]):
#                 Nd = len(corpus[d])
#                 for n in range(Nd):
#                     w = word_list[n]
#                     beta_new[i][w] += phi[n][i] * corpus[d][w]

#         for j in range(V):
#             beta_new[:, j] = beta_new[:, j] / np.sum(beta_new[:, j])

#         beta = beta_new
#         alpha = newton_rhapson(alpha, gamma_list, M, K)
#         likelihood_converged = np.abs(loglikelihood - likelihood_old) < 1e-5
#         print(f"Likelihood_diff: {loglikelihood - likelihood_old}")
#         print(f"likelihood: {loglikelihood}")
#         print(f"M step end: {loop_count}")
#         likelihood_old = loglikelihood
#         loop_count += 1

#     return alpha, beta, phi_list, gamma_list


if __name__ == "__main__":

    # corpus = dict of doc
    # doc = list of words

    # alpha, beta, phi_list, gamma_list = lda_em(K, V, corpus)
    corpus = Corpus(Path("./20newsgroup"))
    lda_model = LDA(corpus, verbose=True)
    lda_model.train()
