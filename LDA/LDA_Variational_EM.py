import numpy as np
from scipy.special import digamma, polygamma
from scipy.special import loggamma


def newton_rhapson(alpha, gamma, M, k):
    log_alpha = np.log(alpha)
    dLda = M * (k * polygamma(1, k * alpha) - k * polygamma(1, alpha)) + np.sum(digamma(gamma) - digamma(np.sum(gamma)), axis=0)
    d2Lda2 = M * (k ** 2 * polygamma(2, k * alpha) - k * polygamma(2, alpha))
    log_new_alpha = log_alpha - dLda / (d2Lda2 * alpha + dLda)
    return np.exp(log_new_alpha)


def L(gamma, phi, alpha, beta, doc, V, K):

    term1 = loggamma(np.sum(alpha)) - np.sum(loggamma(alpha))
    gamma_sum = digamma(np.sum(gamma))

    for i in range(K):
        term1 += (alpha[i] - 1) * (digamma(gamma[i]) - gamma_sum)

    term2 = 0
    for n in range(len(doc)):
        for i in range(K):
            term2 += phi[n, i] * (digamma(gamma[i]) - gamma_sum)

    term3 = 0
    word_list = list(doc.keys())
    for n in range(len(doc)):
        for i in range(K):
            term3 += phi[n, i] * np.log(beta[i, word_list[n]])

    term4 = -loggamma(np.sum(gamma)) + np.sum(loggamma(gamma))
    for i in range(K):
        term4 -= (gamma[i] - 1) * (digamma(gamma[i]) - gamma_sum)

    term5 = 0
    for n in range(len(doc)):
        for i in range(K):
            term5 -= phi[n, i] * np.log(phi[n, i])

    # print(term1, term2, term3, term4, term5)

    return term1 + term2 + term3 + term4 + term5


def lda_em(K, V, corpus):
    # Initialize parameters
    alpha = (50 / K) * np.ones(K)
    phi_list = []
    M = len(corpus)

    gamma_list = []
    for d in range(M):
        Nd = len(corpus[d])
        phi = np.ones((Nd, K))
        phi = phi / K
        phi_list.append(phi)
        gamma = alpha + (Nd / K) * np.ones((K,))
        gamma_list.append(gamma)

    beta = np.ones((K, V)) / K
    likelihood_converged = False
    loglikelihood = 0
    converged = False

    loop_count = 0
    likelihood_old = 0
    while not likelihood_converged and loop_count < 100:
        # E-step
        loglikelihood = 0
        print(f"E step start: {loop_count}")
        for d in range(M):
            phi = phi_list[d]
            phi_old = phi_list[d]
            gamma = gamma_list[d]
            gamma_old = gamma_list[d]
            inner_loop_count = 0
            word_list = list(corpus[d].keys())
            while not converged and inner_loop_count < 100:
                Nd = len(corpus[d])
                for n in range(Nd):
                    for i in range(K):
                        wn = word_list[n]
                        phi[n][i] = beta[i][wn] * np.exp(digamma(gamma[i]))
                    norm_factor = np.sum(phi[n])
                    phi[n] = phi[n] / norm_factor
                gamma = alpha + np.sum(phi, axis=0)

                gamma_diff = np.linalg.norm(gamma - gamma_old)
                phi_diff = np.linalg.norm(phi - phi_old)
                converged = (gamma_diff < 1e-3 and phi_diff < 1e-3)
                gamma_list[d] = gamma
                phi_list[d] = phi
                inner_loop_count += 1

            loglikelihood = loglikelihood + L(gamma, phi, alpha, beta, corpus[d], V, K)
        print(f"E step end: {loop_count}")

        # M-step
        print(f"M step start: {loop_count}")
        beta_new = np.zeros(beta.shape)
        for d in range(M):
            phi = phi_list[d]
            word_list = list(corpus[d].keys())
            for i in range(beta.shape[0]):
                Nd = len(corpus[d])
                for n in range(Nd):
                    w = word_list[n]
                    beta_new[i][w] += phi[n][i] * corpus[d][w]

        for j in range(V):
            beta_new[:, j] = beta_new[:, j] / np.sum(beta_new[:, j])

        beta = beta_new
        alpha = newton_rhapson(alpha, gamma_list, M, K)
        likelihood_converged = np.abs(loglikelihood - likelihood_old) < 1e-5
        print(f"Likelihood_diff: {loglikelihood - likelihood_old}")
        print(f"likelihood: {loglikelihood}")
        print(f"M step end: {loop_count}")
        likelihood_old = loglikelihood
        loop_count += 1

    return alpha, beta, phi_list, gamma_list


if __name__ == "__main__":

    # corpus = dict of doc
    # doc = list of words
    corpus = {}
    words = set()

    # Read data and create corpus
    with open("data/20newsgroup/train.data") as f:
        while True:
            line = f.readline()
            if not line:
                break

            # each line = docIdx wordIdx count
            docIdx, wordIdx, count = line.split()
            docIdx = int(docIdx)
            wordIdx = int(wordIdx)
            count = int(count)
            if docIdx - 1 not in corpus:
                corpus[docIdx - 1] = {wordIdx - 1: count}
            else:
                corpus[docIdx - 1][wordIdx - 1] = count

            if wordIdx not in words:
                words.add(wordIdx)

    V = len(words)
    K = 0
    with open("data/20newsgroup/train.map") as f:
        while True:
            line = f.readline()
            if not line:
                break
            K += 1

    alpha, beta, phi_list, gamma_list = lda_em(K, V, corpus)
    np.save("alpha", alpha)
    np.save("beta", beta)
    np.save("phi_list", phi_list)
    np.save("gamma_list", gamma_list)
