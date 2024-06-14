from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
import numpy as np


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):


    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)

    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j] / num_scored_items)
            if num_ratings == 1:
                num_ratings += 0.0000001
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    if denominator <= 0.0000001:
        denominator = 0.0000001
    return np.round((1.0 - numerator / denominator),4)






def Quadratic_Weighted_Kappa(y_true, y_pred, total_score):
    # 클래스 개수
    num_classes = int(total_score)+1
    
    # 혼동 행렬(confusion matrix) 생성
    conf_matrix = np.zeros((num_classes, num_classes))
    for n in range(num_classes):
        i_c = int(y_true[n])
        j_c = int(y_pred[n])
        conf_matrix[i_c, j_c] += 1
    
    # 가중치 행렬 생성
    weights = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = ((i - j) ** 2)/((num_classes-1) ** 2)
    
    # 가중치 행렬과 혼동 행렬의 합계 계산
    obs_sum = np.sum(conf_matrix)
    exp_sum = np.outer(np.sum(conf_matrix, axis=1), np.sum(conf_matrix, axis=0)) / obs_sum
    
    # quadratic weighted kappa score 계산
    nom = np.sum(weights * conf_matrix)
    denom = np.sum(weights * exp_sum)
    if denom <= 0.0000001:
        denom = 0.0000001
    quadratic_kappa = 1 - (nom / denom)
    
    return quadratic_kappa