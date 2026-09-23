# Default algorithms vs TF-CV-DS (shared preprocessing + centroid classifier)


model             IN_acc HARD_acc HARD_cov HARD_unk HARD_conf  dim  spars      ms
---------------------------------------------------------------------------------
BoW                1.000    0.519    0.556    0.444     0.037  173   0.96     2.2
BinaryBoW          1.000    0.519    0.556    0.444     0.037  173   0.96     1.6
TF                 1.000    0.519    0.556    0.444     0.037  173   0.96     1.6
TF-IDF             1.000    0.519    0.556    0.444     0.037  173   0.96     1.6
SublinearTF-IDF    1.000    0.519    0.556    0.444     0.037  173   0.96     1.7
BM25               1.000    0.519    0.556    0.444     0.037  173   0.96     2.0
TF-CV-DS           1.000    0.519    0.556    0.444     0.037  173   0.96     6.5

IN_VOCAB n=18, HARD n=27. 'unknown' = zero vector (no vocab overlap).
TF-CV-DS here uses plain defaults (no stemming/soft-match/topic-IDF).