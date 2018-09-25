# An-Information-theoretic-Metric-of-Transferability
This repository contains the codes of the experiments in Paper An Information-theoretic Metric of Transferability for Task Transfer Learning
## Requirements
**Python**: see [`requirement.txt`](https://github.com/StanfordVL/taskonomy/blob/master/taskbank/requirement.txt) for complete list of used packages.
## H-score Computation
Given an arbitrary feature function, you can evaluate H-score simply by calling the following function 
```bash
def getCov(X):
    X_mean=X-np.mean(X,axis=0,keepdims=True)
    cov = np.divide(np.dot(X_mean.T, X_mean), len(X)-1) 
    return cov

def getHscore(f,Z):
    #Z=np.argmax(Z, axis=1)
    Covf=getCov(f)
    alphabetZ=list(set(Z))
    g=np.zeros_like(f)
    for z in alphabetZ:
        Ef_z=np.mean(f[Z==z, :], axis=0)
        g[Z==z]=Ef_z
    
    Covg=getCov(g)
    score=np.trace(np.dot(np.linalg.pinv(Covf,rcond=1e-15), Covg))
    return score
```
## Demo
To see how fast H-score can be computed and how amazingly H-score is in accordance with empirical performance, you can reproduce the experiment Validation of H-score within a few minutes. 
