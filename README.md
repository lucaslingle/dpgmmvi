# dpgmmvi

Variational inference for Dirichlet process Gaussian mixture models.

### Contents

My original implementation of the Variational Dirichlet Process algorithm from [Kurihara et al., 2007a](https://proceedings.neurips.cc/paper_files/paper/2006/file/2bd235c31c97855b7ef2dc8b414779af-Paper.pdf) is available in `legacy/nested_vi.py`. 
I also support a simpler algorithm in `legacy/simple_vi.py` from [Kurihara et al., 2007b](https://www.ijcai.org/Proceedings/07/Papers/449.pdf). This algorithm is discussed in-depth on my [blog](https://lucaslingle.substack.com/p/simplified-variational-inference) and I found it performed slightly better in terms of [ELBO](https://en.wikipedia.org/wiki/Evidence_lower_bound). 

The second version is currently the only one iremplemented in `dpgmmvi/algos.py`, which was rewritten from the legacy implementations to support streamlit visualizations. In general, I recommend using `dpgmmvi/algos.py`.

### Installation

Clone this repo, navigate to it, and run
```shell
pip install -e .
```
I recommend using a virtual environment such as [venv](https://docs.python.org/3/library/venv.html) or [miniconda](https://www.anaconda.com/download).

### Basic Usage

Given a numpy array of training data `xs` with shape `[N, D]`, you can fit a model like so: 
```python
from dpgmmvi.algos import Config, vi_loop

config = Config(
    truncation_level=10,   # truncation level of variational model q(c, v, z)
    sigma_c=1.0,           # standard deviation for base distribution p(c)
    sigma_x=0.05,          # standard deviation for observation conditional p(x|c)
    trainset_size=N,       # total number of datapoints in the training set
    data_dim=D,            # dimension of the data vectors being modeled
    kappa=0.001,           # learning rate for stochastic variational inference
)
states, elbos = vi_loop(
    config=config, 
    minibatch_size=20,       # minibatch size for stochastic variational inference
    opt_iters=100,           # training iterations
    xs_train=xs,             # training dataset
)
trained_model = states[-1]
```
Full-batch variational Bayes corresponds to `kappa=1.0` and `minibatch_size=N`. 

### Streamlit Visualizations

To run the streamlit visualizations as a webapp, visit either of
```
https://dpgmmvi-2d.streamlit.app/
https://dpgmmvi-3d.streamlit.app/ 
```
for clustering in 2D or 3D!

To run the streamlit visualizations locally, you can run either of
```
streamlit run ./dpgmmvi/streamlit_2d.py
streamlit run ./dpgmmvi/streamlit_3d.py
```

### References
```
Blei and Jordan, 2006 - Variational Inference for Dirichlet Process Mixtures
URL: https://projecteuclid.org/journals/bayesian-analysis/volume-1/issue-1/Variational-inference-for-Dirichlet-process-mixtures/10.1214/06-BA104.pdf

Kurihara et al., 2007a - Accelerated Variational Dirichlet Process Mixtures
URL: https://proceedings.neurips.cc/paper_files/paper/2006/file/2bd235c31c97855b7ef2dc8b414779af-Paper.pdf

Kurihara et al., 2007b - Collapsed Variational Dirichlet Process Mixture Models
URL: https://www.ijcai.org/Proceedings/07/Papers/449.pdf

Welling et al., 2008 - Deterministic Latent Variable Models and their Pitfalls
URL: https://www.researchgate.net/publication/220907288_Deterministic_Latent_Variable_Models_and_Their_Pitfalls

Hoffman et al., 2012 - Stochastic Variational Inference
URL: https://arxiv.org/abs/1206.7051
```
