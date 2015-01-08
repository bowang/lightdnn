# lightdnn

#### Build
    $ make

#### Run
    $ build/auto-encoder data/patches.mat -N 10000 -S 64,25,64 -K 400 -method LBFGS

#### Usage
To see meanings of parameters, just run without any parameter, i.e. `$ build/auto-encoder`

    USAGE: build/auto-encoder <path-to-dataset> [options]
       -N:      number of training samples
       -S:      layer sizes
       -method: training method (optional)
       -alpha:  learning rate (optional)
       -beta:   weight of sparsity penalty (optional)
       -rho:    sparsity parameter (optional)
       -lambda: weight decay parameter (optional)
       -K:      number of iterations (optional)
       -B:      size of mini-batch (optional)
       -P:      number of processors (optional)
