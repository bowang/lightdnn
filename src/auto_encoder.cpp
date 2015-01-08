#include "net.h"
#include "auto_encoder.h"

using namespace lightdnn;

int main (int argc, char **argv)
{
  if (argc < 2) {
    printf ("USAGE: %s <path-to-dataset> [options]\n"
            "       -N:      number of training samples\n"
            "       -S:      layer sizes\n"
            "       -method: training method (optional)\n"
            "       -alpha:  learning rate (optional)\n"
            "       -beta:   weight of sparsity penalty (optional)\n"
            "       -rho:    sparsity parameter (optional)\n"
            "       -lambda: weight decay parameter (optional)\n"
            "       -K:      number of iterations (optional)\n"
            "       -B:      size of mini-batch (optional)\n"
            "       -P:      number of processors (optional)\n",
            argv[0]);
    exit (1);
  }

  // Parse input command line
  OptMap options = lightdnn::parseCmdLine (argc - 2, &argv[2]);

  // Construct the encoder
  AutoEncoder<double> encoder (argv[1], options);

  // Assemble the network
  encoder.initialize ();

  // Train the network
  encoder.train ();

  // Dump parameters
  encoder.dumpWeights ("weights.mat");

  return 0;
}

