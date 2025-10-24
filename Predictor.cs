using System;
using System.Linq;
using Microsoft.ML.Probabilistic.Distributions;

namespace NaiveBayes {
    public static class Predictor {
        // Predict class probabilities for a new feature vector using inferred model posteriors.
        // featureBetas: Beta[class][feature]
        // classProbsDirichlet: Dirichlet posterior (we'll use its mean)
        public static double[] Predict(Beta[][] featureBetas, Dirichlet classProbsDirichlet, bool[] newFeatures) {
            int classCount = featureBetas.Length;
            int numFeatures = featureBetas[0].Length;
            if (newFeatures.Length != numFeatures) throw new ArgumentException($"Feature vector must have {numFeatures} features");

            var classMeans = classProbsDirichlet.GetMean();
            var logClassProbs = new double[classCount];
            for (int c = 0; c < classCount; c++) logClassProbs[c] = Math.Log(classMeans[c]);

            for (int c = 0; c < classCount; c++) {
                for (int f = 0; f < numFeatures; f++) {
                    var beta = featureBetas[c][f];
                    double p = beta.GetMean();
                    if (newFeatures[f]) logClassProbs[c] += Math.Log(p);
                    else logClassProbs[c] += Math.Log(1 - p);
                }
            }

            double maxLog = logClassProbs.Max();
            var unnorm = logClassProbs.Select(lp => Math.Exp(lp - maxLog)).ToArray();
            var sum = unnorm.Sum();
            return unnorm.Select(x => x / sum).ToArray();
        }

        // Overload accepting class means directly
        public static double[] Predict(Beta[][] featureBetas, double[] classMeans, bool[] newFeatures) {
            int classCount = featureBetas.Length;
            var logClassProbs = new double[classCount];
            for (int c = 0; c < classCount; c++) logClassProbs[c] = Math.Log(classMeans[c]);

            for (int c = 0; c < classCount; c++) {
                for (int f = 0; f < featureBetas[c].Length; f++) {
                    var beta = featureBetas[c][f];
                    double p = beta.GetMean();
                    if (newFeatures[f]) logClassProbs[c] += Math.Log(p);
                    else logClassProbs[c] += Math.Log(1 - p);
                }
            }

            double maxLog = logClassProbs.Max();
            var unnorm = logClassProbs.Select(lp => Math.Exp(lp - maxLog)).ToArray();
            var sum = unnorm.Sum();
            return unnorm.Select(x => x / sum).ToArray();
        }
    }
}
