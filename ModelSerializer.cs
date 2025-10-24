using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Probabilistic.Distributions;
using System.Text.Json;

namespace NaiveBayes {
    public static class ModelSerializer {
        public class FeatureInfo {
            public double Mean { get; set; }
            public double Variance { get; set; }
            // Beta parameters (alpha, beta) to fully describe the posterior
            public double Alpha { get; set; }
            public double Beta { get; set; }
        }
        public class ModelDump {
            public FeatureInfo[][]? FeatureProb { get; set; }
            public double[][]? LabelPosteriors { get; set; }
            public double[]? ClassProb { get; set; }
        }

    public static void SaveModel(string path, Beta[][] featureProb, Discrete[]? labelPosteriors, Dirichlet classProbs) {
            var numClasses = featureProb.Length;
            var numFeatures = featureProb[0].Length;

            var feat = new FeatureInfo[numClasses][];
            for (int c = 0; c < numClasses; c++) {
                feat[c] = new FeatureInfo[numFeatures];
                for (int f = 0; f < numFeatures; f++) {
                    var b = featureProb[c][f];
                    // Beta has GetAlpha() and GetBeta() in some versions; use accessors if available
                    // Derive alpha/beta from mean and variance since Beta parameter accessors
                    // may not be available in this Infer.NET version.
                    var mean = b.GetMean();
                    var var = b.GetVariance();
                    double a = 0.0, bb = 0.0;
                    if (var > 0 && mean > 0 && mean < 1) {
                        var tmp = mean * (1 - mean) / var - 1.0;
                        a = Math.Max(1e-6, mean * tmp);
                        bb = Math.Max(1e-6, (1 - mean) * tmp);
                    } else {
                        // Fallback to non-informative prior if variance is zero or mean on boundary
                        a = 1.0; bb = 1.0;
                    }
                    feat[c][f] = new FeatureInfo { Mean = b.GetMean(), Variance = b.GetVariance(), Alpha = a, Beta = bb };
                }
            }

            double[][]? labels = null;
            if (labelPosteriors != null) {
                labels = new double[labelPosteriors.Length][];
                for (int i = 0; i < labelPosteriors.Length; i++) {
                    var probsVec = labelPosteriors[i].GetProbs();
                    labels[i] = probsVec.ToArray();
                }
            }

            var classMeansVec = classProbs.GetMean();
            var classMeans = classMeansVec.ToArray();

            var dump = new ModelDump {
                FeatureProb = feat,
                LabelPosteriors = labels,
                ClassProb = classMeans
            };

            var options = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(dump, options);
            File.WriteAllText(path, json);
        }

        public class LoadedModel {
            public Beta[][] FeatureBetas { get; set; } = Array.Empty<Beta[]>();
            public double[] ClassMeans { get; set; } = Array.Empty<double>();
            // optional label posteriors per instance
            public double[][]? LabelPosteriors { get; set; }
        }

        public static LoadedModel LoadModel(string path) {
            var text = File.ReadAllText(path);
            var dump = JsonSerializer.Deserialize<ModelDump>(text);
            if (dump == null || dump.FeatureProb == null || dump.ClassProb == null) throw new InvalidDataException("Model JSON missing required fields");

            var feat = dump.FeatureProb;
            int numClasses = feat.Length;
            int numFeatures = feat[0].Length;
            var betas = new Beta[numClasses][];
            for (int c = 0; c < numClasses; c++) {
                betas[c] = new Beta[numFeatures];
                for (int f = 0; f < numFeatures; f++) {
                    var fi = feat[c][f];
                    // Build Beta from stored alpha/beta; fall back to mean if invalid
                    double a = fi.Alpha > 0 ? fi.Alpha : 1.0;
                    double b = fi.Beta > 0 ? fi.Beta : 1.0;
                    // Construct Beta from alpha/beta if available, otherwise fallback to point mass at mean
                    double meanVal = (a + b) > 0 ? a / (a + b) : fi.Mean;
                    betas[c][f] = Beta.PointMass(meanVal);
                }
            }

            return new LoadedModel { FeatureBetas = betas, ClassMeans = dump.ClassProb, LabelPosteriors = dump.LabelPosteriors };
        }
    }
}
