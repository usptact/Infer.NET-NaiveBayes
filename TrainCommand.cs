using System;
using CommandLine;

namespace NaiveBayes {
    [Verb("train", HelpText = "Train a model from CSV and save model JSON.")]
    public class TrainCommand {
        [Option("train", Required = true, HelpText = "Training CSV file")]
        public string TrainingFile { get; set; } = string.Empty;

        [Option('v', "verbose", HelpText = "Verbose output")]
        public bool Verbose { get; set; }

        [Option("out-model", HelpText = "Output model JSON file (default model.json)")]
        public string OutModel { get; set; } = "model.json";

        [Option("predict-unlabeled", HelpText = "Save posteriors for unlabeled training instances to FILE")]
        public string? PredictUnlabeled { get; set; }

        public int Run() {
            // concise header; detailed usage shown only on parse error or when verbose
            Console.WriteLine("Mode: train");
            if (Verbose) {
                Console.WriteLine("Usage: train --train <training.csv> [--out-model <model.json>] [--predict-unlabeled <unlabeled.csv>] [--verbose]");
                Console.WriteLine(" Example: train --train train.csv --out-model model.json --predict-unlabeled preds.csv --verbose");
                Console.WriteLine("CSV format: header row optional. Features then label as last column. Label should be 0/1 or empty for unlabeled.");
            }

            if (!System.IO.File.Exists(TrainingFile)) {
                Console.WriteLine($"Error: Training file not found: '{TrainingFile}'");
                return 1;
            }

            if (Verbose) Console.WriteLine($"Training from: {TrainingFile}");
            if (Verbose) Console.WriteLine($"Model output: {OutModel}");
            if (Verbose) Console.WriteLine($"Predict unlabeled output: {(PredictUnlabeled ?? "(none)")}");
            if (Verbose) Console.WriteLine($"Loading training data from {TrainingFile}");
            var (featuresData, hasLabel, observedLabels) = DataLoader.LoadFromCsv(TrainingFile);
            var numInstances = featuresData.Length;
            var numFeatures = featuresData[0].Length;
            var classCount = 2;

            var priors = new Priors();
            var model = new NaiveBayesModel(numInstances, numFeatures, classCount, priors);
            model.AttachData(featuresData, hasLabel, observedLabels);

            bool inferLabels = PredictUnlabeled != null;
            var (inferredFeatureProb, inferredLabels, inferredClassProbs) = model.Infer(inferLabels);

            ModelSerializer.SaveModel(OutModel, inferredFeatureProb, inferredLabels ?? Array.Empty<Microsoft.ML.Probabilistic.Distributions.Discrete>(), inferredClassProbs);
            Console.WriteLine($"Model saved to {OutModel}");
            if (Verbose) Console.WriteLine("Saved model JSON (verbose mode shows more detail)");

            if (inferLabels && PredictUnlabeled != null) {
                using var w = new System.IO.StreamWriter(PredictUnlabeled);
                w.WriteLine("instance,p0,p1,predicted");
                for (int idx = 0; idx < numInstances; idx++) {
                    if (!hasLabel[idx]) {
                        var post = inferredLabels![idx];
                        var probs = post.GetProbs();
                        int pred = probs[0] > probs[1] ? 0 : 1;
                        w.WriteLine($"{idx},{probs[0]:F6},{probs[1]:F6},{pred}");
                    }
                }
                Console.WriteLine($"Predicted unlabeled instances saved to {PredictUnlabeled}");
                if (Verbose) Console.WriteLine("Unlabeled instance posteriors written (verbose mode shows more detail)");
            }

            return 0;
        }
    }
}
