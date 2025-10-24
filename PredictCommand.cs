using System;
using CommandLine;

namespace NaiveBayes {
    [Verb("predict", HelpText = "Load model JSON and predict unlabeled instances from CSV")]
    public class PredictCommand {
        [Option("model", Required = true, HelpText = "Model JSON file produced by training")]
        public string ModelFile { get; set; } = string.Empty;

        [Option("input", Required = true, HelpText = "Input CSV file containing instances (empty label field for unlabeled)")]
        public string InputCsv { get; set; } = string.Empty;

        [Option("output", Required = true, HelpText = "Output CSV file to write predictions")]
        public string OutputCsv { get; set; } = string.Empty;

        [Option('v', "verbose", HelpText = "Verbose output")]
        public bool Verbose { get; set; }

        public int Run() {
            // concise header; detailed usage shown only on parse error or when verbose
            Console.WriteLine("Mode: predict");
            if (Verbose) {
                Console.WriteLine("Usage: predict --model <model.json> --input <input.csv> --output <output.csv> [--verbose]");
                Console.WriteLine(" Example: predict --model model.json --input unlabeled.csv --output predictions.csv --verbose");
                Console.WriteLine("CSV format: same as training CSV but label column can be all empty");
            }

            if (!System.IO.File.Exists(ModelFile)) {
                Console.WriteLine($"Error: Model file not found: '{ModelFile}'");
                return 1;
            }
            if (!System.IO.File.Exists(InputCsv)) {
                Console.WriteLine($"Error: Input CSV not found: '{InputCsv}'");
                return 1;
            }

            if (Verbose) Console.WriteLine($"Loading model from: {ModelFile}");
            var loaded = ModelSerializer.LoadModel(ModelFile);
            var modelBetas = loaded.FeatureBetas;
            // class means are provided as array
            var classMeans = loaded.ClassMeans;

            if (Verbose) Console.WriteLine($"Loading instances from: {InputCsv}");
            var (featuresData, hasLabel, observedLabels) = DataLoader.LoadFromCsv(InputCsv);
            int numInstances = featuresData.Length;
            int unlabeledCount = hasLabel.Count(x => !x);
            if (Verbose) Console.WriteLine($"Loaded {numInstances} instances, will predict for {unlabeledCount} unlabeled instances");

            if (unlabeledCount == 0) {
                Console.WriteLine("Warning: no unlabeled instances found in input CSV");
            }

            using var w = new System.IO.StreamWriter(OutputCsv);
            w.WriteLine("instance,p0,p1,predicted");
            for (int idx = 0; idx < numInstances; idx++) {
                if (!hasLabel[idx]) {
                    var probs = Predictor.Predict(modelBetas, classMeans, featuresData[idx]);
                    int pred = probs[0] > probs[1] ? 0 : 1;
                    w.WriteLine($"{idx},{probs[0]:F6},{probs[1]:F6},{pred}");
                }
            }

            Console.WriteLine($"Predictions written to {OutputCsv}");
            if (Verbose) Console.WriteLine("Wrote probabilities for each unlabeled instance and predicted class (0/1)");
            return 0;
        }
    }
}
