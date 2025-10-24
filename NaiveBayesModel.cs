using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace NaiveBayes {
    public class NaiveBayesModel {
        public Range InstanceRange { get; private set; }
        public Range FeatureRange { get; private set; }
        public Range ClassRange { get; private set; }

    // Model variables
    public VariableArray<VariableArray<bool>, bool[][]> FeaturesVar { get; private set; }
        public VariableArray<int> LabelsVar { get; private set; }
    public VariableArray<VariableArray<double>, double[][]> FeatureProbVar { get; private set; }
    public Variable<Microsoft.ML.Probabilistic.Math.Vector> ClassProbsDirichlet { get; private set; }

        private Priors priors;
    // constants to be set when attaching data
    private VariableArray<bool> hasLabelConstVar;
    private VariableArray<int> observedLabelsConstVar;

    private InferenceEngine engine;

        public NaiveBayesModel(int numInstances, int numFeatures, int numClasses, Priors priors) {
            this.priors = priors;
            InstanceRange = new Range(numInstances).Named("i");
            FeatureRange = new Range(numFeatures).Named("f");
            ClassRange = new Range(numClasses).Named("c");

            // Variables
            FeaturesVar = Variable.Array(Variable.Array<bool>(FeatureRange), InstanceRange).Named("features");
            LabelsVar = Variable.Array<int>(InstanceRange).Named("labels");
            LabelsVar.SetValueRange(ClassRange);

            // Feature probabilities per class
            FeatureProbVar = Variable.Array(Variable.Array<double>(FeatureRange), ClassRange).Named("featureProb");
            using (Variable.ForEach(ClassRange)) {
                var fp = Variable.Array<double>(FeatureRange);
                using (Variable.ForEach(FeatureRange)) {
                    fp[FeatureRange] = Variable.Beta(priors.FeatureAlpha, priors.FeatureBeta);
                }
                FeatureProbVar[ClassRange] = fp;
            }

            // Class Dirichlet prior
            double[] alphaDir = Enumerable.Repeat(priors.ClassAlpha, numClasses).ToArray();
            ClassProbsDirichlet = Variable.Dirichlet(alphaDir).Named("classProbs");

            // Likelihood wiring: placeholders for label masks/observed labels; set ObservedValue later
            hasLabelConstVar = Variable.Array<bool>(InstanceRange).Named("hasLabel");
            observedLabelsConstVar = Variable.Array<int>(InstanceRange).Named("observedLabels");

            using (Variable.ForEach(InstanceRange)) {
                LabelsVar[InstanceRange] = Variable.Discrete(ClassProbsDirichlet);
                using (Variable.If(hasLabelConstVar[InstanceRange])) {
                    Variable.ConstrainEqual(LabelsVar[InstanceRange], observedLabelsConstVar[InstanceRange]);
                }
                using (Variable.Switch(LabelsVar[InstanceRange])) {
                    using (Variable.ForEach(FeatureRange)) {
                        FeaturesVar[InstanceRange][FeatureRange] = Variable.Bernoulli(FeatureProbVar[LabelsVar[InstanceRange]][FeatureRange]);
                    }
                }
            }

            engine = new InferenceEngine();
        }

        public void AttachData(bool[][] features, bool[] hasLabel, int[] observedLabels) {
            if (features.Length != InstanceRange.SizeAsInt) throw new ArgumentException("features length mismatch");
            if (hasLabel.Length != InstanceRange.SizeAsInt) throw new ArgumentException("hasLabel length mismatch");
            if (observedLabels.Length != InstanceRange.SizeAsInt) throw new ArgumentException("observedLabels length mismatch");

            FeaturesVar.ObservedValue = features;
            // replace the constant arrays by setting values in the generated constants
            hasLabelConstVar.ObservedValue = hasLabel;
            observedLabelsConstVar.ObservedValue = observedLabels;
        }

        /// <summary>
        /// Run inference. Optionally infer labels (Discrete[]) only if requested.
        /// </summary>
        public (Beta[][] featureProb, Discrete[]? labels, Microsoft.ML.Probabilistic.Distributions.Dirichlet classProbs) Infer(bool inferLabels = false) {
            var inferredFeatureProb = engine.Infer<Beta[][]>(FeatureProbVar);
            Discrete[]? inferredLabels = null;
            if (inferLabels) {
                inferredLabels = engine.Infer<Discrete[]>(LabelsVar);
            }
            var inferredClassProbs = engine.Infer<Microsoft.ML.Probabilistic.Distributions.Dirichlet>(ClassProbsDirichlet);
            return (inferredFeatureProb, inferredLabels, inferredClassProbs);
        }
    }
}
