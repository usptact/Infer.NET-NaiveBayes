namespace NaiveBayes {
    public class Priors {
        // Beta prior parameters for feature Bernoullis (per-class)
        public double FeatureAlpha { get; set; } = 1.0;
        public double FeatureBeta { get; set; } = 1.0;

        // Dirichlet alpha for class probabilities (symmetric)
        public double ClassAlpha { get; set; } = 1.0;

        public Priors() { }

        public Priors(double featureAlpha, double featureBeta, double classAlpha) {
            FeatureAlpha = featureAlpha;
            FeatureBeta = featureBeta;
            ClassAlpha = classAlpha;
        }
    }
}