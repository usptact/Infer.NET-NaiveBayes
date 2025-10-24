using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace NaiveBayes 
{
    public class DataLoader 
    {
        public static (bool[][] features, bool[] hasLabel, int[] observedLabels) LoadFromCsv(string filePath) 
        {
            var lines = File.ReadAllLines(filePath).Skip(1).ToList(); // Skip header row
            var numInstances = lines.Count;
            var features = new List<bool[]>();
            var hasLabel = new List<bool>();
            var observedLabels = new List<int>();

            foreach (var line in lines) 
            {
                var parts = line.Split(',');
                // Last column is the label (can be empty)
                var labelStr = parts[parts.Length - 1].Trim();
                
                // Process features (all columns except last)
                var instanceFeatures = new bool[parts.Length - 1];
                for (int i = 0; i < parts.Length - 1; i++) 
                {
                    // Convert feature value to bool (assuming 1/0 or true/false in CSV)
                    var featureStr = parts[i].Trim().ToLower();
                    instanceFeatures[i] = featureStr == "1" || featureStr == "true";
                }
                features.Add(instanceFeatures);

                // Process label
                if (string.IsNullOrWhiteSpace(labelStr)) 
                {
                    // Missing label case
                    hasLabel.Add(false);
                    observedLabels.Add(0); // Placeholder, won't be used
                }
                else 
                {
                    // Known label case
                    hasLabel.Add(true);
                    observedLabels.Add(int.Parse(labelStr));
                }
            }

            return (features.ToArray(), hasLabel.ToArray(), observedLabels.ToArray());
        }
    }
}