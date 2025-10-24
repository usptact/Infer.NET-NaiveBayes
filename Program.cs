using System;
using CommandLine;

namespace NaiveBayes {
    class Program {
        static void Main(string[] args) {
            var parser = new Parser(with => with.HelpWriter = Console.Out);
            var result = parser.ParseArguments<TrainCommand, PredictCommand>(args)
                .MapResult(
                    (TrainCommand tc) => tc.Run(),
                    (PredictCommand pc) => pc.Run(),
                    errs => 1);
            Environment.Exit(result);
        }
    }
}
