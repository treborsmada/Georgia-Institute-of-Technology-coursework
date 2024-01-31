import java.io.File;

import func.nn.NeuralNetwork;
import func.nn.backprop.*;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.example.NeuralNetworkOptimizationProblem;
import shared.*;
import shared.filt.ContinuousToDiscreteFilter;
import shared.filt.LabelSplitFilter;
import shared.reader.ArffDataSetReader;
import shared.reader.DataSetLabelBinarySeperator;
import shared.reader.DataSetReader;
import func.NeuralNetworkClassifier;
import shared.tester.AccuracyTestMetric;
import shared.tester.NeuralNetworkTester;
import shared.tester.PrecisionTestMetric;
import shared.tester.TestMetric;

/**
 * Created by Robert Adams on 10/14/2017.
 */
public class AdultNN {

    public static ErrorMeasure measure = new SumOfSquaresError();
    public static OptimizationAlgorithm oa;

    public static void main(String[] args) throws Exception{
        DataSetReader dsr = new ArffDataSetReader(new File("").getAbsolutePath() + "/src/adult.10%data.arff");
        DataSet ds = dsr.read();
        LabelSplitFilter lsf = new LabelSplitFilter();
        lsf.filter(ds);
        DataSetLabelBinarySeperator.seperateLabels(ds);
        System.out.println(ds);
        System.out.println(new DataSetDescription(ds));

        BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
        BackPropagationNetwork network = factory.createClassificationNetwork(
                new int[] { 104, 54, 2});
        BatchBackPropagationTrainer proptrain = new BatchBackPropagationTrainer(ds, network, new SumOfSquaresError(), new StandardUpdateRule(0.1, 0.1));
        FixedIterationTrainer fit = new FixedIterationTrainer(proptrain, 500);
        fit.train();
        TestMetric metric = new AccuracyTestMetric();
        NeuralNetworkTester tester = new NeuralNetworkTester(network, metric);
        tester.test(ds.getInstances());
        metric.printResults();
        NeuralNetworkOptimizationProblem NNAdult = new NeuralNetworkOptimizationProblem(ds, network, measure);
        oa = new RandomizedHillClimbing(NNAdult);
        fit = new FixedIterationTrainer(oa, 1000);
        fit.train();
        metric = new AccuracyTestMetric();
        tester = new NeuralNetworkTester(network, metric);
        tester.test(ds.getInstances());
        metric.printResults();

    }
}
