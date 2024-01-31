
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;
import shared.filt.LabelSplitFilter;
import shared.reader.ArffDataSetReader;
import shared.reader.DataSetLabelBinarySeperator;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Scanner;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying email into spam and not spam based on word/character
 * frequencies
 *
 * @author James Liu
 * @version 1.0
 */
public class ML2 {
    private static Instance[] instances;

    private static int inputLayer, hiddenLayer, outputLayer = 1, trainingIterations;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set;

    private static BackPropagationNetwork network;
    private static NeuralNetworkOptimizationProblem nnop;

    private static OptimizationAlgorithm oa;
    private static String oaName;
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");
    private static boolean printItermediate;

    public static void main(String[] args) {
        String alg = "rhc";
        String datapath = new File("").getAbsolutePath() + "/src/adult.10%data.arff";
        String hiddenNodes = "54";
        String iterations = "500";
        String cooling = "0.95";
        String temperature = "100000000";
        String popsize = "200";
        String mate = "100";
        String mutate = "10";

        instances = initializeInstances(datapath);
        set = new DataSet(instances);

        inputLayer = instances[0].getData().size() - 1;
        hiddenLayer = Integer.parseInt(hiddenNodes);
        trainingIterations = Integer.parseInt(iterations);
        LabelSplitFilter lsf = new LabelSplitFilter();
        lsf.filter(set);
        DataSetLabelBinarySeperator.seperateLabels(set);
        network = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
        nnop = new NeuralNetworkOptimizationProblem(set, network, measure);

        if(alg.equals("rhc"))
        {
            System.out.println(oaName = "Randomized Hill Climbing");
            oa = new RandomizedHillClimbing(nnop);
        }
        else if(alg.equals("sa"))
        {
            System.out.println(oaName = "Simulated Annealing");
            /*
            if(args.length < 6)
            {
                System.out.println("Provide the initial temperature and cooling factor");
                System.exit(0);
            }
            */
            oa = new SimulatedAnnealing(Double.parseDouble(temperature), Double.parseDouble(cooling), nnop);
        }
        else if(alg.equals("ga"))
        {
            System.out.println(oaName = "Genetic Algorithm");
            /*
            if(args.length < 7)
            {
                System.out.println("Provide the initial population size, amount to mate per iteration, and amount to mutate per generation");
                System.exit(0);
            }
            */
            oa = new StandardGeneticAlgorithm(Integer.parseInt(popsize), Integer.parseInt(mate), Integer.parseInt(mutate), nnop);
        }
        else
        {
            System.out.println("Optimization algorithm not recognized");
            System.exit(0);
        }

        double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
        train(oa, network);
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);

        Instance optimalInstance = oa.getOptimal();
        network.setWeights(optimalInstance.getData());

        double predicted, actual;
        start = System.nanoTime();
        for(int j = 0; j < instances.length; j++) {
            network.setInputValues(instances[j].getData());
            network.run();

            predicted = Double.parseDouble(instances[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        System.out.println("\nResults for " + oaName + ": \nCorrectly classified " + correct + " instances." +
                "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n");
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network) {
        System.out.println("\nError results\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0, predicted, actual;
            int correct = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);

                predicted = Double.parseDouble(instances[j].getLabel().toString());
                actual = Double.parseDouble(network.getOutputValues().toString());

                if(Math.abs(predicted - actual) < 0.5)
                {
                    correct++;
                }
            }

            System.out.println(df.format(error) + ", " + (100 * ((double)correct/(double) instances.length)) + "%");
        }
    }

    private static Instance[] initializeInstances(String file) {
        ArffDataSetReader arffDSreader = new ArffDataSetReader(file);

        try
        {
            return arffDSreader.read().getInstances();

        }
        catch(Exception e)
        {
            e.printStackTrace();
            System.exit(0);
        }

        return null;
    }
}