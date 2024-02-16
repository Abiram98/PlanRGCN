package com.org;

import java.io.IOException;


public class MainDistance {
    public static void main(String[] args) {

        switch (args[0]) {
            case "comb_creator":
                combCreator();
                break;

            default:
                System.out.println("Default Task is distance matrix calculator");
                if (args.length < 3) {
                    System.out.println("Missing Arguments");
                    System.exit(-1);
                }

                String queryLog = args[0];
                String distanceDist = args[1];
                String missPairs = args[2];

                DistanceLoader.runComputationCalculator(queryLog, distanceDist, missPairs);
                break;
        }
    }

    public static void combCreator() {
        System.exit(-1);
        // file : /data/DBpedia2016_0_1_10_weight_loss/train_sampled.tsv
        // combFile: /data/dbpedia_dist2bak/combinations/comb1.csv
        //
        CombinationCreator cr = new CombinationCreator("/data/DBpedia2016_0_1_10_weight_loss/train_sampled.tsv",
                "/data/dbpedia_dist2bak/combinations/comb1.csv", "/data/dbpedia_dist2bak/combinations2");
        try {
            cr.read();
            cr.process();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
