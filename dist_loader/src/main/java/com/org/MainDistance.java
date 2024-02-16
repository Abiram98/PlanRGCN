package com.org;

import java.io.IOException;


public class MainDistance {
    public static void main(String[] args) {

        switch (args[0]) {
            case "comb_creator":
                combCreator();
                break;

            default:
                // java -jar -Xms100G -Xmx100G target/dist_loader-1.0-SNAPSHOT.jar
                // /data/wikidata_0_1_10_v3_weight_loss/train_sampled.tsv
                // /data/wikidata_dists/distances /data/wikidata_dists/combinations/comb1.csv
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
        // file : /data/wikidata_0_1_10_v3_weight_loss/train_sampled.tsv
        // combFile: /data/wikidata_dists/combinations/comb1.csv
        // distance: /data/wikidata_dists/distances

        // file : /data/DBpedia2016_0_1_10_weight_loss/train_sampled.tsv
        // combFile: /data/dbpedia_dist2bak/combinations/comb1.csv
        // combDir: "/data/dbpedia_dist2bak/combinations2"
        CombinationCreator cr = new CombinationCreator("/data/wikidata_0_1_10_v3_weight_loss/train_sampled.tsv",
                "/data/wikidata_dists/combinations/comb1.csv",
                "/data/wikidata_dists/combinations2");
        /*
         * CombinationCreator cr = new
         * CombinationCreator("/data/DBpedia2016_0_1_10_weight_loss/train_sampled.tsv",
         * "/data/dbpedia_dist2bak/combinations/comb1.csv",
         * "/data/dbpedia_dist2bak/combinations2");
         */
        try {
            cr.read();
            cr.process();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
