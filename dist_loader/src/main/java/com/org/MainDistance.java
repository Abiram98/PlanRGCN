package com.org;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

import org.apache.commons.lang3.time.StopWatch;

public class MainDistance {
    public static void main(String[] args) {
        QueryLogReader reader = new QueryLogReader("/data/DBpedia2016_0_1_10_weight_loss/train_sampled.tsv");
        try {
            reader.read();
        } catch (IOException e) {
            e.printStackTrace();
        }
        ArrayList<String> queryIds = reader.queryIDs;
        System.out.println(queryIds.get(0));

        DistanceLoader distLoader = new DistanceLoader();
        File f = new File("/data/dbpedia_dist2bak/distances");
        try {
            FileWriter fw = new FileWriter("/data/dbpedia_dist2bak/combinations/comb.csv");
            int i = 0;
            StopWatch watch = new StopWatch();
            watch.start();
            for (String s : f.list()) {
                distLoader.loadFile(f.getPath() + "/" + s);
                i++;
                System.out.println("File " + i + ": " + ((double) watch.getTime(TimeUnit.MILLISECONDS)) / 1000 + "\n");
            }
            double total = 0;
            int missingPairs = 0;
            for (int j = 0; j < queryIds.size(); j++) {
                for (int v = j; v < queryIds.size(); v++) {
                    double val = distLoader.get(queryIds.get(j), queryIds.get(v), fw);
                    if (val == -1) {
                        missingPairs++;
                    } else {
                        total += val;
                    }
                }
            }
            fw.close();
            System.out.println("Total amount time spent on dist calulcations on training");
            System.out.println(total);
            System.out.println("Missing Pairs");
            System.out.println(missingPairs);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
