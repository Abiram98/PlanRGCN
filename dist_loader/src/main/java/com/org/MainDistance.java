package com.org;

public class MainDistance {
    public static void main(String[] args) {
        DistanceLoader distLoader = new DistanceLoader();
        distLoader.loadFile("/data/wikidata_dists/distances/comb_51465.json");
        System.exit(-1);
        String task = args[0];
        switch (task) {
            case "test": {
                break;
            }
            default: {
                break;
            }
        }

    }
}
